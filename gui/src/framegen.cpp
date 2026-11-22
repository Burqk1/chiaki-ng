// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include "framegen.h"

#include <libplacebo/shaders/custom.h>

#include <algorithm>
#include <cmath>

namespace {

// A block whose mean luma difference stays below this is unchanged: the
// stream repeated the previous picture (30 fps game in a 60 fps stream).
constexpr float kRepeatMaxBlockDiff = 0.004f;
// Mean difference above this is a scene cut: do not interpolate across it.
constexpr float kSceneCutMeanDiff = 0.18f;

// Quarter-resolution blocks are 8x8 (32x32 full-resolution pixels). The
// symmetric search radius is 12 quarter pixels per side, i.e. up to 96 full
// pixels of motion between the two real pictures.
const char *kHeaderCommon = R"GLSL(
const vec3 kLumaW = vec3(0.2126, 0.7152, 0.0722);
)GLSL";

const char *kBodyLuma = R"GLSL(
ivec2 q = ivec2(gl_GlobalInvocationID.xy);
if (q.x < qsize.x && q.y < qsize.y) {
    vec2 inv = 1.0 / vec2(fsize);
    vec2 base = vec2(q) * 4.0;
    float l = 0.0;
    for (int j = 0; j < 4; j++) {
        vec2 o = vec2(float(j & 1) * 2.0 + 1.0, float(j >> 1) * 2.0 + 1.0);
        l += dot(textureLod(src_tex, (base + o) * inv, 0.0).rgb, kLumaW);
    }
    imageStore(dst_img, q, vec4(l * 0.25));
}
)GLSL";

const char *kBodyDiff = R"GLSL(
ivec2 b = ivec2(gl_GlobalInvocationID.xy);
if (b.x < bsize.x && b.y < bsize.y) {
    float s = 0.0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            ivec2 p = min(b * 8 + ivec2(x, y), qsize - ivec2(1));
            s += abs(texelFetch(luma_a, p, 0).r - texelFetch(luma_b, p, 0).r);
        }
    }
    imageStore(diff_img, b, vec4(s / 64.0));
}
)GLSL";

// Symmetric (bilateral) block matching for the midpoint picture: for a block
// of the *midpoint*, find v so that prev(p - v) matches cur(p + v). Searching
// from the midpoint avoids the holes and overlaps of forward projection.
const char *kHeaderMotion = R"GLSL(
float blockCost(vec2 c0, vec2 v) {
    vec2 inv = 1.0 / vec2(qsize);
    float s = 0.0;
    for (int y = 0; y < 8; y += 1) {
        for (int x = 0; x < 8; x += 1) {
            vec2 p = c0 + vec2(float(x), float(y)) + 0.5;
            s += abs(textureLod(luma_a, (p - v) * inv, 0.0).r -
                     textureLod(luma_b, (p + v) * inv, 0.0).r);
        }
    }
    return s / 64.0 + 0.0004 * length(v);
}
)GLSL";

const char *kBodyMotion = R"GLSL(
ivec2 b = ivec2(gl_GlobalInvocationID.xy);
if (b.x < bsize.x && b.y < bsize.y) {
    vec2 c0 = vec2(b * 8);
    // The zero vector gets a small head start: flat areas stay still.
    float best = blockCost(c0, vec2(0.0)) - 0.002;
    vec2 bv = vec2(0.0);
    for (int dy = -12; dy <= 12; dy++) {
        for (int dx = -12; dx <= 12; dx++) {
            vec2 v = vec2(float(dx), float(dy));
            float c = blockCost(c0, v);
            if (c < best) { best = c; bv = v; }
        }
    }
    // Sub-pixel refinement: half, then quarter of a quarter pixel.
    for (int level = 0; level < 2; level++) {
        float stp = level == 0 ? 0.5 : 0.25;
        vec2 centre = bv;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0)
                    continue;
                vec2 v = centre + vec2(float(dx), float(dy)) * stp;
                float c = blockCost(c0, v);
                if (c < best) { best = c; bv = v; }
            }
        }
    }
    imageStore(flow_img, b, vec4(bv, best, 0.0));
}
)GLSL";

// 3x3 vector median: removes isolated wrong block vectors.
const char *kBodyMedian = R"GLSL(
ivec2 b = ivec2(gl_GlobalInvocationID.xy);
if (b.x < bsize.x && b.y < bsize.y) {
    vec2 vs[9];
    for (int k = 0; k < 9; k++) {
        ivec2 c = clamp(b + ivec2(k % 3 - 1, k / 3 - 1), ivec2(0), bsize - ivec2(1));
        vs[k] = texelFetch(flow_tex, c, 0).xy;
    }
    int bi = 4;
    float bd = 1e9;
    for (int i = 0; i < 9; i++) {
        float d = 0.0;
        for (int j = 0; j < 9; j++)
            d += length(vs[i] - vs[j]);
        if (d < bd) { bd = d; bi = i; }
    }
    vec4 own = texelFetch(flow_tex, b, 0);
    imageStore(flow_out, b, vec4(vs[bi], own.z, 0.0));
}
)GLSL";

// Per pixel, pick the best of the 3x3 neighbouring block vectors (and zero)
// by a small luma patch match, so motion boundaries follow the picture rather
// than the block grid (after Veyra's FlowRefine), then blend both pictures
// along the chosen vector.
const char *kBodyInterp = R"GLSL(
ivec2 p = ivec2(gl_GlobalInvocationID.xy);
if (p.x < fsize.x && p.y < fsize.y) {
    vec2 pf = vec2(p) + 0.5;
    vec2 pq = pf * 0.25;
    vec2 qinv = 1.0 / vec2(qsize);
    ivec2 cell = clamp(ivec2(floor(pq / 8.0)), ivec2(0), bsize - ivec2(1));
    float best = 1e9;
    vec2 bv = vec2(0.0);
    for (int k = 0; k < 10; k++) {
        vec2 v = vec2(0.0);
        if (k < 9) {
            ivec2 c = clamp(cell + ivec2(k % 3 - 1, k / 3 - 1), ivec2(0), bsize - ivec2(1));
            v = texelFetch(flow_tex, c, 0).xy;
        }
        float c = 0.0;
        for (int j = 0; j < 5; j++) {
            vec2 o = j == 0 ? vec2(0.0) :
                     j == 1 ? vec2(1.0, 0.0) :
                     j == 2 ? vec2(-1.0, 0.0) :
                     j == 3 ? vec2(0.0, 1.0) : vec2(0.0, -1.0);
            c += abs(textureLod(luma_a, (pq + o - v) * qinv, 0.0).r -
                     textureLod(luma_b, (pq + o + v) * qinv, 0.0).r);
        }
        c *= 0.2;
        if (k != 4)
            c += 0.001; // own cell wins ties
        if (c < best) { best = c; bv = v; }
    }
    vec2 vf = bv * 4.0;
    vec2 finv = 1.0 / vec2(fsize);
    vec3 a = textureLod(rgb_a, (pf - vf) * finv, 0.0).rgb;
    vec3 bb = textureLod(rgb_b, (pf + vf) * finv, 0.0).rgb;
    imageStore(out_img, p, vec4(0.5 * (a + bb), 1.0));
}
)GLSL";

pl_shader_desc sampledDesc(const char *name, pl_tex tex, bool linear)
{
	pl_shader_desc d = {};
	d.desc.name = name;
	d.desc.type = PL_DESC_SAMPLED_TEX;
	d.binding.object = tex;
	d.binding.address_mode = PL_TEX_ADDRESS_CLAMP;
	d.binding.sample_mode = linear ? PL_TEX_SAMPLE_LINEAR : PL_TEX_SAMPLE_NEAREST;
	return d;
}

pl_shader_desc storageDesc(const char *name, pl_tex tex)
{
	pl_shader_desc d = {};
	d.desc.name = name;
	d.desc.type = PL_DESC_STORAGE_IMG;
	d.desc.access = PL_DESC_ACCESS_WRITEONLY;
	d.binding.object = tex;
	return d;
}

pl_shader_var ivec2Var(const char *name, const int *data)
{
	pl_shader_var v = {};
	v.var = pl_var_ivec2(name);
	v.data = data;
	v.dynamic = true;
	return v;
}

bool runCompute(pl_dispatch dp, const char *header, const char *body,
                const std::vector<pl_shader_desc> &descs,
                const std::vector<pl_shader_var> &vars, int w, int h)
{
	pl_shader sh = pl_dispatch_begin(dp);
	if (!sh)
		return false;
	pl_custom_shader cs = {};
	cs.header = header;
	cs.body = body;
	cs.input = PL_SHADER_SIG_NONE;
	cs.output = PL_SHADER_SIG_NONE;
	cs.descriptors = descs.data();
	cs.num_descriptors = static_cast<int>(descs.size());
	cs.variables = vars.data();
	cs.num_variables = static_cast<int>(vars.size());
	cs.compute = true;
	cs.compute_group_size[0] = 16;
	cs.compute_group_size[1] = 16;
	if (!pl_shader_custom(sh, &cs)) {
		pl_dispatch_abort(dp, &sh);
		return false;
	}
	pl_dispatch_compute_params params = {};
	params.shader = &sh;
	params.dispatch_size[0] = (w + 15) / 16;
	params.dispatch_size[1] = (h + 15) / 16;
	params.dispatch_size[2] = 1;
	return pl_dispatch_compute(dp, &params);
}

pl_tex createTex(pl_gpu gpu, pl_fmt fmt, int w, int h, bool renderable, bool host_readable)
{
	pl_tex_params p = {};
	p.w = w;
	p.h = h;
	p.format = fmt;
	p.sampleable = !host_readable;
	p.storable = true;
	p.renderable = renderable;
	p.host_readable = host_readable;
	return pl_tex_create(gpu, &p);
}

} // namespace

FrameGenerator::FrameGenerator(pl_log log, pl_gpu gpu)
	: log_(log), gpu_(gpu)
{
	if (!gpu_ || !gpu_->glsl.compute)
		return;
	renderer_ = pl_renderer_create(log_, gpu_);
	dispatch_ = pl_dispatch_create(log_, gpu_);
	const auto rgba_caps = static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_STORABLE |
	                                                PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_LINEAR);
	const auto r_caps = static_cast<pl_fmt_caps>(PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_STORABLE |
	                                             PL_FMT_CAP_LINEAR);
	const auto diff_caps = static_cast<pl_fmt_caps>(PL_FMT_CAP_STORABLE | PL_FMT_CAP_HOST_READABLE);
	fmt_rgba_ = pl_find_fmt(gpu_, PL_FMT_FLOAT, 4, 16, 0, rgba_caps);
	fmt_r_ = pl_find_fmt(gpu_, PL_FMT_FLOAT, 1, 16, 0, r_caps);
	fmt_diff_ = pl_find_fmt(gpu_, PL_FMT_FLOAT, 1, 32, 32, diff_caps);
	usable_ = renderer_ && dispatch_ && fmt_rgba_ && fmt_r_ && fmt_diff_ &&
	          fmt_rgba_->glsl_format && fmt_r_->glsl_format && fmt_diff_->glsl_format;
}

FrameGenerator::~FrameGenerator()
{
	destroyResources();
	pl_dispatch_destroy(&dispatch_);
	pl_renderer_destroy(&renderer_);
}

void FrameGenerator::destroyResources()
{
	for (auto &t : rgb_)
		pl_tex_destroy(gpu_, &t);
	for (auto &t : luma_)
		pl_tex_destroy(gpu_, &t);
	pl_tex_destroy(gpu_, &mid_);
	pl_tex_destroy(gpu_, &flow_);
	pl_tex_destroy(gpu_, &flow_smooth_);
	pl_tex_destroy(gpu_, &diff_);
	w_ = h_ = qw_ = qh_ = bw_ = bh_ = 0;
	reset();
}

bool FrameGenerator::ensureResources(int w, int h)
{
	if (w == w_ && h == h_ && mid_)
		return true;
	destroyResources();
	w_ = w;
	h_ = h;
	qw_ = (w + 3) / 4;
	qh_ = (h + 3) / 4;
	bw_ = (qw_ + 7) / 8;
	bh_ = (qh_ + 7) / 8;
	bool ok = true;
	for (int i = 0; i < kSlots; i++) {
		rgb_[i] = createTex(gpu_, fmt_rgba_, w_, h_, true, false);
		luma_[i] = createTex(gpu_, fmt_r_, qw_, qh_, false, false);
		ok = ok && rgb_[i] && luma_[i];
	}
	mid_ = createTex(gpu_, fmt_rgba_, w_, h_, true, false);
	flow_ = createTex(gpu_, fmt_rgba_, bw_, bh_, false, false);
	flow_smooth_ = createTex(gpu_, fmt_rgba_, bw_, bh_, false, false);
	diff_ = createTex(gpu_, fmt_diff_, bw_, bh_, false, true);
	diff_host_.assign(static_cast<size_t>(bw_) * bh_, 0.0f);
	ok = ok && mid_ && flow_ && flow_smooth_ && diff_;
	if (!ok)
		destroyResources();
	return ok;
}

void FrameGenerator::reset()
{
	last_ = -1;
	hold_ = -1;
	prev_was_repeat_ = false;
}

pl_frame FrameGenerator::frameFor(pl_tex tex) const
{
	pl_frame f = {};
	f.num_planes = 1;
	f.planes[0].texture = tex;
	f.planes[0].components = 3;
	f.planes[0].component_mapping[0] = PL_CHANNEL_R;
	f.planes[0].component_mapping[1] = PL_CHANNEL_G;
	f.planes[0].component_mapping[2] = PL_CHANNEL_B;
	f.planes[0].component_mapping[3] = PL_CHANNEL_A;
	f.repr = pl_color_repr_rgb;
	f.color = color_;
	f.crop.x0 = 0.0f;
	f.crop.y0 = 0.0f;
	f.crop.x1 = static_cast<float>(w_);
	f.crop.y1 = static_cast<float>(h_);
	return f;
}

int FrameGenerator::freeSlot() const
{
	for (int i = 0; i < kSlots; i++) {
		if (i != last_ && i != hold_)
			return i;
	}
	return 0;
}

bool FrameGenerator::convert(const pl_frame &src, pl_tex dst)
{
	pl_frame target = frameFor(dst);
	pl_render_params params = pl_render_default_params;
	return pl_render_image(renderer_, &src, &target, &params);
}

bool FrameGenerator::computeLuma(pl_tex rgb, pl_tex luma)
{
	const int fsize[2] = { w_, h_ };
	const int qsize[2] = { qw_, qh_ };
	return runCompute(dispatch_, kHeaderCommon, kBodyLuma,
	                  { sampledDesc("src_tex", rgb, true), storageDesc("dst_img", luma) },
	                  { ivec2Var("fsize", fsize), ivec2Var("qsize", qsize) },
	                  qw_, qh_);
}

bool FrameGenerator::computeDiff(pl_tex luma_a, pl_tex luma_b, float *max_diff, float *mean_diff)
{
	const int qsize[2] = { qw_, qh_ };
	const int bsize[2] = { bw_, bh_ };
	if (!runCompute(dispatch_, nullptr, kBodyDiff,
	                { sampledDesc("luma_a", luma_a, false), sampledDesc("luma_b", luma_b, false),
	                  storageDesc("diff_img", diff_) },
	                { ivec2Var("qsize", qsize), ivec2Var("bsize", bsize) },
	                bw_, bh_))
		return false;
	pl_tex_transfer_params tp = {};
	tp.tex = diff_;
	tp.ptr = diff_host_.data();
	if (!pl_tex_download(gpu_, &tp))
		return false;
	float mx = 0.0f;
	double sum = 0.0;
	for (float v : diff_host_) {
		if (!std::isfinite(v))
			continue;
		mx = std::max(mx, v);
		sum += v;
	}
	*max_diff = mx;
	*mean_diff = diff_host_.empty() ? 0.0f : static_cast<float>(sum / diff_host_.size());
	return true;
}

bool FrameGenerator::interpolate(int prev_idx, int cur_idx)
{
	const int fsize[2] = { w_, h_ };
	const int qsize[2] = { qw_, qh_ };
	const int bsize[2] = { bw_, bh_ };
	if (!runCompute(dispatch_, kHeaderMotion, kBodyMotion,
	                { sampledDesc("luma_a", luma_[prev_idx], true), sampledDesc("luma_b", luma_[cur_idx], true),
	                  storageDesc("flow_img", flow_) },
	                { ivec2Var("qsize", qsize), ivec2Var("bsize", bsize) },
	                bw_, bh_))
		return false;
	if (!runCompute(dispatch_, nullptr, kBodyMedian,
	                { sampledDesc("flow_tex", flow_, false), storageDesc("flow_out", flow_smooth_) },
	                { ivec2Var("bsize", bsize) },
	                bw_, bh_))
		return false;
	return runCompute(dispatch_, nullptr, kBodyInterp,
	                  { sampledDesc("flow_tex", flow_smooth_, false),
	                    sampledDesc("luma_a", luma_[prev_idx], true), sampledDesc("luma_b", luma_[cur_idx], true),
	                    sampledDesc("rgb_a", rgb_[prev_idx], true), sampledDesc("rgb_b", rgb_[cur_idx], true),
	                    storageDesc("out_img", mid_) },
	                  { ivec2Var("fsize", fsize), ivec2Var("qsize", qsize), ivec2Var("bsize", bsize) },
	                  w_, h_);
}

const pl_frame *FrameGenerator::process(const pl_frame &src)
{
	if (!usable_ || src.num_planes <= 0)
		return nullptr;
	if (src.repr.dovi) // Dolby Vision reshaping must stay in the main renderer
		return nullptr;
	const int w = static_cast<int>(std::lround(std::fabs(pl_rect_w(src.crop))));
	const int h = static_cast<int>(std::lround(std::fabs(pl_rect_h(src.crop))));
	if (w < 64 || h < 64)
		return nullptr;
	if (!ensureResources(w, h)) {
		usable_ = false;
		return nullptr;
	}
	if (!pl_color_space_equal(&src.color, &color_)) {
		reset();
		color_ = src.color;
	}

	stats_.frames_in++;
	const int slot = freeSlot();
	if (!convert(src, rgb_[slot]) || !computeLuma(rgb_[slot], luma_[slot])) {
		reset();
		return nullptr;
	}

	if (last_ < 0) {
		last_ = slot;
		hold_ = -1;
		prev_was_repeat_ = false;
		stats_.passthrough++;
		out_ = frameFor(rgb_[slot]);
		return &out_;
	}

	float max_diff = 0.0f, mean_diff = 0.0f;
	if (!computeDiff(luma_[last_], luma_[slot], &max_diff, &mean_diff)) {
		reset();
		return nullptr;
	}

	if (max_diff < kRepeatMaxBlockDiff) {
		// Repeated picture: show the real picture we held back, or keep the last one.
		stats_.repeats++;
		prev_was_repeat_ = true;
		int show = last_;
		if (hold_ >= 0) {
			show = hold_;
			hold_ = -1;
			stats_.held_shown++;
		}
		out_ = frameFor(rgb_[show]);
		return &out_;
	}

	const bool cut = mean_diff > kSceneCutMeanDiff;
	if (prev_was_repeat_ && !cut && hold_ < 0 && interpolate(last_, slot)) {
		// 30 fps cadence: show the midpoint now and the real picture on the next (repeated) tick.
		stats_.generated++;
		last_ = slot;
		hold_ = slot;
		prev_was_repeat_ = false;
		out_ = frameFor(mid_);
		return &out_;
	}

	if (cut)
		stats_.scene_cuts++;
	stats_.passthrough++;
	last_ = slot;
	hold_ = -1;
	prev_was_repeat_ = false;
	out_ = frameFor(rgb_[slot]);
	return &out_;
}
