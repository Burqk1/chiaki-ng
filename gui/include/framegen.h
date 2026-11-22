// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#ifndef CHIAKI_FRAMEGEN_H
#define CHIAKI_FRAMEGEN_H

#include <libplacebo/renderer.h>
#include <libplacebo/dispatch.h>

#include <array>
#include <cstdint>
#include <vector>

/**
 * GPU frame generation for game streams.
 *
 * PS5 streams arrive at a fixed 60 fps. A game that renders at 30 fps shows up
 * as every picture being sent twice. FrameGenerator detects these repeats and
 * replaces each repeat with a motion-compensated midpoint between the two real
 * pictures around it, which turns the 30 fps cadence into 60 distinct pictures.
 *
 * Because the midpoint needs the *next* real picture, a real picture that
 * follows a repeat is shown one stream interval later (≈16.7 ms at 60 fps).
 * Content that already changes every frame (60 fps games) passes through with
 * no added latency.
 *
 * Motion is estimated with symmetric block matching on a quarter resolution
 * luma copy and refined per pixel among neighbouring block vectors (idea from
 * Veyra's FlowRefine), all in libplacebo compute shaders, so it runs on any
 * GPU libplacebo supports with storage images.
 */
class FrameGenerator
{
public:
	struct Stats
	{
		uint64_t frames_in = 0;
		uint64_t repeats = 0;
		uint64_t generated = 0;
		uint64_t held_shown = 0;
		uint64_t scene_cuts = 0;
		uint64_t passthrough = 0;
	};

	FrameGenerator(pl_log log, pl_gpu gpu);
	~FrameGenerator();
	FrameGenerator(const FrameGenerator &) = delete;
	FrameGenerator &operator=(const FrameGenerator &) = delete;

	/** False if the GPU lacks the formats/compute support needed. */
	bool usable() const { return usable_; }

	/**
	 * Feed one newly decoded frame and get the frame to present for this
	 * stream tick. Returns nullptr if generation is not possible for this frame
	 * (caller should present `src` directly). The returned frame stays valid
	 * until the next call to process() or reset().
	 */
	const pl_frame *process(const pl_frame &src);

	/** Forget all history (seek, reconnect, resolution change). */
	void reset();

	const Stats &stats() const { return stats_; }

private:
	bool ensureResources(int w, int h);
	void destroyResources();
	bool convert(const pl_frame &src, pl_tex dst);
	bool computeLuma(pl_tex rgb, pl_tex luma);
	bool computeDiff(pl_tex luma_a, pl_tex luma_b, float *max_diff, float *mean_diff);
	bool interpolate(int prev_idx, int cur_idx);
	pl_frame frameFor(pl_tex tex) const;
	int freeSlot() const;

	pl_log log_ = nullptr;
	pl_gpu gpu_ = nullptr;
	pl_renderer renderer_ = nullptr;
	pl_dispatch dispatch_ = nullptr;
	bool usable_ = false;

	pl_fmt fmt_rgba_ = nullptr;
	pl_fmt fmt_r_ = nullptr;
	pl_fmt fmt_rg_ = nullptr;
	pl_fmt fmt_diff_ = nullptr;

	int w_ = 0, h_ = 0;      // full resolution
	int qw_ = 0, qh_ = 0;    // quarter resolution (luma)
	int bw_ = 0, bh_ = 0;    // block grid (8x8 quarter pixels)

	static constexpr int kSlots = 3;
	std::array<pl_tex, kSlots> rgb_{};
	std::array<pl_tex, kSlots> luma_{};
	pl_tex mid_ = nullptr;
	pl_tex flow_ = nullptr;
	pl_tex flow_smooth_ = nullptr;
	pl_tex diff_ = nullptr;
	std::vector<float> diff_host_;

	int last_ = -1;          // slot of the last distinct picture
	int hold_ = -1;          // slot of a real picture waiting for the next tick
	bool prev_was_repeat_ = false;

	pl_color_space color_{};
	pl_frame out_{};
	Stats stats_;
};

#endif // CHIAKI_FRAMEGEN_H
