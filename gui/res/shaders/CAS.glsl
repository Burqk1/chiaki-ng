// SPDX-License-Identifier: MIT
//
// Contrast Adaptive Sharpening for chiaki-ng.
// Algorithm: AMD FidelityFX CAS (MIT, Copyright (c) 2020-2021 Advanced Micro Devices, Inc.).
// Sharpens the decoded stream after scaling; the sharpening strength adapts to
// local contrast, so edges get crisper without ringing on already sharp areas.

//!PARAM cas_sharpness
//!DESC Sharpening strength (0 = off, 1 = maximum)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!HOOK OUTPUT
//!BIND HOOKED
//!DESC chiaki-ng CAS sharpening

vec4 hook()
{
    vec3 a = HOOKED_texOff(vec2(-1.0, -1.0)).rgb;
    vec3 b = HOOKED_texOff(vec2( 0.0, -1.0)).rgb;
    vec3 c = HOOKED_texOff(vec2( 1.0, -1.0)).rgb;
    vec3 d = HOOKED_texOff(vec2(-1.0,  0.0)).rgb;
    vec4 e4 = HOOKED_texOff(vec2(0.0, 0.0));
    vec3 e = e4.rgb;
    vec3 f = HOOKED_texOff(vec2( 1.0,  0.0)).rgb;
    vec3 g = HOOKED_texOff(vec2(-1.0,  1.0)).rgb;
    vec3 h = HOOKED_texOff(vec2( 0.0,  1.0)).rgb;
    vec3 i = HOOKED_texOff(vec2( 1.0,  1.0)).rgb;

    // Soft min/max over the cross and the full 3x3 window.
    vec3 mn = min(min(min(d, e), min(f, b)), h);
    vec3 mn2 = min(mn, min(min(a, c), min(g, i)));
    mn += mn2;
    vec3 mx = max(max(max(d, e), max(f, b)), h);
    vec3 mx2 = max(mx, max(max(a, c), max(g, i)));
    mx += mx2;

    // Amount of sharpening allowed before the result would clip.
    vec3 amp = clamp(min(mn, 2.0 - mx) / max(mx, vec3(1e-5)), 0.0, 1.0);
    amp = sqrt(amp);

    float peak = -1.0 / mix(8.0, 5.0, clamp(cas_sharpness, 0.0, 1.0));
    vec3 w = amp * peak;
    vec3 rcp_weight = 1.0 / (1.0 + 4.0 * w);

    vec3 window = (b + d) + (f + h);
    vec3 result = clamp((window * w + e) * rcp_weight, 0.0, 1.0);
    return vec4(result, e4.a);
}
