#ifndef PROMETHEUS_SVGF_PASS_INCLUDED
#define PROMETHEUS_SVGF_PASS_INCLUDED

// =====================================================================================
//  Prometheus Software Ray-Tracing Engine
//  Spatiotemporal Variance-Guided Filtering - Phase 6 (ADR-002)
//
//  One shadow ray per pixel is pure noise: every sample is 0 or 1, so the estimator has
//  maximum variance. SVGF recovers a usable signal in two stages.
//
//   1. TEMPORAL. Reproject last frame's estimate through the motion vector and blend.
//      This is where nearly all the variance reduction comes from, because it costs a
//      full frame of samples per pixel rather than a wider spatial radius. It is also
//      where ghosting comes from, so the reprojected sample is validated against depth
//      and normal and then clamped to the local neighbourhood.
//
//   2. SPATIAL. Run an a-trous wavelet, widening the tap spacing each iteration. The
//      edge-stopping weights are guided by the running variance, so pixels that have
//      already converged temporally are filtered lightly and freshly disoccluded ones
//      are filtered hard. That variance guidance is what separates SVGF from a plain
//      bilateral blur.
//
//  -----------------------------------------------------------------------------------
//  SIGNAL LAYOUT
//  -----------------------------------------------------------------------------------
//  Colour target, RGBAHalf:      (visibility, second moment, history length, variance)
//  Geometry target, RGBAHalf:    (normal.xyz, linear eye depth)
//
//  The second moment is carried alongside the mean so variance is E[x^2] - E[x]^2 with
//  no extra pass. History length drives the blend weight and is what EC-002 resets.
// =====================================================================================

/// Widest history the temporal blend will trust, in frames.
///
/// The blend weight is 1/n, so without a floor an old pixel becomes immovable and any
/// residual error is frozen in permanently. Capping n bounds the weight from below and
/// keeps the filter responsive to real change.
#define PROMETHEUS_MAX_HISTORY_LENGTH 32.0

/// Relative depth difference beyond which a reprojected sample is a different surface.
#define PROMETHEUS_DEPTH_REJECT 0.05

/// Minimum cosine between current and history normals for the sample to be accepted.
#define PROMETHEUS_NORMAL_REJECT 0.9

/// Sigma terms for the a-trous edge-stopping weights.
///
/// Depth is the tolerance in multiples of the surface's own depth slope across the tap
/// distance. At 0.5 a tap must sit within half a slope-step of the plane the centre pixel
/// lies on, which keeps a shadow boundary that runs diagonally across a receding surface
/// from being averaged with the geometry behind it.
///
/// Normal is the exponent on the cosine, so it controls how fast a crease closes the
/// filter: 128 reaches half weight at about seven degrees, against roughly ten at 64.
///
/// Luma stays at the value SVGF's authors use. It is expressed in standard deviations of
/// the signal's own variance, so it is only meaningful once that variance is propagated
/// correctly; the fix for that is in the a-trous kernel, not in this number.
#define PROMETHEUS_SIGMA_DEPTH  0.5
#define PROMETHEUS_SIGMA_NORMAL 128.0
#define PROMETHEUS_SIGMA_LUMA   4.0

/// Converts a raw depth buffer sample to linear eye depth.
///
/// `zParams` follows Unity's `_ZBufferParams` packing, so the reversed-Z difference is
/// already folded into the constants and this file needs no platform branch.
float PrometheusLinearEyeDepth(float rawDepth, float4 zParams)
{
    return 1.0 / (zParams.z * rawDepth + zParams.w);
}

/// True when a reprojected sample plausibly describes the same surface.
///
/// Depth is compared relatively so the tolerance scales with distance, and normals by
/// dot product. Either test failing means the pixel was disoccluded, and reusing the
/// sample anyway is precisely what produces a ghost trail.
bool PrometheusIsHistoryValid(
    float currentDepth,
    float historyDepth,
    float3 currentNormal,
    float3 historyNormal)
{
    if (currentDepth <= 0.0 || historyDepth <= 0.0)
    {
        return false;
    }

    float relativeDepthDelta = abs(currentDepth - historyDepth) / max(currentDepth, 1e-4);
    if (relativeDepthDelta > PROMETHEUS_DEPTH_REJECT)
    {
        return false;
    }

    return dot(currentNormal, historyNormal) >= PROMETHEUS_NORMAL_REJECT;
}

/// True when the reprojected coordinate still lies on screen.
bool PrometheusIsInsideViewport(float2 uv)
{
    return all(uv >= 0.0) && all(uv <= 1.0);
}

/// Clamps a history sample into the current frame's local neighbourhood.
///
/// Validation alone cannot catch a ghost that reprojects onto a surface of similar depth
/// and orientation, which is exactly what a shadow edge sliding across a flat wall looks
/// like. Constraining the history to the range the current frame actually observed
/// bounds how far it can disagree, and is what holds Gate 6B's no-ghosting requirement
/// during fast translation.
float PrometheusClampHistory(float history, float neighbourhoodMean, float neighbourhoodStdDev, float scale)
{
    float minValue = neighbourhoodMean - (neighbourhoodStdDev * scale);
    float maxValue = neighbourhoodMean + (neighbourhoodStdDev * scale);
    return clamp(history, minValue, maxValue);
}

/// Blend weight for this frame's sample against the accumulated history.
///
/// A simple 1/n running mean converges fastest, so it is used until the history reaches
/// its cap, after which the weight is held at 1/max to keep an exponential tail.
float PrometheusTemporalAlpha(float historyLength)
{
    return 1.0 / max(historyLength + 1.0, 1.0);
}

/// Variance of a mean/second-moment pair, clamped against negative round-off.
float PrometheusVarianceFromMoments(float mean, float secondMoment)
{
    return max(0.0, secondMoment - (mean * mean));
}

/// Edge-stopping weight for one a-trous tap.
///
/// Three independent terms multiply together:
///   depth   - rejects taps across a depth discontinuity, scaled by the local gradient
///             so that steeply sloped surfaces are not mistaken for edges;
///   normal  - rejects taps across a crease;
///   luma    - rejects taps whose value disagrees by more than the noise explains. The
///             denominator carries the variance estimate, which is what makes this
///             variance-guided: a converged pixel gets a tight threshold and stays
///             sharp, a noisy one gets a loose threshold and is smoothed.
///
/// `tapDistance` is how many taps away the sample sits, and the depth tolerance is scaled
/// by it. Without that scaling the tolerance is whatever the slope produces at one step
/// while the outer taps of a 5x5 sit two steps out, so a sloped surface rejects its own
/// far taps and the filter degenerates to a 3x3 exactly where it was asked to widen. The
/// numerator grows with distance on a plane and does not at a discontinuity, which is what
/// lets a single tolerance separate the two.
float PrometheusAtrousWeight(
    float centreDepth,
    float tapDepth,
    float depthGradient,
    float tapDistance,
    float3 centreNormal,
    float3 tapNormal,
    float centreLuma,
    float tapLuma,
    float luminanceStdDev)
{
    float depthWeight = exp(-abs(centreDepth - tapDepth) /
        ((PROMETHEUS_SIGMA_DEPTH * depthGradient * tapDistance) + 1e-4));

    float normalWeight = pow(saturate(dot(centreNormal, tapNormal)), PROMETHEUS_SIGMA_NORMAL);

    float lumaWeight = exp(-abs(centreLuma - tapLuma) /
        ((PROMETHEUS_SIGMA_LUMA * luminanceStdDev) + 1e-4));

    return depthWeight * normalWeight * lumaWeight;
}

/// One tap of the 5x5 B3-spline kernel [1, 4, 6, 4, 1] / 16, separable into an outer product.
float PrometheusAtrousKernelWeight(int offsetX, int offsetY)
{
    const float kernel[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };
    return kernel[offsetX + 2] * kernel[offsetY + 2];
}

#endif // PROMETHEUS_SVGF_PASS_INCLUDED
