#ifndef PROMETHEUS_SPATIAL_UPSAMPLE_PASS_INCLUDED
#define PROMETHEUS_SPATIAL_UPSAMPLE_PASS_INCLUDED

#include "SVGFPass.hlsl"

// =====================================================================================
//  Prometheus Software Ray-Tracing Engine
//  Joint-bilateral upsampling - Phase 6, Gate 6A
//
//  Rays are traced at 0.5x (Section 7.1 Phase 5) and must return to native resolution.
//  A bilinear stretch would be wrong in exactly the place that matters: across a
//  silhouette it blends a lit sample with a shadowed one that belongs to a different
//  surface, and the shadow edge softens into a halo.
//
//  Joint-bilateral upsampling keeps the low-resolution signal but takes its weights from
//  the full-resolution G-buffer. A coarse tap only contributes when its depth and normal
//  agree with the full-resolution pixel being written, so the edge follows the geometry
//  rather than the sampling grid. That is what Gate 6A checks: crisp silhouettes, no
//  bleeding across geometric edges.
//
//  The fallback below matters as much as the main path. When every tap is rejected -
//  a one-pixel-wide feature that the half-resolution trace never sampled - a zero-weight
//  average would produce a black or NaN pixel, so the nearest tap is taken unweighted.
// =====================================================================================

/// Radius of the coarse neighbourhood gathered per full-resolution pixel.
///
/// One coarse texel each way covers the 2x2 footprint of a 0.5x trace plus a ring of
/// slack, which is enough for a tap to survive when the nearest one falls on the far
/// side of an edge.
#define PROMETHEUS_UPSAMPLE_RADIUS 1

/// Depth tolerance for a coarse tap, relative to the full-resolution depth.
#define PROMETHEUS_UPSAMPLE_DEPTH_SIGMA 0.05

/// Normal agreement exponent. Higher values reject creases more aggressively.
#define PROMETHEUS_UPSAMPLE_NORMAL_POWER 32.0

/// Bilinear falloff over the coarse footprint, so the result stays smooth where the
/// geometry is smooth and the bilateral terms are all close to one.
float PrometheusUpsampleSpatialWeight(float2 coarseOffset)
{
    float2 falloff = saturate(1.0 - abs(coarseOffset));
    return falloff.x * falloff.y;
}

/// Combined geometric weight of one coarse tap against the full-resolution pixel.
///
/// Depth is compared relatively so the tolerance widens with distance; normals are
/// compared by dot product raised to a power, which is cheap and falls off sharply at a
/// crease. Returning zero means the tap belongs to different geometry and must not
/// contribute at all.
float PrometheusUpsampleGeometryWeight(
    float fullResDepth,
    float coarseDepth,
    float3 fullResNormal,
    float3 coarseNormal)
{
    if (coarseDepth <= 0.0 || fullResDepth <= 0.0)
    {
        return 0.0;
    }

    float relativeDepthDelta = abs(fullResDepth - coarseDepth) / max(fullResDepth, 1e-4);
    if (relativeDepthDelta > PROMETHEUS_UPSAMPLE_DEPTH_SIGMA)
    {
        return 0.0;
    }

    float depthWeight = exp(-relativeDepthDelta / PROMETHEUS_UPSAMPLE_DEPTH_SIGMA);
    float normalWeight = pow(saturate(dot(fullResNormal, coarseNormal)), PROMETHEUS_UPSAMPLE_NORMAL_POWER);

    return depthWeight * normalWeight;
}

/// Accumulator for a joint-bilateral gather.
struct PrometheusUpsampleAccumulator
{
    float value;
    float weight;
    float nearestValue;
    bool  hasNearest;
};

/// Starts an empty gather.
PrometheusUpsampleAccumulator PrometheusBeginUpsample()
{
    PrometheusUpsampleAccumulator accumulator;
    accumulator.value = 0.0;
    accumulator.weight = 0.0;
    accumulator.nearestValue = 0.0;
    accumulator.hasNearest = false;
    return accumulator;
}

/// Folds one coarse tap into the gather.
///
/// `isNearest` marks the tap the pixel would have taken under point sampling; it is
/// remembered separately so the fallback has something geometrically plausible to use
/// when every weighted tap is rejected.
void PrometheusAccumulateUpsampleTap(
    inout PrometheusUpsampleAccumulator accumulator,
    float tapValue,
    float tapWeight,
    bool isNearest)
{
    accumulator.value += tapValue * tapWeight;
    accumulator.weight += tapWeight;

    if (isNearest)
    {
        accumulator.nearestValue = tapValue;
        accumulator.hasNearest = true;
    }
}

/// Resolves the gather, falling back to the nearest tap when every weight was rejected.
float PrometheusResolveUpsample(PrometheusUpsampleAccumulator accumulator)
{
    if (accumulator.weight > 1e-5)
    {
        return accumulator.value / accumulator.weight;
    }

    // Every tap disagreed with this pixel's geometry. Point sampling is wrong but bounded;
    // dividing by a zero weight would not be.
    return accumulator.hasNearest ? accumulator.nearestValue : 1.0;
}

#endif // PROMETHEUS_SPATIAL_UPSAMPLE_PASS_INCLUDED
