#ifndef PROMETHEUS_INTERSECTION_MATH_INCLUDED
#define PROMETHEUS_INTERSECTION_MATH_INCLUDED

#include "PrometheusStructures.hlsl"

// =====================================================================================
//  Prometheus Software Ray-Tracing Engine
//  Ray/primitive intersection - Phase 5
//
//  Everything here operates on the packed forms declared in PrometheusStructures.hlsl:
//  nodes carry an AABB pair, triangles carry a vertex plus two pre-computed edges. No
//  function in this file reads a buffer; they take values, so the traversal loop decides
//  the memory access pattern.
//
//  The CPU mirror of these routines is CpuBvhValidator.cs, which the Phase 3 accuracy
//  gate holds to bit-exact agreement. Any change here must be made there too.
// =====================================================================================

/// Guards the Moller-Trumbore division against rays travelling parallel to a triangle.
#define PROMETHEUS_INTERSECT_EPSILON 1e-8

/// Reciprocal ray direction with zero components replaced by a tiny finite value.
///
/// A component of exactly zero would make the slab test evaluate 0 * inf, which is NaN,
/// and a NaN compare fails in both directions so the node would be silently skipped.
/// Substituting a tiny magnitude instead pushes both slab planes far outside the ray
/// interval, which is the correct answer for an axis-parallel ray.
float3 PrometheusSafeRcpDirection(float3 direction)
{
    const float kTiny = 1e-20;
    const float kLarge = 1e20;

    float3 safeDir = float3(
        abs(direction.x) < kTiny ? kTiny : direction.x,
        abs(direction.y) < kTiny ? kTiny : direction.y,
        abs(direction.z) < kTiny ? kTiny : direction.z);

    return clamp(1.0 / safeDir, -kLarge, kLarge);
}

/// Branch-free ray/AABB slab test.
///
///     tEnter = max_k min(tk_min, tk_max)
///     tExit  = min_k max(tk_min, tk_max)
///     tk_min = (boundsMin_k - o_k) * invDir_k
///
/// `invDirection` is computed once per ray by the caller, so the inner traversal loop
/// costs six multiplies and four min/max per node with no divides at all.
bool PrometheusIntersectAabb(
    PrometheusBVHNode node,
    float3 origin,
    float3 invDirection,
    float tMax,
    out float tEnter)
{
    float3 t0 = (node.boundsMin - origin) * invDirection;
    float3 t1 = (node.boundsMax - origin) * invDirection;

    float3 tSmall = min(t0, t1);
    float3 tLarge = max(t0, t1);

    tEnter      = max(max(tSmall.x, tSmall.y), tSmall.z);
    float tExit = min(min(tLarge.x, tLarge.y), tLarge.z);

    return (tExit >= max(tEnter, 0.0)) && (tEnter <= tMax);
}

/// Slab test for callers that do not need the entry distance.
bool PrometheusIntersectAabb(PrometheusBVHNode node, float3 origin, float3 invDirection, float tMax)
{
    float unusedEnter;
    return PrometheusIntersectAabb(node, origin, invDirection, tMax, unusedEnter);
}

/// Moller-Trumbore ray/triangle intersection against the pre-computed edge form.
///
///     P = d x e2,   det = e1 . P
///     T = o - A,    u   = (T . P) / det
///     Q = T x e1,   v   = (d . Q) / det
///                   t   = (e2 . Q) / det
///
/// Storing edges rather than vertices is what makes this cheap: the algorithm never
/// references B or C, so the two subtractions are paid once at build time instead of
/// once per ray per candidate triangle.
///
/// When `cullBackFace` is set, triangles whose determinant is non-positive are rejected
/// before the reciprocal, which removes roughly half the work for opaque geometry.
bool PrometheusIntersectTriangle(
    PrometheusTriangle tri,
    float3 origin,
    float3 direction,
    float tMin,
    float tMax,
    bool cullBackFace,
    out float tHit,
    out float2 barycentrics)
{
    tHit = tMax;
    barycentrics = float2(0.0, 0.0);

    float3 pVec = cross(direction, tri.edge2);
    float det = dot(tri.edge1, pVec);

    if (cullBackFace)
    {
        if (det < PROMETHEUS_INTERSECT_EPSILON)
        {
            return false;
        }
    }
    else if (abs(det) < PROMETHEUS_INTERSECT_EPSILON)
    {
        return false;
    }

    float invDet = 1.0 / det;

    float3 tVec = origin - tri.vertex0;
    float u = dot(tVec, pVec) * invDet;
    if (u < 0.0 || u > 1.0)
    {
        return false;
    }

    float3 qVec = cross(tVec, tri.edge1);
    float v = dot(direction, qVec) * invDet;
    if (v < 0.0 || (u + v) > 1.0)
    {
        return false;
    }

    float t = dot(tri.edge2, qVec) * invDet;
    if (t < tMin || t > tMax)
    {
        return false;
    }

    tHit = t;
    barycentrics = float2(u, v);
    return true;
}

/// Occlusion form of the same test: reports only whether the triangle blocks the segment.
///
/// A shadow ray does not care which surface occludes it or where, so the barycentrics,
/// the hit distance and the closest-hit bookkeeping are all dead work. Dropping them
/// lets the caller return the instant anything is hit, and keeps the register footprint
/// of the traversal loop small enough to avoid spilling on Shader Model 5.0 hardware.
bool PrometheusIsTriangleOccluding(
    PrometheusTriangle tri,
    float3 origin,
    float3 direction,
    float tMin,
    float tMax,
    bool cullBackFace)
{
    float3 pVec = cross(direction, tri.edge2);
    float det = dot(tri.edge1, pVec);

    if (cullBackFace)
    {
        if (det < PROMETHEUS_INTERSECT_EPSILON)
        {
            return false;
        }
    }
    else if (abs(det) < PROMETHEUS_INTERSECT_EPSILON)
    {
        return false;
    }

    float invDet = 1.0 / det;

    float3 tVec = origin - tri.vertex0;
    float u = dot(tVec, pVec) * invDet;
    if (u < 0.0 || u > 1.0)
    {
        return false;
    }

    float3 qVec = cross(tVec, tri.edge1);
    float v = dot(direction, qVec) * invDet;
    if (v < 0.0 || (u + v) > 1.0)
    {
        return false;
    }

    float t = dot(tri.edge2, qVec) * invDet;
    return (t >= tMin) && (t <= tMax);
}

#endif // PROMETHEUS_INTERSECTION_MATH_INCLUDED
