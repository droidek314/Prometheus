#ifndef PROMETHEUS_BVH_TRAVERSAL_INCLUDED
#define PROMETHEUS_BVH_TRAVERSAL_INCLUDED

#include "IntersectionMath.hlsl"

// =====================================================================================
//  Prometheus Software Ray-Tracing Engine
//  Stackless two-level traversal - Phase 5
//
//  -----------------------------------------------------------------------------------
//  WHY THERE IS NO STACK
//  -----------------------------------------------------------------------------------
//  A conventional BVH walk pushes the far child and recurses into the near one, which
//  needs a per-thread stack. On Shader Model 5.0 that stack lives in registers, and at
//  the occupancy this engine targets it does not fit: the compiler spills it to scratch
//  memory and traversal becomes bandwidth bound (ADR-001).
//
//  PrometheusBVHNode instead stores an escape ("rope") index and no child pointers at
//  all. Nodes are emitted depth-first, so the left child is always selfIndex + 1, and
//  the escape points at the next node outside the current subtree. The entire walk is
//  therefore a single cursor:
//
//      i = root
//      while i != INVALID
//          if ray misses node[i]      -> i = node[i].escapeIndex   (skip the subtree)
//          else if node[i] is a leaf  -> test primitives; i = escape
//          else                       -> i = i + 1                 (descend left)
//
//  Leaves carry escape == selfIndex + 1, so both leaf branches coincide and the loop
//  stays branch-light. Live state is one cursor plus the ray: nothing to spill.
//
//  -----------------------------------------------------------------------------------
//  SHARED NODE BUFFER LAYOUT
//  -----------------------------------------------------------------------------------
//  _PrometheusBvhNodes holds both hierarchy levels back to back:
//
//      [0, _PrometheusTlasNodeCount)      TLAS nodes, escape indices are absolute
//      [_PrometheusTlasNodeCount, ...)    BLAS nodes, escape indices are BLAS-local
//
//  A BLAS is authored without knowing where it will land, so its ropes and leaf ranges
//  are relative to its own root. Traversal rebases them through the instance's
//  blasNodeOffset and blasTriangleOffset. That is what lets one BLAS be shared by any
//  number of instances with no duplication.
//
//  -----------------------------------------------------------------------------------
//  TLAS LEAVES ARE INDIRECT
//  -----------------------------------------------------------------------------------
//  A TLAS leaf gathers up to four instances. Its packed range addresses
//  _PrometheusInstanceIndices, not _PrometheusInstances, so the Morton sort permutes
//  4-byte indices instead of 64-byte instance records.
// =====================================================================================

/// Number of TLAS nodes at the front of _PrometheusBvhNodes. Set per frame.
uint _PrometheusTlasNodeCount;

/// A closest-hit result in world space.
struct PrometheusHit
{
    float  distance;      ///< Parametric distance along the world-space ray.
    float2 barycentrics;  ///< Moller-Trumbore (u, v) within the hit triangle.
    uint   instanceIndex; ///< Index into _PrometheusInstances.
    uint   primitiveId;   ///< Source mesh triangle index.

    /// Index into _PrometheusTriangles, so the caller can read the hit surface itself.
    ///
    /// primitiveId identifies the triangle within its source mesh, which is enough to look
    /// something up in a per-mesh table but not enough to fetch the triangle record. Shading
    /// a bounce needs the geometric normal, and that lives on the record.
    uint   triangleIndex;
    bool   isHit;
};

/// An empty hit carrying the ray's far clip.
PrometheusHit PrometheusMakeMiss(float tMax)
{
    PrometheusHit hit;
    hit.distance = tMax;
    hit.barycentrics = float2(0.0, 0.0);
    hit.instanceIndex = 0u;
    hit.triangleIndex = 0u;
    hit.primitiveId = 0u;
    hit.isHit = false;
    return hit;
}

// =====================================================================================
//  Bottom level
// =====================================================================================

/// Walks one instance's BLAS looking only for whether anything blocks the segment.
///
/// Returns on the first triangle hit; no closest-hit bookkeeping is performed. The ray
/// is already in the instance's object space, so `t` is directly comparable with the
/// world-space interval (the object-space direction is deliberately left un-normalised).
bool PrometheusBlasAnyHit(
    PrometheusInstance instance,
    float3 localOrigin,
    float3 localDirection,
    float tMin,
    float tMax,
    bool cullBackFace)
{
    float3 invDirection = PrometheusSafeRcpDirection(localDirection);
    uint nodeOffset = instance.blasNodeOffset;
    uint triangleOffset = instance.blasTriangleOffset;
    uint cursor = 0u;

    [loop]
    while (cursor != PROMETHEUS_INVALID_INDEX)
    {
        PrometheusBVHNode node = _PrometheusBvhNodes[nodeOffset + cursor];

        if (!PrometheusIntersectAabb(node, localOrigin, invDirection, tMax))
        {
            cursor = node.escapeIndex;
            continue;
        }

        uint count = PrometheusGetPrimitiveCount(node);
        if (count == 0u)
        {
            cursor = cursor + 1u;
            continue;
        }

        uint first = PrometheusGetFirstPrimitive(node);

        [loop]
        for (uint k = 0u; k < count; k++)
        {
            PrometheusTriangle tri = _PrometheusTriangles[triangleOffset + first + k];

            if (PrometheusIsTriangleOccluding(tri, localOrigin, localDirection, tMin, tMax, cullBackFace))
            {
                return true;
            }
        }

        cursor = node.escapeIndex;
    }

    return false;
}

/// Walks one instance's BLAS narrowing `hit` to the closest triangle found.
///
/// `hit.distance` doubles as the running far clip, so each accepted hit tightens the
/// interval and lets later nodes be rejected by the slab test without being descended.
void PrometheusBlasClosestHit(
    PrometheusInstance instance,
    uint instanceIndex,
    float3 localOrigin,
    float3 localDirection,
    float tMin,
    bool cullBackFace,
    inout PrometheusHit hit)
{
    float3 invDirection = PrometheusSafeRcpDirection(localDirection);
    uint nodeOffset = instance.blasNodeOffset;
    uint triangleOffset = instance.blasTriangleOffset;
    uint cursor = 0u;

    [loop]
    while (cursor != PROMETHEUS_INVALID_INDEX)
    {
        PrometheusBVHNode node = _PrometheusBvhNodes[nodeOffset + cursor];

        if (!PrometheusIntersectAabb(node, localOrigin, invDirection, hit.distance))
        {
            cursor = node.escapeIndex;
            continue;
        }

        uint count = PrometheusGetPrimitiveCount(node);
        if (count == 0u)
        {
            cursor = cursor + 1u;
            continue;
        }

        uint first = PrometheusGetFirstPrimitive(node);

        [loop]
        for (uint k = 0u; k < count; k++)
        {
            uint triangleIndex = triangleOffset + first + k;
            PrometheusTriangle tri = _PrometheusTriangles[triangleIndex];

            float tHit;
            float2 barycentrics;

            if (PrometheusIntersectTriangle(
                    tri, localOrigin, localDirection, tMin, hit.distance, cullBackFace, tHit, barycentrics))
            {
                if (tHit < hit.distance)
                {
                    hit.distance = tHit;
                    hit.barycentrics = barycentrics;
                    hit.instanceIndex = instanceIndex;
                    hit.primitiveId = tri.primitiveId;
                    hit.triangleIndex = triangleIndex;
                    hit.isHit = true;
                }
            }
        }

        cursor = node.escapeIndex;
    }
}

// =====================================================================================
//  Top level
// =====================================================================================

/// Traces a shadow ray through the whole scene, returning on the first occluder.
///
/// This is the hot path for the Phase 5 gate. Early exit matters twice over: it skips
/// the rest of the current BLAS, and it abandons the TLAS walk entirely, so a ray that
/// starts inside dense geometry terminates almost immediately.
bool PrometheusTraceShadowRay(float3 origin, float3 direction, float tMin, float tMax)
{
    if (_PrometheusTlasNodeCount == 0u)
    {
        return false;
    }

    float3 invDirection = PrometheusSafeRcpDirection(direction);
    uint cursor = 0u;

    [loop]
    while (cursor != PROMETHEUS_INVALID_INDEX && cursor < _PrometheusTlasNodeCount)
    {
        PrometheusBVHNode node = _PrometheusBvhNodes[cursor];

        if (!PrometheusIntersectAabb(node, origin, invDirection, tMax))
        {
            cursor = node.escapeIndex;
            continue;
        }

        uint count = PrometheusGetPrimitiveCount(node);
        if (count == 0u)
        {
            cursor = cursor + 1u;
            continue;
        }

        [loop]
        for (uint k = 0u; k < count; k++)
        {
            uint instanceIndex = PrometheusGetLeafInstance(node, k);
            PrometheusInstance instance = _PrometheusInstances[instanceIndex];

            if (!PrometheusInstanceHasFlag(instance, PROMETHEUS_INSTANCE_CASTS_SHADOWS))
            {
                continue;
            }

            float3 localOrigin;
            float3 localDirection;
            PrometheusTransformRayToLocal(instance, origin, direction, localOrigin, localDirection);

            bool cullBackFace = !PrometheusInstanceHasFlag(instance, PROMETHEUS_INSTANCE_DOUBLE_SIDED);

            if (PrometheusBlasAnyHit(instance, localOrigin, localDirection, tMin, tMax, cullBackFace))
            {
                return true;
            }
        }

        cursor = node.escapeIndex;
    }

    return false;
}

/// Traces a closest-hit ray through the whole scene.
///
/// Used by the Phase 6 indirect passes; shadow rays should use the any-hit form above,
/// which is both cheaper and shallower in register pressure.
PrometheusHit PrometheusTraceClosestHit(float3 origin, float3 direction, float tMin, float tMax)
{
    PrometheusHit hit = PrometheusMakeMiss(tMax);

    if (_PrometheusTlasNodeCount == 0u)
    {
        return hit;
    }

    float3 invDirection = PrometheusSafeRcpDirection(direction);
    uint cursor = 0u;

    [loop]
    while (cursor != PROMETHEUS_INVALID_INDEX && cursor < _PrometheusTlasNodeCount)
    {
        PrometheusBVHNode node = _PrometheusBvhNodes[cursor];

        if (!PrometheusIntersectAabb(node, origin, invDirection, hit.distance))
        {
            cursor = node.escapeIndex;
            continue;
        }

        uint count = PrometheusGetPrimitiveCount(node);
        if (count == 0u)
        {
            cursor = cursor + 1u;
            continue;
        }

        [loop]
        for (uint k = 0u; k < count; k++)
        {
            uint instanceIndex = PrometheusGetLeafInstance(node, k);
            PrometheusInstance instance = _PrometheusInstances[instanceIndex];

            float3 localOrigin;
            float3 localDirection;
            PrometheusTransformRayToLocal(instance, origin, direction, localOrigin, localDirection);

            bool cullBackFace = !PrometheusInstanceHasFlag(instance, PROMETHEUS_INSTANCE_DOUBLE_SIDED);

            PrometheusBlasClosestHit(
                instance, instanceIndex, localOrigin, localDirection, tMin, cullBackFace, hit);
        }

        cursor = node.escapeIndex;
    }

    return hit;
}

#endif // PROMETHEUS_BVH_TRAVERSAL_INCLUDED
