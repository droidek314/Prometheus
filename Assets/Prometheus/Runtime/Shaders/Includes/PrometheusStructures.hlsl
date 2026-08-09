#ifndef PROMETHEUS_STRUCTURES_INCLUDED
#define PROMETHEUS_STRUCTURES_INCLUDED

// =====================================================================================
//  Prometheus Software Ray-Tracing Engine
//  Shared GPU memory contracts - Phase 1 (Shared Memory Contracts & Data Layouts)
//
//  This file is the single authoritative GPU-side declaration of every structure that
//  crosses the CPU/GPU boundary. It mirrors, byte for byte:
//
//      Assets/Prometheus/Runtime/Core/Structs/PrometheusBVHNode.cs
//      Assets/Prometheus/Runtime/Core/Structs/PrometheusInstance.cs
//      Assets/Prometheus/Runtime/Core/Structs/PrometheusTriangle.cs
//
//  Every stride below is asserted from managed code by
//  Assets/Prometheus/Tests/EditMode/MemoryLayoutTests.cs using UnsafeUtility.SizeOf<T>()
//  and explicit field offsets. Editing a field here without editing the matching C#
//  struct will fail that gate.
//
//  -----------------------------------------------------------------------------------
//  BUFFER LAYOUT ANNOTATION (Section 4.3 - HLSL Annotation Standard)
//  -----------------------------------------------------------------------------------
//
//   Register | Binding                    | Element type        | Stride | Access
//   ---------+----------------------------+---------------------+--------+-----------
//   t0       | _PrometheusBvhNodes        | PrometheusBVHNode   |  32 B  | read
//   t1       | _PrometheusTriangles       | PrometheusTriangle  |  48 B  | read
//   t2       | _PrometheusInstances       | PrometheusInstance  |  64 B  | read
//   t3       | _PrometheusInstanceIndices | uint                |   4 B  | read
//
//  All structure strides are integer multiples of 16 bytes, so no element straddles a
//  constant register boundary and no implicit padding is inserted by the HLSL packing
//  rules. The indirection buffer at t3 is a flat uint array, not a structure.
//
//  MATRIX STORAGE: no matrix types appear in these structures. Affine transforms are
//  stored as three explicit float4 rows, which packs identically on both sides; a
//  float3x4 would not, because HLSL pads each float3 column out to 16 bytes while
//  Unity.Mathematics packs them contiguously.
// =====================================================================================

// -------------------------------------------------------------------------------------
// Structural bounds (mirrors Section 3.1 and the C# constants)
// -------------------------------------------------------------------------------------

#define PROMETHEUS_INVALID_INDEX            0xFFFFFFFFu

#define PROMETHEUS_NODE_STRIDE              32
#define PROMETHEUS_TRIANGLE_STRIDE          48
#define PROMETHEUS_INSTANCE_STRIDE          64

#define PROMETHEUS_MESH_ID_BITS             16
#define PROMETHEUS_MESH_ID_MASK             0x0000FFFFu

#define PROMETHEUS_PRIMITIVE_INDEX_BITS     22
#define PROMETHEUS_PRIMITIVE_COUNT_BITS     10
#define PROMETHEUS_PRIMITIVE_INDEX_MASK     0x003FFFFFu

#define PROMETHEUS_MAX_SCENE_TRIANGLES      2097152     // 2^21
#define PROMETHEUS_MAX_BVH_NODES            4194303     // 2^22 - 1
#define PROMETHEUS_MAX_INSTANCES            4096

#define PROMETHEUS_MIN_VALID_AREA           1e-7

// Instance behaviour flags - mirrors PrometheusInstanceFlags.
#define PROMETHEUS_INSTANCE_ACTIVE           (1u << 0)
#define PROMETHEUS_INSTANCE_DYNAMIC          (1u << 1)
#define PROMETHEUS_INSTANCE_CASTS_SHADOWS    (1u << 2)
#define PROMETHEUS_INSTANCE_CONTRIBUTES_GI   (1u << 3)
#define PROMETHEUS_INSTANCE_MIRRORED_WINDING (1u << 4)
#define PROMETHEUS_INSTANCE_DOUBLE_SIDED     (1u << 5)

// Guards the division in Moller-Trumbore against edge-on rays.
#define PROMETHEUS_INTERSECT_EPSILON        1e-8

// =====================================================================================
//  PrometheusBVHNode - 32 bytes (2 x 16)
//
//  Offset  Size  Field            Purpose
//  ------  ----  ---------------  -------------------------------------------------
//     0     12   boundsMin        AABB lower corner
//    12      4   escapeIndex      Node to visit on an AABB miss (skip pointer / rope)
//    16     12   boundsMax        AABB upper corner
//    28      4   packedLeafRange  [0..21] first primitive | [22..31] primitive count
//
//  Nodes are stored depth-first, so the left child of an interior node is always
//  selfIndex + 1 and is never stored. Traversal is therefore stackless:
//
//      uint i = rootIndex;
//      [loop] while (i != PROMETHEUS_INVALID_INDEX)
//      {
//          PrometheusBVHNode node = _PrometheusBvhNodes[i];
//          i = PrometheusIntersectAabb(...) ? (i + 1) : node.escapeIndex;
//      }
//
//  Removing the per-thread stack is what keeps register pressure inside the SM 5.0
//  budget and avoids spilling to scratch memory (ADR-001).
// =====================================================================================
struct PrometheusBVHNode
{
    float3 boundsMin;
    uint   escapeIndex;
    float3 boundsMax;
    uint   packedLeafRange;
};

// =====================================================================================
//  PrometheusTriangle - 48 bytes (3 x 16)
//
//  Offset  Size  Field          Purpose
//  ------  ----  -------------  ---------------------------------------------------
//     0     12   vertex0        Object-space vertex A
//    12      4   primitiveId    Source mesh triangle index
//    16     12   edge1          B - A, pre-computed at build time
//    28      4   materialId     Material table index (overrides the instance value)
//    32     12   edge2          C - A, pre-computed at build time
//    44      4   packedNormal   Octahedral geometric normal, 16:16 unorm
//
//  Edges rather than vertices are stored because Moller-Trumbore never references B
//  or C directly; baking the subtractions removes two vector subtracts from the
//  innermost per-candidate-primitive loop.
// =====================================================================================
struct PrometheusTriangle
{
    float3 vertex0;
    uint   primitiveId;
    float3 edge1;
    uint   materialId;
    float3 edge2;
    uint   packedNormal;
};

// =====================================================================================
//  PrometheusInstance - 64 bytes (4 x 16)
//
//  Offset  Size  Field                  Purpose
//  ------  ----  ---------------------  -------------------------------------------
//     0     16   worldToLocalRow0       Row 0 of the world -> object affine transform
//    16     16   worldToLocalRow1       Row 1
//    32     16   worldToLocalRow2       Row 2
//    48      4   blasNodeOffset         First BLAS node in _PrometheusBvhNodes
//    52      4   blasTriangleOffset     First BLAS triangle in _PrometheusTriangles
//    56      4   materialId             Material table index
//    60      4   packedFlagsAndMeshId   [0..15] mesh id | [16..31] PROMETHEUS_INSTANCE_*
//
//  The transform is affine, so its bottom row is always (0,0,0,1) and is not stored.
//  Three float4 rows are used rather than a float3x4: HLSL forbids a member from
//  straddling a 16-byte boundary, so four float3 columns would occupy 64 bytes here
//  while Unity.Mathematics packs them into 48. Explicit rows are 48 bytes under both
//  sets of rules, and match the packing DXR uses for its own instance descriptors.
//
//  The forward localToWorld transform and the world-space AABB were removed to reach
//  this size. Traversal needs neither: the ray is carried into object space and t is
//  preserved, and the TLAS nodes already carry the bounds.
//
//  Object-space normals need the inverse transpose of the forward transform. The rows
//  stored here already are the inverse, so accumulating them weighted by the normal's
//  components applies that transpose directly.
// =====================================================================================
struct PrometheusInstance
{
    float4 worldToLocalRow0;
    float4 worldToLocalRow1;
    float4 worldToLocalRow2;
    uint   blasNodeOffset;
    uint   blasTriangleOffset;
    uint   materialId;
    uint   packedFlagsAndMeshId;
};

// =====================================================================================
//  Canonical acceleration structure bindings.
//
//  Declared once here so that every Prometheus kernel agrees on register assignment
//  (Section 6.1: explicit register allocation for all buffers).
// =====================================================================================
StructuredBuffer<PrometheusBVHNode>  _PrometheusBvhNodes  : register(t0);
StructuredBuffer<PrometheusTriangle> _PrometheusTriangles : register(t1);
StructuredBuffer<PrometheusInstance> _PrometheusInstances : register(t2);

// Sorted instance permutation addressed by TLAS leaves. See PrometheusGetLeafInstance.
StructuredBuffer<uint> _PrometheusInstanceIndices : register(t3);

// Per-instance diffuse albedo, parallel to _PrometheusInstances. Bounce light takes the
// colour of the surface it left, so indirect lighting is meaningless without it.
//
// Held in its own buffer rather than folded into PrometheusInstance because that record is a
// 64-byte layout contract shared with C# and verified by the memory layout tests; widening it
// would move every field after the matrix rows for the sake of data only one kernel reads.
StructuredBuffer<float4> _PrometheusInstanceAlbedo;

// Per-instance surface response: (metallic, smoothness, unused, unused).
//
// Separate from the albedo buffer rather than packed into its unused alpha, because two
// values are needed and because this is the natural place for anything else a surface
// eventually needs to say about itself.
StructuredBuffer<float4> _PrometheusInstanceMaterial;

// Per-instance emitted radiance, in rgb. Zero for everything that does not glow.
//
// This is what turns a neon sign from a bright-looking surface into an actual light. URP
// already draws an emissive material bright, but that is only its own pixels; nothing in a
// rasterised pipeline lets a glowing surface illuminate anything else. A ray that lands on
// one, on the other hand, has simply found a light, and it costs nothing beyond the hit it
// was already paying for.
//
// The shape is preserved for free, which is the entire point for neon. A long tube lights a
// wall like a long tube, because the rays land along its length; approximating it with a
// point light cannot do that at any cost.
StructuredBuffer<float4> _PrometheusInstanceEmission;

// =====================================================================================
//  PrometheusLight - 64 bytes (4 x 16)
//
//  Mirrors URP's own additional-light constants rather than a tidier layout of our own.
//  The host fills these by calling URP's InitializeLightConstants_Common, so the falloff a
//  bounce uses is the identical curve the direct lighting used, evaluated from identical
//  numbers. Deriving it again from the Light component would give something plausible that
//  disagreed, and bounce light that disagrees with its own source looks like a geometry bug
//  rather than a falloff one.
// =====================================================================================
struct PrometheusLight
{
    float4 positionType;   ///< xyz position, w = 1 punctual / 0 directional (xyz is the direction)
    float4 color;          ///< Final colour, intensity and colour space already applied
    float4 spotDirection;  ///< Spot axis; meaningless for other types
    float4 attenuation;    ///< xy distance falloff terms, zw spot cone terms
};

StructuredBuffer<PrometheusLight> _PrometheusLights;

/// Number of entries in _PrometheusLights. The main light is not among them.
uint _PrometheusLightCount;

/// One light's contribution direction and strength at a point.
///
/// Returns false when the light cannot reach the point at all, which lets the caller skip
/// the shadow ray entirely. That early exit is the whole reason this returns a bool rather
/// than a zero: a light facing away, or beyond its range, would otherwise cost a full
/// traversal to prove it contributes nothing.
///
/// `rayLength` is how far a shadow ray towards this light should be allowed to travel.
/// Punctual lights end at the light itself, because geometry behind a lamp cannot shadow it;
/// clipping the ray there also avoids paying for traversal past anything that matters.
bool PrometheusEvaluateLight(
    PrometheusLight light,
    float3 position,
    float3 normal,
    out float3 direction,
    out float3 radiance,
    out float rayLength)
{
    direction = float3(0.0, 1.0, 0.0);
    radiance = float3(0.0, 0.0, 0.0);
    rayLength = 0.0;

    // w carries the type: an offset of zero leaves the direction untouched for a directional
    // light, and one turns it into the vector from the surface to a punctual light.
    float3 toLight = light.positionType.xyz - (position * light.positionType.w);
    float distanceSquared = max(dot(toLight, toLight), 1e-8);
    float inverseDistance = rsqrt(distanceSquared);

    direction = toLight * inverseDistance;

    float normalDotLight = dot(normal, direction);

    if (normalDotLight <= 0.0)
    {
        return false;
    }

    // URP's distance falloff: inverse square, windowed so it reaches exactly zero at the
    // light's range instead of trailing off forever. A directional light packs terms that
    // leave this at one.
    float lightAttenuation = rcp(distanceSquared);
    float factor = distanceSquared * light.attenuation.x;
    float smoothFactor = saturate(1.0 - (factor * factor));
    smoothFactor = smoothFactor * smoothFactor;

    float attenuation = lightAttenuation * smoothFactor;

    // Spot cone, encoded as a scale and offset so it costs one multiply-add. A non-spot light
    // packs terms that leave this at one.
    float spotDot = dot(light.spotDirection.xyz, direction);
    float spotAttenuation = saturate((spotDot * light.attenuation.z) + light.attenuation.w);
    attenuation *= spotAttenuation * spotAttenuation;

    if (attenuation <= 1e-5)
    {
        return false;
    }

    radiance = light.color.rgb * attenuation * normalDotLight;

    // A directional light has no position to stop at, so its ray runs to the far clip.
    rayLength = light.positionType.w > 0.5 ? sqrt(distanceSquared) : 1e5;

    return true;
}

// =====================================================================================
//  Node accessors
// =====================================================================================

/// Returns the number of primitives owned by a leaf; zero identifies an interior node.
uint PrometheusGetPrimitiveCount(PrometheusBVHNode node)
{
    return node.packedLeafRange >> PROMETHEUS_PRIMITIVE_INDEX_BITS;
}

/// Returns the BLAS-local index of a leaf's first primitive.
uint PrometheusGetFirstPrimitive(PrometheusBVHNode node)
{
    return node.packedLeafRange & PROMETHEUS_PRIMITIVE_INDEX_MASK;
}

/// True when the node references primitives directly rather than child nodes.
bool PrometheusIsLeaf(PrometheusBVHNode node)
{
    return PrometheusGetPrimitiveCount(node) != 0u;
}

/// Resolves slot `slot` of a TLAS leaf to an index into _PrometheusInstances.
///
/// What a leaf's packed range means depends on which hierarchy level it belongs to:
///
///   BLAS leaf  -> firstPrimitive indexes _PrometheusTriangles directly, rebased by the
///                 owning instance's blasTriangleOffset.
///   TLAS leaf  -> firstPrimitive indexes _PrometheusInstanceIndices, which in turn holds
///                 the index into _PrometheusInstances.
///
/// That extra hop is what lets one TLAS leaf gather several instances without any
/// instance record being copied or reordered: the sort permutes 4-byte indices only.
/// A TLAS leaf holds up to TlasBuilder.MaxLeafInstances of them, so traversal loops:
///
///     uint first = PrometheusGetFirstPrimitive(node);
///     uint count = PrometheusGetPrimitiveCount(node);
///     [loop] for (uint k = 0; k < count; k++)
///     {
///         uint id = PrometheusGetLeafInstance(node, k);
///         PrometheusInstance inst = _PrometheusInstances[id];
///         // transform the ray and descend into inst.blasNodeOffset
///     }
uint PrometheusGetLeafInstance(PrometheusBVHNode node, uint slot)
{
    return _PrometheusInstanceIndices[PrometheusGetFirstPrimitive(node) + slot];
}

// =====================================================================================
//  Instance accessors
// =====================================================================================

/// Stable mesh handle stored in the low sixteen bits of the packed identity word.
uint PrometheusGetMeshId(PrometheusInstance instance)
{
    return instance.packedFlagsAndMeshId & PROMETHEUS_MESH_ID_MASK;
}

/// PROMETHEUS_INSTANCE_* bit set stored in the high sixteen bits.
uint PrometheusGetInstanceFlags(PrometheusInstance instance)
{
    return instance.packedFlagsAndMeshId >> PROMETHEUS_MESH_ID_BITS;
}

/// True when every bit of `flag` is set on the instance.
bool PrometheusInstanceHasFlag(PrometheusInstance instance, uint flag)
{
    return (PrometheusGetInstanceFlags(instance) & flag) == flag;
}

/// Transforms a world-space point into instance object space.
float3 PrometheusTransformPointToLocal(PrometheusInstance instance, float3 worldPoint)
{
    float4 h = float4(worldPoint, 1.0);
    return float3(
        dot(instance.worldToLocalRow0, h),
        dot(instance.worldToLocalRow1, h),
        dot(instance.worldToLocalRow2, h));
}

/// Transforms a world-space direction into instance object space, un-normalised.
float3 PrometheusTransformDirectionToLocal(PrometheusInstance instance, float3 worldDirection)
{
    return float3(
        dot(instance.worldToLocalRow0.xyz, worldDirection),
        dot(instance.worldToLocalRow1.xyz, worldDirection),
        dot(instance.worldToLocalRow2.xyz, worldDirection));
}

/// Transforms a world-space ray into instance object space.
///
/// The direction is deliberately left un-normalised so that the resulting `t` stays
/// directly comparable with world-space ray distances, removing a rescale per hit.
void PrometheusTransformRayToLocal(
    PrometheusInstance instance,
    float3 worldOrigin,
    float3 worldDirection,
    out float3 localOrigin,
    out float3 localDirection)
{
    localOrigin    = PrometheusTransformPointToLocal(instance, worldOrigin);
    localDirection = PrometheusTransformDirectionToLocal(instance, worldDirection);
}

/// Transforms an object-space normal into world space.
///
/// The correct matrix is the inverse transpose of the forward transform. The stored rows
/// already are the inverse, so accumulating them weighted by the normal's components
/// applies that transpose without reconstructing anything.
float3 PrometheusTransformNormalToWorld(PrometheusInstance instance, float3 localNormal)
{
    float3 n =
        instance.worldToLocalRow0.xyz * localNormal.x +
        instance.worldToLocalRow1.xyz * localNormal.y +
        instance.worldToLocalRow2.xyz * localNormal.z;

    n = normalize(n);
    return PrometheusInstanceHasFlag(instance, PROMETHEUS_INSTANCE_MIRRORED_WINDING) ? -n : n;
}

// =====================================================================================
//  Octahedral normal codec
//
//  Bit-for-bit identical to PrometheusTriangle.EncodeNormal / DecodeNormal in C#.
//  The sphere is projected onto the L1 octahedron and the lower hemisphere is folded
//  outward, giving near-uniform angular error of roughly 0.01 degrees at 16 bits per
//  channel:
//
//      p = n / |n|_1
//      f = (p.z >= 0) ? p.xy : (1 - |p.yx|) * signNonZero(p.xy)
// =====================================================================================

/// Component-wise sign that returns +1 for zero rather than 0.
float2 PrometheusSignNonZero(float2 v)
{
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

/// Packs a unit normal into two 16-bit unorm channels (x in bits [0..15], y in [16..31]).
uint PrometheusOctEncode(float3 normal)
{
    float l1 = abs(normal.x) + abs(normal.y) + abs(normal.z);
    float3 p = normal / max(l1, 1e-20);

    float2 f = (p.z >= 0.0) ? p.xy : (1.0 - abs(p.yx)) * PrometheusSignNonZero(p.xy);
    f = saturate(f * 0.5 + 0.5);

    uint x = (uint)round(f.x * 65535.0);
    uint y = (uint)round(f.y * 65535.0);
    return x | (y << 16);
}

/// Unpacks a normal encoded by PrometheusOctEncode.
float3 PrometheusOctDecode(uint packed)
{
    float2 f = float2(packed & 0xFFFFu, (packed >> 16) & 0xFFFFu) * (1.0 / 65535.0);
    f = f * 2.0 - 1.0;

    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += PrometheusSignNonZero(n.xy) * -t;
    return normalize(n);
}

/// Decoded geometric normal of a triangle, following the counter-clockwise winding
/// convention normalize(cross(edge1, edge2)).
float3 PrometheusGetTriangleNormal(PrometheusTriangle tri)
{
    return PrometheusOctDecode(tri.packedNormal);
}

// Ray/AABB and ray/triangle intersection live in IntersectionMath.hlsl, which includes
// this file. The split keeps this header purely about the CPU/GPU data contract, so the
// layout assertions in MemoryLayoutTests have a single place to look.

#endif // PROMETHEUS_STRUCTURES_INCLUDED
