using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Prometheus.Core.Structs
{
    /// <summary>
    /// Per-instance behaviour flags carried in the high sixteen bits of
    /// <see cref="PrometheusInstance.PackedFlagsAndMeshId"/>.
    /// </summary>
    [Flags]
    public enum PrometheusInstanceFlags : uint
    {
        /// <summary>No special handling.</summary>
        None = 0u,

        /// <summary>The instance participates in ray traversal. Cleared instances are skipped by the TLAS builder.</summary>
        Active = 1u << 0,

        /// <summary>The instance's transform changes per frame and its TLAS entry must be refreshed every build.</summary>
        Dynamic = 1u << 1,

        /// <summary>The instance blocks shadow rays.</summary>
        CastsShadows = 1u << 2,

        /// <summary>The instance contributes bounced radiance to the RTGI pass.</summary>
        ContributesGI = 1u << 3,

        /// <summary>
        /// The instance's transform has a negative determinant, so triangle winding is mirrored
        /// and the interpolated geometric normal must be flipped after transformation.
        /// </summary>
        MirroredWinding = 1u << 4,

        /// <summary>Both triangle faces are intersectable; back-face culling is disabled for this instance.</summary>
        DoubleSided = 1u << 5,

        /// <summary>Default flag set applied to a static opaque scene renderer.</summary>
        DefaultStaticOpaque = Active | CastsShadows | ContributesGI
    }

    /// <summary>
    /// A world-space axis-aligned bounding box.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a CPU-side build input, not a GPU upload structure, so it is packed tightly at 24
    /// bytes rather than padded to a 16-byte boundary. The GPU never needs per-instance bounds:
    /// the TLAS nodes already carry them.
    /// </para>
    /// <para>
    /// Keeping bounds out of <see cref="PrometheusInstance"/> is what allows that record to reach
    /// 64 bytes, and it means the TLAS build passes stream 24 bytes per instance instead of
    /// dragging whole instance records through cache.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusAabb
    {
        /// <summary>Lower corner.</summary>
        public float3 Min;

        /// <summary>Upper corner.</summary>
        public float3 Max;

        /// <summary>Geometric centre, used as the Morton sample point.</summary>
        public float3 Centre
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (Min + Max) * 0.5f;
        }

        /// <summary>An inverted box that acts as the identity element for union accumulation.</summary>
        public static PrometheusAabb Empty
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                PrometheusAabb aabb;
                aabb.Min = new float3(float.PositiveInfinity);
                aabb.Max = new float3(float.NegativeInfinity);
                return aabb;
            }
        }

        /// <summary>Expands this box to also contain <paramref name="other"/>.</summary>
        /// <param name="other">Box to merge in.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Encapsulate(in PrometheusAabb other)
        {
            Min = math.min(Min, other.Min);
            Max = math.max(Max, other.Max);
        }
    }

    /// <summary>
    /// A Top-Level Acceleration Structure entry binding a transform to a Bottom-Level
    /// Acceleration Structure sub-range of the shared node and triangle buffers.
    /// </summary>
    /// <remarks>
    /// <para><b>Memory contract (64 bytes = 4 x 16-byte registers)</b></para>
    /// <code>
    /// Offset  Size  Field                  Purpose
    /// ------  ----  ---------------------  ------------------------------------------------
    ///    0     16   WorldToLocalRow0       Row 0 of the world -> object affine transform
    ///   16     16   WorldToLocalRow1       Row 1
    ///   32     16   WorldToLocalRow2       Row 2
    ///   48      4   BlasNodeOffset         First BLAS node in the shared node buffer
    ///   52      4   BlasTriangleOffset     First BLAS triangle in the shared triangle buffer
    ///   56      4   MaterialId             Material table index
    ///   60      4   PackedFlagsAndMeshId   [0..15] mesh id | [16..31] instance flags
    /// </code>
    /// <para><b>Why three rows rather than a matrix type</b></para>
    /// <para>
    /// The transform is affine, so its bottom row is always <c>(0, 0, 0, 1)</c> and storing it
    /// wastes 16 bytes. The natural C# spelling would be <c>float3x4</c>, which Unity.Mathematics
    /// lays out as four contiguous <c>float3</c> values totalling 48 bytes — but HLSL cannot
    /// reproduce that layout. Its packing rules forbid a member straddling a 16-byte boundary, so
    /// four <c>float3</c> values occupy 64 bytes there, not 48. Three explicit <c>float4</c> rows
    /// are 48 bytes under both sets of rules with no padding anywhere, and are the same packing
    /// DXR uses for its own instance descriptors.
    /// </para>
    /// <para><b>What was removed, and what that costs</b></para>
    /// <para>
    /// The forward <c>localToWorld</c> transform and the world-space AABB used to live here,
    /// making the record 176 bytes. Both were removed to reach 64. The AABB moves to a parallel
    /// <see cref="PrometheusAabb"/> array that only the CPU build reads. The forward transform is
    /// recoverable with <see cref="GetLocalToWorld"/> when a hit position must be returned to
    /// world space; traversal itself never needs it, because the ray is carried into object space
    /// and <c>t</c> is preserved across the change of basis.
    /// </para>
    /// <para><b>Ray transformation</b></para>
    /// <math display="block">
    /// o' = W \begin{bmatrix} o \\ 1 \end{bmatrix}, \qquad
    /// d' = W_{3\times3}\, d
    /// </math>
    /// <para>
    /// The direction is deliberately left un-normalised so <c>t</c> remains directly comparable
    /// against world-space ray distances, which removes a rescale per instance hit.
    /// </para>
    /// <para><b>Normal transformation</b></para>
    /// <para>
    /// Object-space normals require the inverse transpose of the forward transform. Since the
    /// stored rows already are the inverse, that matrix is their transpose, obtained for free by
    /// accumulating the rows weighted by the normal's components rather than dotting against them.
    /// </para>
    /// <para>
    /// The GPU-side mirror of this struct is <c>PrometheusStructures.hlsl :: PrometheusInstance</c>.
    /// Both layouts are pinned by <c>MemoryLayoutTests</c>.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusInstance : IEquatable<PrometheusInstance>
    {
        /// <summary>Exact stride of this struct in bytes. Must equal the HLSL structured buffer stride.</summary>
        public const int SizeInBytes = 64;

        /// <summary>
        /// Maximum number of live instances a single TLAS build accepts. Matches the Phase 3
        /// performance gate (4,096 instances in under 0.20 ms).
        /// </summary>
        public const int MaxInstanceCount = 4096;

        /// <summary>Number of bits reserved for the mesh id inside <see cref="PackedFlagsAndMeshId"/>.</summary>
        public const int MeshIdBits = 16;

        /// <summary>Largest representable mesh id, <c>2^16 - 1</c>.</summary>
        public const uint MaxMeshId = (1u << MeshIdBits) - 1u;

        /// <summary>Sentinel <see cref="MeshId"/> for an instance that has no resolved BLAS entry.</summary>
        public const uint InvalidMeshId = MaxMeshId;

        // --- Registers 0..2 ---------------------------------------------------------------

        /// <summary>Row 0 of the world-to-object affine transform.</summary>
        public float4 WorldToLocalRow0;

        /// <summary>Row 1 of the world-to-object affine transform.</summary>
        public float4 WorldToLocalRow1;

        /// <summary>Row 2 of the world-to-object affine transform.</summary>
        public float4 WorldToLocalRow2;

        // --- Register 3 -------------------------------------------------------------------

        /// <summary>Index of this instance's BLAS root inside the shared <see cref="PrometheusBVHNode"/> buffer.</summary>
        public uint BlasNodeOffset;

        /// <summary>
        /// Index of this instance's first triangle inside the shared <see cref="PrometheusTriangle"/>
        /// buffer. BLAS leaves store BLAS-local primitive indices; this offset rebases them.
        /// </summary>
        public uint BlasTriangleOffset;

        /// <summary>Index into the shared material table used for GI albedo lookups.</summary>
        public uint MaterialId;

        /// <summary>
        /// Packed identity payload: bits <c>[0..15]</c> hold the mesh id, bits <c>[16..31]</c> hold
        /// the <see cref="PrometheusInstanceFlags"/> bit set.
        /// </summary>
        public uint PackedFlagsAndMeshId;

        /// <summary>Stable mesh handle issued by <c>PrometheusMeshCache</c>, used to deduplicate BLAS uploads.</summary>
        public uint MeshId
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => PackedFlagsAndMeshId & MaxMeshId;
        }

        /// <summary>Bit set of <see cref="PrometheusInstanceFlags"/> controlling traversal and shading behaviour.</summary>
        public uint Flags
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => PackedFlagsAndMeshId >> MeshIdBits;
        }

        /// <summary>Whether this instance should be included in the current TLAS build.</summary>
        public bool IsActive
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (Flags & (uint)PrometheusInstanceFlags.Active) != 0u;
        }

        /// <summary>Whether this instance's transform must be refreshed every frame.</summary>
        public bool IsDynamic
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (Flags & (uint)PrometheusInstanceFlags.Dynamic) != 0u;
        }

        /// <summary>Whether this instance occludes shadow rays.</summary>
        public bool CastsShadows
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (Flags & (uint)PrometheusInstanceFlags.CastsShadows) != 0u;
        }

        /// <summary>Whether back-face culling is disabled for this instance.</summary>
        public bool IsDoubleSided
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (Flags & (uint)PrometheusInstanceFlags.DoubleSided) != 0u;
        }

        /// <summary>Tests a single <see cref="PrometheusInstanceFlags"/> bit or bit group.</summary>
        /// <param name="flag">Flag (or flag combination) to test.</param>
        /// <returns><c>true</c> when every bit of <paramref name="flag"/> is set.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool HasFlag(PrometheusInstanceFlags flag) => (Flags & (uint)flag) == (uint)flag;

        /// <summary>Packs a flag set and a mesh id into the <see cref="PackedFlagsAndMeshId"/> encoding.</summary>
        /// <param name="flags">Behaviour flags.</param>
        /// <param name="meshId">Mesh id; clamped to <see cref="MaxMeshId"/>.</param>
        /// <returns>The packed identity word.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint PackFlagsAndMeshId(PrometheusInstanceFlags flags, uint meshId)
        {
            return math.min(meshId, MaxMeshId) | ((uint)flags << MeshIdBits);
        }

        /// <summary>Replaces the flag bits, leaving the mesh id intact.</summary>
        /// <param name="flags">New flag set.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetFlags(PrometheusInstanceFlags flags)
        {
            PackedFlagsAndMeshId = (PackedFlagsAndMeshId & MaxMeshId) | ((uint)flags << MeshIdBits);
        }

        /// <summary>Stores an affine world-to-object transform into the three packed rows.</summary>
        /// <param name="worldToLocal">Inverse transform. Its bottom row is assumed to be <c>(0,0,0,1)</c>.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetWorldToLocal(in float4x4 worldToLocal)
        {
            WorldToLocalRow0 = new float4(worldToLocal.c0.x, worldToLocal.c1.x, worldToLocal.c2.x, worldToLocal.c3.x);
            WorldToLocalRow1 = new float4(worldToLocal.c0.y, worldToLocal.c1.y, worldToLocal.c2.y, worldToLocal.c3.y);
            WorldToLocalRow2 = new float4(worldToLocal.c0.z, worldToLocal.c1.z, worldToLocal.c2.z, worldToLocal.c3.z);
        }

        /// <summary>Rebuilds the full world-to-object matrix from the packed rows.</summary>
        /// <returns>The inverse transform, with an implicit <c>(0,0,0,1)</c> bottom row.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float4x4 GetWorldToLocal()
        {
            return new float4x4(
                new float4(WorldToLocalRow0.x, WorldToLocalRow1.x, WorldToLocalRow2.x, 0.0f),
                new float4(WorldToLocalRow0.y, WorldToLocalRow1.y, WorldToLocalRow2.y, 0.0f),
                new float4(WorldToLocalRow0.z, WorldToLocalRow1.z, WorldToLocalRow2.z, 0.0f),
                new float4(WorldToLocalRow0.w, WorldToLocalRow1.w, WorldToLocalRow2.w, 1.0f));
        }

        /// <summary>
        /// Recovers the forward object-to-world transform by inverting the stored rows.
        /// </summary>
        /// <remarks>
        /// Only shading needs this, and only once per hit, so it is computed rather than stored.
        /// Keeping it out of the record is what halves the instance stream the TLAS build walks.
        /// </remarks>
        /// <returns>The forward transform.</returns>
        public float4x4 GetLocalToWorld()
        {
            return math.inverse(GetWorldToLocal());
        }

        /// <summary>Transforms a world-space point into this instance's object space.</summary>
        /// <param name="worldPoint">World-space position.</param>
        /// <returns>The position in object space.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float3 TransformPointToLocal(in float3 worldPoint)
        {
            float4 homogeneous = new float4(worldPoint, 1.0f);
            return new float3(
                math.dot(WorldToLocalRow0, homogeneous),
                math.dot(WorldToLocalRow1, homogeneous),
                math.dot(WorldToLocalRow2, homogeneous));
        }

        /// <summary>Transforms a world-space direction into object space without normalising it.</summary>
        /// <param name="worldDirection">World-space direction.</param>
        /// <returns>The direction in object space, un-normalised.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float3 TransformDirectionToLocal(in float3 worldDirection)
        {
            return new float3(
                math.dot(WorldToLocalRow0.xyz, worldDirection),
                math.dot(WorldToLocalRow1.xyz, worldDirection),
                math.dot(WorldToLocalRow2.xyz, worldDirection));
        }

        /// <summary>
        /// Transforms a world-space ray into this instance's object space, preserving <c>t</c>.
        /// </summary>
        /// <param name="worldOrigin">World-space ray origin.</param>
        /// <param name="worldDirection">World-space ray direction.</param>
        /// <param name="localOrigin">Resulting object-space ray origin.</param>
        /// <param name="localDirection">Resulting object-space ray direction (un-normalised).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void TransformRayToLocal(
            in float3 worldOrigin,
            in float3 worldDirection,
            out float3 localOrigin,
            out float3 localDirection)
        {
            localOrigin = TransformPointToLocal(worldOrigin);
            localDirection = TransformDirectionToLocal(worldDirection);
        }

        /// <summary>
        /// Transforms an object-space normal into world space using the inverse transpose, then
        /// flips it when the transform mirrors winding.
        /// </summary>
        /// <param name="localNormal">Object-space normal.</param>
        /// <returns>The unit-length world-space normal.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float3 TransformNormalToWorld(in float3 localNormal)
        {
            // Accumulating the rows weighted by the normal applies the transpose of the stored
            // 3x3 block, which is exactly the inverse transpose of the forward transform.
            float3 world =
                (WorldToLocalRow0.xyz * localNormal.x) +
                (WorldToLocalRow1.xyz * localNormal.y) +
                (WorldToLocalRow2.xyz * localNormal.z);

            world = math.normalizesafe(world, new float3(0.0f, 1.0f, 0.0f));
            return HasFlag(PrometheusInstanceFlags.MirroredWinding) ? -world : world;
        }

        /// <summary>
        /// Builds an instance from a forward transform, deriving the inverse, the world-space AABB
        /// and the winding-mirror flag in one pass.
        /// </summary>
        /// <param name="localToWorld">Object-to-world transform of the renderer.</param>
        /// <param name="localBoundsMin">Object-space AABB lower corner of the source mesh.</param>
        /// <param name="localBoundsMax">Object-space AABB upper corner of the source mesh.</param>
        /// <param name="meshId">Handle of the cached mesh whose BLAS backs this instance.</param>
        /// <param name="blasNodeOffset">First BLAS node index in the shared node buffer.</param>
        /// <param name="blasTriangleOffset">First BLAS triangle index in the shared triangle buffer.</param>
        /// <param name="materialId">Material table index.</param>
        /// <param name="flags">Behaviour flags, excluding <see cref="PrometheusInstanceFlags.MirroredWinding"/>.</param>
        /// <param name="worldBounds">Receives the world-space AABB the TLAS build consumes.</param>
        /// <returns>A fully initialised, upload-ready instance record.</returns>
        public static PrometheusInstance Create(
            in float4x4 localToWorld,
            in float3 localBoundsMin,
            in float3 localBoundsMax,
            uint meshId,
            uint blasNodeOffset,
            uint blasTriangleOffset,
            uint materialId,
            PrometheusInstanceFlags flags,
            out PrometheusAabb worldBounds)
        {
            PrometheusInstance instance = default;
            instance.SetWorldToLocal(math.inverse(localToWorld));

            worldBounds = TransformBounds(localToWorld, localBoundsMin, localBoundsMax);

            PrometheusInstanceFlags resolvedFlags = flags;
            if (math.determinant(localToWorld) < 0.0f)
            {
                resolvedFlags |= PrometheusInstanceFlags.MirroredWinding;
            }

            instance.BlasNodeOffset = blasNodeOffset;
            instance.BlasTriangleOffset = blasTriangleOffset;
            instance.MaterialId = materialId;
            instance.PackedFlagsAndMeshId = PackFlagsAndMeshId(resolvedFlags, meshId);
            return instance;
        }

        /// <summary>
        /// Recomputes the stored inverse and the world-space AABB after a transform change, leaving
        /// all BLAS bindings untouched. Used by the per-frame dynamic instance refresh.
        /// </summary>
        /// <param name="localToWorld">Updated object-to-world transform.</param>
        /// <param name="localBoundsMin">Object-space AABB lower corner of the source mesh.</param>
        /// <param name="localBoundsMax">Object-space AABB upper corner of the source mesh.</param>
        /// <param name="worldBounds">Receives the refreshed world-space AABB.</param>
        public void UpdateTransform(
            in float4x4 localToWorld,
            in float3 localBoundsMin,
            in float3 localBoundsMax,
            out PrometheusAabb worldBounds)
        {
            SetWorldToLocal(math.inverse(localToWorld));
            worldBounds = TransformBounds(localToWorld, localBoundsMin, localBoundsMax);

            const uint mirrored = (uint)PrometheusInstanceFlags.MirroredWinding << MeshIdBits;
            PackedFlagsAndMeshId = math.determinant(localToWorld) < 0.0f
                ? (PackedFlagsAndMeshId | mirrored)
                : (PackedFlagsAndMeshId & ~mirrored);
        }

        /// <summary>
        /// Computes the world-space AABB enclosing a transformed object-space AABB using the
        /// centre/extent formulation, which needs three matrix-vector products instead of eight
        /// corner transforms.
        /// </summary>
        /// <remarks>
        /// <math display="block">
        /// c' = M c, \qquad e' = \left| M_{3\times3} \right| e
        /// </math>
        /// where <c>|M|</c> denotes the component-wise absolute value of the upper 3x3 block.
        /// </remarks>
        /// <param name="localToWorld">Object-to-world transform.</param>
        /// <param name="localBoundsMin">Object-space AABB lower corner.</param>
        /// <param name="localBoundsMax">Object-space AABB upper corner.</param>
        /// <returns>The enclosing world-space AABB.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static PrometheusAabb TransformBounds(
            in float4x4 localToWorld,
            in float3 localBoundsMin,
            in float3 localBoundsMax)
        {
            float3 centre = (localBoundsMin + localBoundsMax) * 0.5f;
            float3 extent = (localBoundsMax - localBoundsMin) * 0.5f;

            float3 worldCentre = math.transform(localToWorld, centre);
            float3 worldExtent =
                (math.abs(localToWorld.c0.xyz) * extent.x) +
                (math.abs(localToWorld.c1.xyz) * extent.y) +
                (math.abs(localToWorld.c2.xyz) * extent.z);

            PrometheusAabb bounds;
            bounds.Min = worldCentre - worldExtent;
            bounds.Max = worldCentre + worldExtent;
            return bounds;
        }

        /// <inheritdoc/>
        public bool Equals(PrometheusInstance other)
        {
            return WorldToLocalRow0.Equals(other.WorldToLocalRow0)
                && WorldToLocalRow1.Equals(other.WorldToLocalRow1)
                && WorldToLocalRow2.Equals(other.WorldToLocalRow2)
                && BlasNodeOffset == other.BlasNodeOffset
                && BlasTriangleOffset == other.BlasTriangleOffset
                && MaterialId == other.MaterialId
                && PackedFlagsAndMeshId == other.PackedFlagsAndMeshId;
        }

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is PrometheusInstance other && Equals(other);

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = (int)math.hash(WorldToLocalRow0);
                hash = (hash * 397) ^ (int)math.hash(WorldToLocalRow1);
                hash = (hash * 397) ^ (int)math.hash(WorldToLocalRow2);
                hash = (hash * 397) ^ (int)BlasNodeOffset;
                hash = (hash * 397) ^ (int)BlasTriangleOffset;
                hash = (hash * 397) ^ (int)MaterialId;
                hash = (hash * 397) ^ (int)PackedFlagsAndMeshId;
                return hash;
            }
        }
    }
}
