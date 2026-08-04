using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Prometheus.Core.Structs
{
    /// <summary>
    /// A single node of a Linear Bounding Volume Hierarchy (LBVH), laid out for stackless
    /// skip-pointer (rope tree) traversal on Shader Model 5.0 compute hardware.
    /// </summary>
    /// <remarks>
    /// <para><b>Memory contract (32 bytes = 2 x 16-byte registers)</b></para>
    /// <code>
    /// Offset  Size  Field                Purpose
    /// ------  ----  -------------------  --------------------------------------------------
    ///   0      12   BoundsMin            AABB lower corner (float3)
    ///  12       4   EscapeIndex          Node to jump to when this AABB is missed
    ///  16      12   BoundsMax            AABB upper corner (float3)
    ///  28       4   PackedLeafRange      [0..21] first primitive | [22..31] primitive count
    /// </code>
    /// <para><b>Traversal invariants</b></para>
    /// <para>
    /// Nodes are emitted in depth-first order, therefore the left child of an interior node is
    /// always <c>selfIndex + 1</c> and never needs to be stored. Traversal is a flat loop:
    /// </para>
    /// <code>
    /// i = 0;
    /// while (i != PrometheusBVHNode.InvalidIndex)
    ///     i = IntersectsAabb(ray, node[i]) ? (i + 1) : node[i].EscapeIndex;
    /// </code>
    /// <para>
    /// This removes the per-thread traversal stack entirely, which is what keeps register
    /// pressure low enough to avoid spilling on SM 5.0 hardware (see ADR-001).
    /// </para>
    /// <para><b>Slab intersection</b> evaluated against the packed bounds is the standard
    /// branch-free formulation:</para>
    /// <math display="block">
    /// t_{\text{enter}} = \max_{k \in \{x,y,z\}} \min(t_k^{\min}, t_k^{\max}), \quad
    /// t_{\text{exit}}  = \min_{k \in \{x,y,z\}} \max(t_k^{\min}, t_k^{\max}), \quad
    /// t_k^{\min} = \frac{\text{BoundsMin}_k - o_k}{d_k}
    /// </math>
    /// <para>A hit is reported when <c>t_exit &gt;= max(t_enter, 0)</c> and <c>t_enter &lt;= t_max</c>.</para>
    /// <para>
    /// The CPU-side mirror of this struct is <c>PrometheusStructures.hlsl :: PrometheusBVHNode</c>.
    /// Both layouts are pinned by <c>MemoryLayoutTests</c>.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusBVHNode : IEquatable<PrometheusBVHNode>
    {
        /// <summary>Exact stride of this struct in bytes. Must equal the HLSL structured buffer stride.</summary>
        public const int SizeInBytes = 32;

        /// <summary>
        /// Sentinel written into <see cref="EscapeIndex"/> of the right-most spine of the tree,
        /// signalling that traversal has walked past the root and must terminate.
        /// </summary>
        public const uint InvalidIndex = 0xFFFFFFFFu;

        /// <summary>Number of bits reserved for the first-primitive index inside <see cref="PackedLeafRange"/>.</summary>
        public const int PrimitiveIndexBits = 22;

        /// <summary>Number of bits reserved for the primitive count inside <see cref="PackedLeafRange"/>.</summary>
        public const int PrimitiveCountBits = 10;

        /// <summary>
        /// Largest addressable primitive index (<c>2^22 - 1 = 4,194,303</c>), which covers the
        /// <c>2^21</c> triangle ceiling from the Structural Bounds Specification (Section 3.1).
        /// </summary>
        public const uint MaxPrimitiveIndex = (1u << PrimitiveIndexBits) - 1u;

        /// <summary>Largest number of primitives a single leaf may reference (<c>2^10 - 1 = 1023</c>).</summary>
        public const uint MaxLeafPrimitiveCount = (1u << PrimitiveCountBits) - 1u;

        private const uint PrimitiveIndexMask = MaxPrimitiveIndex;

        // --- Register 0 -------------------------------------------------------------------

        /// <summary>Lower corner of the node's axis-aligned bounding box, in the space of its owning hierarchy level.</summary>
        public float3 BoundsMin;

        /// <summary>
        /// Index of the node to visit when the ray misses <see cref="BoundsMin"/>/<see cref="BoundsMax"/>,
        /// or <see cref="InvalidIndex"/> to terminate traversal. This is the "rope" of the skip-pointer tree.
        /// </summary>
        public uint EscapeIndex;

        // --- Register 1 -------------------------------------------------------------------

        /// <summary>Upper corner of the node's axis-aligned bounding box.</summary>
        public float3 BoundsMax;

        /// <summary>
        /// Packed leaf payload: bits <c>[0..21]</c> hold the first primitive index, bits <c>[22..31]</c>
        /// hold the primitive count. A count of zero identifies an interior node.
        /// </summary>
        public uint PackedLeafRange;

        /// <summary><c>true</c> when this node references primitives directly rather than child nodes.</summary>
        public bool IsLeaf
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (PackedLeafRange >> PrimitiveIndexBits) != 0u;
        }

        /// <summary>Index of this leaf's first primitive within the shared primitive buffer. Meaningless for interior nodes.</summary>
        public uint FirstPrimitive
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => PackedLeafRange & PrimitiveIndexMask;
        }

        /// <summary>Number of consecutive primitives owned by this leaf. Zero for interior nodes.</summary>
        public uint PrimitiveCount
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => PackedLeafRange >> PrimitiveIndexBits;
        }

        /// <summary>Geometric centre of the node's bounding box.</summary>
        public float3 Centroid
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (BoundsMin + BoundsMax) * 0.5f;
        }

        /// <summary>
        /// Surface area of the node's bounding box, the cost term of the Surface Area Heuristic
        /// used by the Phase 2 BLAS builder.
        /// </summary>
        /// <remarks>
        /// <math display="block">
        /// A(N) = 2\left(e_x e_y + e_y e_z + e_z e_x\right), \quad e = \text{BoundsMax} - \text{BoundsMin}
        /// </math>
        /// Returns <c>0</c> for an inverted (empty) box rather than a negative area.
        /// </remarks>
        public float SurfaceArea
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                float3 e = math.max(BoundsMax - BoundsMin, float3.zero);
                return 2.0f * ((e.x * e.y) + (e.y * e.z) + (e.z * e.x));
            }
        }

        /// <summary>Packs a first-primitive index and a primitive count into the <see cref="PackedLeafRange"/> encoding.</summary>
        /// <param name="firstPrimitive">Index of the first primitive; clamped to <see cref="MaxPrimitiveIndex"/>.</param>
        /// <param name="primitiveCount">Primitive count; clamped to <see cref="MaxLeafPrimitiveCount"/>.</param>
        /// <returns>The packed 32-bit leaf range.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint PackLeafRange(uint firstPrimitive, uint primitiveCount)
        {
            uint index = math.min(firstPrimitive, MaxPrimitiveIndex);
            uint count = math.min(primitiveCount, MaxLeafPrimitiveCount);
            return index | (count << PrimitiveIndexBits);
        }

        /// <summary>Creates an interior node covering the supplied bounds.</summary>
        /// <param name="boundsMin">AABB lower corner.</param>
        /// <param name="boundsMax">AABB upper corner.</param>
        /// <param name="escapeIndex">Node index to jump to on an AABB miss, or <see cref="InvalidIndex"/>.</param>
        /// <returns>A fully initialised interior node.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static PrometheusBVHNode CreateInterior(float3 boundsMin, float3 boundsMax, uint escapeIndex)
        {
            PrometheusBVHNode node;
            node.BoundsMin = boundsMin;
            node.EscapeIndex = escapeIndex;
            node.BoundsMax = boundsMax;
            node.PackedLeafRange = 0u;
            return node;
        }

        /// <summary>Creates a leaf node referencing a contiguous run of primitives.</summary>
        /// <param name="boundsMin">AABB lower corner.</param>
        /// <param name="boundsMax">AABB upper corner.</param>
        /// <param name="escapeIndex">Node index to jump to on an AABB miss, or <see cref="InvalidIndex"/>.</param>
        /// <param name="firstPrimitive">Index of the first referenced primitive.</param>
        /// <param name="primitiveCount">Number of referenced primitives; must be at least 1.</param>
        /// <returns>A fully initialised leaf node.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static PrometheusBVHNode CreateLeaf(
            float3 boundsMin,
            float3 boundsMax,
            uint escapeIndex,
            uint firstPrimitive,
            uint primitiveCount)
        {
            PrometheusBVHNode node;
            node.BoundsMin = boundsMin;
            node.EscapeIndex = escapeIndex;
            node.BoundsMax = boundsMax;
            node.PackedLeafRange = PackLeafRange(firstPrimitive, math.max(primitiveCount, 1u));
            return node;
        }

        /// <summary>
        /// Creates a node whose bounds are inverted (min = +inf, max = -inf) so that it acts as the
        /// identity element for AABB union accumulation inside Burst jobs.
        /// </summary>
        /// <returns>An empty interior node with a terminating escape index.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static PrometheusBVHNode CreateEmpty()
        {
            PrometheusBVHNode node;
            node.BoundsMin = new float3(float.PositiveInfinity);
            node.EscapeIndex = InvalidIndex;
            node.BoundsMax = new float3(float.NegativeInfinity);
            node.PackedLeafRange = 0u;
            return node;
        }

        /// <summary>Expands this node's bounds to also contain <paramref name="other"/>.</summary>
        /// <param name="other">Node whose bounds are merged into this one.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void EncapsulateBounds(in PrometheusBVHNode other)
        {
            BoundsMin = math.min(BoundsMin, other.BoundsMin);
            BoundsMax = math.max(BoundsMax, other.BoundsMax);
        }

        /// <summary>Expands this node's bounds to also contain <paramref name="point"/>.</summary>
        /// <param name="point">Position merged into the bounding box.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void EncapsulatePoint(in float3 point)
        {
            BoundsMin = math.min(BoundsMin, point);
            BoundsMax = math.max(BoundsMax, point);
        }

        /// <inheritdoc/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(PrometheusBVHNode other)
        {
            return math.all(BoundsMin == other.BoundsMin)
                && math.all(BoundsMax == other.BoundsMax)
                && EscapeIndex == other.EscapeIndex
                && PackedLeafRange == other.PackedLeafRange;
        }

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is PrometheusBVHNode other && Equals(other);

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = (int)math.hash(BoundsMin);
                hash = (hash * 397) ^ (int)math.hash(BoundsMax);
                hash = (hash * 397) ^ (int)EscapeIndex;
                hash = (hash * 397) ^ (int)PackedLeafRange;
                return hash;
            }
        }
    }
}
