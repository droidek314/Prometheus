using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Prometheus.Core.Structs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Prometheus.Core.BVH
{
    /// <summary>
    /// A Morton key paired with the instance it was derived from, so the sort can permute
    /// instances without moving the (much larger) instance records.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MortonEntry
    {
        /// <summary>30-bit Morton code occupying the low bits of a 32-bit word.</summary>
        public uint Code;

        /// <summary>Index into the caller's submitted instance array, or <c>-1</c> when inactive.</summary>
        public int InstanceIndex;
    }

    /// <summary>
    /// Normalisation frame for a Morton pass: the lower corner of the centroid bounds and the
    /// reciprocal of their extent.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct TlasCentroidBounds
    {
        /// <summary>Lower corner of the AABB enclosing every instance centroid.</summary>
        public float3 Min;

        /// <summary>
        /// Per-axis reciprocal extent. Axes with zero extent store zero, which maps every centroid
        /// on that axis to the same bin rather than producing an infinity.
        /// </summary>
        public float3 InverseExtent;

        /// <summary>Maps a world-space centroid into the unit cube.</summary>
        /// <remarks>
        /// Saturating rather than asserting is deliberate: the frame may be one build stale, and a
        /// centroid that has moved outside it must still produce a valid code. It lands in an edge
        /// bin, which costs a little tree quality and nothing in correctness.
        /// </remarks>
        /// <param name="centroid">World-space point.</param>
        /// <returns>The point expressed in <c>[0, 1]</c> along each axis.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float3 Normalize(float3 centroid)
        {
            return math.saturate((centroid - Min) * InverseExtent);
        }

        /// <summary>Builds a normalisation frame from a centroid AABB.</summary>
        /// <param name="min">Lower corner of the centroid AABB.</param>
        /// <param name="max">Upper corner of the centroid AABB.</param>
        /// <returns>The frame, with degenerate axes collapsed to bin zero.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TlasCentroidBounds FromCentroidAabb(float3 min, float3 max)
        {
            float3 extent = max - min;

            TlasCentroidBounds bounds;
            bounds.Min = min;
            bounds.InverseExtent = math.select(1.0f / extent, float3.zero, extent <= 0.0f);
            return bounds;
        }
    }

    /// <summary>
    /// 30-bit three-dimensional Morton (Z-order) code encoding.
    /// </summary>
    public static class PrometheusMorton
    {
        /// <summary>Bits of precision retained per axis.</summary>
        public const int BitsPerAxis = 10;

        /// <summary>Largest quantised coordinate, <c>2^10 - 1</c>.</summary>
        public const uint AxisMax = (1u << BitsPerAxis) - 1u;

        /// <summary>Largest code the encoder can produce, <c>2^30 - 1</c>.</summary>
        public const uint CodeMax = (1u << (3 * BitsPerAxis)) - 1u;

        /// <summary>
        /// Code written for slots whose instance is inactive.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Reserving the very top code, and clamping real codes to <see cref="MaxActiveCode"/>,
        /// guarantees inactive slots sort strictly after every live instance. That is what lets the
        /// active count be recovered by a binary search for the first sentinel rather than by a
        /// separate counting pass.
        /// </para>
        /// <para>
        /// A plain out-of-range sentinel would not do: only the low thirty bits take part in the
        /// radix sort, so anything above <see cref="CodeMax"/> would be read as
        /// <see cref="CodeMax"/> and could tie with a real instance sitting at the far corner of
        /// the scene, leaving a sentinel stranded ahead of live geometry.
        /// </para>
        /// </remarks>
        public const uint InactiveCode = CodeMax;

        /// <summary>Largest code assigned to a live instance, one below the inactive sentinel.</summary>
        public const uint MaxActiveCode = CodeMax - 1u;

        /// <summary>
        /// Spreads the low ten bits of <paramref name="value"/> so that two zero bits sit between
        /// each original bit.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Each step multiplies by a constant that duplicates the word at a chosen distance and
        /// then masks away everything except the bits that landed where they belong, halving the
        /// size of each contiguous run of bits per step:
        /// </para>
        /// <code>
        /// start   0000000000000000000000aaaaaaaaaa
        /// step 1  0000000000aaaaaaaa00000000000aa0   (x * 0x00010001 &amp; 0xFF0000FF)
        /// step 2  000000aaaa00000000aaaa000000aaaa   (x * 0x00000101 &amp; 0x0F00F00F)
        /// step 3  aa000000aa000000aa000000aa0000aa   (x * 0x00000011 &amp; 0xC30C30C3)
        /// step 4  00a00a00a00a00a00a00a00a00a00a0a   (x * 0x00000005 &amp; 0x49249249)
        /// </code>
        /// <para>
        /// The result occupies bits <c>0, 3, 6, ... 27</c>. Inputs above <see cref="AxisMax"/>
        /// produce undefined placement, so callers must clamp first.
        /// </para>
        /// </remarks>
        /// <param name="value">A value in <c>[0, 1023]</c>.</param>
        /// <returns>The bit-spread value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ExpandBits(uint value)
        {
            value = (value * 0x00010001u) & 0xFF0000FFu;
            value = (value * 0x00000101u) & 0x0F00F00Fu;
            value = (value * 0x00000011u) & 0xC30C30C3u;
            value = (value * 0x00000005u) & 0x49249249u;
            return value;
        }

        /// <summary>
        /// Calculates the 30-bit Morton code for a point already normalised into the unit cube.
        /// </summary>
        /// <remarks>
        /// <para>The code interleaves the quantised coordinates one bit at a time:</para>
        /// <math display="block">
        /// M(x, y, z) = \sum_{i=0}^{9} \left( x_i \cdot 2^{3i+2} + y_i \cdot 2^{3i+1} + z_i \cdot 2^{3i} \right)
        /// </math>
        /// <para>
        /// This matches the reference implementation in Section 4.3, whose final step is
        /// <c>(x &lt;&lt; 2) | (y &lt;&lt; 1) | z</c> and therefore places <c>x</c> in the most
        /// significant slot of each triplet. The LaTeX printed alongside that snippet in Section
        /// 4.3 assigns <c>x</c> to <c>2^{3i}</c> instead; the two disagree. The executable form is
        /// followed here. Axis ordering only permutes which axis dominates the Z-order curve, so
        /// either convention builds a valid hierarchy provided it is applied consistently.
        /// </para>
        /// <para>
        /// A NaN coordinate would survive <c>clamp</c> and produce an undefined conversion, so
        /// coordinates are forced finite before quantisation.
        /// </para>
        /// </remarks>
        /// <param name="normalizedPoint">Point position normalised within scene bounds.</param>
        /// <returns>A 32-bit unsigned integer containing the packed 30-bit Morton code.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Encode(float3 normalizedPoint)
        {
            float3 safePoint = math.select(normalizedPoint, float3.zero, math.isnan(normalizedPoint));
            float3 scaled = math.clamp(safePoint * (AxisMax + 1.0f), 0.0f, AxisMax);

            uint x = ExpandBits((uint)scaled.x);
            uint y = ExpandBits((uint)scaled.y);
            uint z = ExpandBits((uint)scaled.z);

            return (x << 2) | (y << 1) | z;
        }
    }

    /// <summary>
    /// Single fused pass: reads the submitted instances and their world bounds, and emits one
    /// sortable Morton key per slot.
    /// </summary>
    /// <remarks>
    /// <para><b>What had to go for this to be one parallel pass</b></para>
    /// <para>
    /// Two dependencies previously forced a serial collect ahead of the Morton pass, and both were
    /// removed rather than worked around:
    /// </para>
    /// <list type="number">
    /// <item><description>
    /// <b>Compaction.</b> Writing active instances into a dense prefix needs a running index, which
    /// is a prefix sum and cannot be done per-element in parallel. It is gone: TLAS leaves record
    /// the instance's index in the caller's own array, so nothing is moved. That also deletes a
    /// full instance-sized write and a bounds copy from the frame.
    /// </description></item>
    /// <item><description>
    /// <b>The normalisation frame.</b> Morton codes need the centroid AABB of every instance, which
    /// is a reduction and therefore a barrier. Instead the frame arrives from the previous build,
    /// accumulated for free while the flatten pass walked the leaves. Because
    /// <see cref="TlasCentroidBounds.Normalize"/> saturates, a stale frame is always safe; it costs
    /// a little Z-order quality for one build after the scene shifts, and nothing else.
    /// </description></item>
    /// </list>
    /// <para><b>Batched rather than per-element</b></para>
    /// <para>
    /// Implemented over <see cref="IJobParallelForBatch"/> so the single-element normalisation
    /// frame is loaded once per batch instead of once per instance. Under Editor safety checks
    /// that is thousands of bounds-checked reads removed.
    /// </para>
    /// </remarks>
    [BurstCompile(FloatPrecision.Standard, FloatMode.Default, CompileSynchronously = true)]
    public struct CollectAndMortonJob : IJobParallelForBatch
    {
        /// <summary>Instance records submitted by the caller.</summary>
        [ReadOnly] public NativeArray<PrometheusInstance> Source;

        /// <summary>World AABBs parallel to <see cref="Source"/>.</summary>
        [ReadOnly] public NativeArray<PrometheusAabb> SourceBounds;

        /// <summary>Single-element normalisation frame carried over from the previous build.</summary>
        [ReadOnly] public NativeArray<TlasCentroidBounds> Normalization;

        /// <summary>Receives one key per submitted slot.</summary>
        [WriteOnly] public NativeArray<MortonEntry> Entries;

        /// <inheritdoc/>
        public void Execute(int startIndex, int count)
        {
            TlasCentroidBounds frame = Normalization[0];
            int end = startIndex + count;

            for (int i = startIndex; i < end; i++)
            {
                MortonEntry entry;

                if (!Source[i].IsActive)
                {
                    entry.Code = PrometheusMorton.InactiveCode;
                    entry.InstanceIndex = -1;
                    Entries[i] = entry;
                    continue;
                }

                float3 normalized = frame.Normalize(SourceBounds[i].Centre);

                // Clamped one below the sentinel so inactive slots always sort strictly last.
                entry.Code = math.min(PrometheusMorton.Encode(normalized), PrometheusMorton.MaxActiveCode);
                entry.InstanceIndex = i;
                Entries[i] = entry;
            }
        }
    }
}
