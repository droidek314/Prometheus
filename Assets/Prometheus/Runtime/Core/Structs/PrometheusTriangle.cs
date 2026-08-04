using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Prometheus.Core.Structs
{
    /// <summary>
    /// A single object-space triangle pre-conditioned for Möller–Trumbore intersection.
    /// </summary>
    /// <remarks>
    /// <para><b>Memory contract (48 bytes = 3 x 16-byte registers)</b></para>
    /// <code>
    /// Offset  Size  Field                Purpose
    /// ------  ----  -------------------  --------------------------------------------------
    ///    0     12   Vertex0              Object-space vertex A
    ///   12      4   PrimitiveId          Source mesh triangle index (hit reporting)
    ///   16     12   Edge1                B - A
    ///   28      4   MaterialId           Material table index override
    ///   32     12   Edge2                C - A
    ///   44      4   PackedNormal         Octahedral-encoded geometric normal (16:16 unorm)
    /// </code>
    /// <para><b>Why edges instead of vertices</b></para>
    /// <para>
    /// Möller–Trumbore consumes <c>A</c>, <c>B - A</c> and <c>C - A</c>, never <c>B</c> or <c>C</c>
    /// directly. Baking the subtractions at build time removes two vector subtracts from the
    /// innermost GPU loop, which runs once per candidate primitive per ray.
    /// </para>
    /// <math display="block">
    /// \begin{aligned}
    /// P &amp;= d \times e_2, &amp; \det &amp;= e_1 \cdot P \\
    /// T &amp;= o - A,         &amp; u   &amp;= (T \cdot P) / \det \\
    /// Q &amp;= T \times e_1,   &amp; v   &amp;= (d \cdot Q) / \det \\
    ///   &amp;                 &amp; t   &amp;= (e_2 \cdot Q) / \det
    /// \end{aligned}
    /// </math>
    /// <para>
    /// A hit is reported when <c>u &gt;= 0</c>, <c>v &gt;= 0</c>, <c>u + v &lt;= 1</c> and
    /// <c>t</c> lies inside the ray interval.
    /// </para>
    /// <para><b>Degenerate geometry</b></para>
    /// <para>
    /// Edge case <c>EC-003</c> rejects triangles whose area falls below <c>1e-7</c>. Because
    /// <c>2A = \lVert e_1 \times e_2 \rVert</c>, that test is available directly from the stored
    /// edges via <see cref="IsDegenerate"/> without reconstructing the vertices.
    /// </para>
    /// <para>
    /// The GPU-side mirror of this struct is <c>PrometheusStructures.hlsl :: PrometheusTriangle</c>.
    /// Both layouts are pinned by <c>MemoryLayoutTests</c>.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusTriangle : IEquatable<PrometheusTriangle>
    {
        /// <summary>Exact stride of this struct in bytes. Must equal the HLSL structured buffer stride.</summary>
        public const int SizeInBytes = 48;

        /// <summary>
        /// Minimum accepted triangle area. Anything at or below this is treated as degenerate and
        /// dropped during BLAS construction (edge case <c>EC-003: INVALID_GEOMETRY</c>).
        /// </summary>
        public const float MinValidArea = 1e-7f;

        /// <summary>
        /// Hard ceiling on the number of triangles a scene may submit, taken from the Structural
        /// Bounds Specification (Section 3.1). Exceeding it raises <c>ERROR_BVH_CAPACITY_EXCEEDED</c>.
        /// </summary>
        public const int MaxSceneTriangles = 1 << 21;

        /// <summary>Quantisation scale of one octahedral normal channel (16-bit unorm).</summary>
        private const float OctScale = 65535.0f;

        // --- Register 0 -------------------------------------------------------------------

        /// <summary>Object-space position of the triangle's first vertex (<c>A</c>).</summary>
        public float3 Vertex0;

        /// <summary>Index of this triangle within its source mesh, reported on hit for shading lookups.</summary>
        public uint PrimitiveId;

        // --- Register 1 -------------------------------------------------------------------

        /// <summary>First edge vector, <c>B - A</c>, pre-computed at build time.</summary>
        public float3 Edge1;

        /// <summary>
        /// Material table index. Overrides <see cref="PrometheusInstance.MaterialId"/> so that
        /// multi-submesh renderers can share a single BLAS.
        /// </summary>
        public uint MaterialId;

        // --- Register 2 -------------------------------------------------------------------

        /// <summary>Second edge vector, <c>C - A</c>, pre-computed at build time.</summary>
        public float3 Edge2;

        /// <summary>
        /// Geometric normal encoded octahedrally as two 16-bit unorm channels. Decoding costs a few
        /// ALU operations against the 8 extra bytes a raw <c>float3</c> would add to every triangle,
        /// which at the 2^21 triangle ceiling is 16 MB of the VRAM budget.
        /// </summary>
        public uint PackedNormal;

        /// <summary>Second vertex of the triangle (<c>B</c>), reconstructed from <see cref="Edge1"/>.</summary>
        public float3 Vertex1
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Vertex0 + Edge1;
        }

        /// <summary>Third vertex of the triangle (<c>C</c>), reconstructed from <see cref="Edge2"/>.</summary>
        public float3 Vertex2
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Vertex0 + Edge2;
        }

        /// <summary>Centre of mass of the three vertices, used as the Morton code sample point during BLAS builds.</summary>
        /// <remarks>
        /// <math display="block">
        /// c = A + \frac{e_1 + e_2}{3}
        /// </math>
        /// </remarks>
        public float3 Centroid
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Vertex0 + ((Edge1 + Edge2) * (1.0f / 3.0f));
        }

        /// <summary>Surface area of the triangle, <c>0.5 * |e1 x e2|</c>.</summary>
        public float Area
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => math.length(math.cross(Edge1, Edge2)) * 0.5f;
        }

        /// <summary>
        /// <c>true</c> when the triangle is unusable for intersection: any coordinate is NaN or
        /// infinite, or its area is at or below <see cref="MinValidArea"/>.
        /// </summary>
        public bool IsDegenerate
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                bool finite = math.all(math.isfinite(Vertex0)) && math.all(math.isfinite(Edge1)) && math.all(math.isfinite(Edge2));
                return !finite || (math.lengthsq(math.cross(Edge1, Edge2)) <= (4.0f * MinValidArea * MinValidArea));
            }
        }

        /// <summary>Object-space AABB lower corner of the triangle.</summary>
        public float3 BoundsMin
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => math.min(Vertex0, math.min(Vertex0 + Edge1, Vertex0 + Edge2));
        }

        /// <summary>Object-space AABB upper corner of the triangle.</summary>
        public float3 BoundsMax
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => math.max(Vertex0, math.max(Vertex0 + Edge1, Vertex0 + Edge2));
        }

        /// <summary>
        /// Unit-length geometric normal decoded from <see cref="PackedNormal"/>, following the
        /// counter-clockwise winding convention <c>normalize(e1 x e2)</c>.
        /// </summary>
        public float3 Normal
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => DecodeNormal(PackedNormal);
        }

        /// <summary>
        /// Builds a triangle from three object-space vertices, deriving the edges and encoding the
        /// geometric normal.
        /// </summary>
        /// <param name="a">First vertex.</param>
        /// <param name="b">Second vertex.</param>
        /// <param name="c">Third vertex.</param>
        /// <param name="primitiveId">Index of the triangle within its source mesh.</param>
        /// <param name="materialId">Material table index.</param>
        /// <returns>A fully initialised, upload-ready triangle record.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static PrometheusTriangle Create(in float3 a, in float3 b, in float3 c, uint primitiveId, uint materialId)
        {
            float3 edge1 = b - a;
            float3 edge2 = c - a;

            PrometheusTriangle triangle;
            triangle.Vertex0 = a;
            triangle.PrimitiveId = primitiveId;
            triangle.Edge1 = edge1;
            triangle.MaterialId = materialId;
            triangle.Edge2 = edge2;
            triangle.PackedNormal = EncodeNormal(math.normalizesafe(math.cross(edge1, edge2), new float3(0.0f, 1.0f, 0.0f)));
            return triangle;
        }

        /// <summary>
        /// Encodes a unit normal into two 16-bit unorm channels using an octahedral projection.
        /// </summary>
        /// <remarks>
        /// The sphere is projected onto the <c>L1</c> octahedron and the lower hemisphere is folded
        /// outward, giving a near-uniform angular error of roughly 0.01 degrees at 16 bits per channel:
        /// <math display="block">
        /// p = \frac{n}{\lVert n \rVert_1}, \qquad
        /// f = \begin{cases}
        ///   p_{xy} &amp; p_z \ge 0 \\
        ///   \left(1 - \lvert p_{yx} \rvert\right) \cdot \operatorname{sign}(p_{xy}) &amp; p_z &lt; 0
        /// \end{cases}
        /// </math>
        /// This routine is bit-for-bit identical to <c>PrometheusOctEncode</c> in
        /// <c>PrometheusStructures.hlsl</c>.
        /// </remarks>
        /// <param name="normal">Unit-length normal to encode.</param>
        /// <returns>The packed normal: channel <c>x</c> in bits <c>[0..15]</c>, channel <c>y</c> in bits <c>[16..31]</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint EncodeNormal(in float3 normal)
        {
            float l1 = math.abs(normal.x) + math.abs(normal.y) + math.abs(normal.z);
            float3 p = normal / math.max(l1, 1e-20f);

            float2 f = p.z >= 0.0f
                ? p.xy
                : (1.0f - math.abs(p.yx)) * SignNonZero(p.xy);

            f = math.saturate(f * 0.5f + 0.5f);

            uint x = (uint)math.round(f.x * OctScale);
            uint y = (uint)math.round(f.y * OctScale);
            return x | (y << 16);
        }

        /// <summary>Decodes an octahedrally packed normal produced by <see cref="EncodeNormal"/>.</summary>
        /// <param name="packed">Packed normal, channel <c>x</c> in bits <c>[0..15]</c>, channel <c>y</c> in bits <c>[16..31]</c>.</param>
        /// <returns>The reconstructed unit-length normal.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3 DecodeNormal(uint packed)
        {
            float2 f = new float2(packed & 0xFFFFu, (packed >> 16) & 0xFFFFu) * (1.0f / OctScale);
            f = f * 2.0f - 1.0f;

            float3 n = new float3(f.x, f.y, 1.0f - math.abs(f.x) - math.abs(f.y));
            float t = math.saturate(-n.z);
            n.xy += SignNonZero(n.xy) * -t;
            return math.normalize(n);
        }

        /// <summary>Component-wise sign that returns <c>+1</c> for zero rather than <c>0</c>.</summary>
        /// <param name="v">Input vector.</param>
        /// <returns>A vector whose components are <c>+1</c> or <c>-1</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float2 SignNonZero(in float2 v)
        {
            return new float2(v.x >= 0.0f ? 1.0f : -1.0f, v.y >= 0.0f ? 1.0f : -1.0f);
        }

        /// <inheritdoc/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(PrometheusTriangle other)
        {
            return math.all(Vertex0 == other.Vertex0)
                && math.all(Edge1 == other.Edge1)
                && math.all(Edge2 == other.Edge2)
                && PrimitiveId == other.PrimitiveId
                && MaterialId == other.MaterialId
                && PackedNormal == other.PackedNormal;
        }

        /// <inheritdoc/>
        public override bool Equals(object obj) => obj is PrometheusTriangle other && Equals(other);

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = (int)math.hash(Vertex0);
                hash = (hash * 397) ^ (int)math.hash(Edge1);
                hash = (hash * 397) ^ (int)math.hash(Edge2);
                hash = (hash * 397) ^ (int)PrimitiveId;
                hash = (hash * 397) ^ (int)MaterialId;
                hash = (hash * 397) ^ (int)PackedNormal;
                return hash;
            }
        }
    }
}
