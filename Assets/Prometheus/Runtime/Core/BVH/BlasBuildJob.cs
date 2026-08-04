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
    /// Terminal state of a single <see cref="BlasBuildJob"/> execution.
    /// </summary>
    public enum PrometheusBlasBuildStatus
    {
        /// <summary>A complete, traversable hierarchy was produced.</summary>
        Success = 0,

        /// <summary>The submitted geometry contained no triangles, or every triangle was degenerate.</summary>
        EmptyGeometry = 1,

        /// <summary>
        /// The geometry exceeded a Structural Bounds limit from Section 3.1, or the supplied output
        /// buffers were too small. No usable hierarchy was produced.
        /// </summary>
        CapacityExceeded = 2,

        /// <summary>Input buffers were inconsistent (for example an index outside the vertex range).</summary>
        InvalidInput = 3
    }

    /// <summary>
    /// Partitioning strategy used by <see cref="BlasBuildJob"/>.
    /// </summary>
    public enum PrometheusBlasSplitStrategy
    {
        /// <summary>
        /// Binned Surface Area Heuristic across all three axes. The Phase 2 default: slower to
        /// build, but produces the tight, low-overlap hierarchy the GPU traversal depends on.
        /// </summary>
        BinnedSah = 0,

        /// <summary>
        /// Splits at the midpoint of the widest centroid axis, ignoring surface area entirely.
        /// Retained as the reference baseline the SAH is measured against.
        /// </summary>
        SpatialMedian = 1
    }

    /// <summary>
    /// A contiguous run of triangle indices sharing one material, mirroring a Unity sub-mesh.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusSubMeshRange
    {
        /// <summary>Offset of this sub-mesh's first index within the shared index buffer.</summary>
        public int IndexStart;

        /// <summary>Number of indices in this sub-mesh. Always a multiple of three.</summary>
        public int IndexCount;

        /// <summary>Material identifier written into every triangle produced from this range.</summary>
        public uint MaterialId;
    }

    /// <summary>
    /// Diagnostics returned by <see cref="BlasBuildJob"/>, written to a single-element output array
    /// because a job cannot return a value.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct BlasBuildResult
    {
        /// <summary>Number of nodes written to the output node buffer.</summary>
        public int NodeCount;

        /// <summary>Number of triangles written to the output triangle buffer, after degenerate filtering.</summary>
        public int TriangleCount;

        /// <summary>Triangles rejected by the <c>EC-003</c> filter (degenerate area, NaN/Inf, or bad index).</summary>
        public int RejectedTriangleCount;

        /// <summary>Depth of the deepest leaf, with the root counted as depth zero.</summary>
        public int MaxDepth;

        /// <summary>Object-space AABB lower corner of the whole hierarchy.</summary>
        public float3 BoundsMin;

        /// <summary>Object-space AABB upper corner of the whole hierarchy.</summary>
        public float3 BoundsMax;

        /// <summary>Terminal state of the build.</summary>
        public PrometheusBlasBuildStatus Status;
    }

    /// <summary>
    /// Burst-compiled builder that turns a triangle soup into a Bottom-Level Acceleration Structure
    /// using binned Surface Area Heuristic partitioning.
    /// </summary>
    /// <remarks>
    /// <para><b>Why SAH here and Morton codes in Phase 3</b></para>
    /// <para>
    /// BLAS structures are built once at load time, so Section 7.1 Phase 2 trades build speed for
    /// traversal speed. A binned SAH sweep costs far more than the Morton/radix construction used
    /// for the per-frame TLAS, but produces tighter boxes with less sibling overlap, which is what
    /// the GPU pays for on every ray.
    /// </para>
    /// <para><b>The cost model</b></para>
    /// <para>
    /// Each candidate split plane is scored with the standard surface area heuristic: the
    /// probability that a uniformly distributed ray hitting a parent box also hits a child box is
    /// the ratio of their surface areas, so the expected cost of splitting is
    /// </para>
    /// <math display="block">
    /// C_{\text{split}} = C_T + \frac{A(N_L)\,|N_L| + A(N_R)\,|N_R|}{A(N)} \, C_I,
    /// \qquad
    /// C_{\text{leaf}} = |N| \, C_I
    /// </math>
    /// <para>
    /// The node becomes a leaf when <c>C_split >= C_leaf</c>. Because only the ratio matters, the
    /// half-area <c>e_x e_y + e_y e_z + e_z e_x</c> is used throughout and the factor of two cancels.
    /// </para>
    /// <para><b>Emission order is not a free choice</b></para>
    /// <para>
    /// <see cref="PrometheusBVHNode"/> stores no child pointers: the left child is implicitly
    /// <c>self + 1</c> and the only link is the escape rope. That forces depth-first pre-order
    /// emission. This job achieves it with an explicit work stack carrying a "close" marker
    /// beneath each node's two children; when the marker resurfaces, the whole subtree has been
    /// emitted and the parent's escape index is exactly the current node count.
    /// </para>
    /// <para><b>Bounds are derived from reconstructed vertices</b></para>
    /// <para>
    /// Triangle AABBs are computed from <c>v0</c>, <c>v0 + e1</c> and <c>v0 + e2</c> rather than
    /// from the source vertices. In floating point <c>a + (b - a)</c> does not always reproduce
    /// <c>b</c>, and the shader intersects the stored edge form, so deriving bounds from the
    /// reconstruction is what keeps the hierarchy watertight. No epsilon padding is needed.
    /// </para>
    /// <para><b>Indices are BLAS-local</b></para>
    /// <para>
    /// Escape indices and leaf primitive ranges are relative to this BLAS, with the root at zero.
    /// Traversal rebases them through <see cref="PrometheusInstance.BlasNodeOffset"/> and
    /// <see cref="PrometheusInstance.BlasTriangleOffset"/>, so one BLAS is shareable across instances.
    /// </para>
    /// </remarks>
    [BurstCompile(FloatPrecision.Standard, FloatMode.Default, CompileSynchronously = true)]
    public struct BlasBuildJob : IJob
    {
        /// <summary>Number of spatial bins swept per axis. Section 7.1 Phase 2 suggests 16 or 32.</summary>
        public const int BinCount = 32;

        /// <summary>Relative cost of descending through one interior node.</summary>
        public const float TraversalCost = 1.0f;

        /// <summary>Relative cost of one ray/triangle intersection test.</summary>
        public const float IntersectionCost = 1.0f;

        /// <summary>Default leaf capacity. Small leaves suit GPU traversal, which lacks a deep cache.</summary>
        public const int DefaultMaxLeafTriangles = 4;

        /// <summary>
        /// Depth beyond which the SAH is abandoned in favour of an index median split.
        /// </summary>
        /// <remarks>
        /// The SAH may legitimately choose a very uneven split, so on its own it bounds tree depth
        /// only by the triangle count. An index median halves the range every level, so past this
        /// depth the remaining depth is at most <c>log2(2^21) = 21</c>. That is what makes
        /// <see cref="MaxTreeDepth"/> a real bound and lets the work stack be a fixed size.
        /// </remarks>
        public const int MedianFallbackDepth = 56;

        /// <summary>Hard depth ceiling. Unreachable in practice; see <see cref="MedianFallbackDepth"/>.</summary>
        public const int MaxTreeDepth = 80;

        /// <summary>
        /// Work stack capacity. Each processed node pops one entry and pushes at most three
        /// (close marker, right child, left child), a net gain of two per level of descent.
        /// </summary>
        private const int StackCapacity = (2 * MaxTreeDepth) + 8;

        /// <summary>Object-space vertex positions.</summary>
        [ReadOnly] public NativeArray<float3> Vertices;

        /// <summary>Triangle list indices addressing <see cref="Vertices"/>.</summary>
        [ReadOnly] public NativeArray<int> Indices;

        /// <summary>Sub-mesh ranges carrying per-range material identifiers.</summary>
        [ReadOnly] public NativeArray<PrometheusSubMeshRange> SubMeshes;

        /// <summary>
        /// Maximum triangles per leaf. Clamped to
        /// <see cref="PrometheusBVHNode.MaxLeafPrimitiveCount"/> and to at least one.
        /// </summary>
        public int MaxLeafTriangles;

        /// <summary>
        /// Partitioning strategy. Defaults to <see cref="PrometheusBlasSplitStrategy.BinnedSah"/>,
        /// which is the zero value, so leaving this unset selects the Phase 2 behaviour.
        /// </summary>
        public PrometheusBlasSplitStrategy SplitStrategy;

        /// <summary>
        /// Output triangle buffer, in leaf order. Capacity must be at least the source triangle count.
        /// </summary>
        public NativeArray<PrometheusTriangle> Triangles;

        /// <summary>
        /// Output node buffer in depth-first pre-order. Capacity must be at least
        /// <c>2 * sourceTriangleCount - 1</c>.
        /// </summary>
        public NativeArray<PrometheusBVHNode> Nodes;

        /// <summary>Single-element output carrying <see cref="BlasBuildResult"/>.</summary>
        [WriteOnly] public NativeArray<BlasBuildResult> Result;

        /// <inheritdoc/>
        public void Execute()
        {
            BlasBuildResult result = default;
            result.Status = PrometheusBlasBuildStatus.Success;
            result.BoundsMin = new float3(float.PositiveInfinity);
            result.BoundsMax = new float3(float.NegativeInfinity);

            int sourceTriangleCount = 0;
            for (int s = 0; s < SubMeshes.Length; s++)
            {
                sourceTriangleCount += SubMeshes[s].IndexCount / 3;
            }

            if (sourceTriangleCount == 0)
            {
                result.Status = PrometheusBlasBuildStatus.EmptyGeometry;
                Result[0] = result;
                return;
            }

            // Section 3.1 caps primitives at 2^21; PackedLeafRange addresses at most 2^22 - 1.
            if (sourceTriangleCount > PrometheusTriangle.MaxSceneTriangles ||
                sourceTriangleCount > Triangles.Length ||
                ((2 * sourceTriangleCount) - 1) > Nodes.Length)
            {
                result.Status = PrometheusBlasBuildStatus.CapacityExceeded;
                Result[0] = result;
                return;
            }

            NativeArray<PrometheusTriangle> scratchTriangles =
                new NativeArray<PrometheusTriangle>(sourceTriangleCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<float3> triangleMin =
                new NativeArray<float3>(sourceTriangleCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<float3> triangleMax =
                new NativeArray<float3>(sourceTriangleCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<float3> centroids =
                new NativeArray<float3>(sourceTriangleCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<int> primitiveIndices =
                new NativeArray<int>(sourceTriangleCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            int triangleCount = FilterTriangles(
                scratchTriangles, triangleMin, triangleMax, centroids, primitiveIndices,
                out int rejectedCount, out bool invalidInput);

            result.RejectedTriangleCount = rejectedCount;

            if (invalidInput)
            {
                result.Status = PrometheusBlasBuildStatus.InvalidInput;
            }
            else if (triangleCount == 0)
            {
                result.Status = PrometheusBlasBuildStatus.EmptyGeometry;
            }
            else
            {
                int nodeCount = BuildHierarchy(
                    triangleMin, triangleMax, centroids, primitiveIndices, triangleCount,
                    out float3 rootMin, out float3 rootMax, out int maxDepth, out bool depthOverflow);

                result.NodeCount = nodeCount;
                result.TriangleCount = triangleCount;
                result.MaxDepth = maxDepth;
                result.BoundsMin = rootMin;
                result.BoundsMax = rootMax;

                // Section 3.1 caps the node count at 2^22 - 1, the same ceiling the 22-bit
                // primitive index field enforces.
                if (depthOverflow || nodeCount > (int)PrometheusBVHNode.MaxPrimitiveIndex)
                {
                    result.Status = PrometheusBlasBuildStatus.CapacityExceeded;
                }
                else
                {
                    // Reorder triangles so every leaf addresses a contiguous run.
                    for (int i = 0; i < triangleCount; i++)
                    {
                        Triangles[i] = scratchTriangles[primitiveIndices[i]];
                    }
                }
            }

            primitiveIndices.Dispose();
            centroids.Dispose();
            triangleMax.Dispose();
            triangleMin.Dispose();
            scratchTriangles.Dispose();

            Result[0] = result;
        }

        /// <summary>
        /// Converts source indices into <see cref="PrometheusTriangle"/> records, dropping anything
        /// the <c>EC-003</c> rules reject, and caches the per-triangle bounds and centroids the
        /// partitioner needs.
        /// </summary>
        /// <param name="scratchTriangles">Receives the surviving triangles.</param>
        /// <param name="triangleMin">Receives per-triangle AABB lower corners.</param>
        /// <param name="triangleMax">Receives per-triangle AABB upper corners.</param>
        /// <param name="centroids">Receives per-triangle AABB centres, the binning sample point.</param>
        /// <param name="primitiveIndices">Receives the identity permutation over surviving triangles.</param>
        /// <param name="rejectedCount">Number of triangles dropped.</param>
        /// <param name="invalidInput">Set when an index fell outside the vertex range.</param>
        /// <returns>The number of surviving triangles.</returns>
        private int FilterTriangles(
            NativeArray<PrometheusTriangle> scratchTriangles,
            NativeArray<float3> triangleMin,
            NativeArray<float3> triangleMax,
            NativeArray<float3> centroids,
            NativeArray<int> primitiveIndices,
            out int rejectedCount,
            out bool invalidInput)
        {
            int triangleCount = 0;
            int vertexCount = Vertices.Length;
            uint sourceTriangleId = 0;

            rejectedCount = 0;
            invalidInput = false;

            for (int s = 0; s < SubMeshes.Length; s++)
            {
                PrometheusSubMeshRange range = SubMeshes[s];
                int triangles = range.IndexCount / 3;

                for (int t = 0; t < triangles; t++)
                {
                    int cursor = range.IndexStart + (t * 3);
                    int i0 = Indices[cursor];
                    int i1 = Indices[cursor + 1];
                    int i2 = Indices[cursor + 2];

                    uint primitiveId = sourceTriangleId;
                    sourceTriangleId++;

                    // Guard explicitly: NativeArray range checks are compiled out in players.
                    if (i0 < 0 || i1 < 0 || i2 < 0 ||
                        i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
                    {
                        invalidInput = true;
                        rejectedCount++;
                        continue;
                    }

                    PrometheusTriangle triangle = PrometheusTriangle.Create(
                        Vertices[i0], Vertices[i1], Vertices[i2], primitiveId, range.MaterialId);

                    // EC-003: degenerate area or non-finite coordinates.
                    if (triangle.IsDegenerate)
                    {
                        rejectedCount++;
                        continue;
                    }

                    float3 boundsMin = triangle.BoundsMin;
                    float3 boundsMax = triangle.BoundsMax;

                    scratchTriangles[triangleCount] = triangle;
                    triangleMin[triangleCount] = boundsMin;
                    triangleMax[triangleCount] = boundsMax;
                    centroids[triangleCount] = (boundsMin + boundsMax) * 0.5f;
                    primitiveIndices[triangleCount] = triangleCount;
                    triangleCount++;
                }
            }

            return triangleCount;
        }

        /// <summary>
        /// Builds the hierarchy into <see cref="Nodes"/> in depth-first pre-order, permuting
        /// <paramref name="primitiveIndices"/> so each leaf owns a contiguous run.
        /// </summary>
        /// <param name="triangleMin">Per-triangle AABB lower corners.</param>
        /// <param name="triangleMax">Per-triangle AABB upper corners.</param>
        /// <param name="centroids">Per-triangle AABB centres.</param>
        /// <param name="primitiveIndices">Permutation of triangle indices, reordered in place.</param>
        /// <param name="triangleCount">Number of triangles to organise.</param>
        /// <param name="rootMin">Receives the root AABB lower corner.</param>
        /// <param name="rootMax">Receives the root AABB upper corner.</param>
        /// <param name="maxDepth">Receives the depth of the deepest leaf.</param>
        /// <param name="depthOverflow">Set when the depth ceiling forced an oversized leaf.</param>
        /// <returns>The number of nodes emitted.</returns>
        private int BuildHierarchy(
            NativeArray<float3> triangleMin,
            NativeArray<float3> triangleMax,
            NativeArray<float3> centroids,
            NativeArray<int> primitiveIndices,
            int triangleCount,
            out float3 rootMin,
            out float3 rootMax,
            out int maxDepth,
            out bool depthOverflow)
        {
            int leafCapacity = math.clamp(MaxLeafTriangles, 1, (int)PrometheusBVHNode.MaxLeafPrimitiveCount);

            NativeArray<int> binCounts = new NativeArray<int>(BinCount, Allocator.Temp);
            NativeArray<float3> binMin = new NativeArray<float3>(BinCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<float3> binMax = new NativeArray<float3>(BinCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<int> suffixCounts = new NativeArray<int>(BinCount, Allocator.Temp);
            NativeArray<float3> suffixMin = new NativeArray<float3>(BinCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<float3> suffixMax = new NativeArray<float3>(BinCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            NativeArray<BuildStackEntry> stack = new NativeArray<BuildStackEntry>(StackCapacity, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

            rootMin = new float3(float.PositiveInfinity);
            rootMax = new float3(float.NegativeInfinity);
            maxDepth = 0;
            depthOverflow = false;

            int nodeCount = 0;
            int stackTop = 0;

            stack[stackTop++] = BuildStackEntry.Process(0, triangleCount, 0);

            while (stackTop > 0)
            {
                BuildStackEntry entry = stack[--stackTop];

                if (entry.CloseNodeIndex >= 0)
                {
                    // The whole subtree rooted at this node has now been emitted, so the next
                    // node in pre-order is exactly the current count.
                    PrometheusBVHNode parent = Nodes[entry.CloseNodeIndex];
                    parent.EscapeIndex = (uint)nodeCount;
                    Nodes[entry.CloseNodeIndex] = parent;
                    continue;
                }

                int start = entry.Start;
                int count = entry.Count;

                float3 boundsMin = new float3(float.PositiveInfinity);
                float3 boundsMax = new float3(float.NegativeInfinity);
                float3 centroidMin = new float3(float.PositiveInfinity);
                float3 centroidMax = new float3(float.NegativeInfinity);

                for (int i = start; i < start + count; i++)
                {
                    int p = primitiveIndices[i];
                    boundsMin = math.min(boundsMin, triangleMin[p]);
                    boundsMax = math.max(boundsMax, triangleMax[p]);
                    float3 centroid = centroids[p];
                    centroidMin = math.min(centroidMin, centroid);
                    centroidMax = math.max(centroidMax, centroid);
                }

                int nodeIndex = nodeCount++;
                maxDepth = math.max(maxDepth, entry.Depth);

                if (nodeIndex == 0)
                {
                    rootMin = boundsMin;
                    rootMax = boundsMax;
                }

                bool atDepthCeiling = entry.Depth >= MaxTreeDepth;
                bool mustSplit = count > leafCapacity && !atDepthCeiling;
                int mid = -1;

                if (count > 1 && !atDepthCeiling)
                {
                    bool preferMedian = entry.Depth >= MedianFallbackDepth;
                    bool useSah = SplitStrategy == PrometheusBlasSplitStrategy.BinnedSah && !preferMedian;

                    if (useSah && TryFindSahSplit(
                            triangleMin, triangleMax, centroids, primitiveIndices,
                            start, count, centroidMin, centroidMax,
                            binCounts, binMin, binMax, suffixCounts, suffixMin, suffixMax,
                            out int splitAxis, out int splitPlane, out float splitCost))
                    {
                        float parentArea = HalfSurfaceArea(boundsMin, boundsMax);
                        bool worthSplitting = mustSplit;

                        if (!worthSplitting && parentArea > 0.0f)
                        {
                            float sahCost = TraversalCost + ((splitCost / parentArea) * IntersectionCost);
                            worthSplitting = sahCost < (count * IntersectionCost);
                        }

                        if (worthSplitting)
                        {
                            mid = PartitionByBin(
                                centroids, primitiveIndices, start, count,
                                splitAxis, splitPlane, centroidMin, centroidMax);
                        }
                    }

                    if (mid < 0 && (mustSplit || preferMedian))
                    {
                        mid = PartitionByMedian(
                            centroids, primitiveIndices, start, count, centroidMin, centroidMax, preferMedian);
                    }
                }

                if (mid > start && mid < start + count)
                {
                    // Escape is patched when the close marker resurfaces.
                    Nodes[nodeIndex] = PrometheusBVHNode.CreateInterior(boundsMin, boundsMax, 0u);

                    stack[stackTop++] = BuildStackEntry.Close(nodeIndex);
                    stack[stackTop++] = BuildStackEntry.Process(mid, start + count - mid, entry.Depth + 1);
                    stack[stackTop++] = BuildStackEntry.Process(start, mid - start, entry.Depth + 1);
                }
                else
                {
                    // A leaf's escape is its own successor, which keeps the traversal loop uniform.
                    if (count > leafCapacity && atDepthCeiling)
                    {
                        depthOverflow = true;
                    }

                    Nodes[nodeIndex] = PrometheusBVHNode.CreateLeaf(
                        boundsMin, boundsMax, (uint)nodeCount, (uint)start, (uint)count);
                }
            }

            // The rightmost spine escapes past the end of the buffer; that is the terminator.
            for (int i = 0; i < nodeCount; i++)
            {
                PrometheusBVHNode node = Nodes[i];
                if (node.EscapeIndex == (uint)nodeCount)
                {
                    node.EscapeIndex = PrometheusBVHNode.InvalidIndex;
                    Nodes[i] = node;
                }
            }

            stack.Dispose();
            suffixMax.Dispose();
            suffixMin.Dispose();
            suffixCounts.Dispose();
            binMax.Dispose();
            binMin.Dispose();
            binCounts.Dispose();

            return nodeCount;
        }

        /// <summary>
        /// Sweeps all three axes looking for the split plane with the lowest surface area cost.
        /// </summary>
        /// <remarks>
        /// All three axes are evaluated rather than only the widest one. That triples the binning
        /// work, which Section 7.1 Phase 2 explicitly permits in exchange for a tighter tree.
        /// The returned cost is the un-normalised numerator
        /// <c>A(N_L)|N_L| + A(N_R)|N_R|</c>; the caller divides by the parent area.
        /// </remarks>
        /// <returns><c>true</c> when a plane with non-empty sides was found.</returns>
        private static bool TryFindSahSplit(
            NativeArray<float3> triangleMin,
            NativeArray<float3> triangleMax,
            NativeArray<float3> centroids,
            NativeArray<int> primitiveIndices,
            int start,
            int count,
            float3 centroidMin,
            float3 centroidMax,
            NativeArray<int> binCounts,
            NativeArray<float3> binMin,
            NativeArray<float3> binMax,
            NativeArray<int> suffixCounts,
            NativeArray<float3> suffixMin,
            NativeArray<float3> suffixMax,
            out int bestAxis,
            out int bestPlane,
            out float bestCost)
        {
            bestAxis = -1;
            bestPlane = -1;
            bestCost = float.PositiveInfinity;

            for (int axis = 0; axis < 3; axis++)
            {
                float extent = centroidMax[axis] - centroidMin[axis];
                if (!(extent > 0.0f))
                {
                    continue;
                }

                float scale = BinCount / extent;
                float origin = centroidMin[axis];

                for (int b = 0; b < BinCount; b++)
                {
                    binCounts[b] = 0;
                    binMin[b] = new float3(float.PositiveInfinity);
                    binMax[b] = new float3(float.NegativeInfinity);
                }

                for (int i = start; i < start + count; i++)
                {
                    int p = primitiveIndices[i];
                    int bin = BinOf(centroids[p][axis], origin, scale);

                    binCounts[bin] += 1;
                    binMin[bin] = math.min(binMin[bin], triangleMin[p]);
                    binMax[bin] = math.max(binMax[bin], triangleMax[p]);
                }

                // Suffix sweep: bounds and counts for everything from bin b upward.
                int runningCount = 0;
                float3 runningMin = new float3(float.PositiveInfinity);
                float3 runningMax = new float3(float.NegativeInfinity);

                for (int b = BinCount - 1; b >= 0; b--)
                {
                    runningCount += binCounts[b];
                    runningMin = math.min(runningMin, binMin[b]);
                    runningMax = math.max(runningMax, binMax[b]);

                    suffixCounts[b] = runningCount;
                    suffixMin[b] = runningMin;
                    suffixMax[b] = runningMax;
                }

                // Prefix sweep, scoring the plane between bin b and bin b + 1.
                int leftCount = 0;
                float3 leftMin = new float3(float.PositiveInfinity);
                float3 leftMax = new float3(float.NegativeInfinity);

                for (int b = 0; b < BinCount - 1; b++)
                {
                    leftCount += binCounts[b];
                    leftMin = math.min(leftMin, binMin[b]);
                    leftMax = math.max(leftMax, binMax[b]);

                    int rightCount = suffixCounts[b + 1];
                    if (leftCount == 0 || rightCount == 0)
                    {
                        continue;
                    }

                    float cost =
                        (HalfSurfaceArea(leftMin, leftMax) * leftCount) +
                        (HalfSurfaceArea(suffixMin[b + 1], suffixMax[b + 1]) * rightCount);

                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestAxis = axis;
                        bestPlane = b;
                    }
                }
            }

            return bestAxis >= 0;
        }

        /// <summary>
        /// Partitions the primitive range around the chosen bin plane.
        /// </summary>
        /// <remarks>
        /// The bin index is recomputed with the same origin and scale used during scoring, so the
        /// partition reproduces the counts the cost model assumed.
        /// </remarks>
        /// <returns>The split position, or <c>-1</c> when one side came out empty.</returns>
        private static int PartitionByBin(
            NativeArray<float3> centroids,
            NativeArray<int> primitiveIndices,
            int start,
            int count,
            int axis,
            int plane,
            float3 centroidMin,
            float3 centroidMax)
        {
            float extent = centroidMax[axis] - centroidMin[axis];
            if (!(extent > 0.0f))
            {
                return -1;
            }

            float scale = BinCount / extent;
            float origin = centroidMin[axis];

            int left = start;
            int right = start + count - 1;

            while (left <= right)
            {
                int p = primitiveIndices[left];
                if (BinOf(centroids[p][axis], origin, scale) <= plane)
                {
                    left++;
                }
                else
                {
                    (primitiveIndices[left], primitiveIndices[right]) = (primitiveIndices[right], primitiveIndices[left]);
                    right--;
                }
            }

            return (left > start && left < start + count) ? left : -1;
        }

        /// <summary>
        /// Fallback partition used when the SAH declines to split but the leaf capacity forces one.
        /// </summary>
        /// <param name="forceIndexMedian">
        /// When set, splits purely by position in the array. That guarantees the range halves,
        /// which is what bounds the tree depth past <see cref="MedianFallbackDepth"/>.
        /// </param>
        /// <returns>The split position; always valid for <c>count >= 2</c>.</returns>
        private static int PartitionByMedian(
            NativeArray<float3> centroids,
            NativeArray<int> primitiveIndices,
            int start,
            int count,
            float3 centroidMin,
            float3 centroidMax,
            bool forceIndexMedian)
        {
            if (!forceIndexMedian)
            {
                float3 extent = centroidMax - centroidMin;
                int axis = 0;
                if (extent.y > extent.x)
                {
                    axis = 1;
                }

                if (extent.z > extent[axis])
                {
                    axis = 2;
                }

                if (extent[axis] > 0.0f)
                {
                    float pivot = (centroidMin[axis] + centroidMax[axis]) * 0.5f;
                    int left = start;
                    int right = start + count - 1;

                    while (left <= right)
                    {
                        int p = primitiveIndices[left];
                        if (centroids[p][axis] < pivot)
                        {
                            left++;
                        }
                        else
                        {
                            (primitiveIndices[left], primitiveIndices[right]) = (primitiveIndices[right], primitiveIndices[left]);
                            right--;
                        }
                    }

                    if (left > start && left < start + count)
                    {
                        return left;
                    }
                }
            }

            // Splitting by array position always yields two non-empty halves for count >= 2.
            return start + (count / 2);
        }

        /// <summary>Maps a centroid coordinate onto its bin, clamped against floating point drift.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int BinOf(float coordinate, float origin, float scale)
        {
            return math.clamp((int)((coordinate - origin) * scale), 0, BinCount - 1);
        }

        /// <summary>
        /// Half the surface area of an AABB, <c>e_x e_y + e_y e_z + e_z e_x</c>.
        /// </summary>
        /// <remarks>
        /// Only ratios of these values enter the cost model, so the factor of two is dropped.
        /// An inverted (empty) box clamps to zero rather than producing a negative area.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HalfSurfaceArea(float3 boundsMin, float3 boundsMax)
        {
            float3 e = math.max(boundsMax - boundsMin, float3.zero);
            return (e.x * e.y) + (e.y * e.z) + (e.z * e.x);
        }

        /// <summary>
        /// One work stack entry: either a range awaiting partitioning, or a marker that resurfaces
        /// once a node's whole subtree has been emitted.
        /// </summary>
        private struct BuildStackEntry
        {
            /// <summary>First primitive slot of the range, for process entries.</summary>
            public int Start;

            /// <summary>Number of primitives in the range, for process entries.</summary>
            public int Count;

            /// <summary>Depth of the range below the root.</summary>
            public int Depth;

            /// <summary>Node awaiting an escape index, or <c>-1</c> for a process entry.</summary>
            public int CloseNodeIndex;

            /// <summary>Creates an entry describing a range still to be partitioned.</summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static BuildStackEntry Process(int start, int count, int depth)
            {
                BuildStackEntry entry;
                entry.Start = start;
                entry.Count = count;
                entry.Depth = depth;
                entry.CloseNodeIndex = -1;
                return entry;
            }

            /// <summary>Creates a marker that patches <paramref name="nodeIndex"/>'s escape link.</summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static BuildStackEntry Close(int nodeIndex)
            {
                BuildStackEntry entry;
                entry.Start = 0;
                entry.Count = 0;
                entry.Depth = 0;
                entry.CloseNodeIndex = nodeIndex;
                return entry;
            }
        }
    }
}
