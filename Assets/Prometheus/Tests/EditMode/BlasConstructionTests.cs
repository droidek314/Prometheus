using System.Collections.Generic;
using NUnit.Framework;
using Prometheus.Core.BVH;
using Prometheus.Core.Structs;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.TestTools.Constraints;

// UnityEngine.TestTools.Constraints.Is derives from NUnit.Framework.Is, so this alias keeps
// every NUnit constraint available while adding the AllocatingGCMemory() extension.
using Is = UnityEngine.TestTools.Constraints.Is;

namespace Prometheus.Tests.EditMode
{
    /// <summary>
    /// Phase 2 verification gate (Section 7.1): proves the binned SAH builder produces a
    /// structurally valid, tightly packed hierarchy with no degenerate leaves, that root bounds
    /// reproduce the source mesh volume, and that the read path is allocation free.
    /// </summary>
    [TestFixture]
    public sealed class BlasConstructionTests
    {
        /// <summary>Aggregated facts gathered by a full walk of a built hierarchy.</summary>
        private struct HierarchyStats
        {
            public int NodeCount;
            public int LeafCount;
            public int InteriorCount;
            public int ReferencedTriangles;
            public int MaxLeafSize;
            public int MaxDepth;
            public float SahCost;
        }

        private PrometheusMeshCache _cache;
        private GameObject _cubeObject;
        private GameObject _sphereObject;
        private Mesh _cubeMesh;
        private Mesh _sphereMesh;
        private Mesh _denseMesh;

        // Sinks for the allocation probes.
        private float _sinkFloat;
        private uint _sinkUint;

        [OneTimeSetUp]
        public void OneTimeSetUp()
        {
            _cache = new PrometheusMeshCache();

            _cubeObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _sphereObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _cubeMesh = _cubeObject.GetComponent<MeshFilter>().sharedMesh;
            _sphereMesh = _sphereObject.GetComponent<MeshFilter>().sharedMesh;

            // Stands in for the dense reference model named in Section 7.1 Phase 2, which is not
            // redistributable. A pole-free UV sphere gives ~9k non-degenerate triangles.
            _denseMesh = CreateUvSphere(48, 96, 1.0f);
        }

        [OneTimeTearDown]
        public void OneTimeTearDown()
        {
            _cache?.Dispose();
            _cache = null;

            if (_denseMesh != null)
            {
                Object.DestroyImmediate(_denseMesh);
                _denseMesh = null;
            }

            if (_cubeObject != null)
            {
                Object.DestroyImmediate(_cubeObject);
                _cubeObject = null;
            }

            if (_sphereObject != null)
            {
                Object.DestroyImmediate(_sphereObject);
                _sphereObject = null;
            }
        }

        // =================================================================================
        //  Root bounds fidelity
        // =================================================================================

        [Test]
        public void Cube_RootBoundsMatchMeshVolume()
        {
            AssertRootBoundsMatchMesh(_cubeMesh);
        }

        [Test]
        public void Sphere_RootBoundsMatchMeshVolume()
        {
            AssertRootBoundsMatchMesh(_sphereMesh);
        }

        [Test]
        public void DenseMesh_RootBoundsMatchMeshVolume()
        {
            AssertRootBoundsMatchMesh(_denseMesh);
        }

        /// <summary>
        /// Asserts the hierarchy root reproduces the mesh's own AABB.
        /// </summary>
        /// <remarks>
        /// A tolerance is required rather than exact equality: node bounds are derived from the
        /// reconstructed edge form <c>v0 + e1</c> instead of the source vertex, and those differ
        /// by up to an ULP. The tolerance scales with the extent so it stays meaningful.
        /// </remarks>
        /// <param name="mesh">Mesh whose BLAS root bounds are checked.</param>
        private void AssertRootBoundsMatchMesh(Mesh mesh)
        {
            Assert.That(_cache.TryGetOrBuild(mesh, out PrometheusBlasEntry entry), Is.True,
                $"BLAS construction failed for '{mesh.name}'.");

            Bounds expected = mesh.bounds;
            float3 expectedMin = expected.min;
            float3 expectedMax = expected.max;
            float tolerance = math.max(1e-5f, math.cmax(expectedMax - expectedMin) * 1e-5f);

            Assert.That(math.distance(entry.BoundsMin, expectedMin), Is.LessThan(tolerance),
                $"root min {entry.BoundsMin} vs mesh min {expectedMin}");
            Assert.That(math.distance(entry.BoundsMax, expectedMax), Is.LessThan(tolerance),
                $"root max {entry.BoundsMax} vs mesh max {expectedMax}");

            // The root must also enclose every triangle it owns.
            PrometheusBVHNode root = entry.Nodes[0];
            for (int i = 0; i < entry.Triangles.Length; i++)
            {
                PrometheusTriangle triangle = entry.Triangles[i];
                Assert.That(math.all(triangle.BoundsMin >= root.BoundsMin), Is.True,
                    $"triangle {i} escapes the root lower bound");
                Assert.That(math.all(triangle.BoundsMax <= root.BoundsMax), Is.True,
                    $"triangle {i} escapes the root upper bound");
            }
        }

        // =================================================================================
        //  Structural validity
        // =================================================================================

        [Test]
        public void Cube_HierarchyIsStructurallyValid()
        {
            AssertHierarchyValid(_cubeMesh);
        }

        [Test]
        public void Sphere_HierarchyIsStructurallyValid()
        {
            AssertHierarchyValid(_sphereMesh);
        }

        [Test]
        public void DenseMesh_HierarchyIsStructurallyValid()
        {
            AssertHierarchyValid(_denseMesh);
        }

        [Test]
        public void Cube_ProducesNoDegenerateLeaves()
        {
            AssertNoDegenerateLeaves(_cubeMesh);
        }

        [Test]
        public void Sphere_ProducesNoDegenerateLeaves()
        {
            AssertNoDegenerateLeaves(_sphereMesh);
        }

        [Test]
        public void DenseMesh_ProducesNoDegenerateLeaves()
        {
            AssertNoDegenerateLeaves(_denseMesh);
        }

        /// <summary>
        /// Walks the whole hierarchy asserting the depth-first rope encoding is self-consistent.
        /// </summary>
        /// <param name="mesh">Mesh whose BLAS is validated.</param>
        private void AssertHierarchyValid(Mesh mesh)
        {
            Assert.That(_cache.TryGetOrBuild(mesh, out PrometheusBlasEntry entry), Is.True);

            HierarchyStats stats = WalkHierarchy(entry);

            Assert.That(stats.ReferencedTriangles, Is.EqualTo(entry.Triangles.Length),
                "every triangle must be owned by exactly one leaf");

            // A binary hierarchy over T primitives has at most 2T - 1 nodes.
            Assert.That(stats.NodeCount, Is.LessThanOrEqualTo(math.max(1, (2 * entry.Triangles.Length) - 1)));
            Assert.That(stats.InteriorCount, Is.EqualTo(stats.LeafCount - 1),
                "a full binary tree has exactly one fewer interior node than leaves");
            Assert.That(stats.MaxLeafSize, Is.LessThanOrEqualTo(BlasBuildJob.DefaultMaxLeafTriangles));
            Assert.That(stats.NodeCount, Is.LessThanOrEqualTo((int)PrometheusBVHNode.MaxPrimitiveIndex),
                "Section 3.1 caps the node count at 2^22 - 1");

            // Depth must stay logarithmic; a runaway split pattern would blow the traversal budget.
            int balancedDepth = math.max(1, (int)math.ceil(math.log2(math.max(2, stats.LeafCount))));
            Assert.That(stats.MaxDepth, Is.LessThanOrEqualTo((3 * balancedDepth) + 8),
                $"depth {stats.MaxDepth} against a balanced depth of {balancedDepth}");

            Debug.Log(
                $"[BLAS] {mesh.name}: tris={entry.Triangles.Length} nodes={stats.NodeCount} " +
                $"leaves={stats.LeafCount} maxLeaf={stats.MaxLeafSize} depth={stats.MaxDepth} " +
                $"sahCost={stats.SahCost:F2}");
        }

        /// <summary>Asserts no leaf is empty and no surviving triangle is degenerate.</summary>
        /// <param name="mesh">Mesh whose BLAS is validated.</param>
        private void AssertNoDegenerateLeaves(Mesh mesh)
        {
            Assert.That(_cache.TryGetOrBuild(mesh, out PrometheusBlasEntry entry), Is.True);

            Assert.That(entry.RejectedTriangleCount, Is.Zero,
                $"'{mesh.name}' is expected to contain no degenerate source triangles");

            int leafCount = 0;
            for (int i = 0; i < entry.Nodes.Length; i++)
            {
                PrometheusBVHNode node = entry.Nodes[i];

                Assert.That(math.all(math.isfinite(node.BoundsMin)) && math.all(math.isfinite(node.BoundsMax)), Is.True,
                    $"node {i} carries non-finite bounds");
                Assert.That(math.all(node.BoundsMax >= node.BoundsMin), Is.True,
                    $"node {i} carries an inverted AABB");

                if (!node.IsLeaf)
                {
                    continue;
                }

                leafCount++;
                Assert.That(node.PrimitiveCount, Is.GreaterThan(0u), $"leaf {i} is empty");
                Assert.That((int)(node.FirstPrimitive + node.PrimitiveCount), Is.LessThanOrEqualTo(entry.Triangles.Length),
                    $"leaf {i} addresses past the end of the triangle buffer");
            }

            Assert.That(leafCount, Is.GreaterThan(0), "the hierarchy contains no leaves at all");

            for (int i = 0; i < entry.Triangles.Length; i++)
            {
                Assert.That(entry.Triangles[i].IsDegenerate, Is.False,
                    $"triangle {i} survived the EC-003 filter but is degenerate");
            }
        }

        // =================================================================================
        //  SAH quality
        // =================================================================================

        [Test]
        public void DenseMesh_HierarchyIsTightlyPacked()
        {
            Assert.That(_cache.TryGetOrBuild(_denseMesh, out PrometheusBlasEntry entry), Is.True);

            HierarchyStats stats = WalkHierarchy(entry);

            // The expected number of primitive tests for a random ray that hits the root box.
            // A loose hierarchy pushes this up sharply, so it is the direct measure of the
            // "tight spatial volume packing" gate in Section 7.1 Phase 2.
            Assert.That(stats.SahCost, Is.LessThan(40.0f),
                $"SAH cost {stats.SahCost:F2} is too high for {entry.Triangles.Length} triangles");

            // Sibling overlap: children of the root should not each span the whole volume.
            PrometheusBVHNode root = entry.Nodes[0];
            Assert.That(root.IsLeaf, Is.False, "a dense mesh must not collapse into a single leaf");

            PrometheusBVHNode left = entry.Nodes[1];
            uint rightIndex = left.EscapeIndex == PrometheusBVHNode.InvalidIndex
                ? (uint)entry.Nodes.Length
                : left.EscapeIndex;
            PrometheusBVHNode right = entry.Nodes[(int)rightIndex];

            float rootArea = HalfSurfaceArea(root.BoundsMin, root.BoundsMax);
            float childArea = HalfSurfaceArea(left.BoundsMin, left.BoundsMax) +
                              HalfSurfaceArea(right.BoundsMin, right.BoundsMax);

            Assert.That(childArea, Is.LessThan(rootArea * 1.6f),
                $"root children cover {childArea / rootArea:F2}x the root area, indicating heavy overlap");
        }

        /// <summary>
        /// Measures the SAH against a spatial median split on geometry that actually discriminates
        /// between them.
        /// </summary>
        /// <remarks>
        /// <para>
        /// A uniformly tessellated sphere is close to the best case for a median split, so it
        /// barely separates the two strategies. What object-split SAH is actually for is uneven
        /// primitive <em>distribution</em>: a median split halves the box regardless of where the
        /// triangles are, spending levels on empty space, while the surface area term follows the
        /// occupancy. Irregularly placed clusters of differing tessellation density are therefore
        /// the fair discriminating case.
        /// </para>
        /// <para>
        /// Note that extreme scale separation (a unit object inside a 200-unit volume) is
        /// deliberately not used here. With <see cref="BlasBuildJob.BinCount"/> bins spanning the
        /// full extent, such an object collapses into a single bin and binning cannot resolve it
        /// at all. Recovering that case requires spatial splits (SBVH), which Section 7.1 Phase 2
        /// does not call for.
        /// </para>
        /// </remarks>
        [Test]
        public void SahTree_OutperformsMedianSplitOnNonUniformGeometry()
        {
            CreateClusteredGeometry(out float3[] positions, out int[] indices);

            float sahCost = BuildAndMeasure(positions, indices, PrometheusBlasSplitStrategy.BinnedSah);
            float medianCost = BuildAndMeasure(positions, indices, PrometheusBlasSplitStrategy.SpatialMedian);

            Debug.Log(
                $"[BLAS] non-uniform geometry ({indices.Length / 3} tris): " +
                $"SAH cost {sahCost:F2} vs spatial median cost {medianCost:F2} " +
                $"({medianCost / sahCost:F2}x)");

            // Object-split SAH buys on the order of 15% here. That is the realistic margin for
            // binned object splits; the large multiples quoted in the literature come from
            // spatial splits, which this builder does not perform.
            Assert.That(sahCost, Is.LessThan(medianCost * 0.95f),
                $"binned SAH ({sahCost:F2}) must beat a spatial median split ({medianCost:F2}) " +
                "on irregularly distributed geometry");
        }

        // =================================================================================
        //  EC-003 degenerate and non-finite geometry
        // =================================================================================

        [Test]
        public void Builder_RejectsDegenerateAndNonFiniteTriangles()
        {
            float3[] positions =
            {
                new float3(0.0f, 0.0f, 0.0f),
                new float3(1.0f, 0.0f, 0.0f),
                new float3(0.0f, 1.0f, 0.0f),
                new float3(2.0f, 0.0f, 0.0f),
                new float3(float.NaN, 0.0f, 0.0f),
                new float3(0.0f, 0.0f, float.PositiveInfinity)
            };

            int[] triangles =
            {
                0, 1, 2,   // valid
                0, 1, 3,   // collinear, zero area
                0, 1, 1,   // duplicated vertex, zero area
                4, 1, 2,   // NaN coordinate
                5, 1, 2,   // infinite coordinate
                2, 1, 0    // valid, reversed winding
            };

            RunBuilder(positions, triangles, out BlasBuildResult result, out int[] survivingPrimitiveIds);

            Assert.That(result.Status, Is.EqualTo(PrometheusBlasBuildStatus.Success));
            Assert.That(result.TriangleCount, Is.EqualTo(2), "only the two well-formed triangles may survive");
            Assert.That(result.RejectedTriangleCount, Is.EqualTo(4));
            CollectionAssert.AreEquivalent(new[] { 0, 5 }, survivingPrimitiveIds,
                "surviving triangles must retain their original source indices");
        }

        [Test]
        public void Builder_ReportsInvalidInputForOutOfRangeIndices()
        {
            float3[] positions =
            {
                new float3(0.0f, 0.0f, 0.0f),
                new float3(1.0f, 0.0f, 0.0f),
                new float3(0.0f, 1.0f, 0.0f)
            };

            int[] triangles = { 0, 1, 2, 0, 1, 99 };

            RunBuilder(positions, triangles, out BlasBuildResult result, out _);

            Assert.That(result.Status, Is.EqualTo(PrometheusBlasBuildStatus.InvalidInput));
            Assert.That(result.RejectedTriangleCount, Is.EqualTo(1));
        }

        [Test]
        public void Builder_ReportsEmptyGeometryWhenEveryTriangleIsDegenerate()
        {
            float3[] positions =
            {
                new float3(0.0f, 0.0f, 0.0f),
                new float3(1.0f, 0.0f, 0.0f),
                new float3(2.0f, 0.0f, 0.0f)
            };

            int[] triangles = { 0, 1, 2 };

            RunBuilder(positions, triangles, out BlasBuildResult result, out _);

            Assert.That(result.Status, Is.EqualTo(PrometheusBlasBuildStatus.EmptyGeometry));
            Assert.That(result.TriangleCount, Is.Zero);
        }

        [Test]
        public void Builder_ProducesASingleLeafForOneTriangle()
        {
            float3[] positions =
            {
                new float3(0.0f, 0.0f, 0.0f),
                new float3(1.0f, 0.0f, 0.0f),
                new float3(0.0f, 1.0f, 0.0f)
            };

            int[] triangles = { 0, 1, 2 };

            RunBuilder(positions, triangles, out BlasBuildResult result, out _, out PrometheusBVHNode[] nodes);

            Assert.That(result.Status, Is.EqualTo(PrometheusBlasBuildStatus.Success));
            Assert.That(result.NodeCount, Is.EqualTo(1));
            Assert.That(nodes[0].IsLeaf, Is.True);
            Assert.That(nodes[0].PrimitiveCount, Is.EqualTo(1u));
            Assert.That(nodes[0].EscapeIndex, Is.EqualTo(PrometheusBVHNode.InvalidIndex),
                "the only node terminates traversal");
        }

        // =================================================================================
        //  Cache lifecycle
        // =================================================================================

        [Test]
        public void Cache_ReturnsTheSameEntryForRepeatedRequests()
        {
            using (PrometheusMeshCache cache = new PrometheusMeshCache())
            {
                Assert.That(cache.TryGetOrBuild(_cubeMesh, out PrometheusBlasEntry first), Is.True);
                Assert.That(cache.TryGetOrBuild(_cubeMesh, out PrometheusBlasEntry second), Is.True);

                Assert.That(cache.Count, Is.EqualTo(1), "a repeated request must not rebuild");
                Assert.That(first.MeshId, Is.EqualTo(second.MeshId));
                Assert.That(second.Nodes.Length, Is.EqualTo(first.Nodes.Length));

                Assert.That(cache.Release(_cubeMesh), Is.True);
                Assert.That(cache.Count, Is.Zero);
                Assert.That(cache.TryGetEntry(_cubeMesh, out _), Is.False);
            }
        }

        [Test]
        public void Cache_AssignsDistinctMeshIdsAndTracksTotals()
        {
            using (PrometheusMeshCache cache = new PrometheusMeshCache())
            {
                Assert.That(cache.TryGetOrBuild(_cubeMesh, out PrometheusBlasEntry cube), Is.True);
                Assert.That(cache.TryGetOrBuild(_sphereMesh, out PrometheusBlasEntry sphere), Is.True);

                Assert.That(cube.MeshId, Is.Not.EqualTo(sphere.MeshId));
                Assert.That(cache.TotalNodeCount, Is.EqualTo(cube.Nodes.Length + sphere.Nodes.Length));
                Assert.That(cache.TotalTriangleCount, Is.EqualTo(cube.Triangles.Length + sphere.Triangles.Length));
            }
        }

        [Test]
        public void Cache_SubMeshMaterialIdsPropagateToTriangles()
        {
            Assert.That(_cache.TryGetOrBuild(_cubeMesh, out PrometheusBlasEntry entry), Is.True);

            for (int i = 0; i < entry.Triangles.Length; i++)
            {
                Assert.That(entry.Triangles[i].MaterialId, Is.LessThan((uint)_cubeMesh.subMeshCount),
                    "triangle material ids index the source sub-mesh");
            }
        }

        // =================================================================================
        //  Zero managed allocation gate (Section 3.2)
        // =================================================================================

        [Test]
        public void HierarchyTraversal_AllocatesNoManagedMemory()
        {
            Assert.That(_cache.TryGetOrBuild(_denseMesh, out PrometheusBlasEntry entry), Is.True);

            AssertSteadyStateIsAllocationFree(() => TraverseAllNodes(entry));
        }

        [Test]
        public void CacheLookup_AllocatesNoManagedMemory()
        {
            Assert.That(_cache.TryGetOrBuild(_cubeMesh, out _), Is.True);

            AssertSteadyStateIsAllocationFree(LookUpCachedEntry);
        }

        /// <summary>Reads every node and every leaf triangle, mirroring a traversal sweep.</summary>
        /// <param name="entry">Entry to sweep.</param>
        private void TraverseAllNodes(PrometheusBlasEntry entry)
        {
            for (int i = 0; i < entry.Nodes.Length; i++)
            {
                PrometheusBVHNode node = entry.Nodes[i];
                _sinkFloat += node.SurfaceArea + node.Centroid.x;
                _sinkUint += node.EscapeIndex;

                if (!node.IsLeaf)
                {
                    continue;
                }

                uint first = node.FirstPrimitive;
                uint count = node.PrimitiveCount;
                for (uint t = 0; t < count; t++)
                {
                    PrometheusTriangle triangle = entry.Triangles[(int)(first + t)];
                    _sinkFloat += triangle.Area + triangle.Normal.y;
                    _sinkUint += triangle.PrimitiveId;
                }
            }
        }

        /// <summary>Performs a cached lookup, the only cache operation on the per-frame path.</summary>
        private void LookUpCachedEntry()
        {
            if (_cache.TryGetEntry(_cubeMesh, out PrometheusBlasEntry entry))
            {
                _sinkUint += entry.MeshId + (uint)entry.Nodes.Length;
                _sinkFloat += entry.BoundsMin.x;
            }
        }

        /// <summary>
        /// Runs <paramref name="body"/> a few times before measuring, then asserts that a
        /// subsequent invocation allocates nothing.
        /// </summary>
        /// <remarks>
        /// Unity's constraint measures a single invocation, so without a warm-up the Mono JIT
        /// compiling the cold call graph would be counted as managed allocation.
        /// </remarks>
        /// <param name="body">Delegate to measure. Must be allocation free in steady state.</param>
        private static void AssertSteadyStateIsAllocationFree(TestDelegate body)
        {
            for (int i = 0; i < 4; i++)
            {
                body();
            }

            Assert.That(body, Is.Not.AllocatingGCMemory());
        }

        // =================================================================================
        //  Helpers
        // =================================================================================

        /// <summary>
        /// Walks the hierarchy from the root, verifying the rope encoding and accumulating stats.
        /// </summary>
        /// <remarks>
        /// The walk relies on the Phase 1 layout contract: the left child of an interior node is
        /// implicitly <c>i + 1</c>, so the right child is precisely where the left child's subtree
        /// ends, which is the left child's own escape link.
        /// </remarks>
        /// <param name="entry">Entry to inspect.</param>
        /// <returns>Aggregated hierarchy statistics.</returns>
        private static HierarchyStats WalkHierarchy(PrometheusBlasEntry entry)
        {
            int nodeCount = entry.Nodes.Length;
            int triangleCount = entry.Triangles.Length;

            HierarchyStats stats = default;
            stats.NodeCount = nodeCount;

            bool[] triangleSeen = new bool[triangleCount];
            bool[] nodeVisited = new bool[nodeCount];

            PrometheusBVHNode root = entry.Nodes[0];
            float rootArea = HalfSurfaceArea(root.BoundsMin, root.BoundsMax);
            Assert.That(rootArea, Is.GreaterThan(0.0f), "the root AABB is degenerate");

            Stack<(int Index, int Depth)> pending = new Stack<(int, int)>();
            pending.Push((0, 0));

            while (pending.Count > 0)
            {
                (int index, int depth) = pending.Pop();

                Assert.That(index, Is.InRange(0, nodeCount - 1), "node index out of range");
                Assert.That(nodeVisited[index], Is.False, $"node {index} reached twice; the tree is not a tree");
                nodeVisited[index] = true;

                PrometheusBVHNode node = entry.Nodes[index];
                stats.MaxDepth = math.max(stats.MaxDepth, depth);

                uint escape = node.EscapeIndex == PrometheusBVHNode.InvalidIndex
                    ? (uint)nodeCount
                    : node.EscapeIndex;
                Assert.That((int)escape, Is.GreaterThan(index), $"node {index} escapes backwards");
                Assert.That((int)escape, Is.LessThanOrEqualTo(nodeCount), $"node {index} escapes past the buffer");

                float area = HalfSurfaceArea(node.BoundsMin, node.BoundsMax);

                if (node.IsLeaf)
                {
                    stats.LeafCount++;
                    int count = (int)node.PrimitiveCount;
                    int first = (int)node.FirstPrimitive;

                    stats.MaxLeafSize = math.max(stats.MaxLeafSize, count);
                    stats.ReferencedTriangles += count;
                    stats.SahCost += (area / rootArea) * count;

                    Assert.That((int)escape, Is.EqualTo(index + 1),
                        $"leaf {index} must escape to its immediate successor");

                    for (int t = 0; t < count; t++)
                    {
                        int triangleIndex = first + t;
                        Assert.That(triangleIndex, Is.InRange(0, triangleCount - 1));
                        Assert.That(triangleSeen[triangleIndex], Is.False,
                            $"triangle {triangleIndex} is referenced by more than one leaf");
                        triangleSeen[triangleIndex] = true;

                        PrometheusTriangle triangle = entry.Triangles[triangleIndex];
                        Assert.That(math.all(triangle.BoundsMin >= node.BoundsMin), Is.True,
                            $"triangle {triangleIndex} escapes leaf {index}");
                        Assert.That(math.all(triangle.BoundsMax <= node.BoundsMax), Is.True,
                            $"triangle {triangleIndex} escapes leaf {index}");
                    }

                    continue;
                }

                stats.InteriorCount++;
                stats.SahCost += (area / rootArea) * BlasBuildJob.TraversalCost;

                int leftIndex = index + 1;
                Assert.That(leftIndex, Is.LessThan(nodeCount), $"interior node {index} has no left child");

                PrometheusBVHNode left = entry.Nodes[leftIndex];
                uint leftEscape = left.EscapeIndex == PrometheusBVHNode.InvalidIndex
                    ? (uint)nodeCount
                    : left.EscapeIndex;

                int rightIndex = (int)leftEscape;
                Assert.That(rightIndex, Is.InRange(leftIndex + 1, nodeCount - 1),
                    $"interior node {index} has an unreachable right child");
                Assert.That(rightIndex, Is.LessThan((int)escape),
                    $"interior node {index} right child lies outside its own subtree");

                PrometheusBVHNode right = entry.Nodes[rightIndex];

                // Children are built from a subrange of the parent's primitives, so containment
                // holds exactly; no tolerance is warranted.
                AssertContains(node, left, index, leftIndex);
                AssertContains(node, right, index, rightIndex);

                pending.Push((rightIndex, depth + 1));
                pending.Push((leftIndex, depth + 1));
            }

            for (int i = 0; i < nodeCount; i++)
            {
                Assert.That(nodeVisited[i], Is.True, $"node {i} is unreachable from the root");
            }

            for (int i = 0; i < triangleCount; i++)
            {
                Assert.That(triangleSeen[i], Is.True, $"triangle {i} is not referenced by any leaf");
            }

            return stats;
        }

        /// <summary>Asserts a child AABB lies entirely within its parent.</summary>
        private static void AssertContains(PrometheusBVHNode parent, PrometheusBVHNode child, int parentIndex, int childIndex)
        {
            Assert.That(math.all(child.BoundsMin >= parent.BoundsMin), Is.True,
                $"child {childIndex} min {child.BoundsMin} escapes parent {parentIndex} min {parent.BoundsMin}");
            Assert.That(math.all(child.BoundsMax <= parent.BoundsMax), Is.True,
                $"child {childIndex} max {child.BoundsMax} escapes parent {parentIndex} max {parent.BoundsMax}");
        }

        private static float HalfSurfaceArea(float3 boundsMin, float3 boundsMax)
        {
            float3 e = math.max(boundsMax - boundsMin, float3.zero);
            return (e.x * e.y) + (e.y * e.z) + (e.z * e.x);
        }

        private static void RunBuilder(float3[] positions, int[] triangles, out BlasBuildResult result, out int[] survivingPrimitiveIds)
        {
            RunBuilder(positions, triangles, out result, out survivingPrimitiveIds, out _);
        }

        /// <summary>
        /// Drives <see cref="BlasBuildJob"/> directly from managed arrays.
        /// </summary>
        /// <remarks>
        /// Bypassing <see cref="Mesh"/> keeps the malformed-geometry cases out of Unity's own mesh
        /// validation, which would otherwise log errors and fail the test for unrelated reasons.
        /// </remarks>
        private static void RunBuilder(
            float3[] positions,
            int[] triangles,
            out BlasBuildResult result,
            out int[] survivingPrimitiveIds,
            out PrometheusBVHNode[] nodes,
            PrometheusBlasSplitStrategy strategy = PrometheusBlasSplitStrategy.BinnedSah)
        {
            int sourceTriangleCount = triangles.Length / 3;

            NativeArray<float3> vertices = default;
            NativeArray<int> indices = default;
            NativeArray<PrometheusSubMeshRange> subMeshes = default;
            NativeArray<PrometheusTriangle> triangleBuffer = default;
            NativeArray<PrometheusBVHNode> nodeBuffer = default;
            NativeArray<BlasBuildResult> resultBuffer = default;

            try
            {
                vertices = new NativeArray<float3>(positions, Allocator.TempJob);
                indices = new NativeArray<int>(triangles, Allocator.TempJob);
                subMeshes = new NativeArray<PrometheusSubMeshRange>(1, Allocator.TempJob);

                PrometheusSubMeshRange range;
                range.IndexStart = 0;
                range.IndexCount = triangles.Length;
                range.MaterialId = 0u;
                subMeshes[0] = range;

                triangleBuffer = new NativeArray<PrometheusTriangle>(sourceTriangleCount, Allocator.TempJob);
                nodeBuffer = new NativeArray<PrometheusBVHNode>(math.max(1, (2 * sourceTriangleCount) - 1), Allocator.TempJob);
                resultBuffer = new NativeArray<BlasBuildResult>(1, Allocator.TempJob);

                BlasBuildJob job = new BlasBuildJob
                {
                    Vertices = vertices,
                    Indices = indices,
                    SubMeshes = subMeshes,
                    MaxLeafTriangles = BlasBuildJob.DefaultMaxLeafTriangles,
                    SplitStrategy = strategy,
                    Triangles = triangleBuffer,
                    Nodes = nodeBuffer,
                    Result = resultBuffer
                };

                job.Run();

                result = resultBuffer[0];

                survivingPrimitiveIds = new int[result.TriangleCount];
                for (int i = 0; i < result.TriangleCount; i++)
                {
                    survivingPrimitiveIds[i] = (int)triangleBuffer[i].PrimitiveId;
                }

                nodes = new PrometheusBVHNode[math.max(0, result.NodeCount)];
                for (int i = 0; i < nodes.Length; i++)
                {
                    nodes[i] = nodeBuffer[i];
                }
            }
            finally
            {
                if (resultBuffer.IsCreated)
                {
                    resultBuffer.Dispose();
                }

                if (nodeBuffer.IsCreated)
                {
                    nodeBuffer.Dispose();
                }

                if (triangleBuffer.IsCreated)
                {
                    triangleBuffer.Dispose();
                }

                if (subMeshes.IsCreated)
                {
                    subMeshes.Dispose();
                }

                if (indices.IsCreated)
                {
                    indices.Dispose();
                }

                if (vertices.IsCreated)
                {
                    vertices.Dispose();
                }
            }
        }

        /// <summary>
        /// Rebuilds a mesh with the requested strategy and returns the hierarchy's SAH cost, the
        /// expected number of primitive tests for a random ray that enters the root box.
        /// </summary>
        /// <param name="mesh">Mesh to rebuild.</param>
        /// <param name="strategy">Partitioning strategy to measure.</param>
        /// <returns>The SAH cost of the resulting tree.</returns>
        private static float BuildAndMeasure(float3[] positions, int[] indices, PrometheusBlasSplitStrategy strategy)
        {
            RunBuilder(positions, indices, out BlasBuildResult result, out _, out PrometheusBVHNode[] nodes, strategy);
            Assert.That(result.Status, Is.EqualTo(PrometheusBlasBuildStatus.Success));

            float rootArea = HalfSurfaceArea(nodes[0].BoundsMin, nodes[0].BoundsMax);
            float cost = 0.0f;

            for (int i = 0; i < nodes.Length; i++)
            {
                PrometheusBVHNode node = nodes[i];
                float area = HalfSurfaceArea(node.BoundsMin, node.BoundsMax) / rootArea;
                cost += node.IsLeaf ? (area * node.PrimitiveCount) : (area * BlasBuildJob.TraversalCost);
            }

            return cost;
        }

        /// <summary>
        /// Produces irregularly placed clusters of differing tessellation density, spread across a
        /// volume roughly twenty times a single cluster's radius.
        /// </summary>
        /// <remarks>
        /// The spread is kept moderate on purpose so every cluster still spans several bins; the
        /// intent is to vary occupancy, not to defeat the binning resolution. A fixed seed keeps
        /// the arrangement deterministic across runs.
        /// </remarks>
        /// <param name="positions">Receives the combined vertex positions.</param>
        /// <param name="indices">Receives the combined triangle indices.</param>
        private static void CreateClusteredGeometry(out float3[] positions, out int[] indices)
        {
            const int clusterCount = 12;
            const float spread = 20.0f;

            List<float3> vertices = new List<float3>();
            List<int> triangles = new List<int>();
            Unity.Mathematics.Random random = new Unity.Mathematics.Random(0x5EED1234u);

            for (int c = 0; c < clusterCount; c++)
            {
                float3 centre = random.NextFloat3(new float3(-spread), new float3(spread));
                float radius = random.NextFloat(0.5f, 2.5f);

                // Density varies per cluster, so the triangle count per unit volume is uneven.
                int rings = random.NextInt(6, 26);
                int sectors = rings * 2;
                int clusterBase = vertices.Count;

                for (int lat = 0; lat <= rings; lat++)
                {
                    float theta = (lat * math.PI) / rings;
                    float sinTheta = math.sin(theta);
                    float cosTheta = math.cos(theta);

                    for (int lon = 0; lon <= sectors; lon++)
                    {
                        float phi = (lon * 2.0f * math.PI) / sectors;
                        vertices.Add(centre + (new float3(
                            sinTheta * math.cos(phi),
                            cosTheta,
                            sinTheta * math.sin(phi)) * radius));
                    }
                }

                for (int lat = 0; lat < rings; lat++)
                {
                    for (int lon = 0; lon < sectors; lon++)
                    {
                        int a = clusterBase + (lat * (sectors + 1)) + lon;
                        int b = a + sectors + 1;

                        if (lat != 0)
                        {
                            triangles.Add(a);
                            triangles.Add(b);
                            triangles.Add(a + 1);
                        }

                        if (lat != rings - 1)
                        {
                            triangles.Add(b);
                            triangles.Add(b + 1);
                            triangles.Add(a + 1);
                        }
                    }
                }
            }

            positions = vertices.ToArray();
            indices = triangles.ToArray();
        }

        /// <summary>
        /// Builds a UV sphere that deliberately omits the degenerate slivers a naive generator
        /// emits at the poles, so the mesh contains no triangles the EC-003 filter would reject.
        /// </summary>
        /// <param name="rings">Latitude subdivisions.</param>
        /// <param name="sectors">Longitude subdivisions.</param>
        /// <param name="radius">Sphere radius.</param>
        /// <returns>A dense, fully valid triangle mesh.</returns>
        private static Mesh CreateUvSphere(int rings, int sectors, float radius)
        {
            List<Vector3> vertices = new List<Vector3>((rings + 1) * (sectors + 1));

            for (int lat = 0; lat <= rings; lat++)
            {
                float theta = (lat * math.PI) / rings;
                float sinTheta = math.sin(theta);
                float cosTheta = math.cos(theta);

                for (int lon = 0; lon <= sectors; lon++)
                {
                    float phi = (lon * 2.0f * math.PI) / sectors;
                    vertices.Add(new Vector3(
                        sinTheta * math.cos(phi) * radius,
                        cosTheta * radius,
                        sinTheta * math.sin(phi) * radius));
                }
            }

            List<int> indices = new List<int>(rings * sectors * 6);

            for (int lat = 0; lat < rings; lat++)
            {
                for (int lon = 0; lon < sectors; lon++)
                {
                    int a = (lat * (sectors + 1)) + lon;
                    int b = a + sectors + 1;

                    // The first and last rings collapse onto a pole, so emit one triangle there
                    // instead of a quad whose second triangle would have zero area.
                    if (lat != 0)
                    {
                        indices.Add(a);
                        indices.Add(b);
                        indices.Add(a + 1);
                    }

                    if (lat != rings - 1)
                    {
                        indices.Add(b);
                        indices.Add(b + 1);
                        indices.Add(a + 1);
                    }
                }
            }

            Mesh mesh = new Mesh { name = "PrometheusDenseSphere" };
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(vertices);
            mesh.SetTriangles(indices, 0, true);
            return mesh;
        }
    }
}
