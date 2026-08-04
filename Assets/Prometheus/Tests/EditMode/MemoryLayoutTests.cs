using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using NUnit.Framework;
using Prometheus.Core.Structs;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.TestTools.Constraints;

// UnityEngine.TestTools.Constraints.Is derives from NUnit.Framework.Is, so this alias keeps
// every NUnit constraint available while adding the AllocatingGCMemory() extension. Importing
// the namespace without the alias would make the bare name `Is` ambiguous.
using Is = UnityEngine.TestTools.Constraints.Is;

namespace Prometheus.Tests.EditMode
{
    /// <summary>
    /// Phase 1 verification gate (Section 7.1): proves that every structure crossing the
    /// CPU/GPU boundary has an exact, 16-byte aligned stride, that the HLSL mirror in
    /// <c>PrometheusStructures.hlsl</c> agrees field for field, and that the accessor layer
    /// used inside Burst jobs and render passes allocates zero managed memory (Section 3.2).
    /// </summary>
    [TestFixture]
    public sealed class MemoryLayoutTests
    {
        private const int RegisterSize = 16;

        private static readonly string HlslRelativePath =
            Path.Combine("Prometheus", "Runtime", "Shaders", "Includes", "PrometheusStructures.hlsl");

        // Sinks for the zero-allocation probes. Instance fields rather than captured locals so
        // the measured delegates close over nothing.
        private float _sinkFloat;
        private uint _sinkUint;
        private bool _sinkBool;

        // =================================================================================
        //  Stride verification (UnsafeUtility.SizeOf<T>)
        // =================================================================================

        [Test]
        public void BvhNode_StrideIs32Bytes()
        {
            Assert.That(UnsafeUtility.SizeOf<PrometheusBVHNode>(), Is.EqualTo(32),
                "PrometheusBVHNode must occupy exactly two 16-byte registers.");
            Assert.That(PrometheusBVHNode.SizeInBytes, Is.EqualTo(UnsafeUtility.SizeOf<PrometheusBVHNode>()),
                "The declared SizeInBytes constant has drifted from the real struct size.");
        }

        [Test]
        public void Triangle_StrideIs48Bytes()
        {
            Assert.That(UnsafeUtility.SizeOf<PrometheusTriangle>(), Is.EqualTo(48),
                "PrometheusTriangle must occupy exactly three 16-byte registers.");
            Assert.That(PrometheusTriangle.SizeInBytes, Is.EqualTo(UnsafeUtility.SizeOf<PrometheusTriangle>()),
                "The declared SizeInBytes constant has drifted from the real struct size.");
        }

        [Test]
        public void Instance_StrideIs64Bytes()
        {
            Assert.That(UnsafeUtility.SizeOf<PrometheusInstance>(), Is.EqualTo(64),
                "PrometheusInstance must occupy exactly four 16-byte registers.");
            Assert.That(PrometheusInstance.SizeInBytes, Is.EqualTo(UnsafeUtility.SizeOf<PrometheusInstance>()),
                "The declared SizeInBytes constant has drifted from the real struct size.");
        }

        [Test]
        public void Aabb_IsTightlyPackedAndNotUploaded()
        {
            // PrometheusAabb is a CPU-side build input, so it is packed to 24 bytes rather than
            // padded to a register boundary. Keeping it out of the instance record is what allows
            // that record to reach 64 bytes.
            Assert.That(UnsafeUtility.SizeOf<PrometheusAabb>(), Is.EqualTo(24));
            AssertOffset<PrometheusAabb>(nameof(PrometheusAabb.Min), 0);
            AssertOffset<PrometheusAabb>(nameof(PrometheusAabb.Max), 12);
        }

        [Test]
        public void AllStructures_StridesAreMultiplesOf16()
        {
            Assert.That(UnsafeUtility.SizeOf<PrometheusBVHNode>() % RegisterSize, Is.Zero);
            Assert.That(UnsafeUtility.SizeOf<PrometheusTriangle>() % RegisterSize, Is.Zero);
            Assert.That(UnsafeUtility.SizeOf<PrometheusInstance>() % RegisterSize, Is.Zero);
        }

        [Test]
        public void AllStructures_ManagedAndUnmanagedSizesAgree()
        {
            // A divergence here means the marshaller is inserting padding that UnsafeUtility
            // does not see, which would silently corrupt GraphicsBuffer.SetData uploads.
            Assert.That(Marshal.SizeOf<PrometheusBVHNode>(), Is.EqualTo(UnsafeUtility.SizeOf<PrometheusBVHNode>()));
            Assert.That(Marshal.SizeOf<PrometheusTriangle>(), Is.EqualTo(UnsafeUtility.SizeOf<PrometheusTriangle>()));
            Assert.That(Marshal.SizeOf<PrometheusInstance>(), Is.EqualTo(UnsafeUtility.SizeOf<PrometheusInstance>()));
        }

        // =================================================================================
        //  Explicit field offset verification
        // =================================================================================

        [Test]
        public void BvhNode_FieldOffsetsMatchContract()
        {
            AssertOffset<PrometheusBVHNode>(nameof(PrometheusBVHNode.BoundsMin), 0);
            AssertOffset<PrometheusBVHNode>(nameof(PrometheusBVHNode.EscapeIndex), 12);
            AssertOffset<PrometheusBVHNode>(nameof(PrometheusBVHNode.BoundsMax), 16);
            AssertOffset<PrometheusBVHNode>(nameof(PrometheusBVHNode.PackedLeafRange), 28);
        }

        [Test]
        public void Triangle_FieldOffsetsMatchContract()
        {
            AssertOffset<PrometheusTriangle>(nameof(PrometheusTriangle.Vertex0), 0);
            AssertOffset<PrometheusTriangle>(nameof(PrometheusTriangle.PrimitiveId), 12);
            AssertOffset<PrometheusTriangle>(nameof(PrometheusTriangle.Edge1), 16);
            AssertOffset<PrometheusTriangle>(nameof(PrometheusTriangle.MaterialId), 28);
            AssertOffset<PrometheusTriangle>(nameof(PrometheusTriangle.Edge2), 32);
            AssertOffset<PrometheusTriangle>(nameof(PrometheusTriangle.PackedNormal), 44);
        }

        [Test]
        public void Instance_FieldOffsetsMatchContract()
        {
            AssertOffset<PrometheusInstance>(nameof(PrometheusInstance.WorldToLocalRow0), 0);
            AssertOffset<PrometheusInstance>(nameof(PrometheusInstance.WorldToLocalRow1), 16);
            AssertOffset<PrometheusInstance>(nameof(PrometheusInstance.WorldToLocalRow2), 32);
            AssertOffset<PrometheusInstance>(nameof(PrometheusInstance.BlasNodeOffset), 48);
            AssertOffset<PrometheusInstance>(nameof(PrometheusInstance.BlasTriangleOffset), 52);
            AssertOffset<PrometheusInstance>(nameof(PrometheusInstance.MaterialId), 56);
            AssertOffset<PrometheusInstance>(nameof(PrometheusInstance.PackedFlagsAndMeshId), 60);
        }

        [Test]
        public void AllStructures_AreBlittableAndSequential()
        {
            AssertBlittableSequential<PrometheusBVHNode>();
            AssertBlittableSequential<PrometheusTriangle>();
            AssertBlittableSequential<PrometheusInstance>();
        }

        // =================================================================================
        //  CPU <-> GPU layout equivalence (parses the authoritative HLSL header)
        // =================================================================================

        [Test]
        public void HlslMirror_DeclaresIdenticalStrides()
        {
            string source = ReadHlslSource();

            Assert.That(ReadHlslDefine(source, "PROMETHEUS_NODE_STRIDE"),
                Is.EqualTo(UnsafeUtility.SizeOf<PrometheusBVHNode>()));
            Assert.That(ReadHlslDefine(source, "PROMETHEUS_TRIANGLE_STRIDE"),
                Is.EqualTo(UnsafeUtility.SizeOf<PrometheusTriangle>()));
            Assert.That(ReadHlslDefine(source, "PROMETHEUS_INSTANCE_STRIDE"),
                Is.EqualTo(UnsafeUtility.SizeOf<PrometheusInstance>()));
        }

        [Test]
        public void HlslMirror_BvhNodeFieldsAlignWithCSharp()
        {
            AssertHlslLayoutMatches<PrometheusBVHNode>("PrometheusBVHNode", new[]
            {
                ("boundsMin", nameof(PrometheusBVHNode.BoundsMin)),
                ("escapeIndex", nameof(PrometheusBVHNode.EscapeIndex)),
                ("boundsMax", nameof(PrometheusBVHNode.BoundsMax)),
                ("packedLeafRange", nameof(PrometheusBVHNode.PackedLeafRange))
            });
        }

        [Test]
        public void HlslMirror_TriangleFieldsAlignWithCSharp()
        {
            AssertHlslLayoutMatches<PrometheusTriangle>("PrometheusTriangle", new[]
            {
                ("vertex0", nameof(PrometheusTriangle.Vertex0)),
                ("primitiveId", nameof(PrometheusTriangle.PrimitiveId)),
                ("edge1", nameof(PrometheusTriangle.Edge1)),
                ("materialId", nameof(PrometheusTriangle.MaterialId)),
                ("edge2", nameof(PrometheusTriangle.Edge2)),
                ("packedNormal", nameof(PrometheusTriangle.PackedNormal))
            });
        }

        [Test]
        public void HlslMirror_InstanceFieldsAlignWithCSharp()
        {
            AssertHlslLayoutMatches<PrometheusInstance>("PrometheusInstance", new[]
            {
                ("worldToLocalRow0", nameof(PrometheusInstance.WorldToLocalRow0)),
                ("worldToLocalRow1", nameof(PrometheusInstance.WorldToLocalRow1)),
                ("worldToLocalRow2", nameof(PrometheusInstance.WorldToLocalRow2)),
                ("blasNodeOffset", nameof(PrometheusInstance.BlasNodeOffset)),
                ("blasTriangleOffset", nameof(PrometheusInstance.BlasTriangleOffset)),
                ("materialId", nameof(PrometheusInstance.MaterialId)),
                ("packedFlagsAndMeshId", nameof(PrometheusInstance.PackedFlagsAndMeshId))
            });
        }

        [Test]
        public void HlslMirror_StructuralBoundsMatchSpecification()
        {
            string source = ReadHlslSource();

            Assert.That(ReadHlslDefine(source, "PROMETHEUS_MAX_SCENE_TRIANGLES"),
                Is.EqualTo(PrometheusTriangle.MaxSceneTriangles), "Section 3.1 caps scene triangles at 2^21.");
            Assert.That(ReadHlslDefine(source, "PROMETHEUS_MAX_BVH_NODES"),
                Is.EqualTo((int)PrometheusBVHNode.MaxPrimitiveIndex), "Section 3.1 caps LBVH nodes at 2^22 - 1.");
            Assert.That(ReadHlslDefine(source, "PROMETHEUS_MAX_INSTANCES"),
                Is.EqualTo(PrometheusInstance.MaxInstanceCount));
            Assert.That(ReadHlslDefine(source, "PROMETHEUS_PRIMITIVE_INDEX_BITS"),
                Is.EqualTo(PrometheusBVHNode.PrimitiveIndexBits));
            Assert.That(ReadHlslDefine(source, "PROMETHEUS_PRIMITIVE_COUNT_BITS"),
                Is.EqualTo(PrometheusBVHNode.PrimitiveCountBits));
        }

        // =================================================================================
        //  Bit-packing correctness
        // =================================================================================

        [Test]
        public void PackedLeafRange_RoundTripsAcrossTheFullRange()
        {
            uint[] indices = { 0u, 1u, 1023u, 65535u, 2097151u, PrometheusBVHNode.MaxPrimitiveIndex };
            uint[] counts = { 1u, 2u, 8u, 255u, PrometheusBVHNode.MaxLeafPrimitiveCount };

            foreach (uint index in indices)
            {
                foreach (uint count in counts)
                {
                    PrometheusBVHNode leaf = PrometheusBVHNode.CreateLeaf(
                        float3.zero, new float3(1.0f), PrometheusBVHNode.InvalidIndex, index, count);

                    Assert.That(leaf.FirstPrimitive, Is.EqualTo(index), $"index={index} count={count}");
                    Assert.That(leaf.PrimitiveCount, Is.EqualTo(count), $"index={index} count={count}");
                    Assert.That(leaf.IsLeaf, Is.True);
                }
            }
        }

        [Test]
        public void InteriorNode_IsNeverClassifiedAsLeaf()
        {
            PrometheusBVHNode interior = PrometheusBVHNode.CreateInterior(
                new float3(-1.0f), new float3(1.0f), 7u);

            Assert.That(interior.IsLeaf, Is.False);
            Assert.That(interior.PrimitiveCount, Is.Zero);
            Assert.That(interior.EscapeIndex, Is.EqualTo(7u));
        }

        [Test]
        public void PackedLeafRange_ClampsRatherThanOverflowingIntoTheCountField()
        {
            // An out-of-range index must never corrupt the count bits, which would turn an
            // interior node into a phantom leaf during traversal.
            uint packed = PrometheusBVHNode.PackLeafRange(uint.MaxValue, 1u);

            Assert.That(packed & PrometheusBVHNode.MaxPrimitiveIndex, Is.EqualTo(PrometheusBVHNode.MaxPrimitiveIndex));
            Assert.That(packed >> PrometheusBVHNode.PrimitiveIndexBits, Is.EqualTo(1u));
        }

        [Test]
        public void OctahedralNormal_RoundTripsWithinQuantisationError()
        {
            float3[] normals =
            {
                new float3(0.0f, 1.0f, 0.0f),
                new float3(0.0f, -1.0f, 0.0f),
                new float3(1.0f, 0.0f, 0.0f),
                new float3(0.0f, 0.0f, -1.0f),
                math.normalize(new float3(1.0f, 1.0f, 1.0f)),
                math.normalize(new float3(-0.3f, 0.7f, -0.64f)),
                math.normalize(new float3(0.577f, -0.577f, 0.577f))
            };

            foreach (float3 n in normals)
            {
                float3 decoded = PrometheusTriangle.DecodeNormal(PrometheusTriangle.EncodeNormal(n));

                Assert.That(math.length(decoded), Is.EqualTo(1.0f).Within(1e-4f), $"normal {n} decoded non-unit");
                Assert.That(math.dot(decoded, n), Is.GreaterThan(0.99999f), $"normal {n} round-trip drifted to {decoded}");
            }
        }

        [Test]
        public void Triangle_DerivesEdgesNormalAndBoundsFromVertices()
        {
            float3 a = new float3(0.0f, 0.0f, 0.0f);
            float3 b = new float3(2.0f, 0.0f, 0.0f);
            float3 c = new float3(0.0f, 0.0f, -2.0f);

            PrometheusTriangle tri = PrometheusTriangle.Create(a, b, c, 11u, 3u);

            Assert.That(tri.Vertex1.Equals(b), Is.True);
            Assert.That(tri.Vertex2.Equals(c), Is.True);
            Assert.That(tri.PrimitiveId, Is.EqualTo(11u));
            Assert.That(tri.MaterialId, Is.EqualTo(3u));
            Assert.That(tri.Area, Is.EqualTo(2.0f).Within(1e-5f));
            Assert.That(tri.IsDegenerate, Is.False);
            Assert.That(math.dot(tri.Normal, new float3(0.0f, 1.0f, 0.0f)), Is.GreaterThan(0.99999f));
            Assert.That(tri.BoundsMin.Equals(new float3(0.0f, 0.0f, -2.0f)), Is.True);
            Assert.That(tri.BoundsMax.Equals(new float3(2.0f, 0.0f, 0.0f)), Is.True);
        }

        [Test]
        public void Triangle_FlagsDegenerateAndNonFiniteGeometry()
        {
            // EC-003: INVALID_GEOMETRY - collinear vertices and NaN coordinates must both be rejected.
            PrometheusTriangle collinear = PrometheusTriangle.Create(
                float3.zero, new float3(1.0f, 0.0f, 0.0f), new float3(2.0f, 0.0f, 0.0f), 0u, 0u);
            Assert.That(collinear.IsDegenerate, Is.True, "Collinear vertices must be reported as degenerate.");

            PrometheusTriangle nan = PrometheusTriangle.Create(
                float3.zero, new float3(float.NaN, 0.0f, 0.0f), new float3(0.0f, 0.0f, 1.0f), 0u, 0u);
            Assert.That(nan.IsDegenerate, Is.True, "NaN coordinates must be reported as degenerate.");
        }

        [Test]
        public void Instance_DerivesInverseTransformAndWorldBounds()
        {
            float4x4 localToWorld = float4x4.TRS(
                new float3(5.0f, 1.0f, -2.0f),
                quaternion.EulerXYZ(0.0f, math.PI * 0.5f, 0.0f),
                new float3(2.0f, 2.0f, 2.0f));

            PrometheusInstance instance = PrometheusInstance.Create(
                localToWorld,
                new float3(-0.5f),
                new float3(0.5f),
                meshId: 4u,
                blasNodeOffset: 128u,
                blasTriangleOffset: 512u,
                materialId: 9u,
                flags: PrometheusInstanceFlags.DefaultStaticOpaque,
                worldBounds: out PrometheusAabb worldBounds);

            // The forward transform is no longer stored, so recovering it must still invert the
            // packed rows exactly.
            float4x4 identity = math.mul(instance.GetLocalToWorld(), instance.GetWorldToLocal());
            for (int c = 0; c < 4; c++)
            {
                for (int r = 0; r < 4; r++)
                {
                    float expected = c == r ? 1.0f : 0.0f;
                    Assert.That(identity[c][r], Is.EqualTo(expected).Within(1e-4f), $"element [{c}][{r}]");
                }
            }

            // A unit cube scaled 2x and rotated about Y still has a 2-unit world extent.
            // Compared with tolerance: the rotation puts inexact zeros in the basis vectors.
            AssertApproximately(worldBounds.Min, new float3(4.0f, 0.0f, -3.0f), 1e-4f, "world bounds min");
            AssertApproximately(worldBounds.Max, new float3(6.0f, 2.0f, -1.0f), 1e-4f, "world bounds max");
            AssertApproximately(worldBounds.Centre, new float3(5.0f, 1.0f, -2.0f), 1e-4f, "world centroid");

            Assert.That(instance.IsActive, Is.True);
            Assert.That(instance.CastsShadows, Is.True);
            Assert.That(instance.IsDynamic, Is.False);
            Assert.That(instance.HasFlag(PrometheusInstanceFlags.MirroredWinding), Is.False);
            Assert.That(instance.BlasNodeOffset, Is.EqualTo(128u));
            Assert.That(instance.BlasTriangleOffset, Is.EqualTo(512u));
            Assert.That(instance.MeshId, Is.EqualTo(4u), "the mesh id must survive packing alongside the flags");
        }

        [Test]
        public void Instance_DetectsMirroredWindingFromNegativeDeterminant()
        {
            float4x4 mirrored = float4x4.Scale(new float3(-1.0f, 1.0f, 1.0f));

            PrometheusInstance instance = PrometheusInstance.Create(
                mirrored, new float3(-1.0f), new float3(1.0f), 7u, 0u, 0u, 0u,
                PrometheusInstanceFlags.Active, out _);

            Assert.That(instance.HasFlag(PrometheusInstanceFlags.MirroredWinding), Is.True);

            instance.UpdateTransform(float4x4.identity, new float3(-1.0f), new float3(1.0f), out _);
            Assert.That(instance.HasFlag(PrometheusInstanceFlags.MirroredWinding), Is.False,
                "The mirror flag must clear when the transform returns to a positive determinant.");
            Assert.That(instance.MeshId, Is.EqualTo(7u),
                "flipping a flag must not disturb the mesh id sharing the same word");
        }

        [Test]
        public void Instance_RayTransformPreservesTParameterisation()
        {
            float4x4 localToWorld = float4x4.TRS(
                new float3(3.0f, 0.0f, 0.0f), quaternion.identity, new float3(4.0f));

            PrometheusInstance instance = PrometheusInstance.Create(
                localToWorld, new float3(-1.0f), new float3(1.0f), 0u, 0u, 0u, 0u,
                PrometheusInstanceFlags.Active, out _);

            float3 worldOrigin = new float3(-10.0f, 0.0f, 0.0f);
            float3 worldDirection = new float3(1.0f, 0.0f, 0.0f);
            const float t = 9.0f;

            instance.TransformRayToLocal(worldOrigin, worldDirection, out float3 localOrigin, out float3 localDirection);

            // The same t must land on the same physical point in both spaces.
            float3 worldPoint = worldOrigin + worldDirection * t;
            float3 localPoint = localOrigin + localDirection * t;
            float3 roundTripped = math.transform(instance.GetLocalToWorld(), localPoint);

            Assert.That(math.distance(roundTripped, worldPoint), Is.LessThan(1e-3f),
                "Normalising the object-space direction would break t comparability across instances.");
        }

        // =================================================================================
        //  Zero managed allocation gate (Section 3.2 / BAN-002)
        // =================================================================================

        [Test]
        public void BvhNodeAccessors_AllocateNoManagedMemory()
        {
            AssertSteadyStateIsAllocationFree(BuildAndReadNodes);
        }

        [Test]
        public void TriangleAccessors_AllocateNoManagedMemory()
        {
            AssertSteadyStateIsAllocationFree(BuildAndReadTriangles);
        }

        [Test]
        public void InstanceAccessors_AllocateNoManagedMemory()
        {
            AssertSteadyStateIsAllocationFree(BuildAndReadInstances);
        }

        /// <summary>
        /// Verifies that reading and writing pre-allocated native buffers is allocation free.
        /// </summary>
        /// <remarks>
        /// Buffer construction is deliberately outside the measured region. In the Editor,
        /// <c>ENABLE_UNITY_COLLECTIONS_CHECKS</c> attaches managed safety state to every
        /// <see cref="NativeArray{T}"/> as it is created, so allocating inside the probe would
        /// measure the safety system rather than the layout. That matches the intent of Section
        /// 3.2 regardless: buffers are pre-allocated with <c>Allocator.Persistent</c> at scene
        /// scope, and only the element access path runs per frame.
        /// </remarks>
        [Test]
        public void NativeArrayElementAccess_AllocatesNoManagedMemory()
        {
            const int count = 128;

            NativeArray<PrometheusBVHNode> nodes = default;
            NativeArray<PrometheusTriangle> triangles = default;
            NativeArray<PrometheusInstance> instances = default;

            try
            {
                nodes = new NativeArray<PrometheusBVHNode>(count, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                triangles = new NativeArray<PrometheusTriangle>(count, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                instances = new NativeArray<PrometheusInstance>(count, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

                NativeArray<PrometheusBVHNode> capturedNodes = nodes;
                NativeArray<PrometheusTriangle> capturedTriangles = triangles;
                NativeArray<PrometheusInstance> capturedInstances = instances;

                AssertSteadyStateIsAllocationFree(() => RoundTripElements(capturedNodes, capturedTriangles, capturedInstances));
            }
            finally
            {
                // Section 3.2: every NativeArray allocation has an explicitly paired Dispose.
                if (nodes.IsCreated)
                {
                    nodes.Dispose();
                }

                if (triangles.IsCreated)
                {
                    triangles.Dispose();
                }

                if (instances.IsCreated)
                {
                    instances.Dispose();
                }
            }
        }

        /// <summary>
        /// Runs <paramref name="body"/> a few times before measuring, then asserts that a
        /// subsequent invocation allocates nothing.
        /// </summary>
        /// <remarks>
        /// The warm-up matters: Unity's constraint measures a single invocation, so without it
        /// the Mono JIT compiling the cold call graph is counted as managed allocation. The
        /// budget in Section 1.1 is a steady-state per-frame figure, not a one-time JIT cost.
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

        private void BuildAndReadNodes()
        {
            for (uint i = 0; i < 256u; i++)
            {
                PrometheusBVHNode leaf = PrometheusBVHNode.CreateLeaf(
                    new float3(i), new float3(i + 1u), i + 1u, i, (i & 7u) + 1u);
                PrometheusBVHNode interior = PrometheusBVHNode.CreateInterior(
                    new float3(-1.0f), new float3(1.0f), i);

                interior.EncapsulateBounds(leaf);
                interior.EncapsulatePoint(leaf.Centroid);

                _sinkUint += leaf.FirstPrimitive + leaf.PrimitiveCount + interior.EscapeIndex;
                _sinkFloat += interior.SurfaceArea + leaf.Centroid.x;
                _sinkBool ^= leaf.IsLeaf ^ interior.IsLeaf;
            }
        }

        private void BuildAndReadTriangles()
        {
            for (uint i = 0; i < 256u; i++)
            {
                PrometheusTriangle tri = PrometheusTriangle.Create(
                    new float3(i, 0.0f, 0.0f),
                    new float3(i + 1u, 0.0f, 0.0f),
                    new float3(i, 0.0f, 1.0f),
                    i,
                    i & 15u);

                _sinkFloat += tri.Area + tri.Centroid.x + tri.Normal.y + tri.BoundsMin.z + tri.BoundsMax.x;
                _sinkUint += tri.PackedNormal + tri.PrimitiveId + tri.MaterialId;
                _sinkBool ^= tri.IsDegenerate;
            }
        }

        private void BuildAndReadInstances()
        {
            for (uint i = 0; i < 64u; i++)
            {
                float4x4 localToWorld = float4x4.TRS(new float3(i, 0.0f, 0.0f), quaternion.identity, new float3(1.0f));

                PrometheusInstance instance = PrometheusInstance.Create(
                    localToWorld, new float3(-1.0f), new float3(1.0f), i, i * 2u, i * 3u, i & 7u,
                    PrometheusInstanceFlags.DefaultStaticOpaque, out PrometheusAabb bounds);

                instance.UpdateTransform(localToWorld, new float3(-2.0f), new float3(2.0f), out bounds);
                instance.TransformRayToLocal(
                    new float3(0.0f, 5.0f, 0.0f), new float3(0.0f, -1.0f, 0.0f),
                    out float3 localOrigin, out float3 localDirection);

                _sinkFloat += bounds.Centre.x + localOrigin.y + localDirection.y;
                _sinkUint += instance.BlasNodeOffset + instance.BlasTriangleOffset + instance.Flags;
                _sinkBool ^= instance.IsActive ^ instance.IsDoubleSided;
            }
        }

        /// <summary>
        /// Fills and then reads back every element of three pre-allocated native buffers,
        /// exercising the blittable write/read path that the GPU upload depends on.
        /// </summary>
        /// <param name="nodes">Pre-allocated node buffer.</param>
        /// <param name="triangles">Pre-allocated triangle buffer.</param>
        /// <param name="instances">Pre-allocated instance buffer.</param>
        private void RoundTripElements(
            NativeArray<PrometheusBVHNode> nodes,
            NativeArray<PrometheusTriangle> triangles,
            NativeArray<PrometheusInstance> instances)
        {
            int count = nodes.Length;

            for (int i = 0; i < count; i++)
            {
                nodes[i] = PrometheusBVHNode.CreateLeaf(
                    new float3(i), new float3(i + 1), (uint)i + 1u, (uint)i, 1u);
                triangles[i] = PrometheusTriangle.Create(
                    new float3(i, 0.0f, 0.0f), new float3(i + 1, 0.0f, 0.0f), new float3(i, 1.0f, 0.0f),
                    (uint)i, 0u);
                instances[i] = PrometheusInstance.Create(
                    float4x4.identity, new float3(-1.0f), new float3(1.0f),
                    (uint)i, 0u, 0u, 0u, PrometheusInstanceFlags.DefaultStaticOpaque, out _);
            }

            for (int i = 0; i < count; i++)
            {
                _sinkUint += nodes[i].FirstPrimitive + triangles[i].PrimitiveId + instances[i].MeshId;
                _sinkFloat += nodes[i].SurfaceArea + triangles[i].Area + instances[i].WorldToLocalRow0.w;
            }
        }

        // =================================================================================
        //  Helpers
        // =================================================================================

        private static void AssertApproximately(float3 actual, float3 expected, float tolerance, string label)
        {
            Assert.That(math.distance(actual, expected), Is.LessThan(tolerance),
                $"{label}: expected {expected}, got {actual}");
        }

        private static void AssertOffset<T>(string fieldName, int expectedOffset) where T : struct
        {
            int actual = Marshal.OffsetOf<T>(fieldName).ToInt32();
            Assert.That(actual, Is.EqualTo(expectedOffset),
                $"{typeof(T).Name}.{fieldName} sits at byte {actual}, breaking the {expectedOffset}-byte GPU contract.");
        }

        private static void AssertBlittableSequential<T>() where T : struct
        {
            Assert.That(UnsafeUtility.IsBlittable<T>(), Is.True,
                $"{typeof(T).Name} must be blittable to cross into a GraphicsBuffer (BAN-005).");
            Assert.That(typeof(T).IsLayoutSequential, Is.True,
                $"{typeof(T).Name} must carry [StructLayout(LayoutKind.Sequential)] (Section 3.2).");
        }

        private static string ReadHlslSource()
        {
            string path = Path.Combine(Application.dataPath, HlslRelativePath);
            Assert.That(File.Exists(path), Is.True, $"Authoritative HLSL contract missing at {path}");
            return File.ReadAllText(path);
        }

        private static int ReadHlslDefine(string source, string name)
        {
            Match match = Regex.Match(source, $@"^\s*#define\s+{Regex.Escape(name)}\s+(-?\d+)", RegexOptions.Multiline);
            Assert.That(match.Success, Is.True, $"HLSL contract does not define {name}.");
            return int.Parse(match.Groups[1].Value);
        }

        /// <summary>
        /// Parses the named HLSL struct and asserts that each declared member lands on the same
        /// byte offset as its C# counterpart, and that the struct's total stride matches.
        /// </summary>
        /// <remarks>
        /// Offsets are computed with the HLSL packing rule that no member may straddle a 16-byte
        /// register boundary. Any layout satisfying that rule is also tightly packed here, so the
        /// computed offsets are valid for both <c>StructuredBuffer</c> and constant buffer usage.
        /// </remarks>
        private static void AssertHlslLayoutMatches<T>(string hlslStructName, (string HlslField, string CSharpField)[] fields)
            where T : struct
        {
            string body = ExtractHlslStructBody(ReadHlslSource(), hlslStructName);
            List<(string Name, int Offset, int Size)> members = ComputeHlslMemberOffsets(body);

            Assert.That(members.Count, Is.EqualTo(fields.Length),
                $"HLSL struct {hlslStructName} declares {members.Count} members but the C# mirror expects {fields.Length}.");

            for (int i = 0; i < fields.Length; i++)
            {
                (string hlslField, string csharpField) = fields[i];

                Assert.That(members[i].Name, Is.EqualTo(hlslField),
                    $"{hlslStructName} member {i} is '{members[i].Name}', expected '{hlslField}'. Field order defines the layout.");

                int csharpOffset = Marshal.OffsetOf<T>(csharpField).ToInt32();
                Assert.That(members[i].Offset, Is.EqualTo(csharpOffset),
                    $"{hlslStructName}.{hlslField} is at HLSL byte {members[i].Offset} but {typeof(T).Name}.{csharpField} is at {csharpOffset}.");
            }

            int hlslStride = RoundUpToRegister(members[members.Count - 1].Offset + members[members.Count - 1].Size);
            Assert.That(hlslStride, Is.EqualTo(UnsafeUtility.SizeOf<T>()),
                $"HLSL struct {hlslStructName} strides at {hlslStride} bytes, C# {typeof(T).Name} at {UnsafeUtility.SizeOf<T>()}.");
        }

        private static string ExtractHlslStructBody(string source, string structName)
        {
            Match match = Regex.Match(source, $@"struct\s+{Regex.Escape(structName)}\s*\{{(?<body>[^}}]*)\}}\s*;");
            Assert.That(match.Success, Is.True, $"Could not locate 'struct {structName}' in the HLSL contract.");
            return match.Groups["body"].Value;
        }

        private static List<(string Name, int Offset, int Size)> ComputeHlslMemberOffsets(string body)
        {
            var members = new List<(string Name, int Offset, int Size)>();
            int offset = 0;

            foreach (string rawDeclaration in body.Split(';'))
            {
                string declaration = StripHlslComments(rawDeclaration).Trim();
                if (declaration.Length == 0)
                {
                    continue;
                }

                string[] tokens = declaration.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                Assert.That(tokens.Length, Is.GreaterThanOrEqualTo(2), $"Unparsable HLSL declaration: '{declaration}'");

                string name = tokens[tokens.Length - 1];
                string type = tokens[tokens.Length - 2];   // Skips storage qualifiers such as column_major.
                int size = HlslTypeSize(type);

                // No member may straddle a 16-byte register boundary.
                if (size > RegisterSize || (offset / RegisterSize) != ((offset + size - 1) / RegisterSize))
                {
                    offset = RoundUpToRegister(offset);
                }

                members.Add((name, offset, size));
                offset += size;
            }

            return members;
        }

        private static string StripHlslComments(string text)
        {
            return Regex.Replace(text, @"//[^\r\n]*", string.Empty);
        }

        private static int RoundUpToRegister(int value)
        {
            return ((value + RegisterSize - 1) / RegisterSize) * RegisterSize;
        }

        private static int HlslTypeSize(string type)
        {
            switch (type)
            {
                case "float":
                case "uint":
                case "int":
                    return 4;
                case "float2":
                case "uint2":
                case "int2":
                    return 8;
                case "float3":
                case "uint3":
                case "int3":
                    return 12;
                case "float4":
                case "uint4":
                case "int4":
                    return 16;
                case "float3x4":
                    return 48;
                case "float4x4":
                    return 64;
                default:
                    Assert.Fail($"Unsupported HLSL type '{type}' in the shared structure contract.");
                    return 0;
            }
        }
    }
}
