using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Prometheus.Core.Structs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Prometheus.Core.BVH
{
    /// <summary>A world-space ray with an explicit parametric interval.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusRay
    {
        /// <summary>World-space origin.</summary>
        public float3 Origin;

        /// <summary>World-space direction. Need not be normalised, but <c>t</c> is measured in its units.</summary>
        public float3 Direction;

        /// <summary>Near clip of the parametric interval.</summary>
        public float TMin;

        /// <summary>Far clip of the parametric interval.</summary>
        public float TMax;
    }

    /// <summary>Result of a closest-hit query.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusRayHit
    {
        /// <summary>Parametric distance along the ray, or the ray's <c>TMax</c> when nothing was hit.</summary>
        public float Distance;

        /// <summary>Möller–Trumbore barycentric pair <c>(u, v)</c>.</summary>
        public float2 Barycentrics;

        /// <summary>Index of the hit instance within the sorted TLAS instance buffer.</summary>
        public uint InstanceIndex;

        /// <summary>Source mesh triangle index carried by <see cref="PrometheusTriangle.PrimitiveId"/>.</summary>
        public uint PrimitiveId;

        /// <summary>Index of the hit triangle within the shared triangle buffer.</summary>
        public int TriangleIndex;

        /// <summary>Whether anything was hit at all.</summary>
        public bool IsHit;
    }

    /// <summary>
    /// Location of one BLAS inside the shared node and triangle buffers, indexed by
    /// <see cref="PrometheusInstance.MeshId"/>.
    /// </summary>
    /// <remarks>
    /// Only the brute-force reference needs this. The accelerated path never enumerates a BLAS, it
    /// descends into one, so <see cref="PrometheusInstance.BlasNodeOffset"/> alone is sufficient
    /// there and no triangle count is stored on the GPU-side instance record.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusBlasRange
    {
        /// <summary>First node of this BLAS within the shared node buffer.</summary>
        public int NodeStart;

        /// <summary>Number of nodes in this BLAS.</summary>
        public int NodeCount;

        /// <summary>First triangle of this BLAS within the shared triangle buffer.</summary>
        public int TriangleStart;

        /// <summary>Number of triangles in this BLAS.</summary>
        public int TriangleCount;
    }

    /// <summary>
    /// CPU reference tracer over the packed TLAS and BLAS memory, used by the Phase 3 accuracy gate.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The traversal here is a line-for-line mirror of the stackless rope walk in
    /// <c>PrometheusStructures.hlsl</c>, deliberately so: its purpose is to prove that the packed
    /// buffers are traversable exactly as the shader will traverse them. Any divergence between
    /// this and the HLSL would defeat the point of the gate.
    /// </para>
    /// <para>
    /// Both tracers are double-sided. Back-face culling is a shading policy that would only mask
    /// disagreements between the accelerated and reference paths, so it is switched off to make
    /// the comparison as strict as possible.
    /// </para>
    /// </remarks>
    public static class CpuBvhValidator
    {
        /// <summary>Guards the Möller–Trumbore division against edge-on rays; matches the HLSL constant.</summary>
        public const float IntersectEpsilon = 1e-8f;

        /// <summary>
        /// Traverses the TLAS, descending into each candidate instance's BLAS, and returns the
        /// closest hit.
        /// </summary>
        /// <param name="ray">World-space ray.</param>
        /// <param name="tlasNodes">TLAS nodes in depth-first pre-order.</param>
        /// <param name="tlasNodeCount">Number of valid TLAS nodes.</param>
        /// <param name="instanceIndices">Sorted instance permutation that TLAS leaves address.</param>
        /// <param name="instances">Instance records, in the order the caller submitted them.</param>
        /// <param name="blasNodes">Shared BLAS node buffer.</param>
        /// <param name="blasTriangles">Shared BLAS triangle buffer.</param>
        /// <returns>The closest hit, or a miss carrying the ray's far clip.</returns>
        public static PrometheusRayHit TraceClosest(
            in PrometheusRay ray,
            in NativeArray<PrometheusBVHNode> tlasNodes,
            int tlasNodeCount,
            in NativeArray<uint> instanceIndices,
            in NativeArray<PrometheusInstance> instances,
            in NativeArray<PrometheusBVHNode> blasNodes,
            in NativeArray<PrometheusTriangle> blasTriangles)
        {
            PrometheusRayHit hit = default;
            hit.Distance = ray.TMax;
            hit.IsHit = false;

            if (tlasNodeCount <= 0)
            {
                return hit;
            }

            float3 inverseDirection = SafeReciprocal(ray.Direction);
            uint cursor = 0;

            while (cursor != PrometheusBVHNode.InvalidIndex && cursor < (uint)tlasNodeCount)
            {
                PrometheusBVHNode node = tlasNodes[(int)cursor];

                if (!IntersectAabb(node, ray.Origin, inverseDirection, hit.Distance))
                {
                    cursor = node.EscapeIndex;
                    continue;
                }

                if (!node.IsLeaf)
                {
                    cursor++;
                    continue;
                }

                uint first = node.FirstPrimitive;
                uint count = node.PrimitiveCount;

                // A TLAS leaf gathers up to TlasBuilder.MaxLeafInstances instances. Its range
                // addresses the indirection buffer, which resolves to the submitted instance array.
                for (uint k = 0; k < count; k++)
                {
                    uint instanceIndex = instanceIndices[(int)(first + k)];
                    TraceInstance(instanceIndex, ray, instances, blasNodes, blasTriangles, ref hit);
                }

                cursor = node.EscapeIndex;
            }

            return hit;
        }

        /// <summary>
        /// Tests every triangle of every instance with no acceleration structure whatsoever.
        /// </summary>
        /// <remarks>
        /// Uses the same ray transform and the same intersection routine as
        /// <see cref="TraceClosest"/>, so agreement between the two is exact rather than
        /// approximate. Only the visitation order differs.
        /// </remarks>
        /// <param name="ray">World-space ray.</param>
        /// <param name="instances">Instances to test; inactive entries are skipped.</param>
        /// <param name="instanceCount">Number of submitted instances.</param>
        /// <param name="blasRanges">BLAS extents indexed by <see cref="PrometheusInstance.MeshId"/>.</param>
        /// <param name="blasTriangles">Shared BLAS triangle buffer.</param>
        /// <returns>The closest hit, or a miss carrying the ray's far clip.</returns>
        public static PrometheusRayHit TraceBruteForce(
            in PrometheusRay ray,
            in NativeArray<PrometheusInstance> instances,
            int instanceCount,
            in NativeArray<PrometheusBlasRange> blasRanges,
            in NativeArray<PrometheusTriangle> blasTriangles)
        {
            PrometheusRayHit hit = default;
            hit.Distance = ray.TMax;
            hit.IsHit = false;

            for (int index = 0; index < instanceCount; index++)
            {
                PrometheusInstance instance = instances[index];

                // The TLAS never references inactive instances, so the reference tracer must skip
                // them too. Instances are no longer compacted, so the array can contain them.
                if (!instance.IsActive)
                {
                    continue;
                }

                PrometheusBlasRange range = blasRanges[(int)instance.MeshId];

                // Affine 3x4 transform: three dot products for the origin, three for the
                // direction. Identical to the accelerated path, so results stay bit-exact.
                instance.TransformRayToLocal(
                    ray.Origin, ray.Direction, out float3 localOrigin, out float3 localDirection);

                for (int t = 0; t < range.TriangleCount; t++)
                {
                    int triangleIndex = range.TriangleStart + t;
                    PrometheusTriangle triangle = blasTriangles[triangleIndex];

                    if (!IntersectTriangle(
                            triangle, localOrigin, localDirection, ray.TMin, ray.TMax,
                            out float distance, out float2 barycentrics))
                    {
                        continue;
                    }

                    if (distance < hit.Distance)
                    {
                        hit.Distance = distance;
                        hit.Barycentrics = barycentrics;
                        hit.InstanceIndex = (uint)index;
                        hit.PrimitiveId = triangle.PrimitiveId;
                        hit.TriangleIndex = triangleIndex;
                        hit.IsHit = true;
                    }
                }
            }

            return hit;
        }

        /// <summary>Descends one instance's BLAS, narrowing <paramref name="hit"/> as it goes.</summary>
        private static void TraceInstance(
            uint instanceIndex,
            in PrometheusRay ray,
            in NativeArray<PrometheusInstance> instances,
            in NativeArray<PrometheusBVHNode> blasNodes,
            in NativeArray<PrometheusTriangle> blasTriangles,
            ref PrometheusRayHit hit)
        {
            PrometheusInstance instance = instances[(int)instanceIndex];

            // The object-space direction is left un-normalised, so t stays comparable with the
            // world-space interval and no rescale is needed on entry or exit.
            instance.TransformRayToLocal(
                ray.Origin, ray.Direction, out float3 localOrigin, out float3 localDirection);

            float3 inverseDirection = SafeReciprocal(localDirection);
            uint nodeOffset = instance.BlasNodeOffset;
            uint triangleOffset = instance.BlasTriangleOffset;
            uint cursor = 0;

            while (cursor != PrometheusBVHNode.InvalidIndex)
            {
                PrometheusBVHNode node = blasNodes[(int)(nodeOffset + cursor)];

                if (!IntersectAabb(node, localOrigin, inverseDirection, hit.Distance))
                {
                    cursor = node.EscapeIndex;
                    continue;
                }

                if (!node.IsLeaf)
                {
                    cursor++;
                    continue;
                }

                uint first = node.FirstPrimitive;
                uint count = node.PrimitiveCount;

                for (uint k = 0; k < count; k++)
                {
                    int triangleIndex = (int)(triangleOffset + first + k);
                    PrometheusTriangle triangle = blasTriangles[triangleIndex];

                    if (!IntersectTriangle(
                            triangle, localOrigin, localDirection, ray.TMin, hit.Distance,
                            out float distance, out float2 barycentrics))
                    {
                        continue;
                    }

                    if (distance < hit.Distance)
                    {
                        hit.Distance = distance;
                        hit.Barycentrics = barycentrics;
                        hit.InstanceIndex = instanceIndex;
                        hit.PrimitiveId = triangle.PrimitiveId;
                        hit.TriangleIndex = triangleIndex;
                        hit.IsHit = true;
                    }
                }

                cursor = node.EscapeIndex;
            }
        }

        /// <summary>Branch-free ray/AABB slab test; mirrors <c>PrometheusIntersectAabb</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IntersectAabb(in PrometheusBVHNode node, float3 origin, float3 inverseDirection, float tMax)
        {
            float3 t0 = (node.BoundsMin - origin) * inverseDirection;
            float3 t1 = (node.BoundsMax - origin) * inverseDirection;

            float3 small = math.min(t0, t1);
            float3 large = math.max(t0, t1);

            float enter = math.max(math.max(small.x, small.y), small.z);
            float exit = math.min(math.min(large.x, large.y), large.z);

            return exit >= math.max(enter, 0.0f) && enter <= tMax;
        }

        /// <summary>
        /// Möller–Trumbore intersection against the pre-computed edge form; mirrors
        /// <c>PrometheusIntersectTriangle</c> with culling disabled.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IntersectTriangle(
            in PrometheusTriangle triangle,
            float3 origin,
            float3 direction,
            float tMin,
            float tMax,
            out float distance,
            out float2 barycentrics)
        {
            distance = tMax;
            barycentrics = float2.zero;

            float3 pVector = math.cross(direction, triangle.Edge2);
            float determinant = math.dot(triangle.Edge1, pVector);

            if (math.abs(determinant) < IntersectEpsilon)
            {
                return false;
            }

            float inverseDeterminant = 1.0f / determinant;

            float3 tVector = origin - triangle.Vertex0;
            float u = math.dot(tVector, pVector) * inverseDeterminant;
            if (u < 0.0f || u > 1.0f)
            {
                return false;
            }

            float3 qVector = math.cross(tVector, triangle.Edge1);
            float v = math.dot(direction, qVector) * inverseDeterminant;
            if (v < 0.0f || (u + v) > 1.0f)
            {
                return false;
            }

            float t = math.dot(triangle.Edge2, qVector) * inverseDeterminant;
            if (t < tMin || t > tMax)
            {
                return false;
            }

            distance = t;
            barycentrics = new float2(u, v);
            return true;
        }

        /// <summary>
        /// Reciprocal direction with zero components replaced by a tiny finite value, so
        /// axis-aligned rays produce well-defined slab bounds instead of <c>0 * inf</c> NaN.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float3 SafeReciprocal(float3 direction)
        {
            const float tiny = 1e-20f;
            const float large = 1e20f;

            float3 safe = math.select(direction, new float3(tiny), math.abs(direction) < tiny);
            return math.clamp(1.0f / safe, new float3(-large), new float3(large));
        }
    }

    /// <summary>Traces a batch of rays through the packed TLAS and BLAS hierarchy.</summary>
    [BurstCompile(FloatPrecision.High, FloatMode.Strict, CompileSynchronously = true)]
    public struct CpuTraceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<PrometheusRay> Rays;
        [ReadOnly] public NativeArray<PrometheusBVHNode> TlasNodes;
        [ReadOnly] public NativeArray<uint> InstanceIndices;
        [ReadOnly] public NativeArray<PrometheusInstance> Instances;
        [ReadOnly] public NativeArray<PrometheusBVHNode> BlasNodes;
        [ReadOnly] public NativeArray<PrometheusTriangle> BlasTriangles;

        public int TlasNodeCount;

        [WriteOnly] public NativeArray<PrometheusRayHit> Hits;

        /// <inheritdoc/>
        public void Execute(int index)
        {
            Hits[index] = CpuBvhValidator.TraceClosest(
                Rays[index], TlasNodes, TlasNodeCount, InstanceIndices, Instances, BlasNodes, BlasTriangles);
        }
    }

    /// <summary>Traces a batch of rays against every triangle of every instance.</summary>
    /// <remarks>
    /// Compiled with the same float mode as <see cref="CpuTraceJob"/> so the two produce
    /// bit-identical arithmetic and the accuracy gate can demand exact equality.
    /// </remarks>
    [BurstCompile(FloatPrecision.High, FloatMode.Strict, CompileSynchronously = true)]
    public struct CpuBruteForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<PrometheusRay> Rays;
        [ReadOnly] public NativeArray<PrometheusInstance> Instances;
        [ReadOnly] public NativeArray<PrometheusBlasRange> BlasRanges;
        [ReadOnly] public NativeArray<PrometheusTriangle> BlasTriangles;

        public int InstanceCount;

        [WriteOnly] public NativeArray<PrometheusRayHit> Hits;

        /// <inheritdoc/>
        public void Execute(int index)
        {
            Hits[index] = CpuBvhValidator.TraceBruteForce(
                Rays[index], Instances, InstanceCount, BlasRanges, BlasTriangles);
        }
    }
}
