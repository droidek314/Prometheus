using System;
using System.Collections.Generic;
using Prometheus.Core.Structs;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Prometheus.Core.BVH
{
    /// <summary>
    /// A built, scene-lifetime Bottom-Level Acceleration Structure for one mesh.
    /// </summary>
    /// <remarks>
    /// The buffers are owned by the <see cref="PrometheusMeshCache"/> that produced them. Callers
    /// read from them but must never dispose them; releasing happens through
    /// <see cref="PrometheusMeshCache.Release"/> or <see cref="PrometheusMeshCache.Dispose"/>.
    /// </remarks>
    public struct PrometheusBlasEntry
    {
        /// <summary>Hierarchy nodes in depth-first pre-order, with the root at index zero.</summary>
        public NativeArray<PrometheusBVHNode> Nodes;

        /// <summary>Triangles in leaf order, addressed by <see cref="PrometheusBVHNode.FirstPrimitive"/>.</summary>
        public NativeArray<PrometheusTriangle> Triangles;

        /// <summary>Object-space AABB lower corner of the hierarchy.</summary>
        public float3 BoundsMin;

        /// <summary>Object-space AABB upper corner of the hierarchy.</summary>
        public float3 BoundsMax;

        /// <summary>Stable handle written into <see cref="PrometheusInstance.MeshId"/>.</summary>
        public uint MeshId;

        /// <summary>Triangles dropped by the <c>EC-003</c> filter during construction.</summary>
        public int RejectedTriangleCount;

        /// <summary>Depth of the deepest leaf, with the root counted as depth zero.</summary>
        public int MaxDepth;

        /// <summary>Whether this entry holds live buffers.</summary>
        public bool IsValid => Nodes.IsCreated && Triangles.IsCreated;
    }

    /// <summary>
    /// Builds and owns the load-time BLAS buffers for every static mesh in the scene.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Construction is deliberately a load-time operation: it reads mesh data, runs the
    /// Burst-compiled <see cref="BlasBuildJob"/> synchronously, and copies the result into
    /// <see cref="Allocator.Persistent"/> buffers sized exactly to the node and triangle counts.
    /// Section 3.2 permits managed allocation here because none of it runs inside
    /// <c>Update</c>, <c>beginCameraRendering</c> or <c>ScriptableRenderPass.Execute</c>. The
    /// per-frame path is <see cref="TryGetEntry"/>, which is allocation free.
    /// </para>
    /// <para>
    /// Entries are keyed by <c>Mesh.GetInstanceID</c>, so two renderers sharing a mesh share one
    /// BLAS. Per-renderer material assignment lives on <see cref="PrometheusInstance.MaterialId"/>;
    /// the per-triangle <see cref="PrometheusTriangle.MaterialId"/> carries the sub-mesh index so a
    /// multi-material renderer still needs only one BLAS.
    /// </para>
    /// </remarks>
    public sealed class PrometheusMeshCache : IDisposable
    {
        private readonly Dictionary<int, PrometheusBlasEntry> _entries;
        private readonly int _maxLeafTriangles;
        private uint _nextMeshId;
        private bool _disposed;

        /// <summary>Creates an empty cache.</summary>
        /// <param name="maxLeafTriangles">
        /// Triangles per leaf the builder aims for. Defaults to
        /// <see cref="BlasBuildJob.DefaultMaxLeafTriangles"/>, which suits GPU traversal.
        /// </param>
        public PrometheusMeshCache(int maxLeafTriangles = BlasBuildJob.DefaultMaxLeafTriangles)
        {
            _entries = new Dictionary<int, PrometheusBlasEntry>(64);
            _maxLeafTriangles = math.clamp(maxLeafTriangles, 1, (int)PrometheusBVHNode.MaxLeafPrimitiveCount);
            _nextMeshId = 0;
        }

        /// <summary>Number of meshes currently cached.</summary>
        public int Count => _entries.Count;

        /// <summary>Total nodes held across every cached BLAS.</summary>
        public int TotalNodeCount
        {
            get
            {
                int total = 0;
                foreach (KeyValuePair<int, PrometheusBlasEntry> pair in _entries)
                {
                    total += pair.Value.Nodes.Length;
                }

                return total;
            }
        }

        /// <summary>Total triangles held across every cached BLAS.</summary>
        public int TotalTriangleCount
        {
            get
            {
                int total = 0;
                foreach (KeyValuePair<int, PrometheusBlasEntry> pair in _entries)
                {
                    total += pair.Value.Triangles.Length;
                }

                return total;
            }
        }

        /// <summary>
        /// Looks up an already-built BLAS. Allocation free, so it is safe on the per-frame path.
        /// </summary>
        /// <param name="mesh">Mesh to look up.</param>
        /// <param name="entry">Receives the cached entry when one exists.</param>
        /// <returns><c>true</c> when the mesh has a cached BLAS.</returns>
        public bool TryGetEntry(Mesh mesh, out PrometheusBlasEntry entry)
        {
            if (mesh == null)
            {
                entry = default;
                return false;
            }

            return _entries.TryGetValue(mesh.GetInstanceID(), out entry);
        }

        /// <summary>
        /// Returns the cached BLAS for <paramref name="mesh"/>, building it on first request.
        /// </summary>
        /// <param name="mesh">Mesh to acquire a BLAS for.</param>
        /// <param name="entry">Receives the cached entry on success.</param>
        /// <returns><c>false</c> when the mesh is unusable or the build failed; the reason is logged.</returns>
        public bool TryGetOrBuild(Mesh mesh, out PrometheusBlasEntry entry)
        {
            ThrowIfDisposed();

            if (mesh == null)
            {
                Debug.LogError("[Prometheus] PrometheusMeshCache received a null mesh.");
                entry = default;
                return false;
            }

            int key = mesh.GetInstanceID();
            if (_entries.TryGetValue(key, out entry))
            {
                return true;
            }

            if (!TryBuild(mesh, out entry))
            {
                return false;
            }

            _entries.Add(key, entry);
            return true;
        }

        /// <summary>Disposes the BLAS buffers for one mesh and drops it from the cache.</summary>
        /// <param name="mesh">Mesh whose BLAS should be released.</param>
        /// <returns><c>true</c> when an entry was found and released.</returns>
        public bool Release(Mesh mesh)
        {
            if (mesh == null)
            {
                return false;
            }

            int key = mesh.GetInstanceID();
            if (!_entries.TryGetValue(key, out PrometheusBlasEntry entry))
            {
                return false;
            }

            DisposeEntry(ref entry);
            _entries.Remove(key);
            return true;
        }

        /// <summary>Disposes every cached BLAS. Safe to call more than once.</summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            foreach (KeyValuePair<int, PrometheusBlasEntry> pair in _entries)
            {
                PrometheusBlasEntry entry = pair.Value;
                DisposeEntry(ref entry);
            }

            _entries.Clear();
            _disposed = true;
        }

        /// <summary>
        /// Reads mesh geometry, runs the SAH builder, and copies the result into persistent buffers.
        /// </summary>
        /// <param name="mesh">Mesh to process.</param>
        /// <param name="entry">Receives the built entry on success.</param>
        /// <returns><c>true</c> when a traversable hierarchy was produced.</returns>
        private bool TryBuild(Mesh mesh, out PrometheusBlasEntry entry)
        {
            entry = default;

            if (!mesh.isReadable)
            {
                Debug.LogError(
                    $"[Prometheus] Mesh '{mesh.name}' is not readable, so its BLAS cannot be built. " +
                    "Enable Read/Write in the model import settings.");
                return false;
            }

            Mesh.MeshDataArray meshDataArray = Mesh.AcquireReadOnlyMeshData(mesh);

            NativeArray<Vector3> vertices = default;
            NativeArray<int> indices = default;
            NativeArray<PrometheusSubMeshRange> subMeshes = default;
            NativeArray<PrometheusTriangle> triangleScratch = default;
            NativeArray<PrometheusBVHNode> nodeScratch = default;
            NativeArray<BlasBuildResult> resultBuffer = default;

            try
            {
                Mesh.MeshData meshData = meshDataArray[0];

                int triangleSubMeshCount = 0;
                int totalIndexCount = 0;

                for (int s = 0; s < meshData.subMeshCount; s++)
                {
                    SubMeshDescriptor descriptor = meshData.GetSubMesh(s);
                    if (descriptor.topology != MeshTopology.Triangles)
                    {
                        continue;
                    }

                    triangleSubMeshCount++;
                    totalIndexCount += descriptor.indexCount;
                }

                if (totalIndexCount < 3)
                {
                    Debug.LogError($"[Prometheus] Mesh '{mesh.name}' has no triangle topology sub-meshes.");
                    return false;
                }

                vertices = new NativeArray<Vector3>(meshData.vertexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                meshData.GetVertices(vertices);

                indices = new NativeArray<int>(totalIndexCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                subMeshes = new NativeArray<PrometheusSubMeshRange>(triangleSubMeshCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

                int cursor = 0;
                int written = 0;

                for (int s = 0; s < meshData.subMeshCount; s++)
                {
                    SubMeshDescriptor descriptor = meshData.GetSubMesh(s);
                    if (descriptor.topology != MeshTopology.Triangles)
                    {
                        continue;
                    }

                    meshData.GetIndices(indices.GetSubArray(cursor, descriptor.indexCount), s);

                    PrometheusSubMeshRange range;
                    range.IndexStart = cursor;
                    range.IndexCount = descriptor.indexCount;
                    range.MaterialId = (uint)s;
                    subMeshes[written] = range;

                    cursor += descriptor.indexCount;
                    written++;
                }

                int sourceTriangleCount = totalIndexCount / 3;
                int maxNodeCount = math.max(1, (2 * sourceTriangleCount) - 1);

                triangleScratch = new NativeArray<PrometheusTriangle>(sourceTriangleCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                nodeScratch = new NativeArray<PrometheusBVHNode>(maxNodeCount, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                resultBuffer = new NativeArray<BlasBuildResult>(1, Allocator.TempJob);

                BlasBuildJob job = new BlasBuildJob
                {
                    // Vector3 and float3 are both three tightly packed floats, so this aliases
                    // the same memory rather than copying it.
                    Vertices = vertices.Reinterpret<float3>(),
                    Indices = indices,
                    SubMeshes = subMeshes,
                    MaxLeafTriangles = _maxLeafTriangles,
                    Triangles = triangleScratch,
                    Nodes = nodeScratch,
                    Result = resultBuffer
                };

                job.Run();

                BlasBuildResult result = resultBuffer[0];

                if (result.Status != PrometheusBlasBuildStatus.Success)
                {
                    Debug.LogError(
                        $"[Prometheus] BLAS construction for mesh '{mesh.name}' failed with " +
                        $"{result.Status} ({result.RejectedTriangleCount} of {sourceTriangleCount} triangles rejected).");
                    return false;
                }

                if (result.RejectedTriangleCount > 0)
                {
                    Debug.LogWarning(
                        $"[Prometheus] Mesh '{mesh.name}': {result.RejectedTriangleCount} of {sourceTriangleCount} " +
                        "triangles were degenerate or non-finite and were excluded from the BLAS (EC-003).");
                }

                entry.Nodes = new NativeArray<PrometheusBVHNode>(result.NodeCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                NativeArray<PrometheusBVHNode>.Copy(nodeScratch, 0, entry.Nodes, 0, result.NodeCount);

                entry.Triangles = new NativeArray<PrometheusTriangle>(result.TriangleCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
                NativeArray<PrometheusTriangle>.Copy(triangleScratch, 0, entry.Triangles, 0, result.TriangleCount);

                entry.BoundsMin = result.BoundsMin;
                entry.BoundsMax = result.BoundsMax;
                entry.RejectedTriangleCount = result.RejectedTriangleCount;
                entry.MaxDepth = result.MaxDepth;
                entry.MeshId = _nextMeshId;
                _nextMeshId++;

                return true;
            }
            finally
            {
                // Section 3.2: every native allocation on this path has a paired Dispose.
                meshDataArray.Dispose();

                if (resultBuffer.IsCreated)
                {
                    resultBuffer.Dispose();
                }

                if (nodeScratch.IsCreated)
                {
                    nodeScratch.Dispose();
                }

                if (triangleScratch.IsCreated)
                {
                    triangleScratch.Dispose();
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

        /// <summary>Releases the persistent buffers held by one entry.</summary>
        /// <param name="entry">Entry to release; its buffers are left in the default state.</param>
        private static void DisposeEntry(ref PrometheusBlasEntry entry)
        {
            if (entry.Nodes.IsCreated)
            {
                entry.Nodes.Dispose();
            }

            if (entry.Triangles.IsCreated)
            {
                entry.Triangles.Dispose();
            }

            entry.Nodes = default;
            entry.Triangles = default;
        }

        /// <summary>Guards against use after <see cref="Dispose"/>.</summary>
        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(PrometheusMeshCache));
            }
        }
    }
}
