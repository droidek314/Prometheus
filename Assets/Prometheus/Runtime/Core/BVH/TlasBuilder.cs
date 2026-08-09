using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Prometheus.Core.Structs;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// The TLAS passes are implementation detail, but Phase 3 gates their cost, so the test assembly
// needs to time them individually rather than only end to end.
[assembly: InternalsVisibleTo("Prometheus.Tests.EditMode")]

namespace Prometheus.Core.BVH
{
    /// <summary>
    /// Terminal state of a Top-Level Acceleration Structure build.
    /// </summary>
    public enum PrometheusTlasBuildStatus
    {
        /// <summary>A complete, traversable hierarchy was produced.</summary>
        Success = 0,

        /// <summary>No instance carried <see cref="PrometheusInstanceFlags.Active"/>; edge case <c>EC-005</c>.</summary>
        EmptyScene = 1,

        /// <summary>The active instance count exceeded the builder's capacity; edge case <c>EC-001</c>.</summary>
        CapacityExceeded = 2
    }

    /// <summary>
    /// Shared, job-writable state describing one TLAS build.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct TlasBuildState
    {
        /// <summary>Number of active instances, recovered by the sort pass.</summary>
        public int InstanceCount;

        /// <summary>Number of nodes written to the node buffer.</summary>
        public int NodeCount;

        /// <summary>Length of the hierarchy's rightmost spine, a lower bound on depth.</summary>
        public int MaxDepth;

        /// <summary>World-space AABB lower corner of the whole scene.</summary>
        public float3 BoundsMin;

        /// <summary>World-space AABB upper corner of the whole scene.</summary>
        public float3 BoundsMax;

        /// <summary>Terminal state of the build.</summary>
        public PrometheusTlasBuildStatus Status;
    }

    /// <summary>
    /// The child links and covered range of one Karras internal node.
    /// </summary>
    /// <remarks>
    /// A non-negative child value indexes another internal node; a negative value <c>r</c> indexes
    /// the leaf at <c>~r</c>. The range is carried alongside so the flatten pass can decide to
    /// collapse a whole subtree into one multi-instance leaf without re-deriving it.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    internal struct KarrasSplit
    {
        public int Left;
        public int Right;

        /// <summary>First sorted position covered by this node.</summary>
        public int First;

        /// <summary>Last sorted position covered by this node, inclusive.</summary>
        public int Last;
    }

    /// <summary>One work stack entry for the depth-first flatten.</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct TlasFlattenEntry
    {
        /// <summary>Karras reference to visit, or the internal node to close.</summary>
        public int Reference;

        /// <summary>Emitted node awaiting an escape link, or <c>-1</c> for a visit entry.</summary>
        public int CloseNodeIndex;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TlasFlattenEntry Visit(int reference)
        {
            TlasFlattenEntry entry;
            entry.Reference = reference;
            entry.CloseNodeIndex = -1;
            return entry;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TlasFlattenEntry Close(int reference, int nodeIndex)
        {
            TlasFlattenEntry entry;
            entry.Reference = reference;
            entry.CloseNodeIndex = nodeIndex;
            return entry;
        }
    }

    /// <summary>
    /// Karras (2012) parallel binary radix tree construction: one thread per internal node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Over an array sorted by Morton code, the hierarchy is fully determined by where adjacent
    /// codes first differ, and crucially every internal node's range can be derived independently
    /// of every other. Internal node <c>i</c> owns the range that starts or ends at <c>i</c>: the
    /// direction is decided by comparing the common-prefix length with either neighbour, the far
    /// end is found by an exponential probe followed by a binary search, and the split is the last
    /// position still sharing more than the range's own prefix.
    /// </para>
    /// <para><b>Duplicate Morton codes</b></para>
    /// <para>
    /// Karras requires strictly increasing keys; identical codes would make the prefix comparison
    /// ambiguous and can collapse the tree into a list. The standard remedy is applied: the key is
    /// augmented to 64 bits with its own array position in the low half, which is unique by
    /// construction, so <c>delta(i, j) = clz(key_i xor key_j)</c> stays well defined.
    /// </para>
    /// <para>
    /// Batched so the instance count is loaded once per batch rather than once per node.
    /// </para>
    /// </remarks>
    [BurstCompile(CompileSynchronously = true)]
    internal struct KarrasSplitJob : IJobParallelForBatch
    {
        [ReadOnly] public NativeArray<MortonEntry> Entries;
        [ReadOnly] public NativeArray<TlasBuildState> State;

        /// <summary>Clustering limit, mirrored from the flatten pass.</summary>
        public int MaxLeafInstances;

        [NativeDisableParallelForRestriction] public NativeArray<KarrasSplit> Splits;

        /// <inheritdoc/>
        public void Execute(int startIndex, int batchCount)
        {
            int count = State[0].InstanceCount;

            if (count < 2)
            {
                return;
            }

            int internalCount = count - 1;
            int end = math.min(startIndex + batchCount, internalCount);

            for (int index = startIndex; index < end; index++)
            {
                DetermineRange(index, count, out int first, out int last);

                KarrasSplit karrasSplit;
                karrasSplit.First = first;
                karrasSplit.Last = last;

                if ((last - first + 1) <= MaxLeafInstances)
                {
                    // This subtree is small enough that the flatten will collapse it into one leaf
                    // and never look at its children, so the split search would be wasted work.
                    karrasSplit.Left = -1;
                    karrasSplit.Right = -1;
                    Splits[index] = karrasSplit;
                    continue;
                }

                int split = FindSplit(first, last, count);
                karrasSplit.Left = split == first ? ~split : split;
                karrasSplit.Right = (split + 1) == last ? ~(split + 1) : (split + 1);
                Splits[index] = karrasSplit;
            }
        }

        /// <summary>
        /// Length of the common prefix of the augmented keys at two positions. Positions outside
        /// the array return <c>-1</c>, which terminates the range search naturally.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int Delta(int i, int j, int count)
        {
            if (j < 0 || j >= count)
            {
                return -1;
            }

            ulong left = ((ulong)Entries[i].Code << 32) | (uint)i;
            ulong right = ((ulong)Entries[j].Code << 32) | (uint)j;
            return math.lzcnt(left ^ right);
        }

        /// <summary>Finds the range of leaves covered by internal node <paramref name="index"/>.</summary>
        private void DetermineRange(int index, int count, out int first, out int last)
        {
            int deltaRight = Delta(index, index + 1, count);
            int deltaLeft = Delta(index, index - 1, count);
            int direction = deltaRight > deltaLeft ? 1 : -1;

            // Everything inside this node's range shares a longer prefix than its outward neighbour.
            int deltaMin = math.min(deltaLeft, deltaRight);

            // Exponential probe for an upper bound on the range length, then binary search inward.
            int lengthMax = 2;
            while (Delta(index, index + (lengthMax * direction), count) > deltaMin)
            {
                lengthMax <<= 1;
            }

            int length = 0;
            for (int step = lengthMax >> 1; step >= 1; step >>= 1)
            {
                if (Delta(index, index + ((length + step) * direction), count) > deltaMin)
                {
                    length += step;
                }
            }

            int other = index + (length * direction);
            first = math.min(index, other);
            last = math.max(index, other);
        }

        /// <summary>Finds the last position belonging to the left child of <c>[first, last]</c>.</summary>
        private int FindSplit(int first, int last, int count)
        {
            int commonPrefix = Delta(first, last, count);

            int split = first;
            int step = last - first;

            do
            {
                step = (step + 1) >> 1;
                int candidate = split + step;

                if (candidate < last && Delta(first, candidate, count) > commonPrefix)
                {
                    split = candidate;
                }
            }
            while (step > 1);

            return split;
        }
    }

    /// <summary>
    /// Serialises the Karras tree into the depth-first pre-order, skip-pointer form that
    /// <see cref="PrometheusBVHNode"/> requires, merging node bounds as it goes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the only part of hierarchy construction that must stay serial: the node format
    /// stores no child pointers, so the left child has to sit at <c>self + 1</c> and the escape
    /// rope has to point past the whole subtree. Both are properties of the emission order itself.
    /// </para>
    /// <para>
    /// Bounds are merged here rather than in a separate parallel refit. A bottom-up refit needs an
    /// atomic climb whose reads scatter across the whole tree; this walk visits every node once in
    /// depth-first order and keeps its pending sibling bounds on a stack no deeper than the tree.
    /// Measured at 4,096 instances the scattered version cost more than the serialisation it
    /// removed.
    /// </para>
    /// <para>
    /// The walk also accumulates the centroid AABB of every leaf, which becomes the normalisation
    /// frame for the next build. That is what allows the collect and Morton passes to be fused into
    /// a single parallel dispatch, since the reduction they would otherwise have to perform is
    /// already done here for free.
    /// </para>
    /// </remarks>
    [BurstCompile(FloatPrecision.Standard, FloatMode.Default, CompileSynchronously = true)]
    internal struct TlasFlattenJob : IJob
    {
        [ReadOnly] public NativeArray<MortonEntry> Entries;
        [ReadOnly] public NativeArray<PrometheusAabb> SourceBounds;
        [ReadOnly] public NativeArray<KarrasSplit> Splits;

        /// <summary>Maximum instances gathered into one leaf.</summary>
        public int MaxLeafInstances;

        /// <summary>Receives the sorted instance permutation that leaves address.</summary>
        [WriteOnly] public NativeArray<uint> InstanceIndices;

        public NativeArray<TlasFlattenEntry> Stack;
        public NativeArray<PrometheusAabb> BoundsStack;
        public NativeArray<PrometheusBVHNode> Nodes;
        public NativeArray<TlasBuildState> State;
        public NativeArray<TlasCentroidBounds> Normalization;

        /// <inheritdoc/>
        public void Execute()
        {
            TlasBuildState state = State[0];
            int count = state.InstanceCount;

            if (count == 0)
            {
                state.NodeCount = 0;
                State[0] = state;
                return;
            }

            float3 centroidMin = new float3(float.PositiveInfinity);
            float3 centroidMax = new float3(float.NegativeInfinity);

            int nodeCount = 0;
            int stackTop = 0;
            int boundsTop = 0;

            // With one instance there is no internal node, so the root is the single leaf.
            Stack[stackTop++] = TlasFlattenEntry.Visit(count < 2 ? ~0 : 0);

            while (stackTop > 0)
            {
                TlasFlattenEntry entry = Stack[--stackTop];

                if (entry.CloseNodeIndex >= 0)
                {
                    // The right subtree finished last, so its box is on top.
                    PrometheusAabb merged = BoundsStack[--boundsTop];
                    merged.Encapsulate(BoundsStack[--boundsTop]);

                    // The slot was reserved on the way down but never written, so this is a single
                    // clean store rather than a read-modify-write.
                    Nodes[entry.CloseNodeIndex] = PrometheusBVHNode.CreateInterior(
                        merged.Min, merged.Max, (uint)nodeCount);

                    BoundsStack[boundsTop++] = merged;
                    continue;
                }

                int reference = entry.Reference;

                int first;
                int last;

                if (reference < 0)
                {
                    first = ~reference;
                    last = first;
                }
                else
                {
                    KarrasSplit interior = Splits[reference];
                    first = interior.First;
                    last = interior.Last;

                    if ((last - first + 1) > MaxLeafInstances)
                    {
                        int interiorIndex = nodeCount++;

                        Stack[stackTop++] = TlasFlattenEntry.Close(reference, interiorIndex);
                        Stack[stackTop++] = TlasFlattenEntry.Visit(interior.Right);
                        Stack[stackTop++] = TlasFlattenEntry.Visit(interior.Left);
                        continue;
                    }

                    // Small enough to gather: the whole subtree collapses into one leaf, and its
                    // descendants are never emitted at all.
                }

                int nodeIndex = nodeCount++;
                PrometheusAabb bounds = PrometheusAabb.Empty;

                for (int position = first; position <= last; position++)
                {
                    int instanceIndex = Entries[position].InstanceIndex;

                    // Leaves address the indirection buffer, not the instance array, so the sorted
                    // permutation is materialised here rather than by moving instance records.
                    // The flatten visits leaves left to right, so these writes are sequential.
                    InstanceIndices[position] = (uint)instanceIndex;

                    PrometheusAabb instanceBounds = SourceBounds[instanceIndex];
                    bounds.Encapsulate(instanceBounds);

                    float3 centroid = instanceBounds.Centre;
                    centroidMin = math.min(centroidMin, centroid);
                    centroidMax = math.max(centroidMax, centroid);
                }

                Nodes[nodeIndex] = PrometheusBVHNode.CreateLeaf(
                    bounds.Min, bounds.Max, (uint)nodeCount, (uint)first, (uint)(last - first + 1));

                BoundsStack[boundsTop++] = bounds;
            }

            // Exactly the nodes on the rightmost spine escape past the end of the buffer. Walking
            // that spine touches a handful of nodes instead of rescanning all of them.
            int cursor = 0;
            int spine = 0;
            while (true)
            {
                PrometheusBVHNode node = Nodes[cursor];
                bool isLeaf = node.IsLeaf;

                node.EscapeIndex = PrometheusBVHNode.InvalidIndex;
                Nodes[cursor] = node;
                spine++;

                if (isLeaf)
                {
                    break;
                }

                // The left child's escape marks where its subtree ended, which is the right child.
                cursor = (int)Nodes[cursor + 1].EscapeIndex;
            }

            state.NodeCount = nodeCount;
            state.MaxDepth = spine - 1;
            state.BoundsMin = Nodes[0].BoundsMin;
            state.BoundsMax = Nodes[0].BoundsMax;
            State[0] = state;

            Normalization[0] = TlasCentroidBounds.FromCentroidAabb(centroidMin, centroidMax);
        }
    }

    /// <summary>
    /// Rebuilds the Top-Level Acceleration Structure over the active scene instances every frame.
    /// </summary>
    /// <remarks>
    /// <para>
    /// All working memory is allocated once with <see cref="Allocator.Persistent"/> when the
    /// builder is constructed, so <see cref="Schedule"/> performs no allocation of any kind and
    /// satisfies the zero-<c>GC.Alloc</c> mandate in Section 3.2. The pipeline is four dispatches:
    /// </para>
    /// <code>
    /// CollectAndMortonJob  active mask and 30-bit key per instance    (IJobParallelForBatch)
    /// RadixSortJob         all three digit passes, then active count  (IJob)
    /// KarrasSplitJob       parallel binary radix tree split finding   (IJobParallelForBatch)
    /// TlasFlattenJob       DFS emission, AABB merge, rope packing     (IJob)
    /// </code>
    /// <para>
    /// Instances are never copied. TLAS leaves index the array the caller submitted, so that array
    /// is what should be uploaded to the GPU alongside <see cref="Nodes"/>. Inactive entries simply
    /// go unreferenced by any leaf.
    /// </para>
    /// <para>
    /// The Morton normalisation frame is carried over from the previous build, which is what allows
    /// the collect and Morton work to share one dispatch. The very first build seeds it exactly;
    /// after a discontinuity such as edge case <c>EC-002</c>, call
    /// <see cref="InvalidateNormalization"/> to force a fresh seed.
    /// </para>
    /// </remarks>
    public sealed class TlasBuilder : IDisposable
    {
        /// <summary>
        /// Depth a well-formed hierarchy is expected to stay within.
        /// </summary>
        /// <remarks>
        /// Splitting on the highest differing Morton bit bounds the depth at 30 bit levels, and
        /// ranges of identical codes are halved by position, adding at most <c>log2(4096) = 12</c>
        /// more. This is an expected-quality bound used by tests, not a correctness guard: the
        /// work stack is sized for the absolute worst case.
        /// </remarks>
        public const int MaxTreeDepth = 64;

        /// <summary>
        /// Instances gathered into a single TLAS leaf.
        /// </summary>
        /// <remarks>
        /// <para>
        /// One instance per leaf gives the most precise culling but forces <c>2N-1</c> nodes, and
        /// the depth-first emission that the rope format requires is inherently serial, so node
        /// count sets the floor on build time. Collapsing every Karras subtree that covers four or
        /// fewer instances cuts the node count roughly threefold.
        /// </para>
        /// <para>
        /// The GPU pays for this by testing up to four instance boxes at a leaf instead of one,
        /// against far fewer interior nodes to descend. That is the usual trade for a TLAS, and it
        /// is what brings the build inside the Section 7.1 Phase 3 budget.
        /// </para>
        /// </remarks>
        public const int MaxLeafInstances = 4;

        /// <summary>Minimum iterations per parallel batch.</summary>
        private const int ParallelBatchSize = 256;

        private readonly int _capacity;

        private NativeArray<uint> _instanceIndices;
        private NativeArray<PrometheusBVHNode> _nodes;
        private NativeArray<MortonEntry> _entries;
        private NativeArray<MortonEntry> _entriesScratch;
        private NativeArray<int> _histograms;
        private NativeArray<TlasBuildState> _state;
        private NativeArray<TlasCentroidBounds> _normalization;
        private NativeArray<TlasFlattenEntry> _stack;
        private NativeArray<PrometheusAabb> _boundsStack;
        private NativeArray<KarrasSplit> _splits;

        private JobHandle _handle;
        private bool _scheduled;
        private bool _hasNormalization;
        private bool _disposed;

        /// <summary>Creates a builder sized for a fixed maximum number of submitted instances.</summary>
        /// <param name="capacity">
        /// Maximum instances per build. Clamped to
        /// <see cref="PrometheusInstance.MaxInstanceCount"/> from Section 3.1.
        /// </param>
        public TlasBuilder(int capacity = PrometheusInstance.MaxInstanceCount)
        {
            _capacity = math.clamp(capacity, 1, PrometheusInstance.MaxInstanceCount);
            int internalCapacity = math.max(1, _capacity - 1);

            _instanceIndices = Alloc<uint>(_capacity);
            _nodes = Alloc<PrometheusBVHNode>(math.max(1, (2 * _capacity) - 1));
            _entries = Alloc<MortonEntry>(_capacity);
            _entriesScratch = Alloc<MortonEntry>(_capacity);
            _histograms = new NativeArray<int>(RadixSortJob.HistogramLength, Allocator.Persistent);
            _state = new NativeArray<TlasBuildState>(1, Allocator.Persistent);
            _normalization = new NativeArray<TlasCentroidBounds>(1, Allocator.Persistent);

            // Every split leaves both sides non-empty, so the recursion cannot exceed one level per
            // instance. Sizing for that removes any need for a depth cap inside the hot loop.
            _stack = Alloc<TlasFlattenEntry>((2 * _capacity) + 8);

            // At most one completed subtree awaits its sibling per level of descent.
            _boundsStack = Alloc<PrometheusAabb>(_capacity + 8);
            _splits = Alloc<KarrasSplit>(internalCapacity);
        }

        private static NativeArray<T> Alloc<T>(int length) where T : struct
        {
            return new NativeArray<T>(length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }

        /// <summary>Maximum instances this builder accepts per build.</summary>
        public int Capacity => _capacity;

        /// <summary>TLAS nodes in depth-first pre-order. Valid once the build completes.</summary>
        public NativeArray<PrometheusBVHNode> Nodes => _nodes;

        /// <summary>
        /// Sorted instance permutation that TLAS leaves address, uploaded as
        /// <c>_PrometheusInstanceIndices</c>.
        /// </summary>
        /// <remarks>
        /// A leaf's <see cref="PrometheusBVHNode.FirstPrimitive"/> and
        /// <see cref="PrometheusBVHNode.PrimitiveCount"/> describe a contiguous run here, and each
        /// slot holds an index into the instance array the caller submitted. This one level of
        /// indirection is what lets a leaf hold several instances without any instance record ever
        /// being copied or reordered. Only the first <see cref="InstanceCount"/> entries are live.
        /// </remarks>
        public NativeArray<uint> InstanceIndices => _instanceIndices;

        /// <summary>Result of the most recent completed build.</summary>
        public TlasBuildState State => _state[0];

        /// <summary>Number of active instances in the most recent completed build.</summary>
        public int InstanceCount => _state[0].InstanceCount;

        /// <summary>Number of TLAS nodes in the most recent completed build.</summary>
        public int NodeCount => _state[0].NodeCount;

        /// <summary>
        /// Discards the carried-over Morton normalisation frame so the next build reseeds it from
        /// the submitted geometry.
        /// </summary>
        /// <remarks>
        /// Worth calling after a discontinuity that invalidates temporal assumptions, such as the
        /// camera teleport of edge case <c>EC-002</c> or a wholesale scene swap. Skipping it is
        /// safe, only lower quality for one build.
        /// </remarks>
        public void InvalidateNormalization()
        {
            _hasNormalization = false;
        }

        /// <summary>
        /// Schedules the full build pipeline onto worker threads.
        /// </summary>
        /// <param name="source">Instance records. Only active entries are referenced by the TLAS.</param>
        /// <param name="sourceBounds">World AABBs parallel to <paramref name="source"/>.</param>
        /// <param name="sourceCount">Number of valid entries at the front of the source arrays.</param>
        /// <param name="dependency">Optional dependency to chain behind.</param>
        /// <returns>A handle covering the whole pipeline.</returns>
        public JobHandle Schedule(
            NativeArray<PrometheusInstance> source,
            NativeArray<PrometheusAabb> sourceBounds,
            int sourceCount,
            JobHandle dependency = default)
        {
            ValidateRequest(source, sourceBounds, sourceCount);
            EnsureNormalization(source, sourceBounds, sourceCount);

            CreateJobs(source, sourceBounds, sourceCount,
                out CollectAndMortonJob collectAndMorton, out RadixSortJob sort,
                out KarrasSplitJob splitJob, out TlasFlattenJob flatten);

            int length = math.max(1, sourceCount);

            JobHandle handle = collectAndMorton.ScheduleBatch(length, ParallelBatchSize, dependency);
            handle = sort.Schedule(handle);
            handle = splitJob.ScheduleBatch(length, ParallelBatchSize, handle);
            handle = flatten.Schedule(handle);

            _handle = handle;
            _scheduled = true;
            return handle;
        }

        /// <summary>Blocks until the scheduled build finishes.</summary>
        public void Complete()
        {
            if (!_scheduled)
            {
                return;
            }

            _handle.Complete();
            _scheduled = false;
        }

        /// <summary>Schedules a build and blocks until it finishes.</summary>
        /// <param name="source">Instance records.</param>
        /// <param name="sourceBounds">World AABBs parallel to <paramref name="source"/>.</param>
        /// <param name="sourceCount">Number of valid entries at the front of the source arrays.</param>
        /// <returns>The completed build state.</returns>
        public TlasBuildState Build(
            NativeArray<PrometheusInstance> source,
            NativeArray<PrometheusAabb> sourceBounds,
            int sourceCount)
        {
            Schedule(source, sourceBounds, sourceCount);
            Complete();
            return _state[0];
        }

        /// <summary>
        /// Runs the whole pipeline inline on the calling thread, bypassing the job scheduler.
        /// </summary>
        /// <param name="source">Instance records.</param>
        /// <param name="sourceBounds">World AABBs parallel to <paramref name="source"/>.</param>
        /// <param name="sourceCount">Number of valid entries at the front of the source arrays.</param>
        /// <returns>The completed build state.</returns>
        public TlasBuildState BuildImmediate(
            NativeArray<PrometheusInstance> source,
            NativeArray<PrometheusAabb> sourceBounds,
            int sourceCount)
        {
            ValidateRequest(source, sourceBounds, sourceCount);
            EnsureNormalization(source, sourceBounds, sourceCount);

            CreateJobs(source, sourceBounds, sourceCount,
                out CollectAndMortonJob collectAndMorton, out RadixSortJob sort,
                out KarrasSplitJob splitJob, out TlasFlattenJob flatten);

            int length = math.max(1, sourceCount);

            collectAndMorton.RunBatch(length);
            sort.Run();
            splitJob.RunBatch(length);
            flatten.Run();

            return _state[0];
        }

        /// <summary>Releases every persistent buffer. Safe to call more than once.</summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            // A build still in flight holds references to these buffers.
            if (_scheduled)
            {
                _handle.Complete();
                _scheduled = false;
            }

            if (_instanceIndices.IsCreated) { _instanceIndices.Dispose(); _instanceIndices = default; }
            if (_nodes.IsCreated) { _nodes.Dispose(); _nodes = default; }
            if (_entries.IsCreated) { _entries.Dispose(); _entries = default; }
            if (_entriesScratch.IsCreated) { _entriesScratch.Dispose(); _entriesScratch = default; }
            if (_histograms.IsCreated) { _histograms.Dispose(); _histograms = default; }
            if (_state.IsCreated) { _state.Dispose(); _state = default; }
            if (_normalization.IsCreated) { _normalization.Dispose(); _normalization = default; }
            if (_stack.IsCreated) { _stack.Dispose(); _stack = default; }
            if (_boundsStack.IsCreated) { _boundsStack.Dispose(); _boundsStack = default; }
            if (_splits.IsCreated) { _splits.Dispose(); _splits = default; }

            _disposed = true;
        }

        /// <summary>
        /// Seeds the Morton normalisation frame on the first build, or after
        /// <see cref="InvalidateNormalization"/>.
        /// </summary>
        /// <remarks>
        /// Runs on the calling thread because it happens once, not per frame. Every later build
        /// inherits the frame the previous flatten pass produced at no cost.
        /// </remarks>
        private void EnsureNormalization(
            NativeArray<PrometheusInstance> source,
            NativeArray<PrometheusAabb> sourceBounds,
            int sourceCount)
        {
            if (_hasNormalization)
            {
                return;
            }

            float3 centroidMin = new float3(float.PositiveInfinity);
            float3 centroidMax = new float3(float.NegativeInfinity);
            bool any = false;

            for (int i = 0; i < sourceCount; i++)
            {
                if (!source[i].IsActive)
                {
                    continue;
                }

                float3 centroid = sourceBounds[i].Centre;
                centroidMin = math.min(centroidMin, centroid);
                centroidMax = math.max(centroidMax, centroid);
                any = true;
            }

            if (!any)
            {
                centroidMin = float3.zero;
                centroidMax = float3.zero;
            }

            _normalization[0] = TlasCentroidBounds.FromCentroidAabb(centroidMin, centroidMax);
            _hasNormalization = true;
        }

        private void ValidateRequest(
            NativeArray<PrometheusInstance> source,
            NativeArray<PrometheusAabb> sourceBounds,
            int sourceCount)
        {
            ThrowIfDisposed();

            if (sourceCount < 0 || sourceCount > source.Length || sourceCount > sourceBounds.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(sourceCount));
            }

            if (sourceCount > _capacity)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(sourceCount),
                    $"TlasBuilder capacity is {_capacity} instances but {sourceCount} were submitted.");
            }
        }

        /// <summary>Populates the four pass descriptors against this builder's persistent buffers.</summary>
        private void CreateJobs(
            NativeArray<PrometheusInstance> source,
            NativeArray<PrometheusAabb> sourceBounds,
            int sourceCount,
            out CollectAndMortonJob collectAndMorton,
            out RadixSortJob sort,
            out KarrasSplitJob splitJob,
            out TlasFlattenJob flatten)
        {
            collectAndMorton = new CollectAndMortonJob
            {
                Source = source,
                SourceBounds = sourceBounds,
                Normalization = _normalization,
                Entries = _entries
            };

            sort = new RadixSortJob
            {
                Entries = _entries,
                Scratch = _entriesScratch,
                Histograms = _histograms,
                State = _state,
                Count = sourceCount
            };

            splitJob = new KarrasSplitJob
            {
                Entries = _entries,
                State = _state,
                MaxLeafInstances = MaxLeafInstances,
                Splits = _splits
            };

            flatten = new TlasFlattenJob
            {
                Entries = _entries,
                SourceBounds = sourceBounds,
                Splits = _splits,
                MaxLeafInstances = MaxLeafInstances,
                InstanceIndices = _instanceIndices,
                Stack = _stack,
                BoundsStack = _boundsStack,
                Nodes = _nodes,
                State = _state,
                Normalization = _normalization
            };
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(TlasBuilder));
            }
        }
    }
}
