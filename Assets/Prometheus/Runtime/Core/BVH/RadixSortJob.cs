using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace Prometheus.Core.BVH
{
    /// <summary>
    /// Least-significant-digit radix sort over the 30-bit Morton keys produced by
    /// <see cref="Morton3DJob"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>Shape of the sort</b></para>
    /// <para>
    /// Four passes of eight bits cover the full 32-bit word, which is two bits wider than the
    /// 30-bit key space. The surplus costs nothing: the top two bits of every real key are zero,
    /// so that pass is detected as uniform and skipped outright. Histograms for all four digits
    /// are accumulated in a single sweep, so the data is read once for counting and at most four
    /// times for scattering.
    /// </para>
    /// <para><b>Why this is one job rather than a parallel scatter</b></para>
    /// <para>
    /// A parallel radix sort splits each pass into per-block histogram, prefix scan and scatter
    /// stages. At the Section 3.1 ceiling of 4,096 instances the entire sort is a few tens of
    /// microseconds, while each additional job dispatch costs single-digit microseconds of
    /// scheduling latency, so a three-stage split per pass would spend more time synchronising
    /// than sorting. The job is still scheduled onto a worker thread, which is what the Phase 3
    /// performance gate measures. The pass structure below is deliberately the same one a blocked
    /// parallel implementation uses, so raising the instance ceiling later means partitioning the
    /// existing loops rather than rewriting them.
    /// </para>
    /// <para><b>Stability</b></para>
    /// <para>
    /// The scatter walks forward and assigns increasing offsets within each bucket, so equal keys
    /// retain their relative order. That matters because instances sharing a centroid must produce
    /// a deterministic hierarchy from frame to frame; an unstable sort would let the tree flicker
    /// and defeat the temporal denoiser.
    /// </para>
    /// </remarks>
    [BurstCompile(CompileSynchronously = true)]
    public struct RadixSortJob : IJob
    {
        /// <summary>Number of bits consumed per pass.</summary>
        /// <remarks>
        /// Ten bits divides the 30-bit Morton key exactly, so three passes cover the whole key
        /// space where eight-bit digits would need four. The wider 1,024-bucket histogram is still
        /// small enough to stay resident, and dropping a full scatter over every entry is worth
        /// more than the extra buckets cost.
        /// </remarks>
        public const int BitsPerPass = 10;

        /// <summary>Number of buckets per pass, <c>2^10</c>.</summary>
        public const int BucketCount = 1 << BitsPerPass;

        /// <summary>Mask selecting one digit.</summary>
        public const uint DigitMask = BucketCount - 1u;

        /// <summary>Passes needed to cover the 30-bit Morton key space.</summary>
        public const int PassCount = 3;

        /// <summary>Total histogram slots the caller must allocate.</summary>
        public const int HistogramLength = BucketCount * PassCount;

        /// <summary>Keys to sort in place. Sorted ascending by <see cref="MortonEntry.Code"/>.</summary>
        public NativeArray<MortonEntry> Entries;

        /// <summary>Ping-pong buffer, at least as long as <see cref="Entries"/>.</summary>
        public NativeArray<MortonEntry> Scratch;

        /// <summary>Scratch histograms, at least <see cref="HistogramLength"/> entries.</summary>
        public NativeArray<int> Histograms;

        /// <summary>Single-element build state; receives the recovered active instance count.</summary>
        public NativeArray<TlasBuildState> State;

        /// <summary>
        /// Number of entries to sort.
        /// </summary>
        /// <remarks>
        /// Padding slots past the active count carry <see cref="PrometheusMorton.InactiveCode"/>.
        /// Only the low 30 bits participate, so that sentinel is read as the maximum Morton value
        /// and can tie with a real key at the very corner of the scene; the sort's stability then
        /// keeps the padding behind the real entry, because padding always starts further along
        /// the buffer. Passing the full source count is therefore correct.
        /// </remarks>
        public int Count;

        /// <inheritdoc/>
        public void Execute()
        {
            Sort();
            PublishActiveCount();
        }

        /// <summary>
        /// Runs every digit pass back to back inside this one job invocation.
        /// </summary>
        /// <remarks>
        /// Splitting the passes across separate jobs would insert a scheduler wake-up between each
        /// digit shift, and at this element count that latency exceeds the pass itself. The whole
        /// sort therefore lives inside a single execution boundary.
        /// </remarks>
        private void Sort()
        {
            if (Count <= 1)
            {
                return;
            }

            for (int i = 0; i < HistogramLength; i++)
            {
                Histograms[i] = 0;
            }

            // Single counting sweep filling every pass histogram at once.
            for (int i = 0; i < Count; i++)
            {
                uint code = Entries[i].Code;
                Histograms[(int)(code & DigitMask)]++;
                Histograms[BucketCount + (int)((code >> BitsPerPass) & DigitMask)]++;
                Histograms[(2 * BucketCount) + (int)((code >> (2 * BitsPerPass)) & DigitMask)]++;
            }

            bool dataInScratch = false;

            for (int pass = 0; pass < PassCount; pass++)
            {
                int shift = pass * BitsPerPass;
                int histogramBase = pass * BucketCount;

                // If every key shares this digit the pass is a pure copy; skip it and leave the
                // data where it is. This retires the top pass outright for 30-bit keys.
                uint firstDigit = (Entries[0].Code >> shift) & DigitMask;
                if (Histograms[histogramBase + (int)firstDigit] == Count)
                {
                    continue;
                }

                // Exclusive prefix sum turns counts into bucket write cursors.
                int running = 0;
                for (int bucket = 0; bucket < BucketCount; bucket++)
                {
                    int count = Histograms[histogramBase + bucket];
                    Histograms[histogramBase + bucket] = running;
                    running += count;
                }

                if (dataInScratch)
                {
                    Scatter(Scratch, Entries, shift, histogramBase);
                }
                else
                {
                    Scatter(Entries, Scratch, shift, histogramBase);
                }

                dataInScratch = !dataInScratch;
            }

            if (dataInScratch)
            {
                for (int i = 0; i < Count; i++)
                {
                    Entries[i] = Scratch[i];
                }
            }
        }

        /// <summary>
        /// Recovers how many instances were active and records it in the shared build state.
        /// </summary>
        /// <remarks>
        /// Inactive slots carry <see cref="PrometheusMorton.InactiveCode"/>, which is reserved
        /// above every code a live instance can take, so after sorting they form a suffix and the
        /// boundary is a binary search. No counting pass, and no atomics.
        /// </remarks>
        private void PublishActiveCount()
        {
            int low = 0;
            int high = Count;

            while (low < high)
            {
                int mid = (low + high) >> 1;
                if (Entries[mid].Code == PrometheusMorton.InactiveCode)
                {
                    high = mid;
                }
                else
                {
                    low = mid + 1;
                }
            }

            TlasBuildState state = State[0];
            state.InstanceCount = low;
            state.NodeCount = 0;
            state.MaxDepth = 0;

            // EC-005: a scene with nothing visible is a valid early-out, not a failure.
            state.Status = low == 0
                ? PrometheusTlasBuildStatus.EmptyScene
                : PrometheusTlasBuildStatus.Success;

            State[0] = state;
        }

        /// <summary>
        /// Distributes one digit's worth of keys from <paramref name="source"/> into
        /// <paramref name="destination"/>, advancing each bucket's cursor as it goes.
        /// </summary>
        /// <param name="source">Buffer holding the current ordering.</param>
        /// <param name="destination">Buffer receiving the reordered keys.</param>
        /// <param name="shift">Bit offset of the digit being sorted on.</param>
        /// <param name="histogramBase">Offset of this pass's cursors within <see cref="Histograms"/>.</param>
        private void Scatter(
            NativeArray<MortonEntry> source,
            NativeArray<MortonEntry> destination,
            int shift,
            int histogramBase)
        {
            for (int i = 0; i < Count; i++)
            {
                MortonEntry entry = source[i];
                int bucket = histogramBase + (int)((entry.Code >> shift) & DigitMask);
                int cursor = Histograms[bucket];
                destination[cursor] = entry;
                Histograms[bucket] = cursor + 1;
            }
        }
    }
}
