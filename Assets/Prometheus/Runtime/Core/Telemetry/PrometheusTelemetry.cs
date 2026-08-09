using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Prometheus.Core.Structs;
using Unity.Collections;
using UnityEngine;

namespace Prometheus.Core.Telemetry
{
    /// <summary>
    /// Error codes reported by the engine, as defined in Section 6.5.
    /// </summary>
    public enum PrometheusErrorCode : uint
    {
        /// <summary>No error.</summary>
        Success = 0x00000000,

        /// <summary>Scene geometry exceeded the Section 3.1 ceiling; edge case <c>EC-001</c>.</summary>
        ErrBvhCapacityExceeded = 0xE0000001,

        /// <summary>A GPU buffer allocation failed; edge case <c>EC-004</c>.</summary>
        ErrGpuBufferAllocFailed = 0xE0000002,

        /// <summary>Degenerate or non-finite geometry was rejected; edge case <c>EC-003</c>.</summary>
        ErrInvalidGeometryData = 0xE0000003,

        /// <summary>A compute kernel failed to compile or resolve.</summary>
        ErrComputeShaderCompile = 0xE0000004
    }

    /// <summary>What the fallback controller decided to do with a frame's geometry.</summary>
    public enum PrometheusGeometryDecision
    {
        /// <summary>Geometry is within bounds; build and dispatch normally.</summary>
        Proceed = 0,

        /// <summary>
        /// Nothing visible. Edge case <c>EC-005</c>: skip BVH construction and every dispatch,
        /// and leave the command buffer clean. This is not an error.
        /// </summary>
        BypassEmptyScene = 1,

        /// <summary>
        /// Geometry exceeded the Section 3.1 triangle ceiling. Edge case <c>EC-001</c>: report
        /// <see cref="PrometheusErrorCode.ErrBvhCapacityExceeded"/> and hand the frame back to
        /// URP's standard shadow maps.
        /// </summary>
        FallbackToShadowMaps = 2
    }

    /// <summary>Metrics captured for one frame. Blittable, so it lives in native memory.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusFrameMetrics
    {
        /// <summary>Wall time spent building acceleration structures.</summary>
        public double BuildMilliseconds;

        /// <summary>Wall time spent recording and dispatching traversal.</summary>
        public double TraversalMilliseconds;

        /// <summary>Wall time spent in the denoiser.</summary>
        public double DenoiseMilliseconds;

        /// <summary>Active instances submitted to the TLAS.</summary>
        public int InstanceCount;

        /// <summary>Triangles across every BLAS referenced this frame.</summary>
        public int TriangleCount;

        /// <summary>Nodes across both hierarchy levels.</summary>
        public int NodeCount;

        /// <summary>Triangles rejected by the <c>EC-003</c> filter.</summary>
        public int RejectedTriangleCount;

        /// <summary>Bytes reserved across every persistent GPU buffer.</summary>
        public long VramBytes;

        /// <summary>Monotonic frame counter.</summary>
        public int FrameIndex;

        /// <summary>Bit set of error codes raised during the frame.</summary>
        public uint ErrorMask;

        /// <summary>Total CPU time across all three stages.</summary>
        public double TotalMilliseconds => BuildMilliseconds + TraversalMilliseconds + DenoiseMilliseconds;
    }

    /// <summary>
    /// Rolling, allocation-free record of engine performance and error state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Section 7.1 Phase 7 asks for telemetry, and Section 3.2 forbids the frame path from
    /// allocating, so the two requirements meet here: every recorded value is a primitive, the
    /// history is a fixed <see cref="NativeArray{T}"/> ring, and nothing formats a string.
    /// </para>
    /// <para>
    /// Errors are accumulated as a bit mask rather than a list for the same reason. Section 6.5
    /// sketches the handler logging an interpolated message, which its own comment says must not
    /// create managed garbage; the two cannot both hold on a per-frame path. The resolution here
    /// is that recording is always allocation free, and human-readable logging happens at most
    /// once per distinct error code per session, through
    /// <see cref="PrometheusErrorHandler.ReportOnce"/>.
    /// </para>
    /// </remarks>
    public sealed class PrometheusTelemetry : IDisposable
    {
        /// <summary>Frames retained for rolling averages.</summary>
        public const int HistoryLength = 120;

        private NativeArray<PrometheusFrameMetrics> _history;
        private PrometheusFrameMetrics _current;
        private int _writeCursor;
        private int _recordedFrames;
        private bool _disposed;

        /// <summary>Creates a telemetry sink with a fixed-size history.</summary>
        public PrometheusTelemetry()
        {
            _history = new NativeArray<PrometheusFrameMetrics>(HistoryLength, Allocator.Persistent);
        }

        /// <summary>Metrics for the most recently completed frame.</summary>
        public PrometheusFrameMetrics Latest { get; private set; }

        /// <summary>Number of frames recorded since construction.</summary>
        public int RecordedFrameCount => _recordedFrames;

        /// <summary>Bit set of every error code raised since the last <see cref="ClearErrors"/>.</summary>
        public uint ActiveErrorMask { get; private set; }

        /// <summary>The most recent error code raised, or <see cref="PrometheusErrorCode.Success"/>.</summary>
        public PrometheusErrorCode LastError { get; private set; } = PrometheusErrorCode.Success;

        /// <summary>Total errors raised since construction.</summary>
        public int ErrorCount { get; private set; }

        /// <summary>Whether any error is currently latched.</summary>
        public bool HasActiveError => ActiveErrorMask != 0u;

        /// <summary>Starts a new frame, resetting the per-frame accumulators.</summary>
        /// <param name="frameIndex">Monotonic frame counter.</param>
        public void BeginFrame(int frameIndex)
        {
            ThrowIfDisposed();

            _current = default;
            _current.FrameIndex = frameIndex;
        }

        /// <summary>Records acceleration structure build results.</summary>
        /// <param name="milliseconds">Wall time of the build.</param>
        /// <param name="instanceCount">Active instances.</param>
        /// <param name="nodeCount">Nodes emitted.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordBuild(double milliseconds, int instanceCount, int nodeCount)
        {
            _current.BuildMilliseconds = milliseconds;
            _current.InstanceCount = instanceCount;
            _current.NodeCount = nodeCount;
        }

        /// <summary>Records geometry counts, including the <c>EC-003</c> rejection tally.</summary>
        /// <param name="triangleCount">Triangles referenced.</param>
        /// <param name="rejectedTriangleCount">Triangles dropped as degenerate or non-finite.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordGeometry(int triangleCount, int rejectedTriangleCount)
        {
            _current.TriangleCount = triangleCount;
            _current.RejectedTriangleCount = rejectedTriangleCount;

            if (rejectedTriangleCount > 0)
            {
                // EC-003 is a filtering outcome, not a failure: the build continues without the
                // offending triangles, but the count is surfaced so it cannot pass unnoticed.
                RecordError(PrometheusErrorCode.ErrInvalidGeometryData);
            }
        }

        /// <summary>Records traversal recording and dispatch time.</summary>
        /// <param name="milliseconds">Wall time.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordTraversal(double milliseconds)
        {
            _current.TraversalMilliseconds = milliseconds;
        }

        /// <summary>Records denoiser time.</summary>
        /// <param name="milliseconds">Wall time.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordDenoise(double milliseconds)
        {
            _current.DenoiseMilliseconds = milliseconds;
        }

        /// <summary>Records the VRAM currently reserved by persistent buffers.</summary>
        /// <param name="bytes">Reserved bytes.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordVram(long bytes)
        {
            _current.VramBytes = bytes;
        }

        /// <summary>
        /// Latches an error code. Allocation free, so it is safe from the frame path.
        /// </summary>
        /// <param name="code">Code to latch. <see cref="PrometheusErrorCode.Success"/> is ignored.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void RecordError(PrometheusErrorCode code)
        {
            if (code == PrometheusErrorCode.Success)
            {
                return;
            }

            uint bit = ErrorBit(code);
            ActiveErrorMask |= bit;
            _current.ErrorMask |= bit;
            LastError = code;
            ErrorCount++;
        }

        /// <summary>Whether <paramref name="code"/> is currently latched.</summary>
        /// <param name="code">Code to test.</param>
        /// <returns><c>true</c> when the code has been raised since the last clear.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool HasError(PrometheusErrorCode code)
        {
            return (ActiveErrorMask & ErrorBit(code)) != 0u;
        }

        /// <summary>Clears every latched error. Called by an explicit reinitialise.</summary>
        public void ClearErrors()
        {
            ActiveErrorMask = 0u;
            LastError = PrometheusErrorCode.Success;
        }

        /// <summary>Closes the frame and folds it into the rolling history.</summary>
        public void EndFrame()
        {
            ThrowIfDisposed();

            Latest = _current;
            _history[_writeCursor] = _current;
            _writeCursor = (_writeCursor + 1) % HistoryLength;
            _recordedFrames++;
        }

        /// <summary>
        /// Mean total CPU time across the retained history.
        /// </summary>
        /// <remarks>
        /// Walks native memory and returns a primitive, so it is allocation free and can be polled
        /// from a HUD without perturbing what it measures.
        /// </remarks>
        public double AverageTotalMilliseconds
        {
            get
            {
                int count = Mathf.Min(_recordedFrames, HistoryLength);
                if (count == 0)
                {
                    return 0.0;
                }

                double sum = 0.0;
                for (int i = 0; i < count; i++)
                {
                    sum += _history[i].TotalMilliseconds;
                }

                return sum / count;
            }
        }

        /// <summary>Worst total CPU time across the retained history.</summary>
        public double PeakTotalMilliseconds
        {
            get
            {
                int count = Mathf.Min(_recordedFrames, HistoryLength);
                double peak = 0.0;

                for (int i = 0; i < count; i++)
                {
                    double total = _history[i].TotalMilliseconds;
                    if (total > peak)
                    {
                        peak = total;
                    }
                }

                return peak;
            }
        }

        /// <summary>Whether the rolling average sits inside the Section 1.1 frame budget.</summary>
        public bool IsWithinFrameBudget => AverageTotalMilliseconds <= 1.50;

        /// <summary>Releases the native history buffer. Safe to call more than once.</summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            if (_history.IsCreated)
            {
                _history.Dispose();
            }

            _disposed = true;
        }

        /// <summary>Maps an error code onto its bit in the mask.</summary>
        /// <remarks>The Section 6.5 codes differ only in their low nibble, which indexes the bit.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ErrorBit(PrometheusErrorCode code)
        {
            return code == PrometheusErrorCode.Success ? 0u : 1u << (int)((uint)code & 0xFu);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(PrometheusTelemetry));
            }
        }
    }

    /// <summary>
    /// Turns latched error codes into human-readable diagnostics, at most once each.
    /// </summary>
    /// <remarks>
    /// Section 6.5 shows this formatting a message on every call. Formatting allocates, and the
    /// same section's own comment requires the production path not to, so reporting is throttled
    /// to one message per distinct code per session. Repeats are still counted in
    /// <see cref="PrometheusTelemetry.ErrorCount"/>, so nothing is lost, and BAN-007 is honoured
    /// because no failure is silently discarded.
    /// </remarks>
    public sealed class PrometheusErrorHandler
    {
        private uint _reportedMask;

        /// <summary>Codes already reported this session.</summary>
        public uint ReportedMask => _reportedMask;

        /// <summary>
        /// Latches an error, drives the state machine onto the fallback path where the code
        /// warrants it, and logs the code the first time it is seen.
        /// </summary>
        /// <param name="code">Error raised.</param>
        /// <param name="telemetry">Sink that latches the code.</param>
        /// <param name="state">Engine state to transition.</param>
        /// <returns><c>true</c> when the engine was moved to <see cref="PrometheusState.ErrorFallback"/>.</returns>
        public bool HandleError(
            PrometheusErrorCode code,
            PrometheusTelemetry telemetry,
            PrometheusEngineState state)
        {
            telemetry?.RecordError(code);
            ReportOnce(code);

            switch (code)
            {
                case PrometheusErrorCode.ErrBvhCapacityExceeded:
                case PrometheusErrorCode.ErrGpuBufferAllocFailed:
                case PrometheusErrorCode.ErrComputeShaderCompile:
                    // Section 6.5: these are unrecoverable within the frame, so the engine drops
                    // onto the fallback path until an explicit reinitialise.
                    state?.EnterErrorFallback();
                    return true;

                case PrometheusErrorCode.ErrInvalidGeometryData:
                    // Section 6.5: skip the offending elements and keep the pipeline running.
                    return false;

                default:
                    return false;
            }
        }

        /// <summary>Logs a code the first time it is seen, and never again.</summary>
        /// <param name="code">Code to report.</param>
        /// <returns><c>true</c> when a message was emitted.</returns>
        public bool ReportOnce(PrometheusErrorCode code)
        {
            if (code == PrometheusErrorCode.Success)
            {
                return false;
            }

            uint bit = PrometheusTelemetry.ErrorBit(code);
            if ((_reportedMask & bit) != 0u)
            {
                return false;
            }

            _reportedMask |= bit;
            Debug.LogError($"[Prometheus Engine Error 0x{(uint)code:X8}]: {DescribeError(code)}");
            return true;
        }

        /// <summary>Allows every code to be reported again, after an explicit reinitialise.</summary>
        public void Reset()
        {
            _reportedMask = 0u;
        }

        /// <summary>Static description of a code. Constant strings, so no formatting cost.</summary>
        /// <param name="code">Code to describe.</param>
        /// <returns>A human-readable explanation.</returns>
        public static string DescribeError(PrometheusErrorCode code)
        {
            switch (code)
            {
                case PrometheusErrorCode.ErrBvhCapacityExceeded:
                    return "ERROR_BVH_CAPACITY_EXCEEDED: scene triangle count exceeds the 2^21 ceiling. " +
                           "Falling back to standard shadow maps (EC-001).";
                case PrometheusErrorCode.ErrGpuBufferAllocFailed:
                    return "GPU buffer allocation failed. Downgrading ray tracing resolution and " +
                           "flushing temporal history (EC-004).";
                case PrometheusErrorCode.ErrInvalidGeometryData:
                    return "Degenerate or non-finite geometry was rejected during BVH construction (EC-003).";
                case PrometheusErrorCode.ErrComputeShaderCompile:
                    return "A required compute kernel could not be resolved.";
                default:
                    return "Unknown error.";
            }
        }
    }

    /// <summary>
    /// Applies the Section 2.3 boundary rules to a frame before any work is scheduled.
    /// </summary>
    /// <remarks>
    /// Every decision here is a pure function of counts, which keeps it allocation free and
    /// directly testable without a render loop.
    /// </remarks>
    public sealed class PrometheusFallbackController
    {
        /// <summary>Triangle ceiling from Section 3.1, <c>2^21</c>.</summary>
        public const int MaxSceneTriangles = PrometheusTriangle.MaxSceneTriangles;

        /// <summary>Node ceiling from Section 3.1, <c>2^22 - 1</c>.</summary>
        public const int MaxSceneNodes = (int)PrometheusBVHNode.MaxPrimitiveIndex;

        /// <summary>Resolution scale applied after a VRAM exhaustion event (EC-004).</summary>
        public const float VramFallbackResolutionScale = 0.5f;

        private readonly PrometheusErrorHandler _errorHandler = new PrometheusErrorHandler();

        /// <summary>The error handler this controller reports through.</summary>
        public PrometheusErrorHandler ErrorHandler => _errorHandler;

        /// <summary>Times the engine has fallen back to standard shadow maps (EC-001).</summary>
        public int CapacityFallbackCount { get; private set; }

        /// <summary>Times an empty scene was bypassed (EC-005).</summary>
        public int EmptySceneBypassCount { get; private set; }

        /// <summary>Times a VRAM exhaustion downgrade was applied (EC-004).</summary>
        public int VramDowngradeCount { get; private set; }

        /// <summary>
        /// Decides what to do with a frame's geometry before anything is built.
        /// </summary>
        /// <remarks>
        /// The empty-scene test comes first on purpose: a scene with nothing in it is not an
        /// error, and running it through the capacity check would be wasted work on the most
        /// common trivial case.
        /// </remarks>
        /// <param name="instanceCount">Active visible instances.</param>
        /// <param name="triangleCount">Triangles those instances reference.</param>
        /// <param name="telemetry">Sink for counters and error codes.</param>
        /// <param name="state">Engine state to transition on failure.</param>
        /// <returns>The decision the caller must honour.</returns>
        public PrometheusGeometryDecision EvaluateGeometry(
            int instanceCount,
            int triangleCount,
            PrometheusTelemetry telemetry,
            PrometheusEngineState state)
        {
            // EC-005: nothing visible. Bypass construction and every dispatch, and return with a
            // clean command buffer. Not an error, so no code is raised.
            if (instanceCount <= 0 || triangleCount <= 0)
            {
                EmptySceneBypassCount++;
                return PrometheusGeometryDecision.BypassEmptyScene;
            }

            // EC-001: past the Section 3.1 ceiling the packed 22-bit primitive index cannot address
            // the geometry at all, so clamping would silently drop it. Handing the frame back to
            // URP's shadow maps is the only correct answer.
            if (triangleCount > MaxSceneTriangles)
            {
                CapacityFallbackCount++;
                _errorHandler.HandleError(PrometheusErrorCode.ErrBvhCapacityExceeded, telemetry, state);
                return PrometheusGeometryDecision.FallbackToShadowMaps;
            }

            return PrometheusGeometryDecision.Proceed;
        }

        /// <summary>
        /// Applies the <c>EC-004</c> response to a failed GPU allocation.
        /// </summary>
        /// <remarks>
        /// Section 2.3 prescribes deallocating the denoiser history and halving the ray tracing
        /// resolution. Both happen here; the caller performs the flush because it owns the
        /// targets, and receives the reduced scale to apply.
        /// </remarks>
        /// <param name="currentScale">Resolution scale in force when the allocation failed.</param>
        /// <param name="telemetry">Sink for the error code.</param>
        /// <param name="state">Engine state to transition.</param>
        /// <returns>The scale the caller should adopt.</returns>
        public float HandleVramExhaustion(
            float currentScale,
            PrometheusTelemetry telemetry,
            PrometheusEngineState state)
        {
            VramDowngradeCount++;
            _errorHandler.HandleError(PrometheusErrorCode.ErrGpuBufferAllocFailed, telemetry, state);

            return Mathf.Min(currentScale, VramFallbackResolutionScale);
        }

        /// <summary>Clears the counters and lets every error report again.</summary>
        public void Reset()
        {
            CapacityFallbackCount = 0;
            EmptySceneBypassCount = 0;
            VramDowngradeCount = 0;
            _errorHandler.Reset();
        }
    }
}
