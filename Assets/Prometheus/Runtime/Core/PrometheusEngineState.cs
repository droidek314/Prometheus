using System.Runtime.CompilerServices;

namespace Prometheus.Core
{
    /// <summary>
    /// Lifecycle states of the Prometheus engine, mirroring the state machine in Section 3.3.
    /// </summary>
    public enum PrometheusState
    {
        /// <summary>No GPU resources exist yet. The only exit is <see cref="Ready"/>.</summary>
        Uninitialized = 0,

        /// <summary>Resources are live and the engine is waiting for a frame to begin.</summary>
        Ready = 1,

        /// <summary>Acceleration structures are being rebuilt on worker threads.</summary>
        FrameBuilding = 2,

        /// <summary>Ray traversal kernels are recorded or in flight.</summary>
        RayDispatching = 3,

        /// <summary>The denoiser is reconstructing the raw ray output.</summary>
        Filtering = 4,

        /// <summary>Results have been handed back to the render pipeline for compositing.</summary>
        RenderInjected = 5,

        /// <summary>
        /// A hardware or capacity failure took the engine off the ray-traced path. Only an
        /// explicit reinitialise returns it to <see cref="Ready"/>.
        /// </summary>
        ErrorFallback = 6
    }

    /// <summary>
    /// Validating state machine for one engine instance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Section 6.5 sketches this as static state. It is an instance type here instead, because URP
    /// renders several cameras per frame (game view, scene view, reflection probes) through the
    /// same renderer feature; a single global would be driven through conflicting transitions by
    /// cameras that are logically independent. One instance per feature keeps each camera's
    /// lifecycle its own.
    /// </para>
    /// <para>
    /// Every operation is allocation free and branch-only, so it is safe to drive from
    /// <c>ScriptableRenderPass</c> execution under the Section 3.2 zero-allocation rule. Rejected
    /// transitions are recorded on <see cref="LastRejectedFrom"/> and <see cref="LastRejectedTo"/>
    /// rather than logged, because formatting a message would allocate on the very path the rule
    /// protects; callers decide whether and when to surface them.
    /// </para>
    /// </remarks>
    public sealed class PrometheusEngineState
    {
        /// <summary>The engine's current state.</summary>
        public PrometheusState Current { get; private set; } = PrometheusState.Uninitialized;

        /// <summary>Source state of the most recently rejected transition.</summary>
        public PrometheusState LastRejectedFrom { get; private set; } = PrometheusState.Uninitialized;

        /// <summary>Target state of the most recently rejected transition.</summary>
        public PrometheusState LastRejectedTo { get; private set; } = PrometheusState.Uninitialized;

        /// <summary>Number of transitions rejected since construction. Non-zero means a caller drove an illegal sequence.</summary>
        public int RejectedTransitionCount { get; private set; }

        /// <summary>Whether the engine currently holds live resources and can accept a frame.</summary>
        public bool IsOperational
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Current != PrometheusState.Uninitialized && Current != PrometheusState.ErrorFallback;
        }

        /// <summary>
        /// Attempts a transition, recording it as rejected if Section 3.3 does not permit it.
        /// </summary>
        /// <param name="next">Desired state.</param>
        /// <returns><c>true</c> when the transition was legal and applied.</returns>
        public bool TryTransitionTo(PrometheusState next)
        {
            if (!IsValidTransition(Current, next))
            {
                LastRejectedFrom = Current;
                LastRejectedTo = next;
                RejectedTransitionCount++;
                return false;
            }

            Current = next;
            return true;
        }

        /// <summary>
        /// Drops the engine onto the fallback path. Legal from every state, because a hardware
        /// failure can surface at any point.
        /// </summary>
        public void EnterErrorFallback()
        {
            Current = PrometheusState.ErrorFallback;
        }

        /// <summary>Returns the engine to <see cref="PrometheusState.Uninitialized"/>, discarding history.</summary>
        public void Reset()
        {
            Current = PrometheusState.Uninitialized;
            LastRejectedFrom = PrometheusState.Uninitialized;
            LastRejectedTo = PrometheusState.Uninitialized;
            RejectedTransitionCount = 0;
        }

        /// <summary>
        /// The transition table from Section 3.3.
        /// </summary>
        /// <remarks>
        /// <c>ErrorFallback</c> is reachable from every operational state, which the diagram shows
        /// as edges out of each stage, and leaves only toward <c>Ready</c> on an explicit
        /// reinitialise.
        /// </remarks>
        /// <param name="from">Current state.</param>
        /// <param name="to">Desired state.</param>
        /// <returns><c>true</c> when the edge exists.</returns>
        public static bool IsValidTransition(PrometheusState from, PrometheusState to)
        {
            if (to == PrometheusState.ErrorFallback)
            {
                // A failure can be detected in any state that is actually running.
                return from != PrometheusState.Uninitialized;
            }

            switch (from)
            {
                case PrometheusState.Uninitialized:
                    return to == PrometheusState.Ready;
                case PrometheusState.Ready:
                    return to == PrometheusState.FrameBuilding;
                case PrometheusState.FrameBuilding:
                    return to == PrometheusState.RayDispatching;
                case PrometheusState.RayDispatching:
                    return to == PrometheusState.Filtering;
                case PrometheusState.Filtering:
                    return to == PrometheusState.RenderInjected;
                case PrometheusState.RenderInjected:
                    return to == PrometheusState.Ready;
                case PrometheusState.ErrorFallback:
                    return to == PrometheusState.Ready;
                default:
                    return false;
            }
        }
    }
}
