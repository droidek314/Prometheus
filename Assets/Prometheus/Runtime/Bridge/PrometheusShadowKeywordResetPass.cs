using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Prometheus.Bridge
{
    /// <summary>
    /// Restores URP's main-light shadow keywords before transparent geometry is drawn.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="PrometheusRenderPass"/> switches the main light onto the screen-space shadow
    /// path so opaque surfaces sample the traced buffer. Transparent surfaces cannot use that
    /// buffer: it holds one value per screen pixel, describing whichever opaque surface was
    /// nearest, so a transparent fragment in front of one would read its occlusion instead of its
    /// own. URP's shader library encodes exactly that restriction, guarding every screen-space
    /// branch with <c>!defined(_SURFACE_TYPE_TRANSPARENT)</c>.
    /// </para>
    /// <para>
    /// Those guards send transparent fragments down the cascaded shadow map path, which needs the
    /// cascade keywords that the traversal pass cleared. Without this pass they stay cleared for
    /// the rest of the frame and every transparent surface renders unshadowed. URP's own
    /// screen-space shadow feature carries an identical post pass for the identical reason.
    /// </para>
    /// <para>
    /// Keywords are global render state rather than per-pass state, so they also persist into the
    /// next camera and the next frame. Restoring them is what keeps a second camera that does not
    /// run Prometheus from sampling a shadow buffer nothing wrote.
    /// </para>
    /// </remarks>
    public sealed class PrometheusShadowKeywordResetPass : ScriptableRenderPass
    {
        private static readonly GlobalKeyword s_MainLightShadowScreenKeyword =
            GlobalKeyword.Create("_MAIN_LIGHT_SHADOWS_SCREEN");

        private static readonly GlobalKeyword s_MainLightShadowsKeyword =
            GlobalKeyword.Create("_MAIN_LIGHT_SHADOWS");

        private static readonly GlobalKeyword s_MainLightShadowCascadesKeyword =
            GlobalKeyword.Create("_MAIN_LIGHT_SHADOWS_CASCADE");

        private sealed class PassData
        {
            public bool MainLightShadows;
            public bool MainLightShadowCascades;
        }

        /// <summary>Cached so recording the pass allocates nothing (Section 3.2).</summary>
        private static readonly BaseRenderFunc<PassData, UnsafeGraphContext> s_ExecuteFunc = Execute;

        /// <summary>
        /// Pass name, held as a constant because the inherited <c>passName</c> is derived from
        /// <c>profilingSampler</c>, which URP sets to null in non-development builds.
        /// </summary>
        private const string PassName = "Prometheus Restore Shadow Keywords";

        /// <summary>Creates the pass at the point URP itself restores these keywords.</summary>
        public PrometheusShadowKeywordResetPass()
        {
            renderPassEvent = RenderPassEvent.BeforeRenderingTransparents;
            profilingSampler = new ProfilingSampler(PassName);
        }

        /// <inheritdoc/>
        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            UniversalShadowData shadowData = frameData.Get<UniversalShadowData>();

            using (IUnsafeRenderGraphBuilder builder =
                   renderGraph.AddUnsafePass<PassData>(PassName, out PassData passData, profilingSampler))
            {
                // A single cascade uses the plain keyword and several use the cascade variant;
                // they are mutually exclusive, and both stay off when the asset has main light
                // shadows disabled, which is the state URP would have left them in.
                bool supported = shadowData.supportsMainLightShadows;
                int cascades = shadowData.mainLightShadowCascadesCount;

                passData.MainLightShadows = supported && cascades == 1;
                passData.MainLightShadowCascades = supported && cascades > 1;

                builder.AllowGlobalStateModification(true);

                // This pass writes no resource, so RenderGraph would cull it as dead work and the
                // keywords would never be restored.
                builder.AllowPassCulling(false);

                builder.SetRenderFunc(s_ExecuteFunc);
            }
        }

        private static void Execute(PassData data, UnsafeGraphContext context)
        {
            UnsafeCommandBuffer cmd = context.cmd;

            cmd.SetKeyword(s_MainLightShadowScreenKeyword, false);
            cmd.SetKeyword(s_MainLightShadowsKeyword, data.MainLightShadows);
            cmd.SetKeyword(s_MainLightShadowCascadesKeyword, data.MainLightShadowCascades);
        }
    }
}
