using System.Runtime.InteropServices;
using Unity.Mathematics;

namespace Prometheus.Core.Structs
{
    /// <summary>
    /// One light, packed for the ray-tracing kernels. 64 bytes, four 16-byte rows.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The layout deliberately mirrors URP's own additional-light constants rather than inventing
    /// a tidier one. Those four vectors are what URP's <c>InitializeLightConstants_Common</c>
    /// produces, and the host fills this by calling that method directly instead of deriving the
    /// values again.
    /// </para>
    /// <para>
    /// That matters more than it looks. Attenuation is a curve, not a fact: URP applies a
    /// specific smooth window on top of inverse-square falloff, and a spot cone is encoded as a
    /// precomputed scale and offset rather than as angles. Re-deriving any of it from the
    /// <c>Light</c> component would produce something plausible that disagreed with the direct
    /// lighting in the frame, and bounce light that disagrees with its own source reads as a
    /// bug in the geometry rather than in the falloff.
    /// </para>
    /// <para>
    /// Fields follow Section 3.2's 16-byte alignment rule, so the C# and HLSL views match without
    /// padding on either side.
    /// </para>
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct PrometheusGpuLight
    {
        /// <summary>Size in bytes, matching the HLSL declaration.</summary>
        public const int SizeInBytes = 64;

        /// <summary>
        /// World position in <c>xyz</c>, with <c>w</c> distinguishing the two kinds of light.
        /// </summary>
        /// <remarks>
        /// URP's convention: <c>w = 1</c> means a punctual light and <c>xyz</c> is where it is;
        /// <c>w = 0</c> means a directional light and <c>xyz</c> is the direction towards it.
        /// Carrying both cases in one field is what lets the shader loop over every light with a
        /// single branch rather than keeping two lists.
        /// </remarks>
        public float4 PositionType;

        /// <summary>Final colour, with intensity and colour space already applied.</summary>
        public float4 Color;

        /// <summary>Spot axis in <c>xyz</c>. Meaningless for other light types.</summary>
        public float4 SpotDirection;

        /// <summary>
        /// Falloff terms: <c>xy</c> for distance, <c>zw</c> for the spot cone.
        /// </summary>
        /// <remarks>
        /// Precomputed by URP so the shader evaluates each with a single multiply-add. The values
        /// are not angles or radii and are not meaningful to read individually.
        /// </remarks>
        public float4 Attenuation;
    }
}
