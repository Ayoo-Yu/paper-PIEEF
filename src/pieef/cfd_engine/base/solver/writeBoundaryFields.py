import json
import math
from pathlib import Path
import re


KAPPA = 0.41
CMU = 0.09
EPSILON_WALL_E = 9.8
ABL_DEFAULT_Z0_M = 0.03
ABL_C1 = 1.44
ABL_C2 = 1.92
# The neutral log solution must also satisfy the epsilon transport equation:
# kappa^2 = sigmaEps * sqrt(Cmu) * (C2-C1). This is a consistency choice,
# not a calibration to Nanshan or a claim that real turbulence is neutral.
ABL_SIGMA_EPS = KAPPA ** 2 / (math.sqrt(CMU) * (ABL_C2 - ABL_C1))
LEGACY_FIXED_K = 0.5
LEGACY_FIXED_EPSILON = 0.375
# Geometry is reduced while U remains in m/s. Kinematic viscosity and all
# turbulence lengths therefore scale by s; epsilon scales by 1/s. The legacy
# constants came from a model using s=0.001, not an unscaled 10 m length.
LEGACY_REFERENCE_SCALE = 0.001
PHYSICAL_KINEMATIC_VISCOSITY = 1.5e-5
LEGACY_PHYSICAL_EPSILON = LEGACY_FIXED_EPSILON * LEGACY_REFERENCE_SCALE
LEGACY_PHYSICAL_LENGTH_SCALE = (
    CMU ** 0.75 * LEGACY_FIXED_K ** 1.5 / LEGACY_PHYSICAL_EPSILON
)


def fmt_scalar(value):
    return f"{float(value):.10g}"


def fmt_vector(x, y, z):
    return f"({fmt_scalar(x)} {fmt_scalar(y)} {fmt_scalar(z)})"


def foam_header(class_name, object_name, location="0"):
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2212                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    location    "{location}";
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def load_info():
    info_path = Path("../info.json")
    with info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_first(mapping, *keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def ensure_positive(name, value, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {numeric}")
    if allow_zero:
        if numeric < 0:
            raise ValueError(f"{name} must be >= 0, got {numeric}")
    elif numeric <= 0:
        raise ValueError(f"{name} must be > 0, got {numeric}")
    return numeric


def compute_uniform_turbulence(speed, intensity, length_scale):
    """Return physical k and epsilon for a physical length in metres."""
    k = 1.5 * (speed * intensity) ** 2
    epsilon = 0.0
    if k > 0 and length_scale > 0:
        epsilon = (CMU ** 0.75) * (k ** 1.5) / length_scale
    return k, epsilon


def resolve_physical_scaling(info):
    """Resolve explicit physical/model units without changing the velocity scale."""
    scale = ensure_positive("mesh.scale", info["mesh"]["scale"])
    wind = info.get("wind", {})
    speed = ensure_positive("wind.speed", pick_first(wind, "speed", default=10.0))
    profile = str(wind.get("profile") or "legacy_fixed").strip().lower()
    abl = None
    if profile == "abl_log":
        z_ref = ensure_positive("wind.referenceHeight", wind.get("referenceHeight", 80.0))
        z0 = ensure_positive("wind.roughnessLength", wind.get("roughnessLength", ABL_DEFAULT_Z0_M))
        displacement = ensure_positive("wind.displacementHeight", wind.get("displacementHeight", 0.0), allow_zero=True)
        if displacement != 0.0:
            raise ValueError("abl_log currently requires displacementHeight=0: a displaced canopy/wall system has not been verified")
        if float(info.get("roughness", {}).get("vege_times", 0)) != 0:
            raise ValueError("abl_log requires roughness.vege_times=0: additional canopy sources have not been verified with the neutral equilibrium profile")
        friction, k_physical, epsilon_physical, tau = compute_abl_reference(speed, z_ref, z0, displacement)
        length_physical = CMU ** 0.75 * k_physical ** 1.5 / epsilon_physical
        intensity = math.sqrt(k_physical / 1.5) / speed
        abl = {
            "reference_height_physical_m": z_ref,
            "reference_height_model_m": z_ref * scale,
            "roughness_length_physical_m": z0,
            "roughness_length_model_m": z0 * scale,
            "displacement_height_physical_m": displacement,
            "displacement_height_model_m": displacement * scale,
            "friction_velocity_mps": friction,
            "top_kinematic_shear_stress_m2ps2": tau,
            "kappa": KAPPA, "Cmu": CMU, "C1": ABL_C1, "C2": ABL_C2,
            "sigmaEps": ABL_SIGMA_EPS,
            "additional_turbulence_sources": False,
            "roughness_source": "explicit_input" if "roughnessLength" in wind else "standardized_default_not_site_calibration",
            "TI_and_length_inputs_ignored": True,
            "physical_accuracy_validated": False,
        }
    elif profile == "legacy_fixed":
        k_physical = LEGACY_FIXED_K
        epsilon_physical = LEGACY_PHYSICAL_EPSILON
        length_physical = LEGACY_PHYSICAL_LENGTH_SCALE
        intensity = math.sqrt(k_physical / 1.5) / speed
    elif profile == "uniform":
        intensity = ensure_positive(
            "wind.turbulenceIntensity",
            pick_first(wind, "turbulenceIntensity", "TI", default=0.1),
        )
        length_physical = ensure_positive(
            "wind.turbulenceLengthScale",
            pick_first(wind, "turbulenceLengthScale", "lengthScale", "L", default=10.0),
        )
        k_physical, epsilon_physical = compute_uniform_turbulence(
            speed, intensity, length_physical
        )
    else:
        raise ValueError(f"Unsupported wind.profile '{profile}'")
    result = {
        "schema_version": 1,
        "length_scale": scale,
        "velocity_scale": 1.0,
        "profile": profile,
        "speed_mps": speed,
        "turbulence_intensity": intensity,
        "nu_physical_m2ps": PHYSICAL_KINEMATIC_VISCOSITY,
        "nu_model_m2ps": PHYSICAL_KINEMATIC_VISCOSITY * scale,
        "length_physical_m": length_physical,
        "length_model_m": length_physical * scale,
        "k_physical_m2ps2": k_physical,
        "k_model_m2ps2": k_physical,
        "epsilon_physical_m2ps3": epsilon_physical,
        "epsilon_model_m2ps3": epsilon_physical / scale,
        "legacy_ignores_ti_and_length_inputs": profile == "legacy_fixed",
    }
    if abl is not None:
        result["abl"] = abl
    return result


def build_transport_properties(nu_model):
    return (
        foam_header("dictionary", "transportProperties", location="constant")
        + f"""
transportModel  Newtonian;

nu              {fmt_scalar(nu_model)};

// ************************************************************************* //
"""
    )


def resolve_numerics(info):
    scheme = info.get("simulation", {}).get("advectionScheme", "upwind")
    if scheme not in ("upwind", "linearUpwind"):
        raise ValueError("simulation.advectionScheme must be upwind or linearUpwind")
    return {
        "schema_version": 1,
        "velocity_advection_scheme": scheme,
        "velocity_divergence_scheme": "bounded Gauss " + ("upwind" if scheme == "upwind" else "linearUpwind limited"),
        "velocity_formal_order": 1 if scheme == "upwind" else 2,
        "velocity_gradient_scheme": "cellLimited Gauss linear 1",
        "turbulence_advection_scheme": "upwind",
        "turbulence_formal_order": 1,
        "selection_source": "explicit_input" if "advectionScheme" in info.get("simulation", {}) else "runtime_default_without_mutating_frozen_input",
        "engineering_accuracy_validated": False,
        "numerical_diffusion_warning": (
            "一阶迎风具有数值扩散，可能影响尾流亏损和恢复；收敛不代表精度改善，需开展网格敏感性分析。"
            if scheme == "upwind" else
            "速度使用带受限梯度的形式二阶线性迎风，湍流仍用一阶迎风；此格式不保证收敛，需独立验证格式及网格敏感性。"
        ),
    }


def build_fv_schemes(numerics):
    template = Path(__file__).resolve().parents[1] / "initcase/system/fvSchemes"
    text = template.read_text(encoding="utf-8")
    text, velocity_count = re.subn(
        r"(?m)^([ \t]*div\(phi,U\)\s+)[^;]+;",
        lambda match: match.group(1) + numerics["velocity_divergence_scheme"] + ";", text
    )
    text, turbulence_count = re.subn(
        r"(?m)^([ \t]*turbulence\s+)[^;]+;",
        lambda match: match.group(1) + "bounded Gauss upwind;", text
    )
    if velocity_count != 1 or turbulence_count != 1:
        raise ValueError("fvSchemes template must have exactly one velocity and one turbulence advection entry")
    return text


def build_uniform_p():
    """Reference pressure belongs at the outflow with a fixed-velocity inlet."""
    return (
        foam_header("volScalarField", "p")
        + """
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    front { type symmetry; }
    back { type symmetry; }
    bot { type zeroGradient; }
    top { type symmetry; }
}

// ************************************************************************* //
"""
    )


def compute_abl_reference(speed, z_ref, z0, displacement):
    effective_height = z_ref - displacement + z0
    if effective_height <= z0:
        raise ValueError(
            "ABL reference height must satisfy referenceHeight - displacementHeight + roughnessLength > roughnessLength"
        )

    friction_velocity = speed * KAPPA / math.log(effective_height / z0)
    k_ref = (friction_velocity ** 2) / math.sqrt(CMU)
    epsilon_ref = (friction_velocity ** 3) / (KAPPA * effective_height)
    tau = friction_velocity ** 2
    return friction_velocity, k_ref, epsilon_ref, tau


def build_uniform_u(speed):
    return (
        foam_header("volVectorField", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform {fmt_vector(speed, 0.0, 0.0)};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {fmt_vector(speed, 0.0, 0.0)};
    }}

    outlet
    {{
        type            zeroGradient;
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

    bot
    {{
        type            noSlip;
    }}

    top
    {{
        type            symmetry;
    }}
}}

// ************************************************************************* //
"""
    )


def build_uniform_k(k_inlet):
    return (
        foam_header("volScalarField", "k")
        + f"""
kInlet          {fmt_scalar(k_inlet)};

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform $kInlet;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform $kInlet;
    }}

    outlet
    {{
        type            inletOutlet;
        inletValue      uniform $kInlet;
        value           $internalField;
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

    bot
    {{
        type            kqRWallFunction;
        value           uniform 0;
    }}

    top
    {{
        type            symmetry;
    }}
}}

// ************************************************************************* //
"""
    )


def build_uniform_epsilon(epsilon_inlet):
    return (
        foam_header("volScalarField", "epsilon")
        + f"""
epsilonInlet    {fmt_scalar(epsilon_inlet)};

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform $epsilonInlet;

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform $epsilonInlet;
    }}

    outlet
    {{
        type            inletOutlet;
        inletValue      uniform $epsilonInlet;
        value           $internalField;
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

    bot
    {{
        type            epsilonWallFunction;
        value           uniform $epsilonInlet;
    }}

    top
    {{
        type            symmetry;
    }}
}}

// ************************************************************************* //
"""
    )


def build_uniform_nut():
    return (
        foam_header("volScalarField", "nut")
        + """
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            calculated;
        value           uniform 0;
    }

    outlet
    {
        type            calculated;
        value           uniform 0;
    }

    front
    {
        type            symmetry;
    }

    back
    {
        type            symmetry;
    }

    bot
    {
        type            nutkWallFunction;
        value           uniform 0;
    }

    top
    {
        type            symmetry;
    }
}

// ************************************************************************* //
"""
    )


def build_abl_conditions(speed, z_ref, z0, displacement):
    return f"""        kappa           {fmt_scalar(KAPPA)};
        Cmu             {fmt_scalar(CMU)};
        C1              0;
        C2              1;
        flowDir         (1 0 0);
        zDir            (0 0 1);
        Uref            {fmt_scalar(speed)};
        Zref            {fmt_scalar(z_ref)};
        z0              uniform {fmt_scalar(z0)};
        d               uniform {fmt_scalar(displacement)};"""


def build_abl_u(speed, z_ref, z0, displacement, tau):
    abl_conditions = build_abl_conditions(speed, z_ref, z0, displacement)
    return (
        foam_header("volVectorField", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform {fmt_vector(speed, 0.0, 0.0)};

boundaryField
{{
    inlet
    {{
        type            atmBoundaryLayerInletVelocity;
{abl_conditions}
        value           uniform {fmt_vector(0.0, 0.0, 0.0)};
    }}

    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {fmt_vector(0.0, 0.0, 0.0)};
        value           uniform {fmt_vector(speed, 0.0, 0.0)};
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

    bot
    {{
        type            noSlip;
    }}

    top
    {{
        type            fixedShearStress;
        tau             {fmt_vector(tau, 0.0, 0.0)};
        value           uniform {fmt_vector(0.0, 0.0, 0.0)};
    }}
}}

// ************************************************************************* //
"""
    )


def build_abl_p():
    return (
        foam_header("volScalarField", "p")
        + """
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }

    front
    {
        type            symmetry;
    }

    back
    {
        type            symmetry;
    }

    bot
    {
        type            zeroGradient;
    }

    top
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //
"""
    )


def build_abl_k(speed, z_ref, z0, displacement, k_ref):
    abl_conditions = build_abl_conditions(speed, z_ref, z0, displacement)
    return (
        foam_header("volScalarField", "k")
        + f"""
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {fmt_scalar(k_ref)};

boundaryField
{{
    inlet
    {{
        type            atmBoundaryLayerInletK;
{abl_conditions}
        value           uniform {fmt_scalar(k_ref)};
    }}

    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {fmt_scalar(k_ref)};
        value           $internalField;
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

    bot
    {{
        type            kqRWallFunction;
        value           uniform 0;
    }}

    top
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
    )


def build_abl_epsilon(speed, z_ref, z0, displacement, epsilon_ref, top_gradient):
    abl_conditions = build_abl_conditions(speed, z_ref, z0, displacement)
    return (
        foam_header("volScalarField", "epsilon")
        + f"""
dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {fmt_scalar(epsilon_ref)};

boundaryField
{{
    inlet
    {{
        type            atmBoundaryLayerInletEpsilon;
{abl_conditions}
        value           uniform {fmt_scalar(epsilon_ref)};
    }}

    outlet
    {{
        type            inletOutlet;
        inletValue      uniform {fmt_scalar(epsilon_ref)};
        value           uniform {fmt_scalar(epsilon_ref)};
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

    bot
    {{
        type            atmEpsilonWallFunction;
        Cmu             {fmt_scalar(CMU)};
        kappa           {fmt_scalar(KAPPA)};
        z0              uniform {fmt_scalar(z0)};
        lowReCorrection false;
        value           uniform {fmt_scalar(epsilon_ref)};
    }}

    top
    {{
        type            fixedGradient;
        gradient        uniform {fmt_scalar(top_gradient)};
        value           uniform {fmt_scalar(epsilon_ref)};
    }}
}}

// ************************************************************************* //
"""
    )


def build_abl_nut(z0):
    return (
        foam_header("volScalarField", "nut")
        + f"""
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}

    outlet
    {{
        type            calculated;
        value           uniform 0;
    }}

    front
    {{
        type            symmetry;
    }}

    back
    {{
        type            symmetry;
    }}

    bot
    {{
        type            atmNutkWallFunction;
        boundNut        true;
        kappa           {fmt_scalar(KAPPA)};
        Cmu             {fmt_scalar(CMU)};
        z0              uniform {fmt_scalar(z0)};
        value           uniform 0;
    }}

    top
    {{
        type            calculated;
        value           uniform 0;
    }}
}}

// ************************************************************************* //
"""
    )


def build_turbulence_properties(profile):
    sigma_epsilon = ABL_SIGMA_EPS if profile == 'abl_log' else 1.3
    return foam_header('dictionary', 'turbulenceProperties', 'constant') + f"""
simulationType RAS;
RAS
{{
    RASModel kEpsilon;
    turbulence on;
    printCoeffs on;
    kEpsilonCoeffs
    {{
        Cmu {fmt_scalar(CMU)};
        C1 {fmt_scalar(ABL_C1)};
        C2 {fmt_scalar(ABL_C2)};
        sigmak 1;
        sigmaEps {fmt_scalar(sigma_epsilon)};
    }}
}}
"""


def build_uniform_fv_options():
    return (
        foam_header("dictionary", "fvOptions", location="constant")
        + """
atmPlantCanopyTurbSource1
{
    type            atmPlantCanopyTurbSource;
    selectionMode   all;
    Cd              roughCd;
    LAD             roughLAD;
}

// ************************************************************************* //
"""
    )


def write_case_files(
    u_text,
    k_text,
    epsilon_text,
    nut_text,
    fv_options_text=None,
    p_text=None,
):
    zero_dir = Path("0")
    zero_dir.mkdir(exist_ok=True)
    constant_dir = Path("constant")
    constant_dir.mkdir(exist_ok=True)
    (zero_dir / "U").write_text(u_text, encoding="utf-8")
    (zero_dir / "k").write_text(k_text, encoding="utf-8")
    (zero_dir / "epsilon").write_text(epsilon_text, encoding="utf-8")
    (zero_dir / "nut").write_text(nut_text, encoding="utf-8")
    if p_text is not None:
        (zero_dir / "p").write_text(p_text, encoding="utf-8")

    fv_options_path = constant_dir / "fvOptions"
    if fv_options_text is None:
        if fv_options_path.exists():
            fv_options_path.unlink()
    else:
        fv_options_path.write_text(fv_options_text, encoding="utf-8")


def main():
    info = load_info()
    scaling = resolve_physical_scaling(info)
    numerics = resolve_numerics(info)
    fv_schemes = build_fv_schemes(numerics)
    speed = scaling["speed_mps"]
    if scaling["profile"] == "abl_log":
        abl = scaling["abl"]
        height = ensure_positive("domain.h", info.get("domain", {}).get("h"))
        if height <= abl["reference_height_physical_m"]:
            raise ValueError("domain.h must exceed the ABL reference height above the inlet ground")
        z_ref = abl["reference_height_model_m"]
        z0 = abl["roughness_length_model_m"]
        displacement = abl["displacement_height_model_m"]
        roof = height * scaling["length_scale"]
        friction = abl["friction_velocity_mps"]
        top_gradient = -friction ** 3 / (KAPPA * (roof + z0) ** 2)
        abl["top_epsilon_gradient_model"] = top_gradient
        abl["top_epsilon_gradient_definition"] = "analytic_flat_fetch_roof_gradient; terrain_domain_sensitivity_required"
        abl["spatial_initialization_required"] = True
        write_case_files(
            build_abl_u(speed, z_ref, z0, displacement, abl["top_kinematic_shear_stress_m2ps2"]),
            build_abl_k(speed, z_ref, z0, displacement, scaling["k_model_m2ps2"]),
            build_abl_epsilon(speed, z_ref, z0, displacement, scaling["epsilon_model_m2ps3"], top_gradient),
            build_abl_nut(z0), None, p_text=build_abl_p(),
        )
    else:
        write_case_files(
            build_uniform_u(speed),
            build_uniform_k(scaling["k_model_m2ps2"]),
            build_uniform_epsilon(scaling["epsilon_model_m2ps3"]),
            build_uniform_nut(),
            build_uniform_fv_options() if scaling["profile"] == "uniform" else None,
            p_text=build_uniform_p(),
        )
    Path("constant/transportProperties").write_text(
        build_transport_properties(scaling["nu_model_m2ps"]), encoding="utf-8"
    )
    Path("constant/turbulenceProperties").write_text(
        build_turbulence_properties(scaling["profile"]), encoding="utf-8"
    )
    Path("constant/physicalScaling.json").write_text(
        json.dumps(scaling, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    Path("system").mkdir(exist_ok=True)
    Path("system/fvSchemes").write_text(fv_schemes, encoding="utf-8")
    Path("constant/numerics.json").write_text(
        json.dumps(numerics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"[INFO] Velocity advection={numerics['velocity_advection_scheme']}; formal order={numerics['velocity_formal_order']}; turbulence advection=upwind")
    print("[WARNING] " + numerics["numerical_diffusion_warning"])
    print(
        f"[INFO] Wrote {scaling['profile']} inlet fields:",
        f"U={speed}",
        f"physicalTI={scaling['turbulence_intensity']:.6g}",
        f"physicalL={scaling['length_physical_m']:.6g}m",
        f"modelL={scaling['length_model_m']:.6g}m",
        f"k={scaling['k_model_m2ps2']:.6g}",
        f"physicalEpsilon={scaling['epsilon_physical_m2ps3']:.6g}",
        f"modelEpsilon={scaling['epsilon_model_m2ps3']:.6g}",
        f"modelNu={scaling['nu_model_m2ps']:.6g}",
        "pBC=inlet:zeroGradient,outlet:fixedValue(0)",
    )
    if scaling["legacy_ignores_ti_and_length_inputs"]:
        print("[INFO] legacy_fixed preserves historical k/epsilon; TI and length inputs apply only to uniform")
    if scaling["profile"] == "abl_log":
        print("[INFO] Neutral ABL: scaled Zref/z0, matched rough nut/epsilon walls, sigmaEps equilibrium, top shear/epsilon flux; spatial initialization required before decomposition")
        print("[WARNING] ABL z0/defaults are standardized assumptions; field accuracy and domain sensitivity are not established by flat-fetch preservation")


if __name__ == "__main__":
    main()
