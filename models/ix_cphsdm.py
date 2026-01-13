# Import Pyomo libraries
from pyomo.environ import (
    Set,
    Var,
    Param,
    Suffix,
    NonNegativeReals,
    check_optimal_termination,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
import idaes.logger as idaeslog
from idaes.core import declare_process_block_class
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import StrEnum
from idaes.core.util.exceptions import InitializationError, ConfigurationError
from idaes.core.util.constants import Constants
import idaes.core.util.scaling as iscale
from idaes.core.util.math import smooth_bound, smooth_min

from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core import InitializationMixin
from watertap.core.solvers import get_solver
from watertap.core.util.initialization import interval_initializer

# from watertap.costing.unit_models.ion_exchange import cost_ion_exchange
from .ix_cphsdm_costing import cost_ion_exchange


__author__ = "Kurban Sitterley"


class IonExchangeType(StrEnum):
    anion = "anion"


class RegenerantChem(StrEnum):
    HCl = "HCl"
    NaOH = "NaOH"
    H2SO4 = "H2SO4"
    NaCl = "NaCl"
    MeOH = "MeOH"
    single_use = "single_use"


@declare_process_block_class("IonExchangeCPHSDM")
class IonExchangeCPHSDMData(InitializationMixin, UnitModelBlockData):
    """
    Ion exchange constant-pattern homogeneous surface diffusion model (CPHSDM) model.

    ***NOTE***
    This model is intended to be exclusively used for PFAS NAWI-Analysis project.

    This model combines the base and CPHSDM models that are available on my fork:
    https://github.com/kurbansitterley/watertap/tree/ix_reorg

    This model ONLY has a properties_in state block.
    There is no regeneration or properties_out.

    This model is only for single_use resin.
    This model is only for anion exchange resins.


    """

    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False.""",
        ),
    )
    CONFIG.declare(
        "target_component",
        ConfigValue(
            default="",
            domain=str,
            description="Designates targeted species for removal",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False.""",
        ),
    )

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )

    CONFIG.declare(
        "regenerant",
        ConfigValue(
            default=RegenerantChem.single_use,
            domain=In(RegenerantChem),
            description="Chemical used for regeneration of fixed bed",
        ),
    )

    CONFIG.declare(
        "hazardous_waste",
        ConfigValue(
            default=False,
            domain=bool,
            description="Designates if resin and residuals contain hazardous material",
        ),
    )
    CONFIG.declare(
        "regenerant",
        ConfigValue(
            default=RegenerantChem.single_use,
            domain=In(RegenerantChem),
            description="Chemical used for regeneration of fixed bed",
        ),
    )

    def build(self):
        super().build()

        target_component = self.config.target_component

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        self.target_component_set = Set(initialize=[target_component])

        if len(self.target_component_set) > 1:
            raise ConfigurationError(
                f"IonExchange can only accept a single target ion but {len(self.target_component_set)} were provided."
            )

        self.ion_exchange_type = IonExchangeType.anion
        # self.config.regenerant = "single_use"

        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = False

        self.properties_in = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Inlet properties",
            **tmp_dict,
        )

        prop_in = self.properties_in[0]

        self.a0 = Param(
            initialize=0.8,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Stanton equation parameter 0",
        )

        self.a1 = Param(
            initialize=0,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Stanton equation parameter 1",
        )

        self.b0 = Param(
            initialize=0.023,
            mutable=True,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 0",
        )

        self.b1 = Param(
            initialize=0.793673,
            mutable=True,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 1",
        )

        self.b2 = Param(
            initialize=0.039324,
            mutable=True,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 2",
        )

        self.b3 = Param(
            initialize=0.009326,
            mutable=True,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 3",
        )

        self.b4 = Param(
            initialize=0.08275,
            mutable=True,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 4",
        )

        self.freundlich_k = Var(
            initialize=1,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,  # dynamic with freundlich_ninv, ((length ** 3) * (mass ** -1)) ** freundlich_ninv,
            doc="Freundlich isotherm k parameter, must be provided in base [L3/M] units",
        )

        self.underdrain_h = Param(
            initialize=0.5,
            mutable=True,
            units=pyunits.m,
            doc="Underdrain height",  # Perry's
        )

        self.distributor_h = Param(
            initialize=0.5,
            mutable=True,
            units=pyunits.m,
            doc="Distributor height",  # Perry's
        )

        # Particle Peclet number correlation
        # Eq. 4.100 in Inamuddin/Luqman

        self.Pe_p_A = Param(
            initialize=0.05,
            units=pyunits.dimensionless,
            doc="Peclet particle equation A parameter",
        )

        self.Pe_p_exp = Param(
            initialize=0.48,
            units=pyunits.dimensionless,
            doc="Peclet particle equation exponent",
        )

        # Sherwood number as a function of Reynolds and Schmidt number
        # Table 16-9 in Perry's
        # Wilson and Geankoplis, Ind. Eng. Chem. Fundam., 5, 9 (1966)

        self.Sh_A = Param(
            initialize=2.4,
            units=pyunits.dimensionless,
            doc="Sherwood equation A parameter",
        )

        self.Sh_exp_A = Param(
            initialize=0.66,
            units=pyunits.dimensionless,
            doc="Sherwood equation exponent A",
        )

        self.Sh_exp_B = Param(
            initialize=0.34,
            units=pyunits.dimensionless,
            doc="Sherwood equation exponent B",
        )

        self.Sh_exp_C = Param(
            initialize=0.33,
            units=pyunits.dimensionless,
            doc="Sherwood equation exponent C",
        )

        # Pressure drop (psi/m of resin bed depth) is a function of loading rate (loading_rate) in m/hr
        # p_drop (psi/m) = p_drop_A + p_drop_B * loading_rate + p_drop_C * loading_rate**2
        # Default is for strong-base type I acrylic anion exchanger resin (A-850, Purolite), @20C
        # Data extracted from MWH Chap 16, Figure 16-14 and fit with Excel

        self.p_drop_A = Param(
            initialize=0.609,
            mutable=True,
            units=pyunits.psi / pyunits.m,
            doc="Pressure drop equation intercept",
        )

        self.p_drop_B = Param(
            initialize=0.173,
            mutable=True,
            units=(pyunits.psi * pyunits.hr) / pyunits.m**2,
            doc="Pressure drop equation B",
        )

        self.p_drop_C = Param(
            initialize=8.28e-4,
            mutable=True,
            units=(pyunits.psi * pyunits.hr**2) / pyunits.m**3,
            doc="Pressure drop equation C",
        )

        self.pump_efficiency = Param(
            initialize=0.8,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Pump efficiency",
        )

        self.regeneration_time = Param(
            initialize=1800,
            mutable=True,
            units=pyunits.s,
            doc="Regeneration time",
        )

        self.service_to_regen_flow_ratio = Param(
            initialize=3,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Ratio of service flow rate to regeneration flow rate",
        )

        # Bed expansion is calculated as a fraction of the bed_depth
        # These coefficients are used to calculate that fraction (bed_expansion_frac) as a function of backwash rate (backwashing_rate, m/hr)
        # bed_expansion_frac = bed_expansion_A + bed_expansion_B * backwashing_rate + bed_expansion_C * backwashing_rate**2
        # Default is for strong-base type I acrylic anion exchanger resin (A-850, Purolite), @20C
        # Data extracted from MWH Chap 16, Figure 16-15 and fit with Excel

        self.bed_expansion_frac_A = Param(
            initialize=-1.23e-2,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Bed expansion fraction eq intercept",
        )

        self.bed_expansion_frac_B = Param(
            initialize=1.02e-1,
            mutable=True,
            units=pyunits.hr / pyunits.m,
            doc="Bed expansion fraction equation B parameter",
        )

        self.bed_expansion_frac_C = Param(
            initialize=-1.35e-3,
            mutable=True,
            units=pyunits.hr**2 / pyunits.m**2,
            doc="Bed expansion fraction equation C parameter",
        )
        # Rinse, Regen, Backwashing params

        self.rinse_bed_volumes = Param(
            initialize=5,
            mutable=True,
            doc="Number of bed volumes for rinse step",
        )

        self.backwashing_rate = Param(
            initialize=5,
            mutable=True,
            units=pyunits.m / pyunits.hour,
            doc="Backwash loading rate [m/hr]",
        )

        self.backwash_time = Param(
            initialize=600,
            mutable=True,
            units=pyunits.s,
            doc="Backwash time",
        )

        self.redundant_column_freq = Param(
            initialize=4,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Frequency for redundant columns",
        )

        # ==========VARIABLES==========

        self.resin_diam = Var(
            initialize=7e-4,
            bounds=(5e-4, 1.5e-3),  # Perry's
            # domain=NonNegativeReals,
            units=pyunits.m,
            doc="Resin bead diameter",
        )

        self.resin_density = Var(
            initialize=700,
            bounds=(500, 950),  # Perry's
            # domain=NonNegativeReals,
            units=pyunits.kg / pyunits.m**3,
            doc="Resin bulk density",
        )

        self.bed_volume = Var(
            initialize=2,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.m**3,
            doc="Bed volume per column",
        )

        self.bed_volume_total = Var(
            initialize=2,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.m**3,
            doc="Total bed volume",
        )

        self.bed_depth = Var(
            initialize=1,
            bounds=(0.75, 2),  # EPA-WBS guidance
            # domain=NonNegativeReals,
            units=pyunits.m,
            doc="Bed depth",
        )

        self.bed_porosity = Var(
            initialize=0.4,
            bounds=(0.3, 0.8),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Bed porosity",
        )

        self.column_height = Var(
            initialize=2,
            bounds=(0, 4.26),  # EPA-WBS guidance
            # domain=NonNegativeReals,
            units=pyunits.m,
            doc="Column height",
        )

        self.bed_diameter = Var(
            initialize=1,
            bounds=(0.75, 4.26),  # EPA-WBS guidance
            # domain=NonNegativeReals,
            units=pyunits.m,
            doc="Column diameter",
        )

        self.col_height_to_diam_ratio = Var(
            initialize=1,
            bounds=(0, 100),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Min ratio of bed depth to diameter",
        )

        self.number_columns = Var(
            initialize=2,
            bounds=(1, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Number of operational columns for ion exchange process",
        )

        self.number_columns_redundant = Var(
            initialize=1,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Number of redundant columns for ion exchange process",
        )

        self.breakthrough_time = Var(
            initialize=1e5,  # DOW, ~7 weeks max breakthru time
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.s,
            doc="Breakthrough time",
        )

        self.bv = Var(  # BV
            initialize=1e5,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Bed volumes of feed at breakthru concentration",
        )

        self.ebct = Var(
            initialize=520,
            bounds=(90, None),
            # domain=NonNegativeReals,
            units=pyunits.s,
            doc="Empty bed contact time",
        )

        # ====== Hydrodynamic variables ====== #

        self.loading_rate = Var(
            initialize=0.0086,
            bounds=(0, 0.01),  # MWH, Perry's, EPA-WBS
            # domain=NonNegativeReals,
            units=pyunits.m / pyunits.s,
            doc="Superficial velocity through bed",
        )

        self.freundlich_ninv = Var(
            initialize=0.95,
            bounds=(0, 1),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Freundlich isotherm 1/n paramter",
        )

        self.surf_diff_coeff = Var(
            initialize=1e-15,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.m**2 * pyunits.s**-1,
            doc="Surface diffusion coefficient",
        )

        self.film_mass_transfer_coeff = Var(
            initialize=1e-5,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.m * pyunits.s**-1,
            doc="Liquid phase film mass transfer coefficient",
        )

        self.c_norm = Var(
            self.target_component_set,
            initialize=0.5,
            bounds=(0, 1),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Dimensionless (relative) bfreakthrough concentration [Ct/C0] of target ion",
        )

        self.c_eq = Var(
            self.target_component_set,
            initialize=1e-5,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Equilibrium concentration of adsorbed phase with liquid phase",
        )

        self.solute_dist_param = Var(
            initialize=1e5,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Solute distribution parameter",
        )

        self.N_Bi = Var(
            initialize=10,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Biot number",
        )

        self.N_Bi_smooth = Var(
            initialize=10,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Smooth bounded Biot number",
        )

        # correlations using Reynolds number valid in Re < 2e4

        self.N_Sc = Var(
            self.target_component_set,
            initialize=700,
            bounds=(1e-5, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Schmidt number",  # correlations using Schmidt number valid in 0.7 < Sc < 1e4
        )

        self.N_Re = Var(
            initialize=4.3,
            bounds=(0, 60),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Reynolds number",
        )

        self.resin_density_app = Var(
            initialize=1,
            bounds=(1, None),
            # domain=NonNegativeReals,
            units=pyunits.kg / pyunits.m**3,
            doc="Resin apparent density",
        )

        self.min_N_St = Var(
            initialize=10,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Minimum Stanton number to achieve a constant pattern solution",
        )

        self.min_ebct = Var(
            initialize=500,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.s,
            doc="Minimum EBCT to achieve a constant pattern solution",
        )

        self.throughput = Var(
            initialize=1,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Specific throughput from empirical equation",
        )

        self.min_t_contact = Var(
            initialize=1000,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.s,
            doc="Minimum fluid residence time in the bed to achieve a constant pattern solution",
        )

        self.min_breakthrough_time = Var(
            initialize=1e8,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.s,
            doc="Minimum operational time of the bed from fresh to achieve a constant pattern solution",
        )

        self.shape_correction_factor = Var(
            initialize=1,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Shape correction factor",
        )

        self.resin_porosity = Var(
            initialize=1,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Resin bead porosity",
        )

        self.tortuosity = Var(
            initialize=1,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Tortuosity of the path that the adsorbate must take as compared to the radius",
        )

        self.t_contact = Var(
            initialize=100,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.s,
            doc="Contact time (residence time)",
        )

        self.bed_area = Var(
            initialize=100,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.m**2,
            doc="Cross-sectional bed area",
        )

        self.vel_inter = Var(
            initialize=1,
            bounds=(0, None),
            # domain=NonNegativeReals,
            units=pyunits.m / pyunits.s,
            doc="Interstitial bed velocity",
        )

        @self.Expression(doc="Flow per column")
        def flow_per_column(b):
            return prop_in.flow_vol_phase["Liq"] / b.number_columns

        @self.Expression(doc="Pressure drop")
        def pressure_drop(b):
            loading_rate_m_hr = pyunits.convert(
                b.loading_rate, to_units=pyunits.m / pyunits.hr
            )
            return (
                b.p_drop_A
                + b.p_drop_B * loading_rate_m_hr
                + b.p_drop_C * loading_rate_m_hr**2
            ) * b.bed_depth  # for 20C;

        @self.Expression(doc="Rinse time")
        def rinse_time(b):
            return b.ebct * b.rinse_bed_volumes

        @self.Expression(doc="Waste time")
        def waste_time(b):
            return b.regeneration_time + b.backwash_time + b.rinse_time

        @self.Expression(doc="Cycle time")
        def cycle_time(b):
            return b.breakthrough_time + b.waste_time

        # if self.config.regenerant == RegenerantChem.single_use:
        self.regeneration_time.set_value(0)
        self.service_to_regen_flow_ratio.set_value(0)

        @self.Expression(doc="Backwashing flow rate")
        def bw_flow(b):
            return (
                pyunits.convert(b.backwashing_rate, to_units=pyunits.m / pyunits.s)
                * b.bed_area
                * b.number_columns
            )

        @self.Expression(doc="Bed expansion fraction from backwashing")
        def bed_expansion_frac(b):
            return (
                b.bed_expansion_frac_A
                + b.bed_expansion_frac_B * b.backwashing_rate
                + b.bed_expansion_frac_C * b.backwashing_rate**2
            )  # for 20C

        @self.Expression(doc="Rinse flow rate")
        def rinse_flow(b):
            return b.loading_rate * b.bed_area * b.number_columns

        @self.Expression(doc="Backwash pump power")
        def bw_pump_power(b):
            return pyunits.convert(
                (b.pressure_drop * b.bw_flow) / b.pump_efficiency,
                to_units=pyunits.kilowatts,
            ) * (b.backwash_time / b.cycle_time)

        @self.Expression(doc="Rinse pump power")
        def rinse_pump_power(b):
            return pyunits.convert(
                (b.pressure_drop * b.rinse_flow) / b.pump_efficiency,
                to_units=pyunits.kilowatts,
            ) * (b.rinse_time / b.cycle_time)

        @self.Expression(doc="Bed expansion from backwashing")
        def bed_expansion_h(b):
            return b.bed_expansion_frac * b.bed_depth

        @self.Expression(doc="Free board needed")
        def free_board(b):
            return b.distributor_h + b.underdrain_h + b.bed_expansion_h

        @self.Expression(doc="Main pump power")
        def main_pump_power(b):
            return pyunits.convert(
                (b.pressure_drop * prop_in.flow_vol_phase["Liq"]) / b.pump_efficiency,
                to_units=pyunits.kilowatts,
            ) * (b.breakthrough_time / b.cycle_time)

        @self.Expression(doc="Volume per column")
        def column_volume(b):
            return b.column_height * b.bed_area

        @self.Expression(doc="Total column volume required")
        def column_volume_total(b):
            return b.number_columns * b.column_volume

        @self.Expression(doc="Total number of columns")
        def number_columns_total(b):
            return b.number_columns + b.number_columns_redundant

        @self.Expression(
            doc="Sherwood number from laminar conditions"
        )  # Cheng 2024, SI Eq. S2.8
        def Sh_lam(b):
            return 0.664 * b.N_Sc[target_component] ** (1 / 3) * b.N_Re**0.5

        @self.Expression(
            doc="Sherwood number from turbulent conditions"
        )  # Cheng 2024, SI Eq. S2.9
        def Sh_turb(b):
            num = 0.037 * b.N_Re**0.8 * b.N_Sc[target_component]
            denom = 1 + 2.443 * b.N_Re ** (-0.1) * (
                b.N_Sc[target_component] ** (2 / 3) - 1
            )
            return num / denom

        @self.Expression(
            doc="Sherwood number for a single particle"
        )  # Cheng 2024, SI Eq. S2.7
        def Sh_p(b):
            return 2 + (b.Sh_lam**2 + b.Sh_turb**2) ** 0.5

        @self.Expression(
            doc="Sherwood number from Cheng 2024"
        )  # Cheng 2024, SI Eq. S2.6
        def Sh(b):
            return (1 + 1.5 * (1 - b.bed_porosity)) * b.Sh_p

        @self.Expression(doc="Pore Biot number")  # Cheng 2024, SI Eq. S3.1
        def Bi_p(b):
            num = b.film_mass_transfer_coeff * (b.resin_diam / 2) * b.tortuosity
            denom = (
                prop_in.diffus_phase_comp["Liq", target_component] * b.resin_porosity
            )
            return pyunits.convert(num / denom, to_units=pyunits.dimensionless)

        @self.Expression(doc="Surface diffusion parameter")
        def spdfr(b):
            return (
                b.surf_diff_coeff
                * b.tortuosity
                * b.c_eq[target_component]
                * b.resin_density_app
            ) / (
                prop_in.diffus_phase_comp["Liq", target_component]
                * b.resin_porosity
                * prop_in.conc_mass_phase_comp["Liq", target_component]
            )

        @self.Expression(doc="Surface Biot number")  # Cheng 2024, SI Eq. S3.2
        def Bi_s(b):
            return b.Bi_p / b.spdfr

        @self.Expression(doc="Overall Biot Number")  # Cheng 2024, SI Eq. S3.4
        def Bi(b):
            return 1 / ((1 / b.Bi_p) + (1 / b.Bi_s))

        @self.Expression()
        def kf(b):
            return (
                prop_in.diffus_phase_comp["Liq", target_component]
                * b.shape_correction_factor
                * b.Sh
            ) / b.resin_diam
        
        @self.Expression()
        def bed_depth_to_diam_ratio(b):
            return pyunits.convert(b.bed_depth / b.bed_diameter, to_units=pyunits.dimensionless)

        @self.Constraint()
        def eq_t_contact(b):
            return b.t_contact == b.ebct * b.bed_porosity

        @self.Constraint()
        def eq_vel_inter(b):
            return b.loading_rate == b.vel_inter * b.bed_porosity

        @self.Constraint()
        def eq_bed_diameter(b):
            return b.bed_area * 4 == Constants.pi * (b.bed_diameter**2)

        @self.Constraint()
        def eq_bed_area(b):
            return (
                b.bed_area * b.loading_rate
                == prop_in.flow_vol_phase["Liq"] / b.number_columns
            )

        @self.Expression(doc="Bed mass")
        def bed_mass(b):
            return pyunits.convert(b.resin_density * b.bed_volume, to_units=pyunits.kg)

        @self.Constraint(
            self.target_component_set,
            doc="Freundlich isotherm",
        )
        def eq_freundlich(b, j):
            freund_k_units = (pyunits.m**3 * pyunits.kg) ** b.freundlich_ninv
            return b.c_eq[j] == b.freundlich_k * freund_k_units * (
                prop_in.conc_mass_phase_comp["Liq", j] ** b.freundlich_ninv
            )

        @self.Constraint(doc="Bed volume per operational column")
        def eq_bed_volume(b):
            return b.bed_volume == b.bed_area * b.bed_depth

        @self.Constraint(doc="Total bed volume")
        def eq_bed_design(b):
            return b.bed_volume_total == b.bed_volume * b.number_columns

        @self.Constraint(doc="Column height")
        def eq_column_height(b):
            return b.column_height == b.bed_depth + b.free_board

        @self.Constraint(
            self.target_component_set,
            doc="Solute distribution parameter",
        )
        def eq_solute_dist_param(b, j):
            return b.solute_dist_param * b.bed_porosity * prop_in.conc_mass_phase_comp[
                "Liq", j
            ] == b.resin_density_app * b.c_eq[j] * (1 - b.bed_porosity)

        @self.Constraint(doc="Biot number")
        def eq_Bi(b):
            return (
                b.N_Bi * b.solute_dist_param * b.bed_porosity
                == b.film_mass_transfer_coeff
                * (b.resin_diam / 2)
                * (1 - b.bed_porosity)
                / b.surf_diff_coeff
            )

        @self.Constraint(doc="Smooth bounded Biot number")
        def eq_Bi_smooth(b):
            return b.N_Bi_smooth == smooth_bound(b.N_Bi, 0.5, 500, eps=1e-12)
            # return b.N_Bi_smooth == smooth_min(b.N_Bi, 0.501)

        @self.Constraint(doc="Bed porosity")
        def eq_bed_porosity(b):
            dimensionless_density = pyunits.convert(
                b.resin_density / b.resin_density_app, to_units=pyunits.dimensionless
            )
            return b.bed_porosity == 1 - dimensionless_density

        @self.Constraint(doc="Bed depth")
        def eq_bed_depth(b):
            return b.bed_depth == b.loading_rate * b.ebct

        @self.Constraint(
            doc="Minimum Stanton number to achieve constant pattern solution"
        )
        def eq_min_number_st_cps(b):
            return b.min_N_St == b.a0 * b.N_Bi + b.a1

        @self.Constraint(
            doc="Minimum empty bed contact time to achieve constant pattern solution"
        )
        def eq_min_ebct_cps(b):
            return b.min_ebct * (
                1 - b.bed_porosity
            ) * b.film_mass_transfer_coeff == b.min_N_St * (b.resin_diam / 2)

        @self.Constraint(
            self.target_component_set,
            doc="Throughput based on empirical 5-parameter regression",
        )
        def eq_throughput(b, j):
            return b.throughput == b.b0 + b.b1 * (b.c_norm[j] ** b.b2) + b.b3 / (
                1.01 - (b.c_norm[j] ** b.b4)
            )

        @self.Constraint(
            doc="Minimum fluid residence time in the bed to achieve a constant pattern solution"
        )
        def eq_min_t_contact(b):
            return b.min_t_contact == b.bed_porosity * b.min_ebct

        @self.Constraint(
            doc="minimum operational time of the bed from fresh to achieve a constant pattern solution"
        )
        def eq_minimum_breakthrough_time(b):
            return (
                b.min_breakthrough_time
                == b.min_t_contact * (b.solute_dist_param + 1) * b.throughput
            )

        @self.Constraint(
            doc="elapsed operational time between a fresh bed and the theoretical bed replacement"
        )
        def eq_breakthrough_time(b):
            return b.breakthrough_time == b.min_breakthrough_time + (
                b.t_contact - b.min_t_contact
            ) * (b.solute_dist_param + 1)

        @self.Constraint(doc="Bed volumes at breakthrough")
        def eq_bv(b):
            return b.breakthrough_time * b.loading_rate == b.bv * b.bed_depth

        # @self.Constraint(doc="bed volumes treated")
        # def eq_bed_volumes_treated(b):
        #     return (
        #         b.breakthrough_time * b.bed_porosity ==
        #         b.bed_volumes_treated * b.t_contact
        #         # == b.breakthrough_time * b.bed_porosity
        #     )

        @self.Constraint(
            doc="Reynolds number",
        )
        def eq_Re(b):
            return (
                b.N_Re * prop_in.visc_d_phase["Liq"] * b.bed_porosity
                == prop_in.dens_mass_phase["Liq"] * b.resin_diam * b.loading_rate
            )

        @self.Constraint(self.target_component_set, doc="Schmidt number")
        def eq_Sc(b, j):
            return (
                b.N_Sc[j]
                * prop_in.dens_mass_phase["Liq"]
                * prop_in.diffus_phase_comp["Liq", j]
                == prop_in.visc_d_phase["Liq"]
            )

        @self.Constraint(
            self.target_component_set,
            doc="Fluid film mass transfer rate from the Gnielinski correlation",
        )
        def eq_gnielinski(b, j):
            return (
                b.shape_correction_factor
                * (1 + 1.5 * (1 - b.bed_porosity))
                * prop_in.diffus_phase_comp["Liq", j]
                * (2 + 0.644 * (b.N_Re**0.5) * (b.N_Sc[j] ** (1 / 3)))
            ) == (b.film_mass_transfer_coeff * b.resin_diam)

    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        General wrapper for initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        flags = self.properties_in.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info("Initialization Step 1a Complete.")

        # interval_initializer(self)

        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
            if not check_optimal_termination(res):
                init_log.warning(
                    f"Trouble solving unit model {self.name}, trying one more time"
                )
                res = opt.solve(self, tee=slc.tee)

        init_log.info("Initialization Step 2 {}.".format(idaeslog.condition(res)))

        # Release Inlet state
        self.properties_in.release_state(flags, outlvl=outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {self.name} failed to initialize.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        """ scale user provided configuraitons first"""
        if iscale.get_scaling_factor(self.resin_porosity) is None:
            iscale.set_scaling_factor(
                self.resin_porosity, 1 / self.resin_porosity.value
            )

        if iscale.get_scaling_factor(self.tortuosity) is None:
            iscale.set_scaling_factor(self.tortuosity, 1 / self.tortuosity.value)

        if iscale.get_scaling_factor(self.breakthrough_time) is None:
            iscale.set_scaling_factor(self.breakthrough_time, 1e-9)
        if iscale.get_scaling_factor(self.min_breakthrough_time) is None:
            iscale.set_scaling_factor(self.min_breakthrough_time, 1e-9)

        if iscale.get_scaling_factor(self.N_Re) is None:
            iscale.set_scaling_factor(self.N_Re, 1 / 10)

        if iscale.get_scaling_factor(self.number_columns) is None:
            iscale.set_scaling_factor(
                self.number_columns, 1 / self.number_columns.value
            )

        if iscale.get_scaling_factor(self.number_columns_redundant) is None:
            iscale.set_scaling_factor(
                self.number_columns_redundant, 1 / self.number_columns_redundant.value
            )

        if iscale.get_scaling_factor(self.resin_diam) is None:
            iscale.set_scaling_factor(self.resin_diam, 1 / self.resin_diam.value)

        if iscale.get_scaling_factor(self.resin_density) is None:
            iscale.set_scaling_factor(self.resin_density, 1 / self.resin_density.value)

        if iscale.get_scaling_factor(self.bed_volume) is None:
            iscale.set_scaling_factor(self.bed_volume, 1 / self.bed_volume.value)

        if iscale.get_scaling_factor(self.bed_diameter) is None:
            iscale.set_scaling_factor(self.bed_diameter, 1 / self.bed_diameter.value)

        if iscale.get_scaling_factor(self.bed_area) is None:
            iscale.set_scaling_factor(self.bed_area, 1 / self.bed_area.value)

        if iscale.get_scaling_factor(self.bed_volume_total) is None:
            sf = iscale.get_scaling_factor(self.bed_volume)
            sf = sf * iscale.get_scaling_factor(self.number_columns)
            iscale.set_scaling_factor(self.bed_volume_total, sf)

        if iscale.get_scaling_factor(self.bed_depth) is None:
            iscale.set_scaling_factor(self.bed_depth, 1 / self.bed_depth.value)

        if iscale.get_scaling_factor(self.bed_porosity) is None:
            iscale.set_scaling_factor(self.bed_porosity, 1 / self.bed_porosity.value)

        if iscale.get_scaling_factor(self.column_height) is None:
            iscale.set_scaling_factor(self.column_height, 1 / self.column_height.value)

        if iscale.get_scaling_factor(self.ebct) is None:
            iscale.set_scaling_factor(self.ebct, 1 / 100)

        if iscale.get_scaling_factor(self.loading_rate) is None:
            iscale.set_scaling_factor(self.loading_rate, 1 / self.loading_rate.value)

        if iscale.get_scaling_factor(self.bv) is None:
            iscale.set_scaling_factor(self.bv, 1e-5)

        target_component = self.config.target_component
        for j in self.target_component_set:
            sf_solute = iscale.get_scaling_factor(
                self.properties_in[0].flow_mol_phase_comp["Liq", j],
                default=1e4,
                warning=True,
            )

        for j in self.config.property_package.solvent_set:
            sf_solvent = iscale.get_scaling_factor(
                self.properties_in[0].flow_mol_phase_comp["Liq", j],
                default=1e-3,  # default based on typical concentration for treatment
                warning=True,
            )

        if iscale.get_scaling_factor(self.freundlich_k) is None:
            iscale.set_scaling_factor(self.freundlich_k, 1 / self.freundlich_k.value)

        if iscale.get_scaling_factor(self.freundlich_ninv) is None:
            iscale.set_scaling_factor(
                self.freundlich_ninv, 1 / self.freundlich_ninv.value
            )

        if iscale.get_scaling_factor(self.surf_diff_coeff) is None:
            iscale.set_scaling_factor(
                self.surf_diff_coeff, 1 / self.surf_diff_coeff.value
            )

        if iscale.get_scaling_factor(self.c_norm[target_component]) is None:
            iscale.set_scaling_factor(
                self.c_norm[target_component], 1 / self.c_norm[target_component].value
            )

        iscale.set_scaling_factor(
            self.c_eq[target_component], 1 / self.c_eq[target_component].value
        )

        if iscale.get_scaling_factor(self.N_Bi) is None:
            iscale.set_scaling_factor(self.N_Bi, 1e-2)

        if iscale.get_scaling_factor(self.N_Bi_smooth) is None:
            iscale.set_scaling_factor(self.N_Bi_smooth, 1e-2)

        if iscale.get_scaling_factor(self.N_Sc[target_component]) is None:
            iscale.set_scaling_factor(
                self.N_Sc[target_component], 1 / self.N_Sc[target_component].value
            )

        if iscale.get_scaling_factor(self.resin_density_app) is None:
            sf = iscale.get_scaling_factor(self.resin_density)
            iscale.set_scaling_factor(self.resin_density_app, sf)

        if iscale.get_scaling_factor(self.min_N_St) is None:
            iscale.set_scaling_factor(self.min_N_St, 1)

        if iscale.get_scaling_factor(self.min_ebct) is None:
            sf = iscale.get_scaling_factor(self.ebct)
            iscale.set_scaling_factor(self.min_ebct, sf)

        if iscale.get_scaling_factor(self.t_contact) is None:
            sf = iscale.get_scaling_factor(self.ebct)
            sf = sf * iscale.get_scaling_factor(self.bed_porosity)
            iscale.set_scaling_factor(self.t_contact, sf)

        if iscale.get_scaling_factor(self.min_t_contact) is None:
            iscale.set_scaling_factor(
                self.min_t_contact, iscale.get_scaling_factor(self.t_contact)
            )

        if iscale.get_scaling_factor(self.film_mass_transfer_coeff) is None:
            iscale.set_scaling_factor(
                self.film_mass_transfer_coeff, 1 / self.film_mass_transfer_coeff.value
            )

        if iscale.get_scaling_factor(self.shape_correction_factor) is None:
            iscale.set_scaling_factor(
                self.shape_correction_factor, 1 / self.shape_correction_factor.value
            )

        if iscale.get_scaling_factor(self.bed_area) is None:
            sf = self.bed_diameter.value**2 * Constants.pi / 4
            iscale.set_scaling_factor(self.bed_area, 1 / sf)

        if iscale.get_scaling_factor(self.solute_dist_param) is None:
            # should calc from eq
            iscale.set_scaling_factor(self.solute_dist_param, 1 / 1e5)

        if iscale.get_scaling_factor(self.throughput) is None:
            iscale.set_scaling_factor(self.throughput, 1)

        if iscale.get_scaling_factor(self.vel_inter) is None:
            sf = iscale.get_scaling_factor(self.loading_rate)
            sf = sf / iscale.get_scaling_factor(self.bed_porosity)
            iscale.set_scaling_factor(self.vel_inter, sf)

        """constraint scaling"""
        if iscale.get_scaling_factor(self.eq_t_contact) is None:
            iscale.set_constraint_scaling_harmonic_magnitude(self.eq_t_contact)
        if iscale.get_scaling_factor(self.eq_freundlich) is None:
            iscale.constraint_scaling_transform(
                self.eq_freundlich[target_component],
                iscale.get_scaling_factor(self.c_eq[target_component]),
            )
        if iscale.get_scaling_factor(self.eq_solute_dist_param) is None:
            sf = iscale.get_scaling_factor(self.solute_dist_param)
            sf = sf * iscale.get_scaling_factor(self.bed_porosity)
            sf = sf * iscale.get_scaling_factor(
                self.properties_in[0].flow_mol_phase_comp["Liq", target_component],
            )
            iscale.constraint_scaling_transform(
                self.eq_solute_dist_param[target_component], sf
            )
        if iscale.get_scaling_factor(self.eq_Bi) is None:
            sf = iscale.get_scaling_factor(self.film_mass_transfer_coeff)
            sf = sf * iscale.get_scaling_factor(self.resin_diam)
            sf = sf / iscale.get_scaling_factor(self.surf_diff_coeff)
            iscale.constraint_scaling_transform(self.eq_Bi, sf)
        if iscale.get_scaling_factor(self.eq_minimum_breakthrough_time) is None:
            iscale.constraint_scaling_transform(
                self.eq_minimum_breakthrough_time,
                iscale.get_scaling_factor(self.min_breakthrough_time),
            )
        if iscale.get_scaling_factor(self.eq_breakthrough_time) is None:
            iscale.constraint_scaling_transform(
                self.eq_breakthrough_time,
                iscale.get_scaling_factor(self.breakthrough_time),
            )
        if iscale.get_scaling_factor(self.eq_min_ebct_cps) is None:
            sf = iscale.get_scaling_factor(self.min_ebct)
            sf = sf * iscale.get_scaling_factor(self.film_mass_transfer_coeff)
            iscale.constraint_scaling_transform(self.eq_min_ebct_cps, sf)

        if iscale.get_scaling_factor(self.eq_min_t_contact) is None:
            iscale.set_constraint_scaling_harmonic_magnitude(self.eq_min_t_contact)

        if iscale.get_scaling_factor(self.eq_bv) is None:
            sf = iscale.get_scaling_factor(self.breakthrough_time)
            sf = sf * iscale.get_scaling_factor(self.loading_rate)
            iscale.constraint_scaling_transform(self.eq_bv, sf)

        if iscale.get_scaling_factor(self.eq_Sc) is None:
            sf = iscale.get_scaling_factor(self.properties_in[0].visc_d_phase["Liq"])
            iscale.constraint_scaling_transform(self.eq_Sc[target_component], sf)
            # iscale.set_constraint_scaling_harmonic_magnitude(self.eq_Sc)
        if iscale.get_scaling_factor(self.eq_gnielinski) is None:
            sf = iscale.get_scaling_factor(self.film_mass_transfer_coeff)
            sf = sf * iscale.get_scaling_factor(self.resin_diam)
            iscale.constraint_scaling_transform(
                self.eq_gnielinski[target_component], sf
            )

    @property
    def default_costing_method(self):
        return cost_ion_exchange
