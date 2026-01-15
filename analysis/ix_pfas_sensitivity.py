import os
import yaml
import math
import pandas as pd

from pyomo.environ import *
from pyomo.environ import units as pyunits
from pyomo.util.calc_var_value import calculate_variable_from_constraint as cvc

from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from idaes.core.util.scaling import *
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core import FlowsheetBlock, UnitModelCostingBlock


from models.ix_cphsdm import IonExchangeCPHSDM as IX
from watertap.core.solvers import get_solver
from watertap.costing import WaterTAPCosting
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock as MCAS,
    DiffusivityCalculation,
)
from watertap.core.util.model_diagnostics.infeasible import *

from parameter_sweep import (
    parameter_sweep,
    LinearSample,
    ParameterSweep,
)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
absolute_path = os.path.dirname(__file__)
par_dir = os.path.abspath(os.path.join(absolute_path, os.pardir))

with open(f"{par_dir}/data/pfas_properties.yaml", "r") as f:
    species_properties = yaml.load(f, Loader=yaml.FullLoader)

with open(f"{par_dir}/data/resin_properties.yaml", "r") as f:
    resin_properties = yaml.load(f, Loader=yaml.FullLoader)


# EPA limits for PFAS species in kg/m3
assumed_lim = 1e-8  # = 10 ng/L
epa_lims = {
    "PFBA": 1e-08,
    "PFPeA": 1e-08,
    "PFHxA": 1e-08,
    "PFHpA": 1e-08,
    "PFOA": 4e-09,
    "PFNA": 1e-08,
    "PFDA": 1e-08,
    "PFUnDA": 1e-08,
    "PFDoDA": 1e-08,
    "PFTeDA": 1e-08,
    "FOSA": 1e-08,
    "PFBS": 1e-07,
    "PFPeS": 1e-08,
    "PFHxS": 1e-08,
    "PFOS": 4e-09,
    "PFHpS": 1e-08,
}

mw_water = 0.018 * pyunits.kg / pyunits.mol
rho = 1000 * pyunits.kg / pyunits.m**3

# Base parameter values
# These are the values used in the original case study model runs
base_loading_rate = 0.00679  # m3/s per m2 bed area
base_flow_rate = 0.043813  # m3/s, 1 MGD
base_ebct = 360  # seconds
resin_dens_a694 = resin_properties["a694"]["density"]
resin_diam_a694 = resin_properties["a694"]["diameter"]
resin_porosity_a694 = resin_properties["a694"]["porosity"]

solver = get_solver()


def model_build(
    species=None,
    regenerant="single_use",
    regen_composition={},
    regen_soln_density=None,
    **kwargs,
):
    """
    Build ion exchange model for PFAS species and add costing.
    """

    if species is None:
        raise ValueError("Must provide PFAS species for model build.")

    mw = species_properties[species]["mw"]
    mv = species_properties[species]["molar_volume"]

    ix_config = {
        "target_component": species,
        "regenerant": regenerant,
        "regen_composition": regen_composition,
    }

    ion_props = {
        "solute_list": [species],
        "mw_data": {"H2O": mw_water, species: mw},
        "charge": {species: -1},
        "diffus_calculation": DiffusivityCalculation.HaydukLaudie,
        "molar_volume_data": {("Liq", species): mv},
    }

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = MCAS(**ion_props)
    m.fs.properties.visc_d_phase["Liq"] = 1.3097e-3
    m.fs.properties.dens_mass_const = 1000

    ix_config["property_package"] = m.fs.properties

    m.fs.ix = ix = IX(**ix_config)

    m.fs.sample_number = Var(initialize=0)
    m.fs.inlet_conc = Var(initialize=10e-9)
    m.fs.outlet_conc = Var(initialize=4e-9)
    m.fs.optimal_solve = Var(initialize=0)
    m.fs.flow_in = Var(initialize=1, units=pyunits.m**3 / pyunits.s)

    ix.t_breakthru_year = Expression(
        expr=pyunits.convert(ix.breakthrough_time, to_units=pyunits.year)
    )

    ix.t_breakthru_day = Expression(
        expr=pyunits.convert(ix.breakthrough_time, to_units=pyunits.day)
    )
    # Purolite A694 spec sheet - pressure drop
    m.fs.ix.p_drop_A.set_value(0)
    m.fs.ix.p_drop_B.set_value(0.1338)
    m.fs.ix.p_drop_C.set_value(0)
    # Purolite A694 spec sheet - no backwashing permitted with resin
    m.fs.ix.bed_expansion_frac_A.set_value(0)
    m.fs.ix.bed_expansion_frac_B.set_value(0)
    m.fs.ix.bed_expansion_frac_C.set_value(0)

    add_costing(m)
    m.fs.obj = Objective(expr=m.fs.costing.LCOW, sense=minimize)

    m.fs.costing.ion_exchange.anion_exchange_resin_cost.set_value(346)  # EPA-WBS

    if regen_soln_density is not None:
        m.fs.ix.regen_soln_density.set_value(regen_soln_density)

    return m


def add_costing(m):
    """
    Add costing to IX model.
    """

    flow_out = m.fs.ix.properties_in[0].flow_vol_phase["Liq"]
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyunits.USD_2021
    m.fs.costing.utilization_factor.fix(1)
    m.fs.ix.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.costing.cost_process()
    m.fs.costing.add_LCOW(flow_out)
    m.fs.costing.add_specific_energy_consumption(flow_out)

    m.fs.costing.ion_exchange.anion_exchange_resin_cost.set_value(346)  # EPA-WBS
    m.fs.costing.resin_changeout_rate = Param(
        initialize=0.01481,
        mutable=True,
        units=pyunits.hr / pyunits.ft**3,
        doc="Hours of labor required per cubic ft resin",  # From EPA-WBS PFAS IX model; O&M Assumptions tab
    )

    m.fs.costing.resin_replacement_time_required = Expression(
        expr=pyunits.convert(
            m.fs.costing.resin_changeout_rate * m.fs.ix.bed_volume_total,
            to_units=pyunits.hr,
        )
    )


def model_init(
    m,
    solver=None,
    flow_rate=0.043813,
    inlet_conc=10e-9,
    outlet_conc=9e-9,
    sample_number=0,
    loading_rate=0.00679,
    ebct=360,
    kf_init=1,
    ninv_init=0.9,
    ds_init=5e-15,
):
    """
    Initialize ion exchange model and solve stable initial point.
    """

    if solver is None:
        solver = get_solver()

    ix = m.fs.ix
    species = ix.config.target_component
    c_norm = outlet_conc / inlet_conc
    mw = species_properties[species]["mw"] * pyunits.kg / pyunits.mol
    c0 = inlet_conc * pyunits.kg / pyunits.m**3
    q_in = flow_rate * pyunits.m**3 / pyunits.s

    c0_mol_flow = pyunits.convert((c0 * q_in) / mw, to_units=pyunits.mol / pyunits.s)
    c0_mol_flow_sf = 1 / value(c0_mol_flow)

    water_mol_flow = pyunits.convert(
        (q_in * rho) / mw_water,
        to_units=pyunits.mol / pyunits.s,
    )
    water_mol_flow_sf = 1 / value(water_mol_flow)

    m.fs.sample_number.fix(sample_number)
    m.fs.inlet_conc.fix(inlet_conc)
    m.fs.outlet_conc.fix(outlet_conc)
    m.fs.flow_in.fix(flow_rate)

    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", water_mol_flow_sf, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", c0_mol_flow_sf, index=("Liq", species)
    )

    ix.properties_in[0].flow_mol_phase_comp["Liq", "H2O"].fix(water_mol_flow)
    ix.properties_in[0].flow_mol_phase_comp["Liq", species].fix(c0_mol_flow)
    ix.properties_in[0].pressure.fix(101325)
    ix.properties_in[0].temperature.fix(298)
    ix.properties_in[0].flow_vol_phase["Liq"].set_value(flow_rate)
    ix.properties_in[0].conc_mass_phase_comp["Liq", species].set_value(inlet_conc)

    ix.bed_depth.setub(None)
    ix.column_height.setub(None)
    ix.resin_density.setlb(100)
    ix.resin_density_app.setub(None)

    ix.resin_density.fix(resin_dens_a694)
    ix.resin_diam.fix(resin_diam_a694)
    ix.resin_porosity.fix(resin_porosity_a694)

    ix.loading_rate.fix(loading_rate)
    ix.ebct.fix(ebct)

    ix.bed_porosity.fix(0.4)
    ix.shape_correction_factor.fix(1)
    ix.tortuosity.fix(1)
    print(f"dof = {degrees_of_freedom(m)}")

    ncol_op = math.ceil(flow_rate * 2 / 0.04382)
    ncol_redund = math.ceil(ncol_op / 4)
    ix.number_columns.fix(ncol_op)
    ix.number_columns_redundant.fix(ncol_redund)

    ix.c_norm[species].fix(0.5)

    ix.freundlich_ninv.fix(ninv_init)
    ix.freundlich_k.fix(kf_init)
    ix.surf_diff_coeff.fix(ds_init)

    ix.bv.set_value(100000)
    ix.bv.setlb(None)
    ix.N_Re.setlb(None)
    ix.breakthrough_time.setlb(None)
    ix.min_breakthrough_time.setlb(None)

    print(f"dof = {degrees_of_freedom(m)}")
    # calc_ix_from_constr(m, cvc_dict)
    calculate_scaling_factors(m)

    m.fs.ix.initialize()
    results = solver.solve(m)
    assert_optimal_termination(results)
    m.fs.costing.initialize()

    if c_norm <= 0.99:
        ix.c_norm[species].fix(c_norm)
    else:
        ix.c_norm[species].fix(0.99)

    m = activate_surrogate(m)
    m.fs.ix.initialize()

    return m


def activate_surrogate(m):
    """
    Add surrogate models for min St number and throughput.
    """

    min_st_surrogate = PysmoSurrogate.load_from_file(
        f"{par_dir}/data/min_st_pysmo_surr_linear.json"
    )
    throughput_surrogate = PysmoSurrogate.load_from_file(
        f"{par_dir}/data/throughput_pysmo_surr_linear.json"
    )

    ix = m.fs.ix

    ix.eq_min_number_st_cps.deactivate()
    ix.eq_throughput.deactivate()

    m.fs.min_st_surrogate = SurrogateBlock(concrete=True)
    m.fs.min_st_surrogate.build_model(
        min_st_surrogate,
        # input_vars=[ix.freundlich_ninv, ix.N_Bi],
        input_vars=[ix.freundlich_ninv, ix.N_Bi_smooth],
        output_vars=[ix.min_N_St],
    )
    m.fs.throughput_surrogate = SurrogateBlock(concrete=True)
    m.fs.throughput_surrogate.build_model(
        throughput_surrogate,
        # input_vars=[ix.freundlich_ninv, ix.N_Bi, ix.c_norm],
        input_vars=[ix.freundlich_ninv, ix.N_Bi_smooth, ix.c_norm],
        output_vars=[ix.throughput],
    )

    m.fs.ix.N_Bi_smooth.setub(None)
    m.fs.ix.N_Bi_smooth.setlb(0)

    m.fs.ix.N_Bi.setub(None)
    m.fs.ix.N_Bi.setlb(0)

    return m


def calc_ix_from_constr(m, calc_from_constr_dict, return_orignal_state=False):
    """
    Calculate IX variables from constraints.
    """
    ix = m.fs.ix
    for k, v in calc_from_constr_dict.items():
        ixv = getattr(ix, k)
        ixc = getattr(ix, v)
        if all(list(c.is_indexed() for c in [ixv, ixc])):
            for i, vv in ixv.items():
                cvc(vv, ixc[i])
        elif ixv.is_indexed():
            for i, vv in ixv.items():
                cvc(vv, ixc)
        elif ixc.is_indexed():
            for i, cc in ixc.items():
                cvc(ixv, cc)
        else:
            cvc(ixv, ixc)


def model_reinit(m, **kwargs):

    m.fs.ix.initialize()

    return m


def model_solve(m, **kwargs):
    """
    Solve IX model with surrogates.
    """

    solver = get_solver()

    m.fs.optimal_solve.fix(0)

    cvc(m.fs.ix.throughput, m.fs.throughput_surrogate.pysmo_constraint["throughput"])
    # m.fs.throughput_surrogate.pysmo_constraint["throughput"].deactivate()
    # m.fs.ix.throughput.fix()
    cvc(m.fs.ix.min_N_St, m.fs.min_st_surrogate.pysmo_constraint["min_st"])
    # m.fs.min_st_surrogate.pysmo_constraint["min_st"].deactivate()
    # m.fs.ix.min_N_St.fix()

    try:
        results = solver.solve(m)
        print(
            f"\nFirst solve, sample {m.fs.sample_number()}, {results.solver.termination_condition}\n"
        )
        assert_optimal_termination(results)
        m.fs.optimal_solve.fix(1)

    except:
        try:
            results = solver.solve(m)
            print(
                f"\nSecond solve, {m.fs.sample_number()}, {results.solver.termination_condition}\n"
            )
            assert_optimal_termination(results)
            m.fs.optimal_solve.fix(1)
        except:
            _m = ConcreteModel()
            _m.v = Var(initialize=42)
            _m.c = Constraint(expr=_m.v == 42)
            results = solver.solve(_m)
            assert_optimal_termination(results)
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Failed to solve sample {m.fs.sample_number()}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
            print_infeasible_constraints(m)
            print_variables_close_to_bounds(m)
            m.fs.optimal_solve.fix(0)

    return m, results


def build_and_solve(data=None, reinit=False, sweep=None, **kwargs):

    m = model_build(species=data.target_component, **kwargs)

    m = model_init(
        m,
        flow_rate=base_flow_rate,
        inlet_conc=data.c0,
        outlet_conc=data.cb,
        sample_number=data.curve_id,
        loading_rate=base_loading_rate,
        ebct=base_ebct,
        kf_init=1,
        ninv_init=0.9,
        ds_init=5e-15,
    )

    m.fs.ix.resin_diam.fix(data.resin_diam)
    m.fs.ix.resin_density.fix(data.resin_density)
    m.fs.ix.resin_porosity.fix(data.resin_porosity)

    m.fs.ix.freundlich_ninv.fix(data.freundlich_ninv)
    m.fs.ix.freundlich_k.fix(data.freundlich_k)
    m.fs.ix.surf_diff_coeff.fix(data.surf_diff_coeff)

    # if reinit:
    m.fs.ix.initialize()

    results = model_solve(m)
    if sweep in ["ebct", "loading_rate", "ebct+loading_rate"]:
        m.fs.ix.number_columns.unfix()
    if sweep in ["loading_rate", "ebct+loading_rate"]:
        m.fs.ix.loading_rate.setub(1)
    return m


def solve(blk, solver=None, tee=False, check_termination=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    if check_termination:
        assert_optimal_termination(results)
    return results


def build_sweep_params(m, num_samples=5, sweep="ebct", rel_frac=0.25):
    sweep_params = {}

    # EBCT Sensitivity
    # EPA-WBS: EBCT between 1.5-3 min per vessel, 3-6 min total
    # Slack conversation with Alex on Jan 5, 2026 - go up to 9 min total (4.5 min per vessel)
    ebct_lb, ebct_ub = 180, 540  # seconds
    # Loading Rate Sensitivity
    # EPA-WBS: 1-12 gpm/sqft
    lr_lb, lr_ub = 0.000679, 0.00815  # gpm/ft2 to m/s

    # Resin Cost Sensitivity
    # +/- X% base cost of 346 USD/ft3
    resin_cost_base = 346
    resin_cost_lb = resin_cost_base * (1 - rel_frac)
    resin_cost_ub = resin_cost_base * (1 + rel_frac)

    # Freundlich n Sensitivity
    # +./- X% base value
    freundlich_ninv_base = value(m.fs.ix.freundlich_ninv) ** -1
    # freundlich_ninv_lb = freundlich_ninv_base * (1 - rel_frac)
    freundlich_ninv_lb = 1.05
    freundlich_ninv_ub = freundlich_ninv_base * (1 + rel_frac)

    # Freundlich k sensitivity
    # +./- X% base value
    freundlich_k_base = value(m.fs.ix.freundlich_k)
    freundlich_k_lb = freundlich_k_base * (1 - rel_frac)
    freundlich_k_ub = freundlich_k_base * (1 + rel_frac)

    # Surface Diffusivity sensitivity
    # +./- X% base value
    ds_base = value(m.fs.ix.surf_diff_coeff)
    ds_lb = ds_base * (1 - rel_frac)
    ds_ub = ds_base * (1 + rel_frac)

    if sweep == "ebct":
        sweep_params["ebct"] = LinearSample(m.fs.ix.ebct, ebct_lb, ebct_ub, num_samples)

    if sweep == "loading_rate":
        sweep_params["loading_rate"] = LinearSample(
            m.fs.ix.loading_rate, lr_lb, lr_ub, num_samples
        )

    if sweep == "ebct+loading_rate":
        sweep_params["ebct"] = LinearSample(m.fs.ix.ebct, ebct_lb, ebct_ub, num_samples)

        sweep_params["loading_rate"] = LinearSample(
            m.fs.ix.loading_rate, lr_lb, lr_ub, num_samples
        )

    if sweep == "resin_cost":

        sweep_params["resin_cost"] = LinearSample(
            m.fs.costing.ion_exchange.anion_exchange_resin_cost,
            resin_cost_lb,
            resin_cost_ub,
            num_samples,
        )

    if sweep == "freundlich_ninv":
        sweep_params["freundlich_ninv"] = LinearSample(
            m.fs.ix.freundlich_ninv,
            freundlich_ninv_lb,
            freundlich_ninv_ub,
            num_samples,
        )

    if sweep == "freundlich_k":
        sweep_params["freundlich_k"] = LinearSample(
            m.fs.ix.freundlich_k,
            freundlich_k_lb,
            freundlich_k_ub,
            num_samples,
        )

    if sweep == "surf_diff_coeff":
        sweep_params["surf_diff_coeff"] = LinearSample(
            m.fs.ix.surf_diff_coeff,
            ds_lb,
            ds_ub,
            num_samples,
        )

    if sweep == "annual_resin_replacement_factor":
        # only if doing regeneration otherwise doesn't impact cost
        sweep_params["annual_resin_replacement_factor"] = LinearSample(
            m.fs.costing.ion_exchange.annual_resin_replacement_factor,
            0.05,
            0.5,
            num_samples,
        )

    if sweep == "annual_resin_replacement_factor+resin_cost":
        # only if doing regeneration otherwise doesn't impact cost
        sweep_params["annual_resin_replacement_factor"] = LinearSample(
            m.fs.costing.ion_exchange.annual_resin_replacement_factor,
            0.05,
            0.5,
            num_samples,
        )
        sweep_params["resin_cost"] = LinearSample(
            m.fs.costing.ion_exchange.anion_exchange_resin_cost,
            resin_cost_lb,
            resin_cost_ub,
            num_samples,
        )
    return sweep_params


def build_outputs(m):
    outputs = {}

    cols = sorted(
        [
            "fs.costing.LCOW",
            "fs.optimal_solve",
            "fs.ix.freundlich_k",
            "fs.ix.underdrain_h",
            "fs.ix.distributor_h",
            "fs.ix.Pe_p_A",
            "fs.ix.Pe_p_exp",
            "fs.ix.Sh_A",
            "fs.ix.Sh_exp_A",
            "fs.ix.Sh_exp_B",
            "fs.ix.Sh_exp_C",
            "fs.ix.p_drop_A",
            "fs.ix.p_drop_B",
            "fs.ix.p_drop_C",
            "fs.ix.pump_efficiency",
            "fs.ix.regen_bed_volumes",
            "fs.ix.regen_soln_density",
            "fs.ix.regen_tank_vol_factor",
            "fs.ix.regeneration_time",
            "fs.ix.service_to_regen_flow_ratio",
            "fs.ix.bed_expansion_frac_A",
            "fs.ix.bed_expansion_frac_B",
            "fs.ix.bed_expansion_frac_C",
            "fs.ix.rinse_bed_volumes",
            "fs.ix.backwashing_rate",
            "fs.ix.backwash_time",
            "fs.ix.redundant_column_freq",
            "fs.ix.resin_diam",
            "fs.ix.resin_density",
            "fs.ix.bed_volume",
            "fs.ix.bed_volume_total",
            "fs.ix.bed_depth",
            "fs.ix.bed_porosity",
            "fs.ix.column_height",
            "fs.ix.bed_diameter",
            "fs.ix.col_height_to_diam_ratio",
            "fs.ix.number_columns",
            "fs.ix.number_columns_redundant",
            "fs.ix.breakthrough_time",
            "fs.ix.bv",
            "fs.ix.ebct",
            "fs.ix.loading_rate",
            "fs.ix.freundlich_ninv",
            "fs.ix.surf_diff_coeff",
            "fs.ix.film_mass_transfer_coeff",
            "fs.ix.c_norm",
            "fs.ix.c_eq",
            "fs.ix.solute_dist_param",
            "fs.ix.N_Bi",
            "fs.ix.N_Bi_smooth",
            "fs.ix.N_Sc",
            "fs.ix.N_Re",
            "fs.ix.resin_density_app",
            "fs.ix.min_N_St",
            "fs.ix.min_ebct",
            "fs.ix.throughput",
            "fs.ix.min_t_contact",
            "fs.ix.min_breakthrough_time",
            "fs.ix.shape_correction_factor",
            "fs.ix.resin_porosity",
            "fs.ix.tortuosity",
            "fs.ix.t_contact",
            "fs.ix.bed_area",
            "fs.ix.vel_inter",
            "fs.ix.flow_per_column",
            "fs.ix.pressure_drop",
            "fs.ix.rinse_time",
            "fs.ix.waste_time",
            "fs.ix.cycle_time",
            "fs.ix.bw_flow",
            "fs.ix.bed_expansion_frac",
            "fs.ix.rinse_flow",
            "fs.ix.bw_pump_power",
            "fs.ix.rinse_pump_power",
            "fs.ix.bed_expansion_h",
            "fs.ix.free_board",
            "fs.ix.main_pump_power",
            "fs.ix.column_volume",
            "fs.ix.column_volume_total",
            "fs.ix.number_columns_total",
            "fs.ix.Sh_lam",
            "fs.ix.Sh_turb",
            "fs.ix.Sh_p",
            "fs.ix.Sh",
            "fs.ix.Bi_p",
            "fs.ix.spdfr",
            "fs.ix.Bi_s",
            "fs.ix.Bi",
            "fs.ix.kf",
            "fs.ix.bed_depth_to_diam_ratio",
            "fs.ix.bed_mass",
            "fs.ix.t_breakthru_year",
            "fs.ix.t_breakthru_day",
            "fs.ix.costing.capital_cost",
            "fs.ix.costing.fixed_operating_cost",
            "fs.ix.costing.regen_soln_dens",
            "fs.ix.costing.regen_dose",
            "fs.ix.costing.capital_cost_vessel",
            "fs.ix.costing.capital_cost_resin",
            "fs.ix.costing.capital_cost_regen_tank",
            "fs.ix.costing.capital_cost_backwash_tank",
            "fs.ix.costing.operating_cost_hazardous",
            "fs.ix.costing.total_pumping_power",
            "fs.ix.costing.flow_vol_resin",
            "fs.ix.costing.single_use_resin_replacement_cost",
            "fs.ix.costing.backwash_tank_vol",
            "fs.ix.costing.cost_factor",
            "fs.ix.costing.direct_capital_cost",
            "fs.costing.total_investment_factor",
            "fs.costing.maintenance_labor_chemical_factor",
            "fs.costing.utilization_factor",
            "fs.costing.electricity_cost",
            "fs.costing.electrical_carbon_intensity",
            "fs.costing.plant_lifetime",
            "fs.costing.wacc",
            "fs.costing.capital_recovery_factor",
            "fs.costing.TPEC",
            "fs.costing.TIC",
            "fs.costing.HCl_cost",
            "fs.costing.NaOH_cost",
            "fs.costing.methanol_cost",
            "fs.costing.ethanol_cost",
            "fs.costing.acetone_cost",
            "fs.costing.NaCl_cost",
            "fs.costing.aggregate_capital_cost",
            "fs.costing.aggregate_fixed_operating_cost",
            "fs.costing.aggregate_variable_operating_cost",
            "fs.costing.aggregate_flow_electricity",
            "fs.costing.aggregate_flow_ethanol",
            "fs.costing.aggregate_flow_methanol",
            "fs.costing.aggregate_flow_acetone",
            "fs.costing.aggregate_flow_NaCl",
            "fs.costing.aggregate_flow_NaOH",
            "fs.costing.aggregate_flow_costs",
            "fs.costing.aggregate_direct_capital_cost",
            "fs.costing.total_capital_cost",
            "fs.costing.total_operating_cost",
            "fs.costing.maintenance_labor_chemical_operating_cost",
            "fs.costing.total_fixed_operating_cost",
            "fs.costing.total_variable_operating_cost",
            "fs.costing.total_annualized_cost",
            "fs.costing.LCOW_component_direct_capex",
            "fs.costing.LCOW_component_indirect_capex",
            "fs.costing.LCOW_component_fixed_opex",
            "fs.costing.LCOW_component_variable_opex",
            "fs.costing.LCOW_aggregate_direct_capex",
            "fs.costing.LCOW_aggregate_indirect_capex",
            "fs.costing.LCOW_aggregate_fixed_opex",
            "fs.costing.LCOW_aggregate_variable_opex",
            "fs.costing.specific_energy_consumption",
            "fs.costing.specific_energy_consumption_component",
            "fs.costing.resin_changeout_rate",
            "fs.costing.resin_replacement_time_required",
            "fs.costing.ion_exchange.anion_exchange_resin_cost",
            "fs.costing.ion_exchange.cation_exchange_resin_cost",
            "fs.costing.ion_exchange.vessel_A_coeff",
            "fs.costing.ion_exchange.vessel_b_coeff",
            "fs.costing.ion_exchange.backwash_tank_A_coeff",
            "fs.costing.ion_exchange.backwash_tank_b_coeff",
            "fs.costing.ion_exchange.regen_tank_A_coeff",
            "fs.costing.ion_exchange.regen_tank_b_coeff",
            "fs.costing.ion_exchange.annual_resin_replacement_factor",
            "fs.costing.ion_exchange.hazardous_min_cost",
            "fs.costing.ion_exchange.hazardous_resin_disposal",
            "fs.costing.ion_exchange.hazardous_regen_disposal",
            "fs.costing.ion_exchange.regen_recycle",
            "fs.costing.aggregate_flow_costs[electricity]",
            "fs.costing.aggregate_flow_costs[ethanol]",
            "fs.costing.aggregate_flow_costs[methanol]",
            "fs.costing.aggregate_flow_costs[acetone]",
            "fs.costing.aggregate_flow_costs[NaCl]",
            "fs.costing.aggregate_flow_costs[NaOH]",
        ]
    )

    for c in cols:
        comp = m.find_component(c)
        if comp is not None:
            if comp.is_indexed():
                for i, v in comp.items():
                    outputs[c.split(".")[-1]] = v
            else:
                outputs[c.split(".")[-1]] = comp

    return outputs


if __name__ == "__main__":

    df = pd.read_csv(f"{par_dir}/data/ix_case_study_sensitivity_inputs.csv")
    data = df.iloc[0]
    ix_nums = [18, 13, 304]

    for i, row in df.iterrows():
        if row.curve_id not in ix_nums:
            continue
        data = row
        print(f"\n\nRunning sample {data.curve_id} - {data.target_component}\n\n")
        m = build_and_solve(data)
        outputs = build_outputs(m)
        print(f"Sample {data.curve_id} results:")
        print(f"LCOW: {value(outputs['LCOW']):.4f} $/m3")
        # for k, v in outputs.items():
        #     print(f"{k}: {v.name} = {value(v)}")

    # m = build_and_solve(data)
    # outputs = build_outputs(m)
    # for k, v in outputs.items():
    #     print(f"{k}: {v.name}")

    # num_samples = 4
    # num_procs = 4

    # res_file = "ix_pfas_sensitivity-test.h5"

    # num_samples = 100
    # file_save = "parameter_sweep_results.csv"

    # results_array, results_dict = parameter_sweep(
    #     build_model=build_and_solve,
    #     build_model_kwargs={"data": data, "sweep": "ebct"},
    #     build_sweep_params=build_sweep_params,
    #     build_sweep_params_kwargs={"num_samples": num_samples},
    #     build_outputs=build_outputs,
    #     build_outputs_kwargs={},
    #     optimize_function=solve,
    #     num_samples=num_samples,
    #     csv_results_file_name=res_file.replace(".h5", ".csv"),
    # )

    # df = pd.read_csv(res_file.replace(".h5", ".csv"))

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()

    # ax.plot(df["ebct"], df["LCOW"], marker="o")

    # plt.show()
    # # df = pd.read_csv(file_save)
    # # # make_stacked_plot(file_save, parameter="A_comp")

    # # kwargs_dict = {
    # #     # Arguments being used in the demo
    # #     "h5_results_file_name": res_file,
    # #     "build_model": build_and_solve,  # Function that builds the flowsheet model
    # #     "build_model_kwargs": {"data": data},
    # #     "build_sweep_params": build_sweep_params,  # Function for building sweep param dictionary
    # #     "build_sweep_params_kwargs": dict(num_samples=num_samples),
    # #     "build_outputs": build_outputs,  # Function the builds outputs to save
    # #     "build_outputs_kwargs": {},
    # #     "optimize_function": solve,
    # #     "optimize_kwargs": {"solver": solver, "check_termination": True},
    # #     # "initialize_function": None,
    # #     # "initialize_kwargs": {},
    # #     "parallel_back_end": "MultiProcessing",
    # #     "number_of_subprocesses": num_procs,
    # #     "csv_results_file_name": res_file.replace(".h5", ".csv"),
    # #     "h5_parent_group_name": None,  # Useful for loop tool
    # #     "update_sweep_params_before_init": False,
    # #     "initialize_before_sweep": False,
    # #     # "reinitialize_function": None,
    # #     # "reinitialize_kwargs": {},
    # #     # "reinitialize_before_sweep": False,
    # #     # "probe_function": None,
    # #     # # Post-processing arguments
    # #     # "interpolate_nan_outputs": False,
    # #     # # Advanced Users
    # #     # "debugging_data_dir": None,
    # #     # "log_model_states": False,
    # #     # "custom_do_param_sweep": None,  # Advanced users only!
    # #     # "custom_do_param_sweep_kwargs": {},
    # #     # # GUI-related
    # #     # "publish_progress": False,  # Compatibility with WaterTAP GUI
    # #     # "publish_address": "http://localhost:8888",
    # # }
    # # ps = ParameterSweep(**kwargs_dict)
    # # results_array, results_dict = ps.parameter_sweep(
    # #     kwargs_dict["build_model"],
    # #     kwargs_dict["build_sweep_params"],
    # #     build_outputs=kwargs_dict["build_outputs"],
    # #     build_outputs_kwargs=kwargs_dict["build_outputs_kwargs"],
    # #     num_samples=num_samples,
    # #     seed=None,
    # #     build_model_kwargs=kwargs_dict["build_model_kwargs"],
    # #     build_sweep_params_kwargs=kwargs_dict["build_sweep_params_kwargs"],
    # # )
    # # res_df = pd.read_csv(res_file.replace(".h5", ".csv"))
    # # print(res_df)
    # # from watertap.kurby import *
    # # fig, ax = plot_contour(
    # #     d,
    # #     x="fs.ix.ebct",
    # #     y="fs.ix.loading_rate",
    # #     z="fs.costing.LCOW",
    # #     x_adj=x_adj,
    # #     y_adj=y_adj,
    # #     set_dict=set_dict,
    # #     cmap="winter",
    # #     add_contour_labels=True,
    # #     levels=5,
    # #     contour_label_fmt="  %#.1f \$/m$^{3}$ ",
    # #     figsize=figsize,
    # #     cb_fontsize=fontsize - 2,
    # #     cb_title="LCOW (\$/m$^{3}$)",
    # # )
