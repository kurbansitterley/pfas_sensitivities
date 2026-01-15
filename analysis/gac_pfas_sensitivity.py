import os
import yaml
import math
import h5py
import pandas as pd
import idaes.core.util.scaling as iscale
import idaes.core.util.model_statistics as istat

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Expression,
    Constraint,
    Objective,
    SolverFactory,
    TransformationFactory,
    minimize,
    Reals,
    value,
    assert_optimal_termination,
    units as pyunits,
)
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.check_units import assert_units_consistent
from idaes.core import (
    FlowsheetBlock,
    UnitModelCostingBlock,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.core.util.scaling import constraint_autoscale_large_jac
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from idaes.models.unit_models import Feed
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    DiffusivityCalculation,
)
from watertap.core.util.model_diagnostics.infeasible import *

# from watertap.unit_models.gac import GAC
from models.gac_cphsdm import GAC
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.gac import cost_gac, ContactorType
from watertap.costing.multiple_choice_costing_block import MultiUnitModelCostingBlock

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


min_st_surrogate = PysmoSurrogate.load_from_file(
    f"{par_dir}/data/min_st_pysmo_surr_linear.json"
)
throughput_surrogate = PysmoSurrogate.load_from_file(
    f"{par_dir}/data/throughput_pysmo_surr_linear.json"
)

# epa regulation limits
assumed_lim = 10
epa_limits = {
    "PFBA": assumed_lim,
    "PFPeA": assumed_lim,
    "PFHxA": assumed_lim,
    "PFHpA": assumed_lim,
    "PFOA": 4,
    "PFNA": 10,
    "PFDA": assumed_lim,
    "PFUnDA": assumed_lim,
    "PFDoDA": assumed_lim,
    "PFTrDA": assumed_lim,
    "PFTeDA": assumed_lim,
    "FOSA": assumed_lim,
    "NMeFOSAA": assumed_lim,
    "NEtFOSAA": assumed_lim,
    "PFBS": 100,  # 2000
    "PFPeS": assumed_lim,
    "PFHxS": 10,
    "PFOS": 4,
    "HFPO-DA": 10,
    "ADONA": assumed_lim,
    "9Cl-PF3ONS": assumed_lim,
    "11Cl-PF3OUdS": assumed_lim,
}

mw_water = 0.018 * pyunits.kg / pyunits.mol
rho = 1000 * pyunits.kg / pyunits.m**3

solver = get_solver()


def model_build(
    species="PFAS",
    # regenerant="single_use",
    # regen_composition={},
    # regen_soln_density=None,
    gac_config=dict(),
    **kwargs,
):

    pfas_mw = species_properties[species][
        "mw"
    ]  # changing mw after the fact changes concentration
    pfas_mv = species_properties[species]["molar_volume"]

    # create m objects
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=[species],
        mw_data={"H2O": 0.018, species: pfas_mw},
        diffus_calculation=DiffusivityCalculation.HaydukLaudie,
        molar_volume_data={("Liq", species): pfas_mv},
    )
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.feed.properties[0].conc_mass_phase_comp
    m.fs.feed.properties[0].flow_vol_phase

    # gac_config["property_package"] = m.fs.properties
    gac_config["film_transfer_coefficient_type"] = "calculated"
    gac_config["surface_diffusion_coefficient_type"] = "fixed"

    m.fs.gac = GAC(property_package=m.fs.properties, **gac_config)

    # streams
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.gac.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # model customization
    deactivate_ss_calculations(m)

    # build costing blocks
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyunits.USD_2021
    # m.fs.costing.electricity_cost.fix(0.0822)
    m.fs.costing.utilization_factor.fix(1)
    m.fs.gac.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    # m.fs.gac.costing = MultiUnitModelCostingBlock(
    #     flowsheet_costing_block=m.fs.costing,
    #     costing_blocks={
    #         "steel_pressure": {
    #             "costing_method": cost_gac,
    #             "costing_method_arguments": {"contactor_type": "pressure"},
    #         },
    #         "concrete_gravity": {
    #             "costing_method": cost_gac,
    #             "costing_method_arguments": {"contactor_type": "gravity"},
    #         },
    #     },
    # )

    # add flowsheet level blocks
    m.fs.costing.cost_process()
    treated_flow = m.fs.gac.process_flow.properties_out[0].flow_vol_phase["Liq"]
    m.fs.costing.add_LCOW(flow_rate=treated_flow)

    # custom_add_LCOW(
    #     m.fs.costing,
    #     m.fs.gac.costing.costing_blocks["steel_pressure"],
    #     treated_flow,
    #     name="LCOW_pressure",
    # )
    # custom_add_LCOW(
    #     m.fs.costing,
    #     m.fs.gac.costing.costing_blocks["concrete_gravity"],
    #     treated_flow,
    #     name="LCOW_gravity",
    # )

    m.fs.obj = Objective(expr=m.fs.costing.LCOW, sense=minimize)
    # if regen_soln_density is not None:
    #     m.fs.gac.regen_soln_density.set_value(regen_soln_density)

    return m


def model_init(
    m,
    solver=None,
    flow_rate=0.043813,
    inlet_conc=10e-9,
    outlet_conc=9e-9,
    particle_density=510,
    particle_size=0.00065,
    **kwargs,
):

    # touch properties and default scaling
    species = m.fs.properties.solute_set.at(1)

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
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", water_mol_flow_sf, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", c0_mol_flow_sf, index=("Liq", species)
    )
    iscale.set_scaling_factor(m.fs.gac.gac_usage_rate, 1e5)
    iscale.set_scaling_factor(m.fs.gac.operational_time, 1e-7)
    iscale.set_scaling_factor(m.fs.gac.min_operational_time, 1e-7)
    iscale.constraint_scaling_transform(m.fs.gac.eq_minimum_operational_time_cps, 1e-2)
    iscale.constraint_scaling_transform(m.fs.gac.eq_operational_time, 1e-2)
    iscale.constraint_scaling_transform(m.fs.gac.eq_bed_volumes_treated, 1e-2)
    iscale.constraint_scaling_transform(
        m.fs.gac.eq_equilibrium_concentration[0, species], 1e-4
    )
    iscale.constraint_scaling_transform(m.fs.gac.eq_number_bi, 1e-2)
    # default scaling
    iscale.calculate_scaling_factors(m)
    # feed specifications
    m.fs.feed.properties[0].pressure.fix(101325)  # feed pressure [Pa]
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)  # feed temperature [K]

    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].fix(water_mol_flow)
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", species].fix(c0_mol_flow)
    m.fs.feed.initialize()
    propagate_state(m.fs.s01)

    # gac specifications
    # adsorption parameters
    m.fs.gac.freund_k.fix(20)
    m.fs.gac.freund_ninv.fix(0.8)
    m.fs.gac.ds.fix(5e-15)
    # gac particle specifications
    m.fs.gac.particle_dens_app.fix(particle_density)
    # m.fs.gac.particle_dia.fix(particle_size)
    m.fs.gac.particle_dia.fix(1 * pyunits.mm)
    # adsorber bed specifications
    m.fs.gac.ebct.fix(900)  # 15 min
    m.fs.gac.bed_voidage.fix(0.40)
    m.fs.gac.velocity_sup.fix(0.002222)  # 8 m/h
    # design spec
    conc_ratio = outlet_conc / inlet_conc
    if conc_ratio <= 0.95:
        m.fs.gac.conc_ratio_replace.fix(conc_ratio)
        m.fs.gac.conc_ratio_avg.fix(conc_ratio)
    else:
        m.fs.gac.conc_ratio_replace.fix(0.95)
        m.fs.gac.conc_ratio_avg.fix(0.95)
    # parameters, a and b parameters are not used once switching to surrogate
    m.fs.gac.a0.fix(0.8)
    m.fs.gac.a1.fix(0)
    m.fs.gac.b0.fix(0.839439)
    m.fs.gac.b1.fix(0.188966)
    m.fs.gac.b2.fix(0.648124)
    m.fs.gac.b3.fix(0.003006)
    m.fs.gac.b4.fix(0.0157697)
    m.fs.gac.shape_correction_factor.fix()

    for v in m.fs.gac.component_objects(Var, descend_into=True):
        v.domain = Reals
        v.setlb(None)
        v.setub(None)

    m.fs.gac.initialize()

    # costing specifications
    num_op = math.ceil(flow_rate * 2 / 0.04382)
    if flow_rate <= 0.43813:
        num_red = 1
    else:
        num_red = 2
    # for pressure
    m.fs.costing.gac_pressure.regen_frac.fix(0.5)
    m.fs.costing.gac_pressure.num_contactors_op.fix(num_op)
    m.fs.costing.gac_pressure.num_contactors_redundant.fix(math.ceil(num_op / 4))
    # m.fs.gac.costing.costing_blocks["steel_pressure"].regen_frac.fix(0)
    # m.fs.gac.costing.costing_blocks["steel_pressure"].num_contactors_op.fix(num_op)
    # m.fs.gac.costing.costing_blocks["steel_pressure"].num_contactors_redundant.fix(math.ceil(num_op/4))
    # for gravity
    # m.fs.costing.gac_gravity.regen_frac.fix(0)
    # m.fs.costing.gac_gravity.num_contactors_op.fix(num_op)
    # m.fs.costing.gac_gravity.num_contactors_redundant.fix(num_red)
    # m.fs.gac.costing.costing_blocks["concrete_gravity"].regen_frac.fix(0)
    # m.fs.gac.costing.costing_blocks["concrete_gravity"].num_contactors_op.fix(num_op)

    results = solver.solve(m)
    assert_optimal_termination(results)

    # presolve_init(m)
    activate_surrogate(m)
    # calculate_variable_from_constraint(m.fs.gac.min_N_St, m.fs.min_st_surrogate.pysmo_constraint["min_st"])
    # calculate_variable_from_constraint(m.fs.gac.throughput, m.fs.throughput_surrogate.pysmo_constraint["throughput"])

    m.fs.gac.initialize()

    m.fs.optimal_solve = Var(initialize=0)

    return m


def deactivate_ss_calculations(m):

    # deactivate steady state equations
    m.fs.gac.eq_ele_throughput[:].deactivate()
    m.fs.gac.eq_ele_min_operational_time[:].deactivate()
    m.fs.gac.eq_ele_conc_ratio_replace[:].deactivate()
    m.fs.gac.eq_ele_operational_time[:].deactivate()
    m.fs.gac.eq_ele_conc_ratio_term[:].deactivate()
    m.fs.gac.eq_conc_ratio_avg.deactivate()
    m.fs.gac.eq_mass_transfer_cv.deactivate()
    m.fs.gac.gac_removed.deactivate()

    # fix variables used in steady state equations
    m.fs.gac.ele_throughput[:].fix()
    m.fs.gac.ele_min_operational_time[:].fix()
    m.fs.gac.ele_conc_ratio_replace[:].fix()
    m.fs.gac.ele_operational_time[:].fix()
    m.fs.gac.ele_conc_ratio_term[:].fix()
    m.fs.gac.mass_adsorbed.fix()
    m.fs.gac.gac_removed.deactivate()
    m.fs.gac.process_flow.mass_transfer_term.fix(0)
    # pass
    #


def activate_surrogate(m):

    # deactivate empirical equations equations
    m.fs.gac.eq_min_number_st_cps.deactivate()
    m.fs.gac.eq_throughput.deactivate()
    # m.fs.gac.min_N_St.lb = 2

    # establish surrogates
    m.fs.min_st_surrogate = SurrogateBlock(concrete=True)
    m.fs.min_st_surrogate.build_model(
        min_st_surrogate,
        input_vars=[m.fs.gac.freund_ninv, m.fs.gac.N_Bi],
        output_vars=[m.fs.gac.min_N_St],
    )
    m.fs.throughput_surrogate = SurrogateBlock(concrete=True)
    m.fs.throughput_surrogate.build_model(
        throughput_surrogate,
        input_vars=[
            m.fs.gac.freund_ninv,
            m.fs.gac.N_Bi,
            m.fs.gac.conc_ratio_replace,
        ],
        output_vars=[m.fs.gac.throughput],
    )


def model_solve(m, tee=False):

    eps = 1e-8
    iterlim = 2500
    solver = get_solver()

    calculate_variable_from_constraint(
        m.fs.gac.min_N_St, m.fs.min_st_surrogate.pysmo_constraint["min_st"]
    )
    calculate_variable_from_constraint(
        m.fs.gac.throughput, m.fs.throughput_surrogate.pysmo_constraint["throughput"]
    )
    m.fs.gac.initialize()
    assert degrees_of_freedom(m) == 0

    try:
        res = solver.solve(m, tee=tee)
        assert_optimal_termination(res)
        m.fs.optimal_solve.fix(1)
        print("solver termination condition:", res.solver.termination_condition)
    except:
        try:
            # try sequential solution if equation oriented fails
            feasible_breakthrough = sequential_solve(m, eps=eps, iterlim=iterlim)
            if not feasible_breakthrough:
                # m.fs.gac.costing.costing_blocks["steel_pressure"].LCOW = 100
                # m.fs.gac.costing.costing_blocks["concrete_gravity"].LCOW = 100
                res = dummy_pyomo_solve(solver=solver)
                print("solver termination condition:", "skipped")
                pass
            else:
                res = solver.solve(m, tee=tee)
                assert_optimal_termination(res)
                m.fs.optimal_solve.fix(1)
                print("solver termination condition:", res.solver.termination_condition)
        except:
            sequential_solve(m, eps=eps, iterlim=iterlim)
            for x in istat.activated_constraints_set(m):
                residual = abs(value(x.body) - value(x.lb))
                if residual > 1e-8:
                    print(f"{x}\t{residual}")
                    print("solver termination condition:", "failed")
                    assert False
            res = dummy_pyomo_solve(solver=solver)
            print("solver termination condition:", "sequential solution")
            m.fs.optimal_solve.fix(0)

    return res


def sequential_solve(m, eps=1e-12, iterlim=1000):

    species = m.fs.properties.solute_set.at(1)
    calculate_variable_from_constraint(
        m.fs.gac.equil_conc,
        m.fs.gac.eq_equilibrium_concentration[0, species],
        eps=eps,
        iterlim=iterlim,
    )
    calculate_variable_from_constraint(
        m.fs.gac.dg, m.fs.gac.eq_dg[0, species], eps=eps, iterlim=iterlim
    )
    calculate_variable_from_constraint(
        m.fs.gac.N_Bi, m.fs.gac.eq_number_bi, eps=eps, iterlim=iterlim
    )
    calculate_variable_from_constraint(
        m.fs.gac.min_N_St,
        m.fs.min_st_surrogate.pysmo_constraint["min_st"],
        eps=eps,
        iterlim=iterlim,
    )
    calculate_variable_from_constraint(
        m.fs.gac.throughput,
        m.fs.throughput_surrogate.pysmo_constraint["throughput"],
        eps=eps,
        iterlim=iterlim,
    )
    calculate_variable_from_constraint(
        m.fs.gac.min_ebct, m.fs.gac.eq_min_ebct_cps, eps=eps, iterlim=iterlim
    )
    calculate_variable_from_constraint(
        m.fs.gac.min_residence_time,
        m.fs.gac.eq_min_residence_time_cps,
        eps=eps,
        iterlim=iterlim,
    )
    calculate_variable_from_constraint(
        m.fs.gac.min_operational_time,
        m.fs.gac.eq_minimum_operational_time_cps,
        eps=eps,
        iterlim=iterlim,
    )
    calculate_variable_from_constraint(
        m.fs.gac.operational_time,
        m.fs.gac.eq_operational_time,
        eps=eps,
        iterlim=iterlim,
    )
    if m.fs.gac.operational_time.value > 0:
        feasible_breakthrough = True
        calculate_variable_from_constraint(
            m.fs.gac.bed_volumes_treated,
            m.fs.gac.eq_bed_volumes_treated,
            eps=eps,
            iterlim=iterlim,
        )
        # calculate_variable_from_constraint(m.fs.gac.mass_adsorbed, m.fs.gac.eq_mass_adsorbed[species], eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(
            m.fs.gac.gac_usage_rate,
            m.fs.gac.eq_gac_usage_rate,
            eps=eps,
            iterlim=iterlim,
        )
        if hasattr(m.fs.gac.costing, "costing_blocks"):
            for costing_block in (
                m.fs.gac.costing.costing_blocks["steel_pressure"],
                m.fs.gac.costing.costing_blocks["concrete_gravity"],
            ):
                calculate_variable_from_constraint(
                    costing_block.contactor_cost,
                    costing_block.contactor_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.bed_mass_gac_ref,
                    costing_block.bed_mass_gac_ref_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.adsorbent_unit_cost,
                    costing_block.adsorbent_unit_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.adsorbent_cost,
                    costing_block.adsorbent_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.other_process_cost,
                    costing_block.other_process_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.capital_cost,
                    costing_block.capital_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.gac_regen_cost,
                    costing_block.gac_regen_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.gac_makeup_cost,
                    costing_block.gac_makeup_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.fixed_operating_cost,
                    costing_block.fixed_operating_cost_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
                calculate_variable_from_constraint(
                    costing_block.energy_consumption,
                    costing_block.energy_consumption_constraint,
                    eps=eps,
                    iterlim=iterlim,
                )
            # calculate_variable_from_constraint(costing_block.LCOW, costing_block.LCOW_constraint, eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(
            m.fs.costing.aggregate_capital_cost,
            m.fs.costing.aggregate_capital_cost_constraint,
            eps=eps,
            iterlim=iterlim,
        )
        calculate_variable_from_constraint(
            m.fs.costing.aggregate_fixed_operating_cost,
            m.fs.costing.aggregate_fixed_operating_cost_constraint,
            eps=eps,
            iterlim=iterlim,
        )
        calculate_variable_from_constraint(
            m.fs.costing.aggregate_flow_electricity,
            m.fs.costing.aggregate_flow_electricity_constraint,
            eps=eps,
            iterlim=iterlim,
        )
        calculate_variable_from_constraint(
            m.fs.costing.aggregate_flow_costs["electricity"],
            m.fs.costing.aggregate_flow_costs_constraint["electricity"],
            eps=eps,
            iterlim=iterlim,
        )
        calculate_variable_from_constraint(
            m.fs.costing.total_capital_cost,
            m.fs.costing.total_capital_cost_constraint,
            eps=eps,
            iterlim=iterlim,
        )
        calculate_variable_from_constraint(
            m.fs.costing.total_operating_cost,
            m.fs.costing.total_operating_cost_constraint,
            eps=eps,
            iterlim=iterlim,
        )
    else:
        feasible_breakthrough = False

    return feasible_breakthrough


def custom_add_LCOW(fs_costing, unit_costing, flow_rate, name="LCOW"):

    LCOW = Var(
        initialize=0.5,
        doc=f"Levelized Cost of Water based on flow {flow_rate.name}",
        units=fs_costing.base_currency / pyunits.m**3,
    )
    unit_costing.add_component(name, LCOW)

    electricity_flow_cost = pyunits.convert(
        unit_costing.energy_consumption * fs_costing.electricity_cost,
        to_units=fs_costing.base_currency / fs_costing.base_period,
    )
    total_capital_cost = pyunits.convert(
        fs_costing.total_investment_factor * unit_costing.capital_cost,
        to_units=fs_costing.base_currency,
    )
    total_operating_cost = pyunits.convert(
        (fs_costing.maintenance_labor_chemical_factor * total_capital_cost)
        + unit_costing.fixed_operating_cost
        + electricity_flow_cost * fs_costing.utilization_factor,
        to_units=fs_costing.base_currency / fs_costing.base_period,
    )
    LCOW_constraint = Constraint(
        expr=LCOW
        == (
            total_capital_cost * fs_costing.capital_recovery_factor
            + total_operating_cost
        )
        / (
            pyunits.convert(flow_rate, to_units=pyunits.m**3 / fs_costing.base_period)
            * fs_costing.utilization_factor
        ),
        doc=f"Constraint for Levelized Cost of Water based on flow {flow_rate.name}",
    )
    unit_costing.add_component(name + "_constraint", LCOW_constraint)


def dummy_pyomo_solve(solver=None):

    dummy_model = ConcreteModel()
    dummy_model.x = Var(initialize=1)
    dummy_model.eq = Constraint(expr=dummy_model.x == 1)
    res = solver.solve(dummy_model, tee=False)

    return res


def gather_case_study_results():

    gac_input = defaultdict(list)
    gac_input["curve_id"] = []
    gac_input["epa_lim"] = []

    ix_input = pd.read_csv(
        "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/ix_case_study_sensitivity_inputs.csv"
    )

    with h5py.File(
        "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_case_study_regression_results.hdf5",
        "r",
    ) as h5_read:
        for num in h5_read.keys():
            curve_res = {}
            for attr in h5_read[num].keys():
                if attr == "species":
                    species = h5_read[num][attr][()].decode()
                if attr == "source":
                    source = h5_read[num][attr][()].decode()
                if attr == "media":
                    media = h5_read[num][attr][()].decode()
                if isinstance(h5_read[num][attr][()], bytes):
                    gac_input[attr].extend([h5_read[num][attr][()].decode()])
                elif isinstance(h5_read[num][attr][()], float):
                    gac_input[attr].extend([h5_read[num][attr][()]])
                elif isinstance(h5_read[num][attr][()], int):
                    gac_input[attr].extend([h5_read[num][attr][()]])
                elif isinstance(h5_read[num][attr][()], np.ndarray):
                    curve_res[attr] = h5_read[num][attr][()]
                    pass
                else:
                    print(f"unhandled type: {type(h5_read[num][attr][()])} for {attr}")

            ################################################
            curve_res_df = pd.DataFrame({k: pd.Series(v) for k, v in curve_res.items()})
            curve_res_df["curve_id"] = num
            curve_res_df["species"] = species
            curve_res_df["source"] = source
            curve_res_df["media"] = media
            curve_res_df["term_cond_resolve"] = curve_res_df["term_cond_resolve"].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
            if species in ix_input["target_component"].values:
                # Get C0 and Cb from IX input file
                ix_input_row = ix_input[ix_input["target_component"] == species].iloc[0]
                c0 = ix_input_row["c0"]
                cb = ix_input_row["cb"]
            else:
                c0 = np.nan
                cb = np.nan

            curve_res_df["c0"] = c0
            curve_res_df["cb"] = cb

            curve_res_df.to_csv(
                f"/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/kurby/gac/gac_curve{num}_{species}_{source}_{media}.csv",
                index=False,
            )

            ################################################
            fig, ax = plt.subplots()
            opt = curve_res_df[curve_res_df.term_cond_resolve == "optimal"].copy()
            non_opt = curve_res_df[curve_res_df.term_cond_resolve != "optimal"].copy()
            ax.scatter(
                opt.bed_volume_filter,
                opt.effluent_conc_ratio_filter,
                marker=".",
                label="Data",
            )
            ax.scatter(
                non_opt.bed_volume_filter,
                non_opt.effluent_conc_ratio_filter,
                marker="x",
                color="k",
            )

            ax.plot(
                opt.bed_volume_resolve,
                opt.effluent_conc_ratio_resolve,
                color="red",
                label="WaterTAP",
                linewidth=2,
            )

            ax.set_xlabel("Bed Volumes Treated")
            ax.set_ylabel("C/C0")
            ax.set_title(f"Curve {num}: {species} - {source} - {media}")
            ax.legend()

            fig.tight_layout()
            fig.savefig(
                f"/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/kurby/gac/gac_curve{num}_{species}_{source}_{media}.png",
                bbox_inches="tight",
            )
            plt.close()
            ################################################
            gac_input["curve_id"].append(num)
            gac_input["epa_lim"].append(epa_limits[species])
            gac_input["c0"].append(c0)
            gac_input["cb"].append(cb)

    df = pd.DataFrame(gac_input)
    df.to_csv(
        "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_case_study_sensitivity_inputs.csv",
        index=False,
    )


def solve(blk, solver=None, tee=False, check_termination=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    if check_termination:
        assert_optimal_termination(results)
    return results


def build_and_solve(data, sweep=None, gac_config={}, **kwargs):

    m = model_build(species=data.species, gac_config=gac_config, **kwargs)
    model_init(
        m,
        solver=solver,
        flow_rate=0.043813,
        inlet_conc=data.c0,
        outlet_conc=data.cb,
    )

    m.fs.gac.freund_k.fix(data.freund_k)
    m.fs.gac.freund_ninv.fix(data.freund_ninv)
    m.fs.gac.ds.fix(data.ds)
    res = model_solve(m, tee=False)

    if sweep == "adsorbent_unit_cost":
        m.fs.gac.costing.adsorbent_cost_constraint.deactivate()

    return m


def build_sweep_params(m, num_samples=5, sweep="ebct", rel_frac=0.25):
    sweep_params = {}

    # EPA-WBS GAC documentation:
    # https://www.epa.gov/system/files/documents/2022-03/gac-documentation-.pdf_0.pdf

    # GAC EBCT Sensitivity
    # EPA-WBS: PFAS removal use total EBCTs between 7.6 and 26 minutes
    ebct_lb = 7.6 * 60  # seconds
    ebct_ub = 26 * 60  # seconds

    # GAC Loading Rate Sensitivity
    # EPA-WBS: Loading rate between 0.5 and 10 gpm/ft2
    loading_rate_lb = value(
        pyunits.convert(
            0.5 * pyunits.gallon / (pyunits.minute * pyunits.ft**2),
            to_units=pyunits.m / pyunits.s,
        )
    )  # gpm/ft2
    loading_rate_ub = value(
        pyunits.convert(
            10 * pyunits.gallon / (pyunits.minute * pyunits.ft**2),
            to_units=pyunits.m / pyunits.s,
        )
    )  # gpm/ft2

    # GAC Regeneration Fraction Sensitivity
    regen_frac_lb = 0.1
    regen_frac_ub = 0.9

    # GAC Adsorbent Unit Cost Sensitivity
    adsorbent_cost_base = value(m.fs.costing.gac_pressure.makeup_unit_cost)
    adsorbent_unit_cost_lb = adsorbent_cost_base * (1 - rel_frac)
    adsorbent_unit_cost_ub = adsorbent_cost_base * (1 + rel_frac)

    # GAC Regen Unit Cost Sensitivity
    # EPA-WBS: Off-site thermal regeneration cost between 1.21-2.03 $/lb GAC
    # adsorbent_cost_base = value(m.fs.costing.gac_pressure.makeup_unit_cost)
    # adsorbent_unit_cost_lb = adsorbent_cost_base * (1 - rel_frac)
    # adsorbent_unit_cost_ub = adsorbent_cost_base * (1 + rel_frac)

    if sweep == "ebct":
        sweep_params["ebct"] = LinearSample(
            m.fs.gac.ebct, ebct_lb, ebct_ub, num_samples
        )

    if sweep == "loading_rate":
        sweep_params["loading_rate"] = LinearSample(
            m.fs.gac.velocity_sup,
            loading_rate_lb,
            loading_rate_ub,
            num_samples,
        )

    if sweep == "regen_frac":
        sweep_params["regen_frac"] = LinearSample(
            m.fs.costing.gac_pressure.regen_frac,
            regen_frac_lb,
            regen_frac_ub,
            num_samples,
        )

    if sweep == "makeup_unit_cost":
        sweep_params["makeup_unit_cost"] = LinearSample(
            m.fs.costing.gac_pressure.makeup_unit_cost,
            adsorbent_unit_cost_lb,
            adsorbent_unit_cost_ub,
            num_samples,
        )

    if sweep == "regen_frac+makeup_unit_cost":
        sweep_params["regen_frac"] = LinearSample(
            m.fs.costing.gac_pressure.regen_frac,
            regen_frac_lb,
            regen_frac_ub,
            num_samples,
        )

        sweep_params["makeup_unit_cost"] = LinearSample(
            m.fs.costing.gac_pressure.makeup_unit_cost,
            adsorbent_unit_cost_lb,
            adsorbent_unit_cost_ub,
            num_samples,
        )
    return sweep_params


def build_outputs(m):
    outputs = {}

    cols = [
        "fs.optimal_solve",
        "fs.costing.LCOW",
        "fs.gac.costing.costing_blocks[steel_pressure].LCOW_pressure",
        "fs.gac.costing.costing_blocks[concrete_gravity].LCOW_gravity",
        "fs.gac.freund_k",
        "fs.gac.freund_ninv",
        "fs.gac.ds",
        "fs.gac.ebct",
        "fs.gac.velocity_sup",
        "fs.gac.kf",
        "fs.gac.equil_conc",
        "fs.gac.dg",
        "fs.gac.N_Bi",
        "fs.gac.velocity_int",
        "fs.gac.bed_voidage",
        "fs.gac.bed_length",
        "fs.gac.bed_diameter",
        "fs.gac.bed_area",
        "fs.gac.bed_volume",
        "fs.gac.residence_time",
        "fs.gac.bed_mass_gac",
        "fs.gac.particle_dens_app",
        "fs.gac.particle_dens_bulk",
        "fs.gac.particle_dia",
        "fs.gac.min_N_St",
        "fs.gac.min_ebct",
        "fs.gac.throughput",
        "fs.gac.min_residence_time",
        "fs.gac.min_operational_time",
        "fs.gac.conc_ratio_replace",
        "fs.gac.operational_time",
        "fs.gac.bed_volumes_treated",
        "fs.gac.ele_throughput",
        "fs.gac.ele_min_operational_time",
        "fs.gac.ele_conc_ratio_replace",
        "fs.gac.ele_operational_time",
        "fs.gac.ele_conc_ratio_term",
        "fs.gac.conc_ratio_avg",
        "fs.gac.mass_adsorbed",
        "fs.gac.gac_usage_rate",
        "fs.gac.regen_bed_volumes",
        "fs.gac.regen_soln_density",
        "fs.gac.regen_tank_vol_factor",
        "fs.gac.regeneration_time",
        "fs.gac.service_to_regen_flow_ratio",
        "fs.gac.regen_flow_vol",
        "fs.gac.regen_flow_mass",
        "fs.gac.regen_soln_flow_vol",
        "fs.gac.regen_time",
        "fs.gac.regen_tank_vol",
        "fs.gac.N_Re",
        "fs.gac.N_Sc",
        "fs.gac.shape_correction_factor",
        "fs.gac.process_flow.mass_transfer_term",
        "fs.gac.costing.direct_capital_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].capital_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].contactor_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].bed_mass_gac_ref",
        "fs.gac.costing.costing_blocks[steel_pressure].adsorbent_unit_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].adsorbent_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].other_process_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].energy_consumption",
        "fs.gac.costing.costing_blocks[steel_pressure].cost_factor",
        "fs.gac.costing.costing_blocks[steel_pressure].direct_capital_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].fixed_operating_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].gac_regen_cost",
        "fs.gac.costing.costing_blocks[steel_pressure].gac_makeup_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].capital_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].contactor_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].bed_mass_gac_ref",
        "fs.gac.costing.costing_blocks[concrete_gravity].adsorbent_unit_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].adsorbent_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].other_process_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].energy_consumption",
        "fs.gac.costing.costing_blocks[concrete_gravity].cost_factor",
        "fs.gac.costing.costing_blocks[concrete_gravity].direct_capital_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].fixed_operating_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].gac_regen_cost",
        "fs.gac.costing.costing_blocks[concrete_gravity].gac_makeup_cost",
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
        "fs.costing.aggregate_capital_cost",
        "fs.costing.aggregate_fixed_operating_cost",
        "fs.costing.aggregate_variable_operating_cost",
        "fs.costing.aggregate_flow_electricity",
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
        "fs.costing.gac_pressure.num_contactors_op",
        "fs.costing.gac_pressure.num_contactors_redundant",
        "fs.costing.gac_pressure.regen_frac",
        "fs.costing.gac_pressure.bed_mass_max_ref",
        "fs.costing.gac_pressure.contactor_cost_coeff",
        "fs.costing.gac_pressure.adsorbent_unit_cost_coeff",
        "fs.costing.gac_pressure.other_cost_param",
        "fs.costing.gac_pressure.regen_unit_cost",
        "fs.costing.gac_pressure.makeup_unit_cost",
        "fs.costing.gac_pressure.energy_consumption_coeff",
        "fs.costing.gac_gravity.num_contactors_op",
        "fs.costing.gac_gravity.num_contactors_redundant",
        "fs.costing.gac_gravity.regen_frac",
        "fs.costing.gac_gravity.bed_mass_max_ref",
        "fs.costing.gac_gravity.contactor_cost_coeff",
        "fs.costing.gac_gravity.adsorbent_unit_cost_coeff",
        "fs.costing.gac_gravity.other_cost_param",
        "fs.costing.gac_gravity.regen_unit_cost",
        "fs.costing.gac_gravity.makeup_unit_cost",
        "fs.costing.gac_gravity.energy_consumption_coeff",
        "fs.gac.costing.onsite_thermal_regen_energy_consumption",
        "fs.gac.costing.onsite_thermal_regen_capital_cost",
        "fs.costing.aggregate_flow_costs[electricity]",
        "fs.costing.aggregate_flow_costs[ethanol]",
        "fs.costing.aggregate_flow_costs[methanol]",
        "fs.costing.aggregate_flow_costs[acetone]",
        "fs.costing.aggregate_flow_costs[NaCl]",
        "fs.costing.aggregate_flow_costs[NaOH]",
    ]

    for c in cols:
        comp = m.find_component(c)
        if comp is not None:
            if comp.is_indexed():
                for i, cc in comp.items():
                    outputs[c.split(".")[-1]] = cc
            else:
                outputs[c.split(".")[-1]] = comp

    return outputs


if __name__ == "__main__":

    # gather_case_study_results()

    inputs_file = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_case_study_sensitivity_inputs.csv"
    gac_inputs = pd.read_csv(inputs_file)
    path = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities"
    num_samples = 20
    num_procs = 6
    sweep = "loading_rate"
    # sweep = "adsorbent_unit_cost"
    cs_nums = [14, 21, 0]
    # cs_nums = [0]

    # ebct_lb = 7.6 * 60  # seconds
    # ebct_ub = 26 * 60  # seconds
    # ebcts = np.linspace(ebct_lb, ebct_ub, num_samples)
    data = gac_inputs[gac_inputs.curve_id == cs_nums[0]].iloc[0]

    # m = build_and_solve(data, sweep=None)
    # m.fs.costing.LCOW.display()
    # m.fs.gac.costing.costing_blocks["steel_pressure"].LCOW_pressure.display()
    # m.fs.gac.costing.costing_blocks["concrete_gravity"].LCOW_gravity.display()

    from analysis.run_ix_sens import get_regen_soln_info

    solvent_vv = {"ethanol": 0.70}
    solutes_molar = {"NaCl": 0.17}
    density, mass_fractions = get_regen_soln_info(
        solvent_vv=solvent_vv,
        solutes_molar=solutes_molar,
    )
    gac_config = {}
    gac_config["regenerant"] = "single_use"
    gac_config["regen_composition"] = mass_fractions
    gac_config["regen_soln_density"] = density
    # m = build_and_solve(data, sweep=None, gac_config=gac_config)

    # m.fs.costing.LCOW.display()
    # m.fs.costing.total_operating_cost.display()
    # m.fs.gac.costing.gac_makeup_cost.display()
    # m.fs.gac.costing.fixed_operating_cost.display()
    # m.fs.costing.gac_pressure.regen_frac.display()
    # m = build_and_solve(data, sweep=None, regenerant="single_use", regen_composition={}, regen_soln_density=None)
    # m.fs.costing.LCOW.display()
    # m.fs.costing.total_operating_cost.display()
    # m.fs.gac.costing.gac_makeup_cost.display()
    # m.fs.gac.costing.fixed_operating_cost.display()
    # m.fs.costing.gac_pressure.regen_frac.display()
    # m.fs.costing.total_operating_cost.display()
    # m.fs.gac.costing.costing_blocks["steel_pressure"].LCOW_pressure.display()
    # m.fs.gac.costing.costing_blocks["concrete_gravity"].LCOW_gravity.display()
    # # for x in ebcts:
    # #     m.fs.gac.ebct.fix(x)
    # #     # m.fs.costing.initialize()
    # #     print(f"dof =    {degrees_of_freedom(m)}")
    # #     res = model_solve(m, tee=False)
    # #     print(f"Adsorbent Unit Cost: {x}")
    # #     print(f"LCOW: {value(m.fs.costing.LCOW)}")
    # #     print("")

    # # for sweep in ["ebct", "loading_rate", "regen_frac"]:

    for sweep in ["regen_frac+makeup_unit_cost"]:
        for cs in cs_nums:
            data = gac_inputs[gac_inputs.curve_id == cs].iloc[0]
            save_file = f"{path}/results/gac/gac_pfas_{sweep}_sensitivity-{data.source}_curve{data.curve_id}_{data.species}_{data.media}_{gac_config['regenerant']}.h5"
            results_array, results_dict = parameter_sweep(
                build_model=build_and_solve,
                build_model_kwargs={
                    "data": data,
                    "sweep": None,
                    "gac_config": gac_config,
                },
                build_sweep_params=build_sweep_params,
                build_sweep_params_kwargs={"num_samples": num_samples, "sweep": sweep},
                build_outputs=build_outputs,
                build_outputs_kwargs={},
                h5_results_file_name=save_file,
                optimize_function=solve,
                num_samples=num_samples,
                csv_results_file_name=save_file.replace(".h5", ".csv"),
                number_of_subprocesses=num_procs,
            )

            df = pd.read_csv(save_file.replace(".h5", ".csv"))
            # Get results for base case
            _m = build_and_solve(data)
            o = build_outputs(_m)
            d = {}

            for k, v in o.items():
                d[k] = value(v)
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
            if gac_config["regenerant"] == "custom":
                # for k, v in mass_fractions.items():
                #     df[f"regen_solute"] = k
                #     df[f"regen_solute_{k}_mass_frac"] = v
                #     df[f"regen_solute_{k}_conc_molar"] = solutes_molar[k]
                # for k, v in solu
                for k, v in solvent_vv.items():
                    df[f"regen_solvent"] = k
                    df[f"regen_solvent_{k}_vol_frac"] = v
                    df[f"regen_solution_mass_frac_{k}"] = mass_fractions[k]
                for k, v in solutes_molar.items():
                    df[f"regen_solution"] = k
                    df[f"regen_solute_{k}_conc_molar"] = v
                    df[f"regen_solution_mass_frac_{k}"] = mass_fractions[k]

            df["regenerant"] = gac_config["regenerant"]
            df.to_csv(save_file.replace(".h5", ".csv"), index=False)
            #     df
            break
