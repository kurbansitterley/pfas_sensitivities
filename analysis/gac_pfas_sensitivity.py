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

import pyomo.environ as pyo
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
from watertap.unit_models.gac import GAC
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.gac import cost_gac, ContactorType
from watertap.costing.multiple_choice_costing_block import MultiUnitModelCostingBlock
from watertap.core.util.initialization import assert_degrees_of_freedom

# from models.gac_cphsdm import GAC
# from analysisWaterTAP.utils.model_state_tool import modelStateStorage

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

mw_water = 0.018 * pyo.units.kg / pyo.units.mol
rho = 1000 * pyo.units.kg / pyo.units.m**3

solver = get_solver()


def model_build(
    species="PFAS",
    **kwargs,
):

    pfas_mw = species_properties[species][
        "mw"
    ]  # changing mw after the fact changes concentration
    pfas_mv = species_properties[species]["molar_volume"]

    # create m objects
    m = pyo.ConcreteModel()
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

    m.fs.gac = GAC(
        property_package=m.fs.properties,
        film_transfer_coefficient_type="calculated",
        surface_diffusion_coefficient_type="fixed",
    )

    # streams
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.gac.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # model customization
    deactivate_ss_calculations(m)

    # build costing blocks
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2021
    m.fs.costing.electricity_cost.fix(0.0822)
    m.fs.gac.costing = MultiUnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_blocks={
            "steel_pressure": {
                "costing_method": cost_gac,
                "costing_method_arguments": {"contactor_type": "pressure"},
            },
            "concrete_gravity": {
                "costing_method": cost_gac,
                "costing_method_arguments": {"contactor_type": "gravity"},
            },
        },
    )

    # add flowsheet level blocks
    m.fs.costing.cost_process()
    treated_flow = m.fs.gac.process_flow.properties_in[0].flow_vol_phase["Liq"]
    m.fs.costing.add_LCOW(flow_rate=treated_flow)

    custom_add_LCOW(
        m.fs.costing,
        m.fs.gac.costing.costing_blocks["steel_pressure"],
        treated_flow,
    )
    custom_add_LCOW(
        m.fs.costing,
        m.fs.gac.costing.costing_blocks["concrete_gravity"],
        treated_flow,
    )

    m.fs.obj = pyo.Objective(expr=m.fs.costing.LCOW, sense=pyo.minimize)

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

    mw = species_properties[species]["mw"] * pyo.units.kg / pyo.units.mol
    c0 = inlet_conc * pyo.units.kg / pyo.units.m**3
    q_in = flow_rate * pyo.units.m**3 / pyo.units.s

    c0_mol_flow = pyo.units.convert(
        (c0 * q_in) / mw, to_units=pyo.units.mol / pyo.units.s
    )
    c0_mol_flow_sf = 1 / value(c0_mol_flow)

    water_mol_flow = pyo.units.convert(
        (q_in * rho) / mw_water,
        to_units=pyo.units.mol / pyo.units.s,
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
    m.fs.gac.particle_dia.fix(1 * pyo.units.mm)
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

    for v in m.fs.gac.component_objects(pyo.Var, descend_into=True):
        v.domain = pyo.Reals
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
    m.fs.costing.gac_pressure.regen_frac.fix(0)
    m.fs.costing.gac_pressure.num_contactors_op.fix(num_op)
    m.fs.costing.gac_pressure.num_contactors_redundant.fix(math.ceil(num_op / 4))
    # m.fs.gac.costing.costing_blocks["steel_pressure"].regen_frac.fix(0)
    # m.fs.gac.costing.costing_blocks["steel_pressure"].num_contactors_op.fix(num_op)
    # m.fs.gac.costing.costing_blocks["steel_pressure"].num_contactors_redundant.fix(math.ceil(num_op/4))
    # for gravity
    m.fs.costing.gac_gravity.regen_frac.fix(0)
    m.fs.costing.gac_gravity.num_contactors_op.fix(num_op)
    m.fs.costing.gac_gravity.num_contactors_redundant.fix(num_red)
    # m.fs.gac.costing.costing_blocks["concrete_gravity"].regen_frac.fix(0)
    # m.fs.gac.costing.costing_blocks["concrete_gravity"].num_contactors_op.fix(num_op)

    results = solver.solve(m)
    pyo.assert_optimal_termination(results)

    # presolve_init(m)
    activate_surrogate(m)
    # calculate_variable_from_constraint(m.fs.gac.min_N_St, m.fs.min_st_surrogate.pysmo_constraint["min_st"])
    # calculate_variable_from_constraint(m.fs.gac.throughput, m.fs.throughput_surrogate.pysmo_constraint["throughput"])

    m.fs.gac.initialize()

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
        pyo.assert_optimal_termination(res)
        print("solver termination condition:", res.solver.termination_condition)
    except:
        try:
            # try sequential solution if equation oriented fails
            feasible_breakthrough = sequential_solve(m, eps=eps, iterlim=iterlim)
            if not feasible_breakthrough:
                m.fs.gac.costing.costing_blocks["steel_pressure"].LCOW = 100
                m.fs.gac.costing.costing_blocks["concrete_gravity"].LCOW = 100
                res = dummy_pyomo_solve(solver=solver)
                print("solver termination condition:", "skipped")
                pass
            else:
                res = solver.solve(m, tee=tee)
                pyo.assert_optimal_termination(res)
                print("solver termination condition:", res.solver.termination_condition)
        except:
            sequential_solve(m, eps=eps, iterlim=iterlim)
            for x in istat.activated_constraints_set(m):
                residual = abs(pyo.value(x.body) - pyo.value(x.lb))
                if residual > 1e-8:
                    print(f"{x}\t{residual}")
                    print("solver termination condition:", "failed")
                    assert False
            res = dummy_pyomo_solve(solver=solver)
            print("solver termination condition:", "sequential solution")

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

    LCOW = pyo.Var(
        initialize=0.5,
        doc=f"Levelized Cost of Water based on flow {flow_rate.name}",
        units=fs_costing.base_currency / pyo.units.m**3,
    )
    unit_costing.add_component(name, LCOW)

    electricity_flow_cost = pyo.units.convert(
        unit_costing.energy_consumption * fs_costing.electricity_cost,
        to_units=fs_costing.base_currency / fs_costing.base_period,
    )
    total_capital_cost = pyo.units.convert(
        fs_costing.total_investment_factor * unit_costing.capital_cost,
        to_units=fs_costing.base_currency,
    )
    total_operating_cost = pyo.units.convert(
        (fs_costing.maintenance_labor_chemical_factor * total_capital_cost)
        + unit_costing.fixed_operating_cost
        + electricity_flow_cost * fs_costing.utilization_factor,
        to_units=fs_costing.base_currency / fs_costing.base_period,
    )
    LCOW_constraint = pyo.Constraint(
        expr=LCOW
        == (
            total_capital_cost * fs_costing.capital_recovery_factor
            + total_operating_cost
        )
        / (
            pyo.units.convert(
                flow_rate, to_units=pyo.units.m**3 / fs_costing.base_period
            )
            * fs_costing.utilization_factor
        ),
        doc=f"Constraint for Levelized Cost of Water based on flow {flow_rate.name}",
    )
    unit_costing.add_component(name + "_constraint", LCOW_constraint)


def dummy_pyomo_solve(solver=None):

    dummy_model = pyo.ConcreteModel()
    dummy_model.x = pyo.Var(initialize=1)
    dummy_model.eq = pyo.Constraint(expr=dummy_model.x == 1)
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


def build_and_solve(data):
    
    m = model_build(species=data.species)
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

    return m

if __name__ == "__main__":
    # pass
    gac_inputs = pd.read_csv(
        "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_case_study_sensitivity_inputs.csv"
    )
    # data = gac_inputs.iloc[0]
    data = gac_inputs[gac_inputs.curve_id == 0].iloc[0]

    m = build_and_solve(data)
    