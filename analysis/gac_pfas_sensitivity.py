import os
import yaml
import math
import h5py
import pandas as pd
import pyomo.environ as pyo
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
import idaes.core.util.model_statistics as istat

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# import analysisWaterTAP.utils.step_optimize_tool as stepTool
# import analysisWaterTAP.analysis_scripts.pfas_treatment_analysis.pfas_gac_analysis.pfas_gac_regression.statistics_analysis as custom_stats

from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.check_units import assert_units_consistent
from idaes.core import (
    FlowsheetBlock,
    UnitModelCostingBlock,
)
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
from watertap.unit_models.gac import GAC
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.gac import cost_gac, ContactorType
from watertap.costing.multiple_choice_costing_block import MultiUnitModelCostingBlock
from watertap.core.util.initialization import assert_degrees_of_freedom

# from analysisWaterTAP.utils.model_state_tool import modelStateStorage

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

cwd = os.getcwd()

solver = get_solver()


def run_cases():

    cases = {}
    with h5py.File(cwd + "/case_studies/case_study_regression_results.hdf5", "r") as h5_read:
        for num in h5_read.keys():
            cases[num] = {}
            for attr in h5_read[num].keys():
                cases[num][attr] = h5_read[num][attr][()]

    designs = pd.read_csv(cwd + "/case_studies/cases_design.csv")

    # epa regulation limits
    assumed_lim = 10
    epa_limits = {
        'PFBA': assumed_lim,
        'PFPeA': assumed_lim,
        'PFHxA': assumed_lim,
        'PFHpA': assumed_lim,
        'PFOA': 4,
        'PFNA': 10,
        'PFDA': assumed_lim,
        'PFUnDA': assumed_lim,
        'PFDoDA': assumed_lim,
        'PFTrDA': assumed_lim,
        'PFTeDA': assumed_lim,
        'FOSA': assumed_lim,
        'NMeFOSAA': assumed_lim,
        'NEtFOSAA': assumed_lim,
        'PFBS': 100,  # 2000
        'PFPeS': assumed_lim,
        'PFHxS': 10,
        'PFOS': 4,
        'HFPO-DA': 10,
        'ADONA': assumed_lim,
        '9Cl-PF3ONS': assumed_lim,
        '11Cl-PF3OUdS': assumed_lim,
    }

    results = pd.DataFrame()
    for num in cases.keys():

        source = cases[num]["source"].decode()
        species = cases[num]["species"].decode()
        media = cases[num]["media"].decode()
        print(f"Starting case: {num} - {source} - {media} - {species}")

        # pull information from case
        source_lookup = designs.loc[designs["Source"] == source]
        media_lookup = source_lookup.loc[source_lookup["Media"] == media]
        design_data = designs.iloc[media_lookup.index[0]]
        particle_density = design_data["Published Adsorbent Density (g/cm3)"] * 1000
        particle_size = design_data["Mean Particle Size (cm)"] / 100

        # build
        m = model_build(
            species=species,
            particle_density=particle_density,
            particle_size=particle_size,
        )

        # # init
        model_init(
            m,
            solver=solver,
            flow_rate=0.43813,
            inlet_conc=cases[num]["inlet_concentration"],
            outlet_conc=epa_limits[species],
        )

        # solve
        param = [cases[num]["freund_k"], cases[num]["freund_ninv"], cases[num]["ds"]]
        res = model_solve(m, param, solver=solver, tee=False)

        if m.fs.gac.costing.costing_blocks["steel_pressure"].LCOW.value < m.fs.gac.costing.costing_blocks["concrete_gravity"].LCOW.value:
            costing_method = "steel_pressure"
        else:
            costing_method = "concrete_gravity"

        # ---------------------------------------------------------------------
        # save
        unit_costing = m.fs.gac.costing.costing_blocks[costing_method]
        fs_costing = m.fs.costing
        electricity_flow_cost = pyo.units.convert(unit_costing.energy_consumption * fs_costing.electricity_cost * fs_costing.utilization_factor, to_units=fs_costing.base_currency / fs_costing.base_period)
        results_row = pd.DataFrame({
            "source": [cases[num]["source"].decode()],
            "species": [species],
            "media": [media],
            "inlet_concentration": [cases[num]["inlet_concentration"]*1e9],
            "outlet_concentration": [epa_limits[species]],
            "breakthrough_time": [m.fs.gac.operational_time.value],
            "energy_consumption": [unit_costing.energy_consumption.value],
            "costing_method": [costing_method],
            "contactor_cost": [pyo.value(pyo.units.convert(unit_costing.contactor_cost, to_units=fs_costing.base_currency))],
            "adsorbent_cost": [pyo.value(pyo.units.convert(unit_costing.adsorbent_cost, to_units=fs_costing.base_currency))],
            "other_process_cost": [pyo.value(pyo.units.convert(unit_costing.other_process_cost, to_units=fs_costing.base_currency))],
            "TIC": [fs_costing.TIC.value],
            "factor_total_investment": [fs_costing.factor_total_investment.value],
            "investment_cost": [pyo.value(pyo.units.convert((fs_costing.factor_total_investment * unit_costing.capital_cost) - (unit_costing.contactor_cost+unit_costing.adsorbent_cost+unit_costing.other_process_cost), to_units=fs_costing.base_currency))],
            "electricity_flow_cost": [pyo.value(electricity_flow_cost)],
            "factor_maintenance_labor_chemical": [fs_costing.factor_maintenance_labor_chemical.value],
            "adsorbent_replacement_cost": [pyo.value(pyo.units.convert(unit_costing.fixed_operating_cost, to_units=fs_costing.base_currency / fs_costing.base_period))],
            "utilization_factor": [fs_costing.utilization_factor.value],
            "factor_capital_annualization": [fs_costing.factor_capital_annualization.value],
            "maintenance_labor_chemical": [pyo.value(pyo.units.convert((fs_costing.factor_maintenance_labor_chemical * (fs_costing.factor_total_investment * unit_costing.capital_cost)), to_units=fs_costing.base_currency / fs_costing.base_period))],
            "total_operating_cost": [pyo.value(pyo.units.convert((fs_costing.factor_maintenance_labor_chemical * (fs_costing.factor_total_investment * unit_costing.capital_cost)) + unit_costing.fixed_operating_cost + electricity_flow_cost, to_units=fs_costing.base_currency / fs_costing.base_period))],
            "LCOW": [pyo.value(pyo.units.convert(unit_costing.LCOW, to_units=fs_costing.base_currency / pyo.units.meter**3))],
            "annual_flow_rate": [pyo.value(pyo.units.convert(m.fs.gac.process_flow.properties_out[0].flow_vol, to_units=pyo.units.m**3 / fs_costing.base_period) * fs_costing.utilization_factor)],
        })
        results = pd.concat([results, results_row], ignore_index=True)

    results.to_csv(cwd + "/case_studies/case_study_simulation_results.csv")


def model_build(
        species="PFAS",
        **kwargs,
):

    # load surrogate objects
    global min_st_surrogate, throughput_surrogate
    min_st_surrogate = PysmoSurrogate.load_from_file(cwd + "/gac_surrogate_training/trained_surrogate_models/min_st_pysmo_surr_linear.json")
    throughput_surrogate = PysmoSurrogate.load_from_file(cwd + "/gac_surrogate_training/trained_surrogate_models/throughput_pysmo_surr_linear.json")

    # lookup table for property data of sweep species
    with open("pfas_properties.yaml", "r") as f:
        species_properties = yaml.load(f, Loader=yaml.FullLoader)
    pfas_mw = species_properties[species]["mw"]  # changing mw after the fact changes concentration
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
        }
    )

    # add flowsheet level blocks
    m.fs.costing.cost_process()
    treated_flow = m.fs.gac.process_flow.properties_out[0].flow_vol
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

    return m


def model_init(
        m,
        solver=None,
        flow_rate=0.043813,
        inlet_conc=10e-9,
        outlet_conc=9e-9,
        particle_density=500,
        particle_size=0.00065,
        **kwargs,
):

    # touch properties and default scaling
    species = m.fs.properties.solute_set.at(1)
    water_sf = 10**-math.ceil(math.log10(abs(flow_rate*1000/m.fs.properties.mw_comp["H2O"].value)))
    pfas_sf = 10**-math.ceil(math.log10(abs(flow_rate*inlet_conc/m.fs.properties.mw_comp[species].value)))
    m.fs.properties.set_default_scaling("flow_mol_phase_comp", water_sf, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mol_phase_comp", pfas_sf, index=("Liq", species))
    m.fs.feed.properties[0].conc_mass_phase_comp
    m.fs.feed.properties[0].flow_vol_phase["Liq"]
    model_scale(m)

    # feed specifications
    m.fs.feed.properties[0].pressure.fix(101325)  # feed pressure [Pa]
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)  # feed temperature [K]
    m.fs.feed.properties.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): flow_rate,
            ("conc_mass_phase_comp", ("Liq", species)): inlet_conc,
        },
        hold_state=True,  # fixes the calculated component mass flow rates
    )

    # gac specifications
    # adsorption parameters
    m.fs.gac.freund_k.fix(20)
    m.fs.gac.freund_ninv.fix(0.8)
    m.fs.gac.ds.fix(5e-15)
    # gac particle specifications
    m.fs.gac.particle_dens_app.fix(particle_density)
    m.fs.gac.particle_dia.fix(particle_size)
    # adsorber bed specifications
    m.fs.gac.ebct.fix(900)  # 15 min
    m.fs.gac.bed_voidage.fix(0.40)
    m.fs.gac.velocity_sup.fix(0.002222)  # 8 m/h
    # design spec
    conc_ratio = outlet_conc/inlet_conc
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

    # costing specifications
    num_op = math.ceil(flow_rate*2/0.04382)
    # for pressure
    m.fs.gac.costing.costing_blocks["steel_pressure"].regen_frac.fix(0)
    m.fs.gac.costing.costing_blocks["steel_pressure"].num_contactors_op.fix(num_op)
    m.fs.gac.costing.costing_blocks["steel_pressure"].num_contactors_redundant.fix(math.ceil(num_op/4))
    # for gravity
    m.fs.gac.costing.costing_blocks["concrete_gravity"].regen_frac.fix(0)
    m.fs.gac.costing.costing_blocks["concrete_gravity"].num_contactors_op.fix(num_op)
    if flow_rate <= 0.43813:
        num_red = 1
    else:
        num_red = 2
    m.fs.gac.costing.costing_blocks["concrete_gravity"].num_contactors_redundant.fix(num_red)

    assert_units_consistent(m)
    assert_degrees_of_freedom(m, 0)

    presolve_init(m)

    return m


def deactivate_ss_calculations(m):

    # deactivate steady state equations
    m.fs.gac.eq_ele_throughput[:].deactivate()
    m.fs.gac.eq_ele_min_operational_time[:].deactivate()
    m.fs.gac.eq_ele_conc_ratio_replace[:].deactivate()
    m.fs.gac.eq_ele_operational_time[:].deactivate()
    m.fs.gac.eq_ele_conc_ratio_term[:].deactivate()
    m.fs.gac.eq_conc_ratio_avg.deactivate()

    # fix variables used in steady state equations
    m.fs.gac.ele_throughput[:].fix()
    m.fs.gac.ele_min_operational_time[:].fix()
    m.fs.gac.ele_conc_ratio_replace[:].fix()
    m.fs.gac.ele_operational_time[:].fix()
    m.fs.gac.ele_conc_ratio_term[:].fix()


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

def model_scale(m):

    # custom scaling
    species = m.fs.properties.solute_set.at(1)
    iscale.set_scaling_factor(m.fs.gac.gac_usage_rate, 1e5)
    iscale.set_scaling_factor(m.fs.gac.operational_time, 1e-7)
    iscale.set_scaling_factor(m.fs.gac.min_operational_time, 1e-7)
    iscale.constraint_scaling_transform(m.fs.gac.eq_minimum_operational_time_cps, 1e-2)
    iscale.constraint_scaling_transform(m.fs.gac.eq_operational_time, 1e-2)
    iscale.constraint_scaling_transform(m.fs.gac.eq_bed_volumes_treated, 1e-2)
    iscale.constraint_scaling_transform(m.fs.gac.eq_equilibrium_concentration[0, species], 1e-4)
    iscale.constraint_scaling_transform(m.fs.gac.eq_number_bi, 1e-2)

    # default scaling
    iscale.calculate_scaling_factors(m)


def presolve_init(m):

    m.fs.feed.initialize()
    propagate_state(m.fs.s01)
    m.fs.gac.initialize()
    m.fs.gac.costing.initialize()
    m.fs.costing.initialize()

    activate_surrogate(m)


def model_solve(m, param, solver=None, tee=False):

    eps = 1e-8
    iterlim = 2500

    # pull parameter values from case study
    m.fs.gac.freund_k.fix(param[0])
    m.fs.gac.freund_ninv.fix(param[1])
    m.fs.gac.ds.fix(param[2])
    species = m.fs.properties.solute_set.at(1)

    # check that units are consistent
    assert_units_consistent(m)
    assert_degrees_of_freedom(m, 0)
    # print(
    #     f"solving iteration {round(m.fs.sample_number.value)} for a "
    #     f"{round(m.fs.gac.conc_ratio_replace.value, 4)} reduction of "
    #     f"{round(1e9*m.fs.feed.properties[0].conc_mass_phase_comp['Liq', species].value, 2)} ppt {species} at "
    #     f"{round(m.fs.feed.properties[0].flow_vol_phase['Liq'].value/0.043812636574074, 2)} MGD"
    # )

    m.presolve_state = modelStateStorage(m)

    # solve m
    try:
        res = solver.solve(m, tee=tee)
        pyo.assert_optimal_termination(res)
        print("solver termination condition:", res.solver.termination_condition)
    except:
        try:
            m.presolve_state.restore_state()
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
            m.presolve_state.restore_state()
            # try sequential solution if equation oriented fails
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
    calculate_variable_from_constraint(m.fs.gac.equil_conc, m.fs.gac.eq_equilibrium_concentration[0, species], eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.dg, m.fs.gac.eq_dg[0, species], eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.N_Bi, m.fs.gac.eq_number_bi, eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.min_N_St, m.fs.min_st_surrogate.pysmo_constraint["min_st"], eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.throughput, m.fs.throughput_surrogate.pysmo_constraint["throughput"], eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.min_ebct, m.fs.gac.eq_min_ebct_cps, eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.min_residence_time, m.fs.gac.eq_min_residence_time_cps, eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.min_operational_time, m.fs.gac.eq_minimum_operational_time_cps, eps=eps, iterlim=iterlim)
    calculate_variable_from_constraint(m.fs.gac.operational_time, m.fs.gac.eq_operational_time, eps=eps, iterlim=iterlim)
    if m.fs.gac.operational_time.value > 0:
        feasible_breakthrough = True
        calculate_variable_from_constraint(m.fs.gac.bed_volumes_treated, m.fs.gac.eq_bed_volumes_treated, eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.gac.mass_adsorbed, m.fs.gac.eq_mass_adsorbed[species], eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.gac.gac_usage_rate, m.fs.gac.eq_gac_usage_rate, eps=eps, iterlim=iterlim)
        for costing_block in (m.fs.gac.costing.costing_blocks["steel_pressure"], m.fs.gac.costing.costing_blocks["concrete_gravity"]):
            calculate_variable_from_constraint(costing_block.contactor_cost, costing_block.contactor_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.bed_mass_gac_ref, costing_block.bed_mass_gac_ref_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.adsorbent_unit_cost, costing_block.adsorbent_unit_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.adsorbent_cost, costing_block.adsorbent_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.other_process_cost, costing_block.other_process_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.capital_cost, costing_block.capital_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.gac_regen_cost, costing_block.gac_regen_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.gac_makeup_cost, costing_block.gac_makeup_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.fixed_operating_cost, costing_block.fixed_operating_cost_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.energy_consumption, costing_block.energy_consumption_constraint, eps=eps, iterlim=iterlim)
            calculate_variable_from_constraint(costing_block.LCOW, costing_block.LCOW_constraint, eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.costing.aggregate_capital_cost, m.fs.costing.aggregate_capital_cost_constraint, eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.costing.aggregate_fixed_operating_cost, m.fs.costing.aggregate_fixed_operating_cost_constraint, eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.costing.aggregate_flow_electricity, m.fs.costing.aggregate_flow_electricity_constraint, eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.costing.aggregate_flow_costs["electricity"], m.fs.costing.aggregate_flow_costs_constraint["electricity"], eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.costing.total_capital_cost, m.fs.costing.total_capital_cost_constraint, eps=eps, iterlim=iterlim)
        calculate_variable_from_constraint(m.fs.costing.total_operating_cost, m.fs.costing.total_operating_cost_constraint, eps=eps, iterlim=iterlim)
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

    electricity_flow_cost = pyo.units.convert(unit_costing.energy_consumption * fs_costing.electricity_cost,
        to_units=fs_costing.base_currency / fs_costing.base_period)
    total_capital_cost = pyo.units.convert(fs_costing.factor_total_investment * unit_costing.capital_cost,
        to_units=fs_costing.base_currency)
    total_operating_cost = pyo.units.convert((fs_costing.factor_maintenance_labor_chemical * total_capital_cost) +
        unit_costing.fixed_operating_cost + electricity_flow_cost * fs_costing.utilization_factor,
        to_units=fs_costing.base_currency / fs_costing.base_period)
    LCOW_constraint = pyo.Constraint(
        expr=LCOW == (total_capital_cost * fs_costing.factor_capital_annualization + total_operating_cost) / (
                pyo.units.convert(flow_rate, to_units=pyo.units.m**3 / fs_costing.base_period) * fs_costing.utilization_factor),
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
#     run_cases()

    # def main():

    # cwd = os.getcwd()

    # plot options
    # plot_width = 5
    # plot_height = 5
    # markersize = 20
    # linewidth = 1.5

    gac_input = defaultdict(list)
    gac_input["curve_id"] = []
    gac_input["epa_lim"] = []

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
            curve_res_df.to_csv(
                f"/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_curve{num}_{species}_{source}_{media}.csv",
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

            plt.tight_layout()
            plt.savefig(
                f"/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_curve{num}_{species}_{source}_{media}.png",
                bbox_inches="tight",
            )
            ################################################
            gac_input["curve_id"].append(num)
            gac_input["epa_lim"].append(epa_limits[species])

    df = pd.DataFrame(gac_input)
    df.to_csv(
        "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_case_study_regression_results.hdf5".replace(
            ".hdf5", ".csv"
        ),
        index=False,
    )
    # print(df)
    # import pprint
    # pprint.pprint(gac_input)

if __name__ == "__main__":
    pass