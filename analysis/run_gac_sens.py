import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import (
    Var,
    Param,
    Constraint,
    Expression,
    ConcreteModel,
    Objective,
    Block,
    value,
    assert_optimal_termination,
    check_optimal_termination,
    units as pyunits,
)
from pyomo.util.calc_var_value import calculate_variable_from_constraint as cvc

from idaes.core.util import DiagnosticsToolbox
from idaes.core.util.scaling import *
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver as get_idaes_solver
from idaes.core import (
    FlowsheetBlock,
    UnitModelCostingBlock,
)

from watertap.core.util.model_diagnostics.infeasible import *
from watertap.core.solvers import get_solver
from watertap.costing import WaterTAPCosting
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock as MCAS,
    DiffusivityCalculation,
)
from watertap.core.util.model_diagnostics.infeasible import *

# import watertap.kurby.analysis.PFAS.sensitivities.ix_pfas_sensitivity as ixm
from IPython.display import clear_output

import analysis.gac_pfas_sensitivity as gac
from parameter_sweep import parameter_sweep

# from watertap.kurby import *

with open(
    f"/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/pfas_properties.yaml",
    "r",
) as f:
    species_properties = yaml.load(f, Loader=yaml.FullLoader)

with open(
    f"/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/resin_properties.yaml",
    "r",
) as f:
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


inputs_file = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/ix_case_study_sensitivity_inputs.csv"
ix_inputs = pd.read_csv(inputs_file)
path = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities"

ix_nums = [18, 13, 304]
gac_nums = [0, 14, 21]
reinit_curves = [302, 305, 310]

solv_dens = {
    "water": 998.0,  # kg/m3
    "methanol": 792.0,
    "ethanol": 789.0,
    "acetone": 784.0,
}

mol_mass = {
    "NaCl": 58.44e-3,  # kg/mol
    "KCl": 74.55e-3,
    "NH4Cl": 53.49e-3,
    "NaOH": 40.00e-3,
    "NH4OH": 35.05e-3,  # treated as NH3(aq)
}

# Apparent molar volumes (m3/mol)
# Order-of-magnitude correct for dilute aqueous solutions
app_mol_vol = {
    "NaCl": 16.6e-6,
    "KCl": 27.0e-6,
    "NH4Cl": 38.0e-6,
    "NaOH": 18.8e-6,
    "NH4OH": 35.0e-6,
}


def get_regen_soln_info(
    solvent_vv=None,
    solutes_molar=None,
):
    """
    Calculate solution density and component mass fractions.

    Parameters
    ----------
    solvent_vv : dict
        Volume fractions, e.g. {"methanol": 0.30}
        Water is assumed to be the remainder.
    solutes_molar : dict
        Molar concentrations [mol/L], e.g. {"NaCl": 0.17}

    Returns
    -------
    density : float
        Solution density [kg/m3]
    mass_fractions : dict
        Mass fraction of each component
    """

    if solvent_vv is None:
        solvent_vv = {}
    if solutes_molar is None:
        solutes_molar = {}

    v_tot = 1.0  # m3 basis
    masses = {}
    v_solv = 0.0
    v_sol = 0.0

    # ---- solvents ----
    for solvent, frac in solvent_vv.items():
        V = frac * v_tot
        m = V * solv_dens[solvent]
        masses[solvent] = m
        v_solv += V

    # ---- water ----
    v_water = v_tot - v_solv
    m_water = v_water * solv_dens["water"]
    # masses["water"] = v_water * solv_dens["water"]

    # ---- solutes ----
    for solute, M in solutes_molar.items():
        n = M * 1000.0  # mol/m3
        m = n * mol_mass[solute]
        masses[solute] = m
        v_sol += n * app_mol_vol[solute]

    # ---- total mass and density ----
    total_mass = sum(masses.values()) + m_water
    v_eff = v_tot + v_sol
    density = total_mass / v_eff

    # ---- mass fractions ----
    mass_fractions = {comp: m / total_mass for comp, m in masses.items()}

    return density, mass_fractions

if __name__ == "__main__":

    inputs_file = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities/data/gac_case_study_sensitivity_inputs.csv"
    gac_inputs = pd.read_csv(inputs_file)
    path = "/Users/ksitterl/Documents/Python/pfas_sensitivities/pfas_sensitivities"
    num_samples = 20
    num_procs = 8

    # sweep = "adsorbent_unit_cost"
    cs_nums = [14, 21, 0]

    gac_config = {}
    regen_tag = "single_use"

    for sweep in ["ebct", "loading_rate", "makeup_unit_cost", "regen_frac+makeup_unit_cost"]:
        for cs in cs_nums:
            data = gac_inputs[gac_inputs.curve_id == cs].iloc[0]
            save_file = f"{path}/results/gac/gac_pfas_{sweep}_sensitivity-{data.source}_curve{data.curve_id}_{data.species}_{data.media}_{regen_tag}.h5"
            results_array, results_dict = parameter_sweep(
                build_model=gac.build_and_solve,
                build_model_kwargs={
                    "data": data,
                    "sweep": None,
                    "gac_config": gac_config,
                },
                build_sweep_params=gac.build_sweep_params,
                build_sweep_params_kwargs={"num_samples": num_samples, "sweep": sweep},
                build_outputs=gac.build_outputs,
                build_outputs_kwargs={},
                h5_results_file_name=save_file,
                optimize_function=gac.solve,
                num_samples=num_samples,
                csv_results_file_name=save_file.replace(".h5", ".csv"),
                number_of_subprocesses=num_procs,
            )

            df = pd.read_csv(save_file.replace(".h5", ".csv"))
            # Get results for base case
            _m = gac.build_and_solve(data)
            o = gac.build_outputs(_m)
            d = {}

            for k, v in o.items():
                d[k] = value(v)
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

            df["regenerant"] = regen_tag
            df.to_csv(save_file.replace(".h5", ".csv"), index=False)
            #     df
            # break


    gac_config = {}
    gac_config["regenerant"] = "custom"
    regen_tag = "methanol_nacl"
    solvent_vv = {"methanol": 0.70}
    solutes_molar = {"NaCl": 0.17}
    regen_soln_density, regen_composition = get_regen_soln_info(
        solvent_vv=solvent_vv,
        solutes_molar=solutes_molar,
    )
    gac_config["regen_composition"] = regen_composition
    gac_config["regen_soln_density"] = regen_soln_density


    for sweep in ["ebct", "loading_rate", "makeup_unit_cost", "regen_frac+makeup_unit_cost"]:
        for cs in cs_nums:
            data = gac_inputs[gac_inputs.curve_id == cs].iloc[0]
            save_file = f"{path}/results/gac/gac_pfas_{sweep}_sensitivity-{data.source}_curve{data.curve_id}_{data.species}_{data.media}_{regen_tag}.h5"
            results_array, results_dict = parameter_sweep(
                build_model=gac.build_and_solve,
                build_model_kwargs={
                    "data": data,
                    "sweep": None,
                    "gac_config": gac_config,
                },
                build_sweep_params=gac.build_sweep_params,
                build_sweep_params_kwargs={"num_samples": num_samples, "sweep": sweep},
                build_outputs=gac.build_outputs,
                build_outputs_kwargs={},
                h5_results_file_name=save_file,
                optimize_function=gac.solve,
                num_samples=num_samples,
                csv_results_file_name=save_file.replace(".h5", ".csv"),
                number_of_subprocesses=num_procs,
            )

            df = pd.read_csv(save_file.replace(".h5", ".csv"))
            # Get results for base case
            _m = gac.build_and_solve(data)
            o = gac.build_outputs(_m)
            d = {}

            for k, v in o.items():
                d[k] = value(v)
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
            if gac_config["regenerant"] == "custom":
                for k, v in solvent_vv.items():
                    df[f"regen_solvent"] = k
                    df[f"regen_solvent_{k}_vol_frac"] = v
                    df[f"regen_solution_mass_frac_{k}"] = regen_composition[k]
                for k, v in solutes_molar.items():
                    df[f"regen_solution"] = k
                    df[f"regen_solute_{k}_conc_molar"] = v
                    df[f"regen_solution_mass_frac_{k}"] = regen_composition[k]

            df["regenerant"] = regen_tag
            df.to_csv(save_file.replace(".h5", ".csv"), index=False)
            #     df
            # break

    gac_config = {}
    gac_config["regenerant"] = "custom"
    regen_tag = "acetone_nacl"
    solvent_vv = {"acetone": 0.70}
    solutes_molar = {"NaCl": 0.17}
    regen_soln_density, regen_composition = get_regen_soln_info(
        solvent_vv=solvent_vv,
        solutes_molar=solutes_molar,
    )
    gac_config["regen_composition"] = regen_composition
    gac_config["regen_soln_density"] = regen_soln_density


    for sweep in ["ebct", "loading_rate", "makeup_unit_cost", "regen_frac+makeup_unit_cost"]:
        for cs in cs_nums:
            data = gac_inputs[gac_inputs.curve_id == cs].iloc[0]
            save_file = f"{path}/results/gac/gac_pfas_{sweep}_sensitivity-{data.source}_curve{data.curve_id}_{data.species}_{data.media}_{regen_tag}.h5"
            results_array, results_dict = parameter_sweep(
                build_model=gac.build_and_solve,
                build_model_kwargs={
                    "data": data,
                    "sweep": None,
                    "gac_config": gac_config,
                },
                build_sweep_params=gac.build_sweep_params,
                build_sweep_params_kwargs={"num_samples": num_samples, "sweep": sweep},
                build_outputs=gac.build_outputs,
                build_outputs_kwargs={},
                h5_results_file_name=save_file,
                optimize_function=gac.solve,
                num_samples=num_samples,
                csv_results_file_name=save_file.replace(".h5", ".csv"),
                number_of_subprocesses=num_procs,
            )

            df = pd.read_csv(save_file.replace(".h5", ".csv"))
            # Get results for base case
            _m = gac.build_and_solve(data)
            o = gac.build_outputs(_m)
            d = {}

            for k, v in o.items():
                d[k] = value(v)
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
            if gac_config["regenerant"] == "custom":
                for k, v in solvent_vv.items():
                    df[f"regen_solvent"] = k
                    df[f"regen_solvent_{k}_vol_frac"] = v
                    df[f"regen_solution_mass_frac_{k}"] = regen_composition[k]
                for k, v in solutes_molar.items():
                    df[f"regen_solution"] = k
                    df[f"regen_solute_{k}_conc_molar"] = v
                    df[f"regen_solution_mass_frac_{k}"] = regen_composition[k]

            df["regenerant"] = regen_tag
            df.to_csv(save_file.replace(".h5", ".csv"), index=False)
            #     df
            # break

    gac_config = {}
    gac_config["regenerant"] = "custom"
    regen_tag = "ethanol_nacl"
    solvent_vv = {"ethanol": 0.70}
    solutes_molar = {"NaCl": 0.17}
    regen_soln_density, regen_composition = get_regen_soln_info(
        solvent_vv=solvent_vv,
        solutes_molar=solutes_molar,
    )
    gac_config["regen_composition"] = regen_composition
    gac_config["regen_soln_density"] = regen_soln_density


    for sweep in ["ebct", "loading_rate", "makeup_unit_cost", "regen_frac+makeup_unit_cost"]:
        for cs in cs_nums:
            data = gac_inputs[gac_inputs.curve_id == cs].iloc[0]
            save_file = f"{path}/results/gac/gac_pfas_{sweep}_sensitivity-{data.source}_curve{data.curve_id}_{data.species}_{data.media}_{regen_tag}.h5"
            results_array, results_dict = parameter_sweep(
                build_model=gac.build_and_solve,
                build_model_kwargs={
                    "data": data,
                    "sweep": None,
                    "gac_config": gac_config,
                },
                build_sweep_params=gac.build_sweep_params,
                build_sweep_params_kwargs={"num_samples": num_samples, "sweep": sweep},
                build_outputs=gac.build_outputs,
                build_outputs_kwargs={},
                h5_results_file_name=save_file,
                optimize_function=gac.solve,
                num_samples=num_samples,
                csv_results_file_name=save_file.replace(".h5", ".csv"),
                number_of_subprocesses=num_procs,
            )

            df = pd.read_csv(save_file.replace(".h5", ".csv"))
            # Get results for base case
            _m = gac.build_and_solve(data)
            o = gac.build_outputs(_m)
            d = {}

            for k, v in o.items():
                d[k] = value(v)
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
            if gac_config["regenerant"] == "custom":
                for k, v in solvent_vv.items():
                    df[f"regen_solvent"] = k
                    df[f"regen_solvent_{k}_vol_frac"] = v
                    df[f"regen_solution_mass_frac_{k}"] = regen_composition[k]
                for k, v in solutes_molar.items():
                    df[f"regen_solution"] = k
                    df[f"regen_solute_{k}_conc_molar"] = v
                    df[f"regen_solution_mass_frac_{k}"] = regen_composition[k]

            df["regenerant"] = regen_tag
            df.to_csv(save_file.replace(".h5", ".csv"), index=False)
            #     df
            # break


    gac_config = {}
    gac_config["regenerant"] = "onsite_thermal"
    regen_tag = "onsite_thermal"


    for sweep in ["ebct", "loading_rate", "makeup_unit_cost", "regen_frac+makeup_unit_cost"]:
        for cs in cs_nums:
            data = gac_inputs[gac_inputs.curve_id == cs].iloc[0]
            save_file = f"{path}/results/gac/gac_pfas_{sweep}_sensitivity-{data.source}_curve{data.curve_id}_{data.species}_{data.media}_{regen_tag}.h5"
            results_array, results_dict = parameter_sweep(
                build_model=gac.build_and_solve,
                build_model_kwargs={
                    "data": data,
                    "sweep": None,
                    "gac_config": gac_config,
                },
                build_sweep_params=gac.build_sweep_params,
                build_sweep_params_kwargs={"num_samples": num_samples, "sweep": sweep},
                build_outputs=gac.build_outputs,
                build_outputs_kwargs={},
                h5_results_file_name=save_file,
                optimize_function=gac.solve,
                num_samples=num_samples,
                csv_results_file_name=save_file.replace(".h5", ".csv"),
                number_of_subprocesses=num_procs,
            )

            df = pd.read_csv(save_file.replace(".h5", ".csv"))
            # Get results for base case
            _m = gac.build_and_solve(data)
            o = gac.build_outputs(_m)
            d = {}

            for k, v in o.items():
                d[k] = value(v)
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

            df["regenerant"] = regen_tag
            df.to_csv(save_file.replace(".h5", ".csv"), index=False)
            #     df
            # break
