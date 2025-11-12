import json
import logging
import os
import sys
import typing
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pathos as pa
import pycountry
import country_converter as coco

from climada.engine import Impact
from climada.util.config import CONFIG as CLIMADA_CONFIG
from climada_petals.engine.supplychain import DirectShocksSet, get_mriot, StaticIOModel
from climada_petals.engine.supplychain.core import (
    translate_exp_to_sectors,
    distribute_reg_impact_to_sectors,
)
from climada_petals.engine.supplychain.utils import calc_G, calc_B

from nccs.pipeline.direct.calc_yearset import combine_yearsets, yearset_from_imp
from nccs.pipeline.direct.direct import get_sector_exposure, nccs_direct_impacts_simple
from nccs.pipeline.indirect.indirect import (
    dump_direct_to_csv,
    dump_supchain_events_to_csv,
    dump_supchain_to_csv,
    SUPER_SEC,
    MRIOT_TYPE
)
from nccs.utils import folder_naming
from nccs.pipeline.indirect.event_aggregations import aggregate_yearset_return_periods

LOGGER = logging.getLogger(__name__)


def run_pipeline_from_config(
    config: dict
):
    """Run the full model NCCS supply chain from a config dictionary.

    The method uses the input config object to create a dataframe with one
    row for each analysis required for the run, based on the requested
    scenarios, sectors, countries and hazards. It creates an impact object
    for each analysis, a yearset based on these impacts, and runs the
    supply chain model for each yearset, and writes outputs as it goes.

    Parameters
    ----------
    config : dict
        A dictionary describing the full model run configuration. See the
        examples in run_configurations/ for how these are constructed
    direct_output_dir : str or os.PathLike
        Location to store direct impact calculation results
        (both impact objects and the yearsets created from them). Generated
        automatically from the config run name if not provided
    indirect_output_dir : str or os.PathLike
        location to store indirect impact calculation results.
        Generated automatically from the config run name if not provided
    """
    if not direct_output_dir:
        if "direct_output_dir" in config.keys():
            direct_output_dir = config["direct_output_dir"]
        else:
            direct_output_dir = folder_naming.get_direct_output_dir(config["run_title"])
    if not indirect_output_dir:
        if "indirect_output_dir" in config.keys():
            indirect_output_dir = config["indirect_output_dir"]
        else:
            indirect_output_dir = folder_naming.get_indirect_output_dir(config["run_title"])

    config["direct_output_dir"] = direct_output_dir
    config["indirect_output_dir"] = indirect_output_dir

    os.makedirs(direct_output_dir, exist_ok=True)
    os.makedirs(indirect_output_dir, exist_ok=True)

    time_now = datetime.now()
    config["time_run"] = str(time_now)
    with open(Path(indirect_output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    LOGGER.info(f"Direct output will be saved to {direct_output_dir}")

    ### --------------------------------- ###
    ### CALCULATE DIRECT ECONOMIC IMPACTS ###
    ### --------------------------------- ###

    ### Read the config to create a dataframe of simulations with metadata, and filepaths for each analysis
    analysis_df = config_to_dataframe(config)

    direct_output_dir_impact = Path(direct_output_dir, "impact_raw")
    direct_output_dir_yearsets = Path(direct_output_dir, "yearsets")
    direct_output_dir_supchain_direct = Path(direct_output_dir, "supchain_direct")
    os.makedirs(direct_output_dir_impact, exist_ok=True)
    os.makedirs(direct_output_dir_yearsets, exist_ok=True)
    os.makedirs(direct_output_dir_supchain_direct, exist_ok=True)

    analysis_df["_direct_impact_already_exists"] = [
        exists_impact_file(p) for p in analysis_df["direct_impact_path"]
    ]
    analysis_df["_direct_impact_calculate"] = (
        True if config["force_recalculation"] else ~analysis_df["_direct_impact_already_exists"]
    )
    n_direct_calculations = np.sum(analysis_df["_direct_impact_calculate"])
    n_direct_exists = np.sum(analysis_df["_direct_impact_already_exists"])

    LOGGER.info(f"Config:\n{config}")
    if config["do_direct"]:
        LOGGER.info("\n\nRUNNING DIRECT IMPACT CALCULATIONS")
        LOGGER.info(
            f"There are {n_direct_calculations} direct impacts to calculate. ({n_direct_exists} exist already. Full "
            f"analysis has {analysis_df.shape[0]} impacts.)"
        )
        calculate_direct_impacts_from_df(analysis_df, config)
    else:
        LOGGER.info(
            "Skipping direct impact calculations. Set do_direct: True in your config to change this"
        )

    analysis_df["_direct_impact_exists"] = [
        exists_impact_file(p) for p in analysis_df["direct_impact_path"]
    ]

    analysis_df_filename = f'calculations_report_{time_now.strftime("%Y-%m-%d_%H%M")}.csv'
    analysis_df_path = Path(indirect_output_dir, analysis_df_filename)
    analysis_df.to_csv(analysis_df_path)

    ### ------------------- ###
    ### SAMPLE IMPACT YEARS ###
    ### ------------------- ###

    # Create a yearset for each row of the analysis dataframe
    # This gives us an impact object where each event is a fictional year of events
    yearset_output_dir = Path(direct_output_dir, "yearsets")
    os.makedirs(yearset_output_dir, exist_ok=True)

    analysis_df["_yearset_already_exists"] = [
        exists_impact_file(p) for p in analysis_df["yearset_path"]
    ]
    analysis_df["_yearset_calculate"] = (
        True if config["force_recalculation"] else ~analysis_df["_yearset_already_exists"]
    ) * analysis_df["_direct_impact_exists"]
    n_yearset_calculations = np.sum(analysis_df["_yearset_calculate"])
    n_yearset_exists = np.sum(analysis_df["_yearset_already_exists"])
    n_missing_direct = np.sum(
        ~analysis_df["_yearset_already_exists"] * ~analysis_df["_direct_impact_exists"]
    )

    if config["do_yearsets"]:
        LOGGER.info("\n\nCREATING IMPACT YEARSETS")
        LOGGER.info(
            f"There are {n_yearset_calculations} yearsets to create. ({n_yearset_exists} already exist, "
            f"{n_missing_direct} of the remaining are missing direct impact data, full analysis has "
            f"{analysis_df.shape[0]} yearsets.)"
        )
        LOGGER.info(f"yearsets will be saved in {yearset_output_dir}")
        calculate_yearsets_from_df(analysis_df, config)
    else:
        LOGGER.info(
            "Skipping yearset calculations. Set do_yearsets: True in your config to change this"
        )

    analysis_df["_yearset_exists"] = [
        exists_impact_file(p) for p in analysis_df["yearset_path"]
    ]
    analysis_df.to_csv(analysis_df_path)

    # Combine the yearsets for each agriculture crop type to one agriculture yearset
    analysis_df_crop = analysis_df[analysis_df["hazard"].str.contains("relative_crop_yield")]
    analysis_df_no_crop = analysis_df[~analysis_df["hazard"].str.contains("relative_crop_yield")]
    if analysis_df_crop.shape[0] > 0:
        LOGGER.info("\n\nCOMBINING CROP YIELD YEARSETS")
        grouping_cols = ["i_scenario", "country"]
        df_aggregated_yearsets = df_extend_with_multihazard(
            analysis_df_crop,
            df_create_combined_hazard_yearsets_agriculture,
            config,
            grouping_cols,
            False,
            True,
        )

        analysis_df = pd.concat([analysis_df_no_crop, df_aggregated_yearsets]).reset_index(
            drop=True
        )
        analysis_df = sort_analysis_df(analysis_df)

    # _ = _check_config_valid_for_indirect_aggregations(config)

    # Combine yearsets by hazard to create multihazard yearsets
    grouping_cols = ["i_scenario", "sector", "country"]
    if config["do_multihazard"]:
        LOGGER.info("\n\nCOMBINING HAZARDS TO MULTIHAZARD YEARSETS")
        analysis_df = df_extend_with_multihazard(
            analysis_df,
            df_create_combined_hazard_yearsets,
            config,
            grouping_cols,
            True,
            True,
        )
    else:
        LOGGER.info(
            "Skipping multihazard impact calculations. Set do_multihazard: True in your config to change this"
        )

    ### ----------------------------------- ###
    ### CALCULATE INDIRECT ECONOMIC IMPACTS ###
    ### ----------------------------------- ###

    # Generate supply chain impacts from the yearsets
    # Create a folder to output the data
    # indirect_output_dir = Path(indirect_output_dir, "results")
    LOGGER.info("\n\nMODELLING SUPPLY CHAINS")
    os.makedirs(indirect_output_dir, exist_ok=True)
    analysis_df["_indirect_leontief_already_exists"] = [
        os.path.exists(p) for p in analysis_df["supchain_indirect_leontief_path"]
    ]
    analysis_df["_indirect_ghosh_already_exists"] = [
        os.path.exists(p) for p in analysis_df["supchain_indirect_ghosh_path"]
    ]

    if "leontief" in config["io_approach"]:
        analysis_df["_indirect_leontief_calculate"] = (
            True
            if config["force_recalculation"]
            else analysis_df["_yearset_exists"] * ~analysis_df["_indirect_leontief_already_exists"]
        )
    else:
        analysis_df["_indirect_leontief_calculate"] = False

    if "ghosh" in config["io_approach"]:
        analysis_df["_indirect_ghosh_calculate"] = (
            True
            if config["force_recalculation"]
            else (analysis_df["_yearset_exists"] * ~analysis_df["_indirect_ghosh_already_exists"])
        )
    else:
        analysis_df["_indirect_ghosh_calculate"] = False

    analysis_df["_indirect_leontief_exists"] = analysis_df["_indirect_leontief_already_exists"]
    analysis_df["_indirect_ghosh_exists"] = analysis_df["_indirect_ghosh_already_exists"]

    n_supchain_calculations_leontief = np.sum(analysis_df["_indirect_leontief_calculate"])
    n_supchain_calculations_ghosh = np.sum(analysis_df["_indirect_ghosh_calculate"])

    LOGGER.info(
        f"There are {n_supchain_calculations_leontief} out of {analysis_df.shape[0]} leontief supply chains to "
        f"calculate"
    )
    LOGGER.info(
        f"There are {n_supchain_calculations_ghosh} out of {analysis_df.shape[0]} ghosh supply chains to calculate"
    )

    # Run the Supply Chain for each country and sector and output the data needed to csv
    if config["do_indirect"]:
        calculate_indirect_impacts_from_df(analysis_df, config)
        aggregate_yearset_return_periods(config["run_title"], 100, config["n_sim_years"])
    else:
        LOGGER.info(
            "Skipping supply chain calculations. Set do_indirect: True in your config to change this"
        )

    analysis_df["indirect_leontief_exists"] = [
        os.path.exists(f) for f in analysis_df["supchain_indirect_leontief_path"]
    ]
    analysis_df["indirect_ghosh_exists"] = [
        os.path.exists(f) for f in analysis_df["supchain_indirect_ghosh_path"]
    ]

    # AGGREGATE ACROSS DIFFERENT DIMENSIONS
    # aggregation_axes = ['hazard', 'sector', 'country']

    # for axes in chain.from_iterable(combinations(aggregation_axes, n) for n in range(1, len(aggregation_axes)+1)):
    #     LOGGER.info(f'Aggregating supply chain output on {axes}')
    #     analysis_df = aggregate_supply_chain_impacts(analysis_df, axes)

    analysis_df.to_csv(analysis_df_path)

    LOGGER.info("\n\nDone!\nTo show the Dashboard run:\nbokeh serve dashboard.py --show")
    LOGGER.info(
        "Don't forget to update the current run title within the dashboard.py script: RUN_TITLE"
    )


def config_to_dataframe(config: dict) -> pd.DataFrame:
    """Convert a run config to a dataframe of required model runs.
    Note: these don't include model runs that combine hazards, sectors and
    countries, which are created after this first set is run.

    Parameters
    ----------
    config : dict
        A config object. See run_configurations/ for the format

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row for each simulation that will be run in the
        supply chain modelling, and the parameters required to run the
        simulations.
    """
    df = pd.DataFrame(
        [
            {
                "hazard": run["hazard"],
                "sector": sector,
                "country": country,
                "scenario": scenario["scenario"],
                "i_scenario": i,
                "ref_year": scenario["ref_year"],
            }
            for run in config["runs"]
            for i, scenario in enumerate(run["scenario_years"])
            for country in run["countries"]
            for sector in run["sectors"]
        ]
    )
    # unfold the agriculture sector into the different crop types
    for crop_type in ["whe", "mai", "ric", "soy"]:
        df_crop_yield = df[df["hazard"] == "relative_crop_yield"].copy()
        df_crop_yield["sector"] = df_crop_yield["sector"].apply(lambda x: f"{x}_{crop_type}")
        df_crop_yield["hazard"] = df_crop_yield["hazard"].apply(lambda x: f"{x}_{crop_type}")
        df = pd.concat([df, df_crop_yield])
    df = df.reset_index(drop=True)
    # Drop the original agriculture rows
    df = df[df["hazard"] != "relative_crop_yield"]

    for i, row in df.iterrows():
        direct_impact_filename = folder_naming.get_filename_direct(row)
        direct_impact_path = Path(config["direct_output_dir"], "impact_raw", direct_impact_filename)
        df.loc[i, "direct_impact_path"] = direct_impact_path

        yearset_filename = folder_naming.get_filename_yearset(row)
        yearset_path = Path(config["direct_output_dir"], "yearsets", yearset_filename)
        df.loc[i, "yearset_path"] = yearset_path

        supchain_direct_filename = folder_naming.get_filename_supchain_direct(row)
        supchain_direct_path = Path(
            config["direct_output_dir"], "supchain_direct", supchain_direct_filename
        )
        df.loc[i, "supchain_direct_path"] = supchain_direct_path

        supchain_indirect_leontief_filename = folder_naming.get_filename_supchain_indirect(
            row, io_approach="leontief"
        )
        supchain_indirect_leontief_path = Path(
            config["indirect_output_dir"], supchain_indirect_leontief_filename
        )
        df.loc[i, f"supchain_indirect_leontief_path"] = supchain_indirect_leontief_path

        supchain_indirect_ghosh_filename = folder_naming.get_filename_supchain_indirect(
            row, io_approach="ghosh"
        )
        supchain_indirect_ghosh_path = Path(
            config["indirect_output_dir"], supchain_indirect_ghosh_filename
        )
        df.loc[i, f"supchain_indirect_ghosh_path"] = supchain_indirect_ghosh_path

        supchain_indirect_events_leontief_path = Path(
            config["indirect_output_dir"], "events", supchain_indirect_leontief_filename
        )
        df.loc[i, f"supchain_indirect_events_leontief_path"] = (
            supchain_indirect_events_leontief_path
        )

        supchain_indirect_events_ghosh_path = Path(
            config["indirect_output_dir"], "events", supchain_indirect_ghosh_filename
        )
        df.loc[i, f"supchain_indirect_events_ghosh_path"] = supchain_indirect_events_ghosh_path

    return sort_analysis_df(df)


def sort_analysis_df(df):
    sorting_cols = ["i_scenario", "country", "hazard", "sector"]
    return df.sort_values(sorting_cols)


def calculate_direct_impacts_from_df(df, config):
    def direct_impacts_from_row(row):
        if not row["_direct_impact_calculate"]:
            return

        logging_dict = {k: row[k] for k in ["hazard", "sector", "country", "scenario", "ref_year"]}
        LOGGER.info(f"Calculating direct impacts for for {logging_dict}")
        try:
            imp = nccs_direct_impacts_simple(
                haz_type=row["hazard"],
                sector=row["sector"],
                country=row["country"],
                scenario=row["scenario"],
                ref_year=row["ref_year"],
                data_path=config["data_path"],
                business_interruption=config["business_interruption"],
                calibrated=config["calibrated"],
                use_sector_bi_scaling=config["use_sector_bi_scaling"],
            )
            # throw error if there are no impacts
            # TODO actually, return an empty impact object
            if imp.imp_mat.shape[1] == 0:
                raise ValueError(f"No impacts for {logging_dict}")
            write_impact_to_file(imp, row["direct_impact_path"])
        except Exception as e:
            LOGGER.error(f"Error calculating direct impacts for {logging_dict}:", exc_info=True)

    if config["do_parallel"]:
        with pa.multiprocessing.ProcessPool(config["ncpus"]) as pool:
            pool.map(direct_impacts_from_row, [row for _, row in df.iterrows()])
    else:
        for _, row in df.iterrows():
            direct_impacts_from_row(row)


def calculate_yearsets_from_df(df, config):
    def yearset_from_row(row):
        if not row["_yearset_calculate"]:
            return

        logging_dict = {k: row[k] for k in ["hazard", "sector", "country", "scenario", "ref_year"]}
        LOGGER.info(f"Generating yearsets for {logging_dict}")
        try:
            imp_yearset = create_single_yearset(
                row,
                n_sim_years=config["n_sim_years"],
                seed=config["seed"],
                config=config
            )
            write_impact_to_file(imp_yearset, row["yearset_path"])
        except Exception as e:
            LOGGER.error(
                f"Error calculating an indirect yearset for {logging_dict}",
                exc_info=True,
            )

    if config["do_parallel"]:
        with pa.multiprocessing.ProcessPool(config["ncpus"]) as pool:
            pool.map(yearset_from_row, [row for _, row in df.iterrows()])
    else:
        for _, row in df.iterrows():
            yearset_from_row(row)


def df_extend_with_multihazard(
    df, combining_function, config, grouping_cols, extend, calculate_yearsets=True
):
    def calc_partial(df):
        partial_combine_yearsets = partial(
            combining_function, config=config, calculate_yearsets=True
        )
        cols_to_keep = list(
            {"hazard", "sector", "scenario", "ref_year", "yearset_path"}.union(set(grouping_cols))
        )
        out = df.groupby(grouping_cols)[cols_to_keep].apply(partial_combine_yearsets).reset_index()
        # out[grouping_cols] = df[grouping_cols].drop_duplicates().reset_index(drop=True)
        return out

    # TODO this doesn't work in parallel yet - we get duplicated results. Turning off for this calculation.
    if config["do_parallel"] and False:
        chunk_size = int(np.ceil(df.shape[0] / config["ncpus"]))
        df_chunked = [df[i : i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
        with pa.multiprocessing.ProcessPool(config["ncpus"]) as pool:
            df_aggregated_yearsets_list = pool.map(calc_partial, df_chunked)
        df_aggregated_yearsets = pd.concat(df_aggregated_yearsets_list).reset_index(drop=True)
    else:
        df_aggregated_yearsets = calc_partial(df)
    if extend:
        df_aggregated_yearsets = pd.concat([df, df_aggregated_yearsets]).reset_index(drop=True)
    return sort_analysis_df(df_aggregated_yearsets)


def df_create_combined_hazard_yearsets(df, config, calculate_yearsets=True):
    """For each grouping of scenario, country and sector, combine hazard yearsets

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing analyses metadata created by config_to_dataframe

    Returns
    -------
    pandas.DataFrame
        A dataframe containing analysis metadata for a supply chain analysis
        for all hazards combined.

    Notes
    -----
    This function adapts pymrio.tools.iomath.calc_x to compute
    value added (v).
    """

    r = df.iloc[0].to_dict()
    yearset_output_dir = os.path.dirname(r["yearset_path"])
    LOGGER.debug(r)
    r["haz_type"] = "COMBINED"

    out = {
        "hazard": "COMBINED",
        "scenario": r["scenario"],
        "ref_year": r["ref_year"],
    }

    combined_filename = folder_naming.get_filename_direct(r | out)
    combined_path = Path(yearset_output_dir, combined_filename)
    supchain_direct_filename = folder_naming.get_filename_supchain_direct(r | out)
    supchain_direct_path = Path(
        config["direct_output_dir"], "supchain_direct", supchain_direct_filename
    )
    supchain_indirect_leontief_filename = folder_naming.get_filename_supchain_indirect(
        r | out, io_approach="leontief"
    )
    supchain_indirect_leontief_path = Path(
        config["indirect_output_dir"], supchain_indirect_leontief_filename
    )
    supchain_indirect_ghosh_filename = folder_naming.get_filename_supchain_indirect(
        r | out, io_approach="ghosh"
    )
    supchain_indirect_ghosh_path = Path(
        config["indirect_output_dir"], supchain_indirect_ghosh_filename
    )

    impact_list = [get_impact_from_file(f) for f in df["yearset_path"] if os.path.exists(f)]

    # validate
    impact_exp_size = pd.Series([imp.imp_mat.shape[1] for imp in impact_list])

    out = out | {
        "yearset_path": combined_path,
        "supchain_direct_path": supchain_direct_path,
        "supchain_indirect_leontief_path": supchain_indirect_leontief_path,
        "supchain_indirect_ghosh_path": supchain_indirect_ghosh_path,
        "supchain_indirect_events_leontief_path": Path(
            config["indirect_output_dir"], "events", supchain_indirect_leontief_filename
        ),
        "supchain_indirect_events_ghosh_path": Path(
            config["indirect_output_dir"], "events", supchain_indirect_ghosh_filename
        ),
    }

    out["_yearset_exists"] = False
    if len(impact_list) > 0:
        if os.path.exists(combined_path):  # TODO include overwrite command here if necessary
            out["_yearset_exists"] = True
        if calculate_yearsets and (~os.path.exists(combined_path) or config["force_recalculation"]):
            try:
                combined = combine_yearsets(
                    impact_list=impact_list,
                    cap_exposure=get_sector_exposure(r["sector"], r["country"], config["data_path"]),
                )
                # TODO drop the impact matrix to save RAM/HD space once SupplyChain is updated and doesn't need it
                combined.write_hdf5(combined_path)
                out["_yearset_exists"] = True
            except Exception as e:
                LOGGER.error(f"Error calculating combining yearsets for {out}", exc_info=True)

    return pd.Series(out)


def df_create_combined_hazard_yearsets_agriculture(
    df: pd.DataFrame, config, calculate_yearsets=True
):
    """For each grouping of scenario, country and sector, combine hazard yearsets

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing analyses metadata created by config_to_dataframe

    Returns
    -------
    pandas.DataFrame
        A dataframe containing analysis metadata for a supply chain analysis
        for all hazards combined.

    Notes
    -----
    This function adapts pymrio.tools.iomath.calc_x to compute
    value added (v).
    """

    r = df.iloc[0].to_dict()
    yearset_output_dir = os.path.dirname(r["yearset_path"])
    LOGGER.debug(r)
    r["hazard"] = "relative_crop_yield"
    r["sector"] = "agriculture"

    out = {
        "hazard": r["hazard"],
        "scenario": r["scenario"],
        "ref_year": r["ref_year"],
        "sector": r["sector"],
    }

    combined_filename = folder_naming.get_filename_direct(r | out)
    combined_path = Path(yearset_output_dir, combined_filename)
    supchain_direct_filename = folder_naming.get_filename_supchain_direct(r | out)
    supchain_direct_path = Path(
        config["direct_output_dir"], "supchain_direct", supchain_direct_filename
    )
    supchain_indirect_leontief_filename = folder_naming.get_filename_supchain_indirect(
        r | out, io_approach="leontief"
    )
    supchain_indirect_leontief_path = Path(
        config["indirect_output_dir"], supchain_indirect_leontief_filename
    )
    supchain_indirect_ghosh_filename = folder_naming.get_filename_supchain_indirect(
        r | out, io_approach="ghosh"
    )
    supchain_indirect_ghosh_path = Path(
        config["indirect_output_dir"], supchain_indirect_ghosh_filename
    )

    impact_list = [get_impact_from_file(f) for f in df["yearset_path"] if os.path.exists(f)]

    out = out | {
        "yearset_path": combined_path,
        "supchain_direct_path": supchain_direct_path,
        "supchain_indirect_leontief_path": supchain_indirect_leontief_path,
        "supchain_indirect_ghosh_path": supchain_indirect_ghosh_path,
        "supchain_indirect_events_leontief_path": Path(
            config["indirect_output_dir"], "events", supchain_indirect_leontief_filename
        ),
        "supchain_indirect_events_ghosh_path": Path(
            config["indirect_output_dir"], "events", supchain_indirect_ghosh_filename
        ),
    }

    out["_yearset_exists"] = False
    if len(impact_list) > 0:
        if os.path.exists(combined_path):
            out["_yearset_exists"] = True
        elif calculate_yearsets:
            try:
                combined = combine_yearsets(impact_list=impact_list)
                combined.write_hdf5(combined_path)
                out["_yearset_exists"] = True
            except Exception as e:
                LOGGER.error(f"Error combining agricultural yearset for {out}", exc_info=True)

    return pd.Series(out)


def create_single_yearset(
    analysis_spec: pd.Series,
    n_sim_years: int,
    seed: int,
    config,
):
    """Take the metadata for an analysis and create an impact yearset if it
    doesn't already exist. These are created as files and a `yearset_path` added
    to the input dataframe.

    Parameters
    ----------
    analysis_spec : pd.Series
        A row of a dataframe created by config_to_dataframe
    n_sim_years : int
        Number of years to create for each output yearset
    seed : int
        The random number seed to use in each yearset's sampling
    """
    row = analysis_spec.copy().to_dict()
    imp = get_impact_from_file(row["direct_impact_path"])

    poisson_hazards = ["tropical_cyclone", "sea_level_rise"]
    poisson = row["hazard"] in poisson_hazards

    # TODO we don't actually want to generate a yearset if we're looking at observed events
    imp_yearset = yearset_from_imp(
        imp,
        n_sim_years,
        poisson=poisson,
        cap_exposure=get_sector_exposure(row["sector"], row["country"], config["data_path"]),
        seed=seed,
    )
    # TODO drop the impact matrix to save RAM/HD space once SupplyChain is updated and doesn't need it
    return imp_yearset


def calculate_indirect_impacts_from_df(df: pd.DataFrame, config: dict):
    """
    Calculate indirect impacts for each configuration specified in the dataframe.

    The function optimizes the calculation by grouping configurations that share
    the same input data (exposures and hazards) to avoid reloading them
    repeatedly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row represents a calculation configuration.
        It must contain columns for 'country', 'sector', 'yearset_path',
        '_indirect_leontief_calculate', and '_indirect_ghosh_calculate'.
    config : dict
        The run configuration dictionary.
    """
    mriot = get_mriot(config["mriot_name"], config["mriot_year"])
    mriot.G = calc_G(calc_B(mriot.Z, mriot.x))
    # 1. Filter out all rows that do not require any computation.
    df_to_process = df[df["_indirect_leontief_calculate"] | df["_indirect_ghosh_calculate"]].copy()

    if df_to_process.empty:
        LOGGER.info("No indirect impacts to calculate.")
        return

    LOGGER.info(f"Starting indirect impact calculations for {len(df_to_process)} configurations.")

    # 2. Group rows by shared input data to avoid redundant loading.
    # First, group by exposure ('country' and 'sector').
    exposure_groups = df_to_process.groupby(["country", "sector"])

    for (country, sector), exposure_df in exposure_groups:
        LOGGER.info(f"Processing exposure for country: '{country}', sector: '{sector}'")
        try:
            # Load the shared exposure data once for this group.
            exp = get_sector_exposure(sector=sector, country=country, data_path=config["data_path"])
            impacted_secs = SUPER_SEC[sector]
            #impacted_secs = mriot.get_sectors()[sec_range].tolist()
            assert len(exp.gdf["region_id"].unique()) == 1
        except Exception as e:
            LOGGER.error(
                f"Failed to load exposure for country: '{country}', sector: '{sector}'. Skipping group.",
                exc_info=True,
            )
            continue

        # Second, group by hazard ('yearset_path') within the exposure group.
        impact_groups = exposure_df.groupby("yearset_path")

        for yearset_path, impact_df in impact_groups:
            LOGGER.info(f"Loading hazard from: {yearset_path}")
            try:
                # Load the shared hazard yearset data once.
                imp = Impact.from_hdf5(yearset_path)
                if not imp.at_event.any():
                    LOGGER.info(f"No non-zero impacts in {yearset_path}. Skipping.")
                    continue
            except Exception as e:
                LOGGER.error(
                    f"Failed to load impact from {yearset_path}. Skipping group.",
                    exc_info=True,
                )
                continue

            # Now, iterate through each specific configuration in this group.
            country_iso3alpha = pycountry.countries.get(name=country).alpha_3
            cc = coco.CountryConverter()
            country_in_mriot = cc.convert(country_iso3alpha, to=MRIOT_TYPE[mriot.name]).upper()
            direct_shock = DirectShocksSet._init_with_mriot(
                mriot=mriot,
                exposure_assets=translate_exp_to_sectors(
                    exp, affected_sectors=impacted_secs, mriot=mriot, value_col="value"
                ),
                impacted_assets=distribute_reg_impact_to_sectors(
                    imp.impact_at_reg().sum(axis=1).to_frame(country_in_mriot),
                    mriot.x.loc[
                        pd.IndexSlice[country_in_mriot, impacted_secs],
                        mriot.x.columns[0],
                    ],
                ),
                event_dates=imp.date,
                shock_name=f"{country}_{sector}",
            )  # renamed the function from
            for _, row in impact_df.iterrows():
                # The core computation logic for a single row will be placed here.
                # Both `exp` and `imp` are already loaded.
                #
                # TODO: Implement the calculation and result-saving logic from the
                # original `indirect_impacts_from_row` function here. This will
                # involve:
                #   - Looping through the `io_approach` from the config.
                #   - Checking if the specific calculation (leontief/ghosh) is needed.
                #   - Calling `supply_chain_climada(exp, imp, ...)`.
                #   - Dumping results to the various CSV files.
                logging_dict = {
                    k: row[k] for k in ["hazard", "sector", "country", "scenario", "ref_year"]
                }
                LOGGER.info(f"Running calculation for: {logging_dict}")
                try:
                    dump_direct_to_csv(
                        direct_shocks=direct_shock,
                        mriot=mriot,
                        haz_type=row["hazard"],
                        sector=row["sector"],
                        scenario=row["scenario"],
                        ref_year=row["ref_year"],
                        country=row["country"],
                        n_sim=config["n_sim_years"],
                        return_period=100,
                        output_file=row[f"supchain_direct_path"],
                    )
                except Exception as e:
                    LOGGER.error(
                        f"Failed to dump direct imapcts.",
                        exc_info=True,
                    )
                    continue

                try:
                    model = StaticIOModel(mriot=mriot, direct_shocks=direct_shock)
                    res = model.calc_indirect_impacts()
                    for io_a in config["io_approach"]:
                        dump_supchain_events_to_csv(
                            model=model,
                            results=res,
                            haz_type=row["hazard"],
                            sector=row["sector"],
                            scenario=row["scenario"],
                            ref_year=row["ref_year"],
                            country=row["country"],
                            io_approach=io_a,
                            output_file=row[f"supchain_indirect_events_{io_a}_path"],
                        )
                        dump_supchain_to_csv(
                            model=model,
                            results=res,
                            haz_type=row["hazard"],
                            sector=row["sector"],
                            scenario=row["scenario"],
                            ref_year=row["ref_year"],
                            country=row["country"],
                            n_sim=config["n_sim_years"],
                            return_period=100,
                            io_approach=io_a,
                            output_file=row[f"supchain_indirect_{io_a}_path"],
                        )

                except Exception as e:
                    LOGGER.error(
                        f"Error calculating indirect impacts for {logging_dict}:",
                        exc_info=True,
                    )


def exists_impact_file(filepath: str):
    """Check if an impact object exists at a filepath.

    Parameters
    ----------
    filepath : str
        Path to requested file

    Returns
    -------
    bool
        Whether the impact file exists
    """
    return os.path.exists(filepath)


def get_impact_from_file(filepath: str):
    """Load an impact object from a filepath.

    Parameters
    ----------
    filepath : str
        Path to requested file

    Returns
    -------
    climada.engine.impact.Impact
        CLIMADA Impact object loaded from the filepath
    """
    if os.path.exists(filepath):
        return Impact.from_hdf5(filepath)
    
    raise FileExistsError(f"Could not find an impact object at {filepath}")


def write_impact_to_file(imp: Impact, filepath: str):
    imp.write_hdf5(filepath)


def _check_config_valid_for_indirect_aggregations(config):
    # Check all scenarios lists are the same length OR length 1
    scenarios_list_list = [run["scenario_years"] for run in config["runs"]]
    n_scenarios_list = [
        len(scenario_list) for scenario_list in scenarios_list_list if len(scenario_list) > 1
    ]
    if len(np.unique(n_scenarios_list)) > 1:
        raise ValueError(
            "To continue with generation of yearsets and indirect impacts, the config needs to have the "
            "same number of scenarios specified for each hazard in the config, or just one scenario."
        )
    return 1 if len(n_scenarios_list) == 0 else n_scenarios_list[0]


if __name__ == "__main__":
    # This is the full run
    # from run_configurations.config import CONFIG
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG to see all messages from all loggers
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Important for overriding previous basicConfig calls
    )
    LOGGER.setLevel("INFO")
    # Note: the logging level for CLIMADA is set separately in the climada.conf file

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    # This is for testing
    from run_configurations.test_config import (
        CONFIG,
        CONFIG2,
        CONFIG3,
        CONFIG4,
        CONFIG5,
        CONFIG6,
    )  # change here to test_config if needed

    run_pipeline_from_config(CONFIG)
