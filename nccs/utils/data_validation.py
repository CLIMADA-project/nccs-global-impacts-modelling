import sys
import os
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
import pycountry
import logging
from pathos.multiprocessing import Pool
from climada.engine import Impact

sys.path.append(
    "../"
)  # HELP: what's the correct way to load things from the parent directory?
from nccs.analysis import (
    config_to_dataframe,
    df_extend_with_multihazard,
    get_impact_from_file,
    df_create_combined_hazard_yearsets_agriculture,
    df_create_combined_hazard_yearsets,
    _check_config_valid_for_indirect_aggregations,
    sort_analysis_df,
)
from nccs.pipeline.direct.direct import get_hazard, get_sector_exposure
from nccs.utils import folder_naming

use_s3 = False  # Not ready yet
log_level = "INFO"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(log_level)
FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
CONSOLE = logging.StreamHandler(stream=sys.stdout)
CONSOLE.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE)

# Validate all the data!


def validate_from_config(config):
    LOGGER.info(f'Analysing inputs and outputs for {config["run_title"]}')

    if "direct_output_dir" in config.keys():
        direct_output_dir = config["direct_output_dir"]
    else:
        direct_output_dir = folder_naming.get_direct_output_dir(config["run_title"])
        config["direct_output_dir"] = direct_output_dir
    direct_output_dir_impact = Path(direct_output_dir, "impact_raw")
    direct_output_dir_yearsets = Path(direct_output_dir, "yearsets")

    if "indirect_output_dir" in config.keys():
        indirect_output_dir = config["indirect_output_dir"]
    else:
        indirect_output_dir = folder_naming.get_indirect_output_dir(config["run_title"])
        config["indirect_output_dir"] = indirect_output_dir

    analysis_df = config_to_dataframe(config)
    LOGGER.info(f"There are {analysis_df.shape[0]} analysis calculations in this run")
    os.makedirs(Path(direct_output_dir).parent, exist_ok=True)

    # HAZARD
    # ------
    LOGGER.info("Validating hazard data")
    haz_id_cols = ["hazard", "country", "scenario", "i_scenario", "ref_year"]
    hazard_df = analysis_df[haz_id_cols].drop_duplicates()
    LOGGER.info(f"{hazard_df.shape[0]} hazard objects to validate")

    def validate_hazard(row):
        country_iso3alpha = pycountry.countries.get(name=row["country"]).alpha_3
        d = {
            "hazard": row["hazard"],
            "country": row["country"],
            "scenario": row["scenario"],
            "i_scenario": row["i_scenario"],
            "ref_year": row["ref_year"],
            "haz_exists": False,
            "haz_exists_error": None,
            "haz_has_events": None,
            "haz_nonzero": None,
        }
        try:
            LOGGER.debug(f"Hazard check: {d}")
            haz = get_hazard(
                row["hazard"], country_iso3alpha, row["scenario"], row["ref_year"]
            )
            d["haz_exists"] = True
            d["haz_has_events"] = haz.intensity.shape[0] > 0
            d["haz_nonzero"] = (
                ~(haz.intensity.max() == 0 and haz.intensity.min() == 0)
                if d["haz_has_events"]
                else None
            )
        except Exception as e:
            # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = e
            LOGGER.debug(f"Error: {msg}:")
            d["haz_exists_error"] = msg
        return d

    if not config["do_parallel"]:
        haz_results = [validate_hazard(row) for _, row in hazard_df.iterrows()]
    else:
        with Pool(processes=config["ncpus"]) as pool:
            haz_results = pool.map(
                validate_hazard, [row for _, row in hazard_df.iterrows()]
            )

    haz_results = pd.DataFrame(haz_results)
    analysis_df = analysis_df.merge(haz_results, how="left", on=haz_id_cols)

    # EXPOSURE
    # --------
    LOGGER.info("Validating exposure data")
    exposure_id_cols = ["sector", "country"]
    exposure_df = analysis_df[exposure_id_cols].drop_duplicates()
    LOGGER.info(f"{exposure_df.shape[0]} exposure objects to validate")

    def validate_exposure(row):
        country_iso3alpha = pycountry.countries.get(name=row["country"]).alpha_3
        d = {
            "sector": row["sector"],
            "country": row["country"],
            "exp_exists": False,
            "exp_exists_error": None,
            "exp_nonzero": None,
        }
        try:
            LOGGER.debug(f"Exposure check: {d}")
            exp = get_sector_exposure(row["sector"], row["country"], data_path=config['data_path'])
            d["exp_exists"] = True
            d["exp_has_values"] = exp.gdf.shape[0] > 0
            d["exp_nonzero"] = exp.gdf.value.max() != 0 if d["exp_has_values"] else None
        except Exception as e:
            # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = e
            LOGGER.debug(f"Error: {msg}:")
            d["exp_exists_error"] = msg
        return d

    if not config["do_parallel"]:
        exp_results = [validate_exposure(row) for _, row in exposure_df.iterrows()]
    else:
        with Pool(processes=config["ncpus"]) as pool:
            exp_results = pool.map(
                validate_exposure, [row for _, row in exposure_df.iterrows()]
            )

    exp_results = pd.DataFrame(exp_results)
    analysis_df = analysis_df.merge(exp_results, how="left", on=exposure_id_cols)

    # DIRECT IMPACTS
    # --------------
    LOGGER.info("Validating direct impact data")
    LOGGER.info(f"{analysis_df.shape[0]} impact objects to validate")
    impact_id_cols = [
        "hazard",
        "sector",
        "country",
        "scenario",
        "i_scenario",
        "ref_year",
    ]

    def validate_direct_impact(row):
        d = {
            "hazard": row["hazard"],
            "sector": row["sector"],
            "country": row["country"],
            "scenario": row["scenario"],
            "i_scenario": row["i_scenario"],
            "ref_year": row["ref_year"],
            "imp_exists": False,
            "imp_exists_error": None,
            "imp_has_events": None,
            "imp_nonzero": None,
        }
        try:
            LOGGER.debug(f"Direct impacts check: {d}")
            imp = get_impact_from_file(row["direct_impact_path"], use_s3=use_s3)
            d["imp_exists"] = True
            d["imp_has_events"] = len(imp.at_event) > 0
            d["imp_nonzero"] = (
                ~(imp.at_event.max() == 0 and imp.at_event.min() == 0)
                if d["imp_has_events"]
                else None
            )
        except Exception as e:
            # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = e
            LOGGER.debug(f"Error: {msg}:")
            d["imp_exists_error"] = msg
        return d

    if not config["do_parallel"]:
        imp_results = [validate_direct_impact(row) for _, row in analysis_df.iterrows()]
    else:
        with Pool(processes=config["ncpus"]) as pool:
            imp_results = pool.map(
                validate_direct_impact, [row for _, row in analysis_df.iterrows()]
            )

    imp_results = pd.DataFrame(imp_results)
    analysis_df = analysis_df.merge(imp_results, how="left", on=impact_id_cols)

    # (RE)COMBINE AGRICULTURE
    # -----------------------
    LOGGER.info("Reformatting and aggregating agriculture data")
    analysis_df_crop = analysis_df[
        analysis_df["hazard"].str.contains("relative_crop_yield")
    ]
    analysis_df_no_crop = analysis_df[
        ~analysis_df["hazard"].str.contains("relative_crop_yield")
    ]

    def first_nonmissing(series):
        return None if not any(series) else series.dropna().iloc[0]

    if analysis_df_crop.shape[0] > 0:
        grouping_cols = ["i_scenario", "country"]
        df_aggregated_counts = (
            analysis_df_crop.groupby(grouping_cols)
            .aggregate(
                {
                    "haz_exists": "all",
                    "haz_exists_error": first_nonmissing,
                    "haz_has_events": "any",
                    "haz_nonzero": "any",
                    "exp_exists": "all",
                    "exp_exists_error": first_nonmissing,
                    "exp_nonzero": "any",
                    "exp_has_values": "any",
                    "imp_exists": "all",
                    "imp_exists_error": first_nonmissing,
                    "imp_has_events": "any",
                    "imp_nonzero": "any",
                }
            )
            .reset_index()
        )

        df_aggregated_agriculture = df_extend_with_multihazard(
            analysis_df_crop,
            df_create_combined_hazard_yearsets_agriculture,
            config,
            grouping_cols,
            False,
            False,
        )
        df_aggregated_agriculture = df_aggregated_agriculture[
            grouping_cols
            + [
                "hazard",
                "scenario",
                "ref_year",
                "sector",
                "yearset_path",
                "supchain_direct_path",
                "supchain_indirect_leontief_path",
                "supchain_indirect_ghosh_path",
            ]
        ]

        df_aggregated_counts = df_aggregated_counts.merge(
            df_aggregated_agriculture, on=grouping_cols
        )

        # df_aggregated_counts[grouping_cols] = analysis_df_crop[grouping_cols].drop_duplicates().reset_index(drop=True)
        analysis_df = pd.concat(
            [analysis_df_no_crop, df_aggregated_counts]
        ).reset_index(drop=True)

    # COMBINED YEARSETS FOR MULTIHAZARD
    # ---------------------------------
    grouping_cols = ["i_scenario", "sector", "country"]
    if config["do_multihazard"]:
        LOGGER.info("Expanding validation tasks for multihazard")
        analysis_df = df_extend_with_multihazard(
            analysis_df,
            df_create_combined_hazard_yearsets,
            config,
            grouping_cols,
            True,
            False,
        ).drop(columns=["_yearset_exists"])

    # YEARSETS
    # --------
    LOGGER.info("Validating yearset data")
    LOGGER.info(f"{analysis_df.shape[0]} yearsets to validate")

    def validate_yearsets(row):
        d = {
            "hazard": row["hazard"],
            "sector": row["sector"],
            "country": row["country"],
            "scenario": row["scenario"],
            "i_scenario": row["i_scenario"],
            "ref_year": row["ref_year"],
            "yearset_exists": False,
            "yearset_exists_error": None,
            "yearset_has_events": None,
            "yearset_nonzero": None,
        }
        try:
            LOGGER.debug(f"Yearset check: {d}")
            yearset = get_impact_from_file(row["yearset_path"], use_s3=use_s3)
            d["yearset_exists"] = True
            d["yearset_has_events"] = len(yearset.at_event) > 0
            d["yearset_nonzero"] = (
                ~(yearset.at_event.max() == 0 and yearset.at_event.min() == 0)
                if d["yearset_has_events"]
                else None
            )
        except Exception as e:
            # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = e
            LOGGER.debug(f"Error: {msg}:")
            d["yearset_exists_error"] = msg
        return d

    if not config["do_parallel"]:
        yearset_results = [validate_yearsets(row) for _, row in analysis_df.iterrows()]
    else:
        with Pool(processes=config["ncpus"]) as pool:
            yearset_results = pool.map(
                validate_yearsets, [row for _, row in analysis_df.iterrows()]
            )

    yearset_results = pd.DataFrame(yearset_results)
    analysis_df = analysis_df.merge(yearset_results, how="left", on=impact_id_cols)

    # SUPPLY CHAIN DIRECT IMPACTS
    # ---------------------------
    LOGGER.info("Validating direct supply chain impacts")
    LOGGER.info(f"{analysis_df.shape[0]} supply chain objects to validate")

    def validate_supply_chain_direct(row):
        d = {
            "hazard": row["hazard"],
            "sector": row["sector"],
            "country": row["country"],
            "scenario": row["scenario"],
            "i_scenario": row["i_scenario"],
            "ref_year": row["ref_year"],
            "supchain_direct_exists": False,
            "supchain_direct_exists_error": None,
            "supchain_direct_nonzero": None,
        }
        try:
            LOGGER.debug(f"Supply chain direct check: {d}")
            supchain_direct = pd.read_csv(row["supchain_direct_path"])
            d["supchain_direct_exists"] = True
            d["supchain_direct_nonzero"] = (
                supchain_direct.shape[0] > 0 and supchain_direct["AAPL"].max() > 0
            )
        except Exception as e:
            # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = e
            LOGGER.debug(f"Error: {msg}:")
            d["supchain_direct_exists_error"] = msg
        return d

    if not config["do_parallel"]:
        supchain_direct_results = [
            validate_supply_chain_direct(row) for _, row in analysis_df.iterrows()
        ]
    else:
        with Pool(processes=config["ncpus"]) as pool:
            supchain_direct_results = pool.map(
                validate_supply_chain_direct, [row for _, row in analysis_df.iterrows()]
            )

    supchain_direct_results = pd.DataFrame(supchain_direct_results)
    analysis_df = analysis_df.merge(
        supchain_direct_results, how="left", on=impact_id_cols
    )

    # SUPPLY CHAIN INDIRECT IMPACTS: LEONTIEF
    # ---------------------------------------
    LOGGER.info("Validating indirect supply chain impacts: Leontief")
    LOGGER.info(f"{analysis_df.shape[0]} indirect impacts to validate")

    def validate_supply_chain_indirect_leontief(row):
        d = {
            "hazard": row["hazard"],
            "sector": row["sector"],
            "country": row["country"],
            "scenario": row["scenario"],
            "i_scenario": row["i_scenario"],
            "ref_year": row["ref_year"],
            "supchain_indirect_leontief_exists": False,
            "supchain_indirect_leontief_exists_error": None,
            "supchain_indirect_leontief_nonzero": None,
        }
        try:
            LOGGER.debug(f"Supply chain indirect Leontief check: {d}")
            supchain_indirect_leontief = pd.read_csv(
                row["supchain_indirect_leontief_path"]
            )
            d["supchain_indirect_leontief_exists"] = True
            d["supchain_indirect_leontief_nonzero"] = (
                supchain_indirect_leontief.shape[0] > 0
                and supchain_indirect_leontief["AAPL"].max() > 0
                if d["supchain_indirect_leontief_exists"]
                else None
            )
        except Exception as e:
            # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = e
            LOGGER.debug(f"Error: {msg}:")
            d["supchain_indirect_leontief_exists_error"] = msg
        return d

    if not config["do_parallel"]:
        supchain_indirect_leontief_results = [
            validate_supply_chain_indirect_leontief(row)
            for _, row in analysis_df.iterrows()
        ]
    else:
        with Pool(processes=config["ncpus"]) as pool:
            supchain_indirect_leontief_results = pool.map(
                validate_supply_chain_indirect_leontief,
                [row for _, row in analysis_df.iterrows()],
            )

    supchain_indirect_leontief_results = pd.DataFrame(
        supchain_indirect_leontief_results
    )
    analysis_df = analysis_df.merge(
        supchain_indirect_leontief_results, how="left", on=impact_id_cols
    )

    # SUPPLY CHAIN INDIRECT IMPACTS: GHOSH
    # ------------------------------------
    LOGGER.info("Validating indirect supply chain impacts: Ghosh")
    LOGGER.info(f"{analysis_df.shape[0]} indirect impacts to validate")

    def validate_supply_chain_indirect_ghosh(row):
        d = {
            "hazard": row["hazard"],
            "sector": row["sector"],
            "country": row["country"],
            "scenario": row["scenario"],
            "i_scenario": row["i_scenario"],
            "ref_year": row["ref_year"],
            "supchain_indirect_ghosh_exists": False,
            "supchain_indirect_ghosh_exists_error": None,
            "supchain_indirect_ghosh_nonzero": None,
        }
        try:
            LOGGER.debug(f"Supply chain indirect Ghosh check: {d}")
            supchain_indirect_ghosh = pd.read_csv(row["supchain_indirect_ghosh_path"])
            d["supchain_indirect_ghosh_exists"] = True
            d["supchain_indirect_ghosh_nonzero"] = (
                supchain_indirect_ghosh.shape[0] > 0
                and supchain_indirect_ghosh["iAAPL"].max() > 0
                if d["supchain_indirect_ghosh_exists"]
                else None
            )
        except Exception as e:
            # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            msg = e
            LOGGER.debug(f"Error: {msg}:")
            d["supchain_indirect_ghosh_exists_error"] = msg
        return d

    if not config["do_parallel"]:
        supchain_indirect_ghosh_results = [
            validate_supply_chain_indirect_ghosh(row)
            for _, row in analysis_df.iterrows()
        ]
    else:
        with Pool(processes=config["ncpus"]) as pool:
            supchain_indirect_ghosh_results = pool.map(
                validate_supply_chain_indirect_ghosh,
                [row for _, row in analysis_df.iterrows()],
            )

    supchain_indirect_ghosh_results = pd.DataFrame(supchain_indirect_ghosh_results)
    analysis_df = analysis_df.merge(
        supchain_indirect_ghosh_results, how="left", on=impact_id_cols
    )

    analysis_df = sort_analysis_df(analysis_df)

    # Internal logic validation checks
    # --------------------------------
    # Note, these can sometimes fail if some intermediate data is missing or moved,
    # or if file names weren't what was expected, etc. It's not automatically an error
    LOGGER.info("Checking for logical errors in the validation itself")
    LOGGER.info(f"{analysis_df.shape[0]} rows to validate")

    analysis_df["validation_makes_sense"] = True
    for i, row in analysis_df.iterrows():
        valid = True
        # Missing hazard -> missing impacts
        if not row["haz_exists"] and row["imp_exists"]:
            valid = 1
        # Missing hazard -> empty hazard
        if not row["haz_exists"] and (row["haz_has_events"] or row["haz_nonzero"]):
            valid = 2
        # no hazard events -> zero hazard
        if not row["haz_has_events"] and row["haz_nonzero"]:
            valid = 3
        # zero hazard -> zero impacts
        if not row["haz_nonzero"] and row["imp_nonzero"]:
            valid = 4

        # Missing exposure -> missing impacts
        if not row["exp_exists"] and row["imp_exists"]:
            valid = 5
        # Missing exp -> empty exp
        if not row["exp_exists"] and (row["exp_has_values"] or row["exp_nonzero"]):
            valid = 6
        # No exposure values -> zero exposure
        if not row["exp_has_values"] and row["exp_nonzero"]:
            valid = 7
        # Zero exposure -> zero impacts
        if not row["exp_nonzero"] and row["imp_nonzero"]:
            valid = 8

        # Missing impact -> missing yearset
        if not row["imp_exists"] and row["yearset_exists"]:
            valid = 9
        # Missing impact -> empty impact
        if not row["imp_exists"] and (row["imp_has_events"] or row["imp_nonzero"]):
            valid = 10
        # No imp events -> zero impact
        if not row["imp_has_events"] and row["imp_nonzero"]:
            valid = 11

        # Missing yearset -> missing supchain direct
        if not row["yearset_exists"] and row["supchain_direct_exists"]:
            valid = 12
        # Missing yearset -> empty yearset
        if not row["yearset_exists"] and (
            row["yearset_has_events"] or row["yearset_nonzero"]
        ):
            valid = 13

        # Missing supchain direct -> missing supchain
        if not row["supchain_direct_exists"] and (
            row["supchain_indirect_leontief_exists"]
            or row["supchain_indirect_ghosh_exists"]
        ):
            valid = 14
        # Missing supchain -> empty supchain
        if not row["supchain_direct_exists"] and row["supchain_direct_nonzero"]:
            valid = 15

        # Missing leontief or ghosh -> empty output
        if (
            not row["supchain_indirect_leontief_exists"]
            and row["supchain_indirect_leontief_nonzero"]
        ):
            valid = 16
        if (
            not row["supchain_indirect_ghosh_exists"]
            and row["supchain_indirect_ghosh_nonzero"]
        ):
            valid = 17

        analysis_df.loc[i, "validation_makes_sense"] = valid

    # TODO also check existing outputs against the xlsx output

    # Write output
    # ------------

    out_path = Path(folder_naming.get_run_dir(config["run_title"]), "validation.csv")
    analysis_df.to_csv(out_path)

    LOGGER.info("DATA VALIDATION SUMMARY:")
    LOGGER.info("========================")
    LOGGER.info(f'Run name:                 {config["run_title"]}')
    LOGGER.info(
        f'Hazard calculations:      {[run["hazard"] for run in config["runs"]]}'
    )
    LOGGER.info(
        f'Hazard data exists:       {haz_results["haz_exists"].sum()} / {haz_results.shape[0]}'
    )
    LOGGER.info(
        f'Exposure data exists:     {exp_results["exp_exists"].sum()} / {exp_results.shape[0]}'
    )
    LOGGER.info(
        f'Impact data exists:       {imp_results["imp_exists"].sum()} / {imp_results.shape[0]}'
    )
    LOGGER.info(
        f'Yearset data exists:      {yearset_results["yearset_exists"].sum()} / {yearset_results.shape[0]}'
    )
    LOGGER.info(
        f'Supchain direct exists:   {supchain_direct_results["supchain_direct_exists"].sum()} / {supchain_direct_results.shape[0]}'
    )
    LOGGER.info(
        f'Supchain Leontief exists: {supchain_indirect_leontief_results["supchain_indirect_leontief_exists"].sum()} / {supchain_indirect_leontief_results.shape[0]}'
    )
    LOGGER.info(
        f'Supchain Ghosh exists:    {supchain_indirect_ghosh_results["supchain_indirect_ghosh_exists"].sum()} / {supchain_indirect_ghosh_results.shape[0]}'
    )
    LOGGER.info("")
    LOGGER.info(f"More details and diagnostics in {out_path}")


if __name__ == "__main__":
    from nccs.run_configurations.test_multi import (
        CONFIG,
    )  # change here to test_config if needed

    # from nccs.run_configurations.test.test_config import CONFIG  # change here to test_config if needed
    validate_from_config(CONFIG)
