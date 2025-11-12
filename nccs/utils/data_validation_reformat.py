from pathlib import Path
import pandas as pd
import numpy as np
import sys
import logging

from nccs.utils import folder_naming

# Script that takes the output from data_validation and reformats it to be more human-friendly
# NOTE: this assumes that the configuration is the same as the main NCCS runs, with fixed scenarios

use_s3 = False  # Not ready yet
log_level = "INFO"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(log_level)
FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
CONSOLE = logging.StreamHandler(stream=sys.stdout)
CONSOLE.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE)


def reformat_validation(config=None, input_csv_path=None):
    if not config and not input_csv_path:
        raise ValueError("Need at least one input pls")

    if not input_csv_path:
        input_csv_path = Path(
            folder_naming.get_run_dir(config["run_title"]), "validation.csv"
        )

    LOGGER.info(f"Reformatting file at {input_csv_path}")

    out_dir = Path(input_csv_path).absolute().parent
    out_file = Path(input_csv_path).absolute().parts[-1]
    out_file = str(out_file).split(".")[0] + "_reformatted.csv"
    out_path = Path(out_dir, out_file)

    df = pd.read_csv(input_csv_path)

    # Build a new, better output table
    out = pd.DataFrame()
    out["Hazard"] = df["hazard"]
    out["Sector"] = df["sector"]
    out["Country"] = df["country"]

    # Convert scenarios to human readable.
    scenario_mapping = [
        "Historical",
        "Future: low climate change",
        "Future: high climate change",
    ]
    assert len(np.unique(df["i_scenario"])) == 3
    out["Scenario"] = [scenario_mapping[x] for x in df["i_scenario"]]
    out["Hazard file exists"] = df["haz_exists"]
    out["Hazard not all zero"] = np.multiply(
        df["haz_has_events"].fillna(False), df["haz_nonzero"].fillna(False)
    ).astype(bool)
    out["Exposure file exists"] = df["exp_exists"]
    out["Exposure not all zero"] = np.multiply(
        df["exp_nonzero"].fillna(False), df["exp_has_values"].fillna(False)
    ).astype(bool)
    out["Input to supply chain exists"] = df["yearset_exists"]
    out["Input to supply chain not all zero"] = np.multiply(
        df["yearset_has_events"].fillna(False), df["yearset_nonzero"].fillna(False)
    ).astype(bool)
    out["Supply chain shocks calculated"] = df["supchain_direct_exists"]
    out["Supply chain shocks not all zero"] = df["supchain_direct_nonzero"].fillna(
        False
    )
    out["Leontief supply chain output exists"] = df["supchain_indirect_leontief_exists"]
    out["Ghosh supply chain output exists"] = df["supchain_indirect_ghosh_exists"]

    LOGGER.info(f"Writing output to {out_path}")
    out.to_csv(out_path, index=False)


if __name__ == "main":
    from nccs.run_configurations.test_multi import CONFIG

    reformat_validate_from_config(config=CONFIG)
