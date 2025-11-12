import logging
import glob
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from nccs.utils.folder_naming import get_indirect_output_dir, get_run_dir

LOGGER = logging.getLogger(__name__)
#RUN_TITLE = "test_sea"


def aggregate(
    df: pd.DataFrame, groupby: List[str], return_period: int = 100, n_sim: int = 1000
) -> pd.DataFrame:
    """Aggregate impacts to a given return period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with event impacts. Must contain 'impact', 'event_id',
        'scenario', and 'ref_year' columns, plus columns specified in `groupby`.
    groupby : list[str]
        List of columns to group by for the aggregation.
    return_period : int, optional
        The return period to calculate the impact for, by default 100.
    n_sim : int, optional
        Total number of simulation years, by default 1000.

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated impacts for the specified return period.
    """

    def sum_extract_rp(x: pd.DataFrame) -> pd.Series:
        """Sort by impact and extract the impact at the return period index."""
        res = x.sort_values(ascending=False, by="impact")
        try:
            return pd.Series(
                {
                    "impact": res["impact"].iloc[index_rp],
                    "event_id": res.event_id.iloc[index_rp],
                }
            )
        except IndexError:
            return pd.Series({"impact": 0, "event_id": None})

    index_rp = np.floor(n_sim / return_period).astype(int) - 1
    groupby = ["scenario", "ref_year"] + groupby
    dd: pd.DataFrame = (
        df.groupby(groupby + ["event_id"])[["impact"]].sum().reset_index()
    )
    dd = dd.groupby(groupby).apply(sum_extract_rp).reset_index()
    return dd


def aggregate_yearset_return_periods(
    run_title: str, return_period: int, n_sim: int = 1000
) -> None:
    """Performs aggregations on the yearsets after the supply chain simulation.

    Reads all event CSVs for a given run, aggregates them based on different
    criteria (e.g., by sector, by hazard), and saves the results to new CSV
    files.

    Parameters
    ----------
    run_title : str
        The title of the run, used to locate input and output directories.
    return_period : int
        The return period in years for which to calculate impacts.
    n_sim : int, optional
        The total number of simulation years, by default 1000.

    Raises
    ------
    ValueError
        If the output directory for the given `run_title` does not exist.
    """
    events_dir = Path(get_indirect_output_dir(run_title), "events")
    aggregates_dir = Path(get_run_dir(run_title), "aggregates")

    if not os.path.exists(get_run_dir(run_title)):
        raise ValueError(
            f"Could not locate existing output directory {get_run_dir(run_title)}"
        )
    os.makedirs(aggregates_dir, exist_ok=True)

    files = [f for f in glob.glob(str(events_dir) + '/*.csv')]
    LOGGER.info(f"Reading event files from {events_dir}:\n{files})")
    df: pd.DataFrame = pd.concat(
        [pd.read_csv(f) for f in files]
    ).reset_index(drop=True)
    # We need to replace the future with 2060 in the ref_year
    df.loc[df.ref_year == 2060, "ref_year"] = "future"
    # Ensure the historic period is grouped as well
    df.loc[pd.isna(df["scenario"]), "scenario"] = "historic"

    aggrs: List[Dict[str, Any]] = [
        {"groupby": ["sector"], "filename": "by_sector.csv"},
        {"groupby": ["hazard_type"], "filename": "by_hazard.csv"},
        {"groupby": ["country_of_impact"], "filename": "by_country.csv"},
        {"groupby": ["sector", "hazard_type"], "filename": "by_sector_hazard.csv"},
        {
            "groupby": ["sector", "country_of_impact"],
            "filename": "by_sector_country.csv",
        },
        {
            "groupby": ["hazard_type", "country_of_impact"],
            "filename": "by_hazard_country.csv",
        },
        {
            "groupby": ["sector", "hazard_type", "country_of_impact"],
            "filename": "by_sector_hazard_country.csv",
        },
    ]

    for io_approach in ["ghosh", "leontief"]:
        for aggr in aggrs:
            df_agg = aggregate(df, aggr["groupby"], return_period, n_sim)
            df_agg.to_csv(
                Path(aggregates_dir, f"{io_approach}_{aggr['filename']}"), index=False
            )


if __name__ == "__main__":
    aggregate_yearset_return_periods(RUN_TITLE, 100)
