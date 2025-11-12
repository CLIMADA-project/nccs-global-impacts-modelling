import pandas as pd
import numpy as np

from climada.entity import Exposures
from climada_petals.engine import get_mriot

import logging

LOGGER = logging.getLogger()

from exposures.utils import root_dir
from nccs.utils.s3client import upload_to_s3_bucket

# --- CONSTANTS ---
MRIOT_TYPE_DEFAULT = "WIOD16"
MRIOT_YEAR_DEFAULT = 2011

# File Paths
GDP_WB_PATH = "exposures/utilities/refinement_1/GDP_Worldbank_modified_without_regions.csv"
BASE_DATA_PATH = "exposures/utilities/refinement_1/intermediate_data"
OUTPUT_DIR = "exposures/utilities/refinement_1"

# Column Names / Keys
REGION_ID_COL = "region_id"
COUNTRY_NORMALIZED_COL = "country_normalized"
NORMALIZED_AREA_COL = "normalized_area"
VALUE_COL = "value"
COUNTRY_CODE_COL = "Country Code"
NORMALIZED_GDP_COL = "Normalized_GDP"
ROW_REGION = "ROW"
HDF_KEY = "data"
HDF_MODE = "w"

# Subscores and their corresponding MRIO sectors
SUBSCORES = ["Subscore_energy", "Subscore_water", "Subscore_waste"]
SECTOR_MAP = {
    "Subscore_energy": "Electricity, gas, steam and air conditioning supply",
    "Subscore_water": "Water collection, treatment and supply",
    "Subscore_waste": "Sewerage; waste collection, treatment and disposal activities; materials recovery; remediation activities and other waste management services ",
}
# --- END CONSTANTS ---


# Get the root directory
project_root = root_dir()


def get_utilities_exp(
    countries=None,
    mriot_type=MRIOT_TYPE_DEFAULT,
    mriot_year=MRIOT_YEAR_DEFAULT,
    repr_sectors=None,
    data=None,
):

    glob_prod, repr_sectors, IO_countries = get_prod_secs(
        mriot_type, mriot_year, repr_sectors
    )

    ## option 2: distribute ROW production value according to GDP (implemented)
    ROW_gdp_lookup = get_ROW_factor_GDP(mriot_year, IO_countries, countries)

    cnt_dfs = []

    for iso3_cnt in countries:

        cnt_df = data.loc[data[REGION_ID_COL] == iso3_cnt]

        # calculate total amount of infrastructure per country
        country_sum_area = cnt_df[COUNTRY_NORMALIZED_COL].sum()

        try:
            if country_sum_area != 0:
                # Normalize 'amount' values by dividing by total amount
                cnt_df[NORMALIZED_AREA_COL] = (
                    cnt_df[COUNTRY_NORMALIZED_COL] / country_sum_area
                )
            else:
                cnt_df[NORMALIZED_AREA_COL] = cnt_df[COUNTRY_NORMALIZED_COL]
        except Exception:
            print(f"Area of {cnt_df} is not zero")

        try:
            cnt_df[VALUE_COL] = (
                glob_prod.loc[iso3_cnt].loc[repr_sectors].sum().values[0]
                * cnt_df[NORMALIZED_AREA_COL]
            )
        except KeyError:
            LOGGER.warning(
                "You are simulating a country for which there are no production data in the chosen IOT"
            )
            
            # code under option 2:
            try:
                ROW_gdp_factor = ROW_gdp_lookup.loc[
                    ROW_gdp_lookup[COUNTRY_CODE_COL] == iso3_cnt, NORMALIZED_GDP_COL
                ].values[0]
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_gdp_factor
                cnt_df[VALUE_COL] = ROW_country_production * cnt_df[NORMALIZED_AREA_COL]
            except:
                print(
                    f"For the country {iso3_cnt} there is no GDP value available, 0 value is assigned"
                )
                ROW_gdp_factor = 0
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_gdp_factor
                cnt_df[VALUE_COL] = ROW_country_production * cnt_df[NORMALIZED_AREA_COL]

        cnt_dfs.append(cnt_df)

    exp = Exposures(pd.concat(cnt_dfs).reset_index(drop=True))
    exp.set_geometry_points()

    return exp


def get_prod_secs(mriot_type, mriot_year, repr_sectors):
    mriot = get_mriot(mriot_type=mriot_type, mriot_year=mriot_year)

    if isinstance(repr_sectors, (range, np.ndarray)):
        repr_sectors = mriot.get_sectors()[repr_sectors].tolist()

    elif isinstance(repr_sectors, str):
        repr_sectors = [repr_sectors]

    return mriot.x, repr_sectors, mriot.get_regions()


def get_ROW_factor_GDP(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the GDP of counries
    gdp_worldbank = pd.read_csv(
        f"{project_root}/{GDP_WB_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code'
    ROW_gdp_worldbank = gdp_worldbank[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~gdp_worldbank[COUNTRY_CODE_COL].isin(IO_countries)
    ]
    # Assuming ROW_gdp_worldbank is your DataFrame and country is your list of countries
    filtered_gdp_worldbank = ROW_gdp_worldbank[
        ROW_gdp_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]

    ROW_total_GDP = filtered_gdp_worldbank[str(mriot_year)].sum()
    # Create a new column with normalized GDP values
    ROW_gdp_worldbank[NORMALIZED_GDP_COL] = (
        ROW_gdp_worldbank[str(mriot_year)] / ROW_total_GDP
    )

    return ROW_gdp_worldbank


for subscore in SUBSCORES:

    repr_sectors = SECTOR_MAP.get(subscore)

    data = pd.read_hdf(
        f"{project_root}/{BASE_DATA_PATH}/{subscore}_ISO3_normalized.h5"
    )
    countries = data[REGION_ID_COL].unique().tolist()
    countries.sort()

    # apply function that alters the value using MRIO
    exp = get_utilities_exp(
        countries=countries,
        mriot_type=MRIOT_TYPE_DEFAULT,
        mriot_year=MRIOT_YEAR_DEFAULT,
        repr_sectors=repr_sectors,
        data=data,
    )

    # Save final file to a climada available format h5
    df = exp.gdf.drop(columns="geometry")
    filename_h5 = (
        f"{project_root}/{OUTPUT_DIR}/{subscore}/{subscore}_MRIO.h5"
    )
    s3_filename_h5 = f"{OUTPUT_DIR}/{subscore}/{subscore}_MRIO.h5"
    df.to_hdf(
        filename_h5, key=HDF_KEY, mode=HDF_MODE
    )
    # upload the file to the s3 Bucket
    upload_to_s3_bucket(filename_h5, s3_filename_h5)
    print(f"upload of {s3_filename_h5} to s3 bucket successful")

    # Split exposure into countries
    # Save individual country files
    for region_id in df[REGION_ID_COL].unique():
        subset_df = df[df[REGION_ID_COL] == region_id]
        filename_country = f"{project_root}/{OUTPUT_DIR}/{subscore}/country_split/{subscore}_MRIO_{region_id}.h5"
        s3_filename_country = f"{OUTPUT_DIR}/{subscore}/country_split/{subscore}_MRIO_{region_id}.h5"
        subset_df.to_hdf(filename_country, key=HDF_KEY, mode=HDF_MODE)
        # upload the individual country files to s3 bucket
        upload_to_s3_bucket(filename_country, s3_filename_country)
        print(f"upload of {s3_filename_country} to s3 bucket successful")