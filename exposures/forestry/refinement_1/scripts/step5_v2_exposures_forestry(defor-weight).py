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
REPR_SECTORS_DEFAULT = "Forestry and logging"
DATA_H5_PATH = "exposures/forestry/refinement_1/intermediate_data/forest_exp_osm_defor(v2).h5"
WB_CSV_PATH = "exposures/forestry/refinement_1/WorldBank_forestry_production.csv"
REGION_ID_COL = "region_id"
WEIGHT_NORM_COL = "weight_norm"
VALUE_COL = "value"
COUNTRY_CODE_COL = "Country Code"
NORMALIZED_WB_COL = "Normalized_WB"
ROW_REGION = "ROW"
HDF_KEY = "data"
HDF_MODE = "w"

# Final output file base name (used for H5, SHP, and country splits)
OUTPUT_FILE_BASE = "forestry_values"
OUTPUT_DIR = "exposures/forestry/refinement_1"
COUNTRY_SPLIT_DIR = "exposures/forestry/refinement_1/country_split"
# --- END CONSTANTS ---


# Get the root directory
project_root = root_dir()


def get_forestry_exp_new_2(
    countries=None,
    mriot_type=MRIOT_TYPE_DEFAULT,
    mriot_year=MRIOT_YEAR_DEFAULT,
    repr_sectors=REPR_SECTORS_DEFAULT,
):

    glob_prod, repr_sectors, IO_countries = get_prod_secs(
        mriot_type, mriot_year, repr_sectors
    )

    ROW_WB_lookup = get_ROW_factor_WB_forestry(mriot_year, IO_countries, countries)

    cnt_dfs = []

    data = pd.read_hdf(
        f"{project_root}/{DATA_H5_PATH}"
    )

    for iso3_cnt in countries:

        cnt_df = data.loc[data[REGION_ID_COL] == iso3_cnt]

        try:
            cnt_df[VALUE_COL] = (
                glob_prod.loc[iso3_cnt].loc[repr_sectors].sum().values[0] / len(cnt_df)
            ) * cnt_df[WEIGHT_NORM_COL]
        except KeyError:
            LOGGER.warning(
                "You are simulating a country for which there are no production data in the chosen IOT"
            )

            try:
                ROW_WB_factor = ROW_WB_lookup.loc[
                    ROW_WB_lookup[COUNTRY_CODE_COL] == iso3_cnt, NORMALIZED_WB_COL
                ].values[0]
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_WB_factor
                cnt_df[VALUE_COL] = (ROW_country_production / len(cnt_df)) * cnt_df[
                    WEIGHT_NORM_COL
                ]
            except:
                print(
                    f"For the country {iso3_cnt} there is no WB value available, 0 value is assigned"
                )
                ROW_WB_factor = 0
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_WB_factor
                cnt_df[VALUE_COL] = (ROW_country_production / len(cnt_df)) * cnt_df[
                    WEIGHT_NORM_COL
                ]

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


def get_ROW_factor_WB_forestry(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the forestry production of counries
    forestry_prod_WB = pd.read_csv(
        f"{project_root}/{WB_CSV_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code'
    ROW_forestry_prod_WB = forestry_prod_WB[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~forestry_prod_WB[COUNTRY_CODE_COL].isin(IO_countries)
    ]
    # Assuming ROW_forestry_prod_WB is your DataFrame and country is your list of countries
    filtered_forestry_prod_WB = ROW_forestry_prod_WB[
        ROW_forestry_prod_WB[COUNTRY_CODE_COL].isin(countries)
    ]

    ROW_total_WB = filtered_forestry_prod_WB[str(mriot_year)].sum()
    # Create a new column with normalized WB values
    ROW_forestry_prod_WB[NORMALIZED_WB_COL] = (
        ROW_forestry_prod_WB[str(mriot_year)] / ROW_total_WB
    )

    return ROW_forestry_prod_WB


data = pd.read_hdf(
    f"{project_root}/{DATA_H5_PATH}"
)
countries = data[REGION_ID_COL].unique().tolist()
countries.sort()
del data

# apply function that alters the value using MRIO
exp = get_forestry_exp_new_2(
    countries=countries,
    mriot_type=MRIOT_TYPE_DEFAULT,
    mriot_year=MRIOT_YEAR_DEFAULT,
    repr_sectors=REPR_SECTORS_DEFAULT,
)


# # Save a shape file to check it in QGIS
# df_shape = exp.gdf
# filename_shp = f"{project_root}/{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.shp"
# s3_filename_shp = f"{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.shp"
# df_shape.to_file(filename_shp,driver="ESRI Shapefile")
# # upload the file to the s3 Bucket
# upload_to_s3_bucket(filename_shp, s3_filename_shp)
# print(f"upload of {s3_filename_shp} to s3 bucket successful")


# Save final file to a climada available format h5
df = exp.gdf.drop(columns="geometry")
filename_h5 = (
    f"{project_root}/{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.h5"
)
s3_filename_h5 = f"{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.h5"
df.to_hdf(
    filename_h5, key=HDF_KEY, mode=HDF_MODE
)
# upload the file to the s3 Bucket
upload_to_s3_bucket(filename_h5, s3_filename_h5)
print(f"upload of {s3_filename_h5} to s3 bucket successful")

# Save individual country files #TODO save country splited files to S3 bucket
for region_id in df[REGION_ID_COL].unique():
    subset_df = df[df[REGION_ID_COL] == region_id]
    filename_country = f"{project_root}/{COUNTRY_SPLIT_DIR}/{OUTPUT_FILE_BASE}_{region_id}.h5"
    s3_filename_country = f"{COUNTRY_SPLIT_DIR}/{OUTPUT_FILE_BASE}_{region_id}.h5"
    subset_df.to_hdf(filename_country, key=HDF_KEY, mode=HDF_MODE)
    # upload the individual country files to s3 bucket
    upload_to_s3_bucket(filename_country, s3_filename_country)
    print(f"upload of {s3_filename_country} to s3 bucket successful")