import pandas as pd
import numpy as np
import geopandas as gpd


from climada.entity import Exposures
from climada_petals.engine import get_mriot

import logging

LOGGER = logging.getLogger()

from exposures.utils import root_dir
from nccs.utils.s3client import upload_to_s3_bucket

# --- CONSTANTS ---
MRIOT_TYPE_DEFAULT = "WIOD16"
MRIOT_YEAR_DEFAULT = 2011
EDGAR_YEAR = 2011
EMISSION_THRESHOLD_DEFAULT = 0
EMISSION_THRESHOLD_SCALED = 100

# File Paths
GENERAL_MANUFAC_WB_PATH = "exposures/manufacturing/manufacturing_general_exposure/refinement_1/WorldBank_Manufac_output_without_regions.csv"
FOOD_MANUFAC_WB_PATH = "exposures/manufacturing/manufacturing_sub_exposures/refinement_1/WorldBank_food_perc_of_manufac_without_regions.csv"
CHEMICAL_MANUFAC_WB_PATH = "exposures/manufacturing/manufacturing/manufacturing_sub_exposures/refinement_1/WorldBank_chemical_perc_of_manufac_without_regions.csv"
BASE_DATA_PATH = "exposures/manufacturing/manufacturing_sub_exposures/refinement_1/intermediate_data_EDGAR"
OUTPUT_DIR = "exposures/manufacturing/manufacturing_sub_exposures/refinement_1"

# Column Names / Keys
REGION_ID_COL = "region_id"
EMISSION_COL = "emission_t"
NORMALIZED_EMISSIONS_COL = "normalized_emissions"
VALUE_COL = "value"
COUNTRY_CODE_COL = "Country Code"
NORMALIZED_PROD_COL = "Normalized_Prod"
ROW_REGION = "ROW"
HDF_KEY = "data"
HDF_MODE = "w"
SHAPEFILE_DRIVER = "ESRI Shapefile"

# File Keys and Sector Names (Dictionaries)
FILE_KEYS = {
    'food_and_paper': 'food_and_paper_manufacture_values',
    'refin_and_transform': 'refin_and_transform_manufacture_values',
    "chemical_process": "chemical_process_manufacture_values",
    "non_metallic_mineral": "non_metallic_mineral_manufacture_values",
    "basic_metals": "basic_metals_manufacture_values",
    "pharmaceutical": "pharmaceutical_manufacture_values",
    "wood": "wood_manufacture_values",
    "rubber_and_plastic": "rubber_and_plastic_manufacture_values",
}

SECTORS = {
    # 'food_and_paper': "Manufacture of food products, beverages and tobacco products",
    # 'refin_and_transform': "Manufacture of coke and refined petroleum products ",
    "chemical_process": "Manufacture of chemicals and chemical products ",
    "non_metallic_mineral": "Manufacture of other non-metallic mineral products",
    "basic_metals": "Manufacture of basic metals",
    "pharmaceutical": "Manufacture of basic pharmaceutical products and pharmaceutical preparations",
    "wood": "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials",
    "rubber_and_plastic": "Manufacture of rubber and plastic products",
}
# --- END CONSTANTS ---

# Get the root directory
project_root = root_dir()

worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


def get_manufacturing_exp(
    data, countries, mriot_type, mriot_year, repr_sectors, variable
):

    glob_prod, repr_sectors, IO_countries = get_prod_secs(
        mriot_type, mriot_year, repr_sectors
    )

    cnt_dfs = []
    for iso3_cnt in countries:
        cnt_df = data.loc[data[REGION_ID_COL] == iso3_cnt]

        # calculate total emssions per country (tons)
        country_sum_emissions = cnt_df[EMISSION_COL].sum()

        # normalize each emission with the total emisions
        # Attempt to calculate total area and normalize if it's non-zero
        try:
            if country_sum_emissions != 0:
                # Normalize 'emissions' values by dividing by total area
                cnt_df[NORMALIZED_EMISSIONS_COL] = (
                    cnt_df[EMISSION_COL] / country_sum_emissions
                )
                print(f"Total emissions {iso3_cnt} is {country_sum_emissions}")
            else:
                cnt_df[NORMALIZED_EMISSIONS_COL] = cnt_df[EMISSION_COL]
                print(
                    f"Total area of {iso3_cnt} is zero. Cannot perform normalization of emissions."
                )
        except Exception:
            print(f"Emissions of {cnt_df} is not zero")

        try:
            # For countries that are explicitely in the MRIO  table:
            cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * (
                glob_prod.loc[iso3_cnt].loc[repr_sectors].sum().values[0]
            )
        except KeyError:
            # For Rest of the world countries:
            LOGGER.warning(
                "your are simulating a country for which there are no specific production data in the chose IO --> ROW country"
            )

            ROW_manufac_prod = get_ROW_factor_WorldBank_manufac(
                mriot_year, IO_countries, countries
            )
            try:
                ROW_manufac_prod_factor = ROW_manufac_prod.loc[
                    ROW_manufac_prod[COUNTRY_CODE_COL] == iso3_cnt, NORMALIZED_PROD_COL
                ].values[0]
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_manufac_prod_factor
                cnt_df[VALUE_COL] = (
                    cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production
                )
            except:
                print(
                    f"For the country {iso3_cnt} there is no MP value available, 0 value is assigned"
                )
                ROW_manufac_prod_factor = 0
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_manufac_prod_factor
                cnt_df[VALUE_COL] = (
                    cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production
                )

            # # TODO Insert if statements to call different functions depending on the variable

            # if variable == 'food_and_paper':
            #     #Distribute value according to WorldBank Food production
            #     ROW_food_prod = get_ROW_factor_WorldBank_food(mriot_year, IO_countries, countries)
            #     try:
            #         ROW_food_prod_factor = ROW_food_prod.loc[ROW_food_prod[COUNTRY_CODE_COL] == iso3_cnt, NORMALIZED_PROD_COL].values[0]
            #         ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_food_prod_factor)
            #         cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production
            #     except:
            #         print(f"For the country {iso3_cnt} there is no Food production value available, 0 value is assigned")
            #         ROW_food_prod_factor = 0
            #         ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_food_prod_factor)
            #         cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production

            # if variable == 'chemical_process':
            #     #Distribute value according to WorldBank chemcial production
            #     ROW_chemical_prod = get_ROW_factor_WorldBank_chemical(mriot_year, IO_countries, countries)
            #     try:
            #         ROW_chemical_prod_factor = ROW_chemical_prod.loc[ROW_chemical_prod[COUNTRY_CODE_COL] == iso3_cnt, NORMALIZED_PROD_COL].values[0]
            #         ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_chemical_prod_factor)
            #         cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production
            #     except:
            #         print(f"For the country {iso3_cnt} there is no Chemical Production value available, 0 value is assigned")
            #         ROW_chemical_prod_factor = 0
            #         ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_chemical_prod_factor)
            #         cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production

            # elif variable == 'refin_and_transform':
            #     #Distribute value according to WorldBank manufacturing production
            #     ROW_manufac_prod = get_ROW_factor_WorldBank_manufac(mriot_year, IO_countries, countries)
            #     try:
            #         ROW_manufac_prod_factor = ROW_manufac_prod.loc[ROW_manufac_prod[COUNTRY_CODE_COL] == iso3_cnt, NORMALIZED_PROD_COL].values[0]
            #         ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_manufac_prod_factor)
            #         cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production
            #     except:
            #         print(f"For the country {iso3_cnt} there is no MP value available, 0 value is assigned")
            #         ROW_manufac_prod_factor = 0
            #         ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_manufac_prod_factor)
            #         cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * ROW_country_production


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


def get_ROW_factor_WorldBank_food(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the Worldbank data
    WB_manufac = pd.read_csv(
        f"{project_root}/{GENERAL_MANUFAC_WB_PATH}"
    )
    Food_manufact = pd.read_csv(
        f"{project_root}/{FOOD_MANUFAC_WB_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code',
    # select only the countries with are not within the IO table
    ROW_manufac_worldbank = WB_manufac[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~WB_manufac[COUNTRY_CODE_COL].isin(IO_countries)
    ]
    ROW_food_worldbank = Food_manufact[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~Food_manufact[COUNTRY_CODE_COL].isin(IO_countries)
    ]

    # select only the countries which are in the countries list (and not in the IO from before)
    filtered_manufac_worldbank = ROW_manufac_worldbank[
        ROW_manufac_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]
    filtered_food_worldbank = ROW_food_worldbank[
        ROW_food_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]

    # Combine the GDP with the mineral rent to get back to a production
    merged_df = pd.merge(
        filtered_manufac_worldbank,
        filtered_food_worldbank,
        on=COUNTRY_CODE_COL,
        suffixes=("_manufac", "_food"),
    )

    # Divide the manufac by 100 and multiply it with the food (in % of manufac) to get back to a production in USD per country
    merged_df[f"{str(mriot_year)}_food_production"] = (
        merged_df[f"{str(mriot_year)}_manufac"] / 100
    ) * merged_df[f"{str(mriot_year)}_food"]
    food_production_worldbank = merged_df[
        [COUNTRY_CODE_COL, f"{str(mriot_year)}_food_production"]
    ].copy()

    ROW_total_food_prodoction = food_production_worldbank[
        f"{str(mriot_year)}_food_production"
    ].sum()
    # Create a new column with normalized GDP values
    food_production_worldbank[NORMALIZED_PROD_COL] = (
        food_production_worldbank[f"{str(mriot_year)}_food_production"]
        / ROW_total_food_prodoction
    )
    return food_production_worldbank


def get_ROW_factor_WorldBank_chemical(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the Worldbank data
    WB_manufac = pd.read_csv(
        f"{project_root}/{GENERAL_MANUFAC_WB_PATH}"
    )
    Chemical_manufact = pd.read_csv(
        f"{project_root}/{CHEMICAL_MANUFAC_WB_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code',
    # select only the countries with are not within the IO table
    ROW_manufac_worldbank = WB_manufac[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~WB_manufac[COUNTRY_CODE_COL].isin(IO_countries)
    ]
    ROW_chemical_worldbank = Chemical_manufact[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~Chemical_manufact[COUNTRY_CODE_COL].isin(IO_countries)
    ]

    # select only the countries which are in the countries list (and not in the IO from before)
    filtered_manufac_worldbank = ROW_manufac_worldbank[
        ROW_manufac_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]
    filtered_chemical_worldbank = ROW_chemical_worldbank[
        ROW_chemical_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]

    # Combine the GDP with the mineral rent to get back to a production
    merged_df = pd.merge(
        filtered_manufac_worldbank,
        filtered_chemical_worldbank,
        on=COUNTRY_CODE_COL,
        suffixes=("_manufac", "_chemcial"),
    )

    # Divide the manufac by 100 and multiply it with the food (in % of manufac) to get back to a production in USD per country
    merged_df[f"{str(mriot_year)}_food_production"] = (
        merged_df[f"{str(mriot_year)}_manufac"] / 100
    ) * merged_df[f"{str(mriot_year)}_chemcial"]
    chemical_production_worldbank = merged_df[
        [COUNTRY_CODE_COL, f"{str(mriot_year)}_chemcial_production"]
    ].copy()

    ROW_total_food_prodoction = chemical_production_worldbank[
        f"{str(mriot_year)}_chemcial_production"
    ].sum()
    # Create a new column with normalized GDP values
    chemical_production_worldbank[NORMALIZED_PROD_COL] = (
        chemical_production_worldbank[f"{str(mriot_year)}_chemcial_production"]
        / ROW_total_food_prodoction
    )
    return chemical_production_worldbank


###Using the WorldBank Manufacturing data (providing Total manufacturing sector output MRIO)
def get_ROW_factor_WorldBank_manufac(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the Manufacturing of countries
    WB_manufac = pd.read_csv(
        f"{project_root}/{GENERAL_MANUFAC_WB_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code',
    # select only the countries with are not within the IO table
    ROW_ManufacProd = WB_manufac[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~WB_manufac[COUNTRY_CODE_COL].isin(IO_countries)
    ]
    # select only the countries which are in the countries list (and not in the IO from before)
    filtered_ManufacProd = ROW_ManufacProd[
        ROW_ManufacProd[COUNTRY_CODE_COL].isin(countries)
    ]

    ROW_total_prod = filtered_ManufacProd[str(mriot_year)].sum()
    # Create a new column with normalized GDP values
    filtered_ManufacProd[NORMALIZED_PROD_COL] = (
        filtered_ManufacProd[str(mriot_year)] / ROW_total_prod
    )
    return filtered_ManufacProd


year = EDGAR_YEAR
emission_threshold = EMISSION_THRESHOLD_DEFAULT
emission_threshold_scaled = EMISSION_THRESHOLD_SCALED


files = FILE_KEYS
sectors = SECTORS

# only activate this to run for individual runs
# files = {
#         'wood':f'wood_NOx_emissions_{year}_above_{emission_threshold_scaled}t_0.1deg_ISO3',
# }
#
# sectors = {
#         'wood':'Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials',
#
# }

mriot_type = MRIOT_TYPE_DEFAULT
mriot_year = MRIOT_YEAR_DEFAULT

for variable, filename in files.items():
    data = pd.read_hdf(
        f"{project_root}/{BASE_DATA_PATH}/{filename}.h5"
    )
    repr_sectors = sectors[variable]

    # get the countries within each sub-exposure
    countries = data[REGION_ID_COL].unique().tolist()
    countries.sort()
    print(f"Total number of countries within {variable} exposure", len(countries))

    # Get the sector for the current variable
    current_sector = sectors.get(variable, "")

    # Apply the function that alters the value using MRIO
    exp = get_manufacturing_exp(
        data=data,
        countries=countries,
        mriot_type=mriot_type,
        mriot_year=mriot_year,
        repr_sectors=repr_sectors,
        variable=variable,
    )

    """
    Saving of file, first, locally and secondly also to the s3 Bukcet
    """

    # Save a shape file to check it in QGIS
    df_shape = exp.gdf.drop(columns=[EMISSION_COL, NORMALIZED_EMISSIONS_COL])
    filename_shp = f"{project_root}/{OUTPUT_DIR}/{variable}/{filename}_values_Manfac_scaled.shp"
    s3_filename_shp = f"{OUTPUT_DIR}/{variable}/{filename}_values_Manfac_scaled.shp"
    df_shape.to_file(filename_shp, driver=SHAPEFILE_DRIVER)
    # upload the file to the s3 Bucket
    upload_to_s3_bucket(filename_shp, s3_filename_shp)
    print(f"upload of {s3_filename_shp} to s3 bucket successful")

    # Save the final complete file to a climada available format h5
    df = exp.gdf.drop(columns=["geometry", EMISSION_COL, NORMALIZED_EMISSIONS_COL])
    filename_h5 = f"{project_root}/{OUTPUT_DIR}/{variable}/{filename}_values_Manfac_scaled.h5"
    s3_filename_h5 = f"{OUTPUT_DIR}/{variable}/{filename}_values_Manfac_scaled.h5"
    df.to_hdf(filename_h5, key=HDF_KEY, mode=HDF_MODE)
    # upload the file to the s3 Bucket
    upload_to_s3_bucket(filename_h5, s3_filename_h5)
    print(f"upload of {s3_filename_h5} to s3 bucket successful")

    # Save individual country files
    for region_id in df[REGION_ID_COL].unique():
        subset_df = df[df[REGION_ID_COL] == region_id]
        filename_country = f"{project_root}/{OUTPUT_DIR}/{variable}/country_split/{filename}_values_Manfac_scaled_{region_id}.h5"
        s3_filename_country = f"{OUTPUT_DIR}/{variable}/country_split/{filename}_values_Manfac_scaled_{region_id}.h5"
        subset_df.to_hdf(filename_country, key=HDF_KEY, mode=HDF_MODE)
        # upload the individual country files to s3 bucket
        upload_to_s3_bucket(filename_country, s3_filename_country)
        print(f"upload of {s3_filename_country} to s3 bucket successful")