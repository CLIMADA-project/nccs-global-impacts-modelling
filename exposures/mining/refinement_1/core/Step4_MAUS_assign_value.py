import os.path

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from climada.entity import Exposures
from climada_petals.engine import get_mriot
import logging

LOGGER = logging.getLogger()
worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

from exposures.utils import root_dir
from nccs.utils.s3client import upload_to_s3_bucket

# --- CONSTANTS ---
MRIOT_TYPE_DEFAULT = "WIOD16"
MRIOT_YEAR_DEFAULT = 2011
REPR_SECTORS_DEFAULT = "Mining and quarrying"
TOTAL_COUNTRIES = 195 # Used in commented-out Option 1 for equally distributing ROW production

# File Paths
MINING_H5_PATH = "exposures/mining/refinement_1/intermediate_data_MAUS/global_miningarea_v2_30arcsecond_converted_ISO3_improved.h5"
GDP_WB_PATH = "exposures/mining/refinement_1/core/GDP_Worldbank_modified_without_regions.csv"
MINERAL_RENT_WB_PATH = "exposures/mining/refinement_1/core/WorldBank_mineral_rents_modified_without_regions.csv"
WORLD_MINING_DATA_PATH = "exposures/mining/refinement_1/core/WorldMiningData_2021_Total_Mineral_Production.xlsx"

# Column Names
REGION_ID_COL = "region_id"
AREA_COL = "area"
NORMALIZED_AREA_COL = "normalized_area"
VALUE_COL = "value"
COUNTRY_CODE_COL = "Country Code"
ISO3_COL = "ISO3"
NORMALIZED_GDP_COL = "Normalized_GDP"
NORMALIZED_PROD_COL = "Normalized_Prod"
MINERAL_PRODUCTION_COL = "_mineral_production"
TOTAL_MINING_VALUE_COL = "Total Value Mineral Production (incl. Bauxite)"
ROW_REGION = "ROW"
HDF_KEY = "data"
HDF_MODE = "w"
SHAPEFILE_DRIVER = "ESRI Shapefile"

# Output Files
OUTPUT_DIR = "exposures/mining/refinement_1"
OUTPUT_FILE_BASE = "mining_values"
COUNTRY_SPLIT_DIR = f"{OUTPUT_DIR}/country_split"

# List of WIOD16 countries (used for filtering ROW checkpoint)
WIOD16_REGIONS = [
    "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHE", "CHN", "CYP", "CZE",
    "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HRV", "HUN",
    "IDN", "IND", "IRL", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX",
    "MLT", "NLD", "NOR", "POL", "PRT", "ROU", "RUS", "SVK", "SVN", "SWE",
    "TUR", "TWN", "USA", ROW_REGION,
]
# --- END CONSTANTS ---

# Get the root directory
project_root = root_dir()
print(os.path.abspath(project_root))


def get_mining_exp(
    countries=None,
    mriot_type=MRIOT_TYPE_DEFAULT,
    mriot_year=MRIOT_YEAR_DEFAULT,
    repr_sectors=REPR_SECTORS_DEFAULT,
):
    glob_prod, repr_sectors, IO_countries = get_prod_secs(
        mriot_type, mriot_year, repr_sectors
    )

    data = pd.read_hdf(
        f"{project_root}/{MINING_H5_PATH}"
    )
    cnt_dfs = []
    for iso3_cnt in countries:
        cnt_df = data.loc[data[REGION_ID_COL] == iso3_cnt]

        # calculate total area of mines per country
        country_sum_area = cnt_df[AREA_COL].sum()

        # normalize each area with the total area
        # Attempt to calculate total area and normalize if it's non-zero
        try:
            if country_sum_area != 0:
                # Normalize 'area' values by dividing by total area
                cnt_df[NORMALIZED_AREA_COL] = cnt_df[AREA_COL] / country_sum_area
                print(f"Total area of {iso3_cnt} mining is {country_sum_area}")
            else:
                cnt_df[NORMALIZED_AREA_COL] = cnt_df[AREA_COL]
                print(
                    f"Total area of {iso3_cnt} is zero. Cannot perform normalization of area."
                )
        except Exception:
            print(f"Area of {cnt_df} is not zero")

        try:
            # For countries that are explicitely in the MRIO  table:
            cnt_df[VALUE_COL] = cnt_df[NORMALIZED_AREA_COL] * (
                glob_prod.loc[iso3_cnt].loc[repr_sectors].sum().values[0]
            )
        except KeyError:
            # For Rest of the world countries:
            LOGGER.warning(
                "your are simulating a country for which there are no specific production data in the chose IO --> ROW country"
            )

            # #### OPTION 1: Distribute ROW production value equally --> Not suggested
            # #Get a percentage of the total ROW production for each ROW country
            # n_total = TOTAL_COUNTRIES
            # #get the factor to scale each production
            # ROW_factor = (1 / (n_total - (len(set(r[0] for r in glob_prod.axes[0])) - 1)))
            # ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_factor)
            # cnt_df[VALUE_COL] = cnt_df[NORMALIZED_AREA_COL] * ROW_country_production

            # #### OPTION 2: Distribute value according to GDP
            # ROW_gdp_lookup = get_ROW_factor_GDP(mriot_year, IO_countries, countries)
            # try:
            #     ROW_gdp_factor = ROW_gdp_lookup.loc[ROW_gdp_lookup[COUNTRY_CODE_COL] == iso3_cnt, NORMALIZED_GDP_COL].values[0]
            #     ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_gdp_factor)
            #     cnt_df[VALUE_COL] = cnt_df[NORMALIZED_AREA_COL] * ROW_country_production
            # except:
            #     print(f"For the country {iso3_cnt} there is no GDP value available, 0 value is assigned")
            #     ROW_gdp_factor = 0
            #     ROW_country_production = ((glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()).values[0] * ROW_gdp_factor)
            #     cnt_df[VALUE_COL] = cnt_df[NORMALIZED_AREA_COL] * ROW_country_production

            #### OPTION 3: Distribute value according to WorldMiningData
            ROW_mineral_prod = get_ROW_factor_WorldMiningData(
                mriot_year, IO_countries, countries
            )
            try:
                ROW_mineral_prod_factor = ROW_mineral_prod.loc[
                    ROW_mineral_prod[ISO3_COL] == iso3_cnt, NORMALIZED_PROD_COL
                ].values[0]
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_mineral_prod_factor
                cnt_df[VALUE_COL] = cnt_df[NORMALIZED_AREA_COL] * ROW_country_production
            except:
                print(
                    f"For the country {iso3_cnt} there is no MP value available, 0 value is assigned"
                )
                ROW_mineral_prod_factor = 0
                ROW_country_production = (
                    glob_prod.loc[ROW_REGION].loc[repr_sectors].sum()
                ).values[0] * ROW_mineral_prod_factor
                cnt_df[VALUE_COL] = cnt_df[NORMALIZED_AREA_COL] * ROW_country_production

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


####### Different Option to scale the ROW mining production value of the MRIO table


## Using just the GDP of the ROW countries to scale it
def get_ROW_factor_GDP(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the GDP of counries
    gdp_worldbank = pd.read_csv(
        f"{project_root}/{GDP_WB_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code',
    # select only the countries with are not within the IO table
    ROW_gdp_worldbank = gdp_worldbank[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~gdp_worldbank[COUNTRY_CODE_COL].isin(IO_countries)
    ]
    # select only the countries which are in the countries list (and not in the IO from before)
    filtered_gdp_worldbank = ROW_gdp_worldbank[
        ROW_gdp_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]

    ROW_total_GDP = filtered_gdp_worldbank[str(mriot_year)].sum()
    # Create a new column with normalized GDP values
    filtered_gdp_worldbank[NORMALIZED_GDP_COL] = (
        filtered_gdp_worldbank[str(mriot_year)] / ROW_total_GDP
    )
    return filtered_gdp_worldbank


### Using the Mineral Rent (provided by the WorldBank) and multiplied back with the GPD to get to a production
def get_ROW_factor_mineral_rent_GDP(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the GDP of counries
    gdp_worldbank = pd.read_csv(
        f"{project_root}/{GDP_WB_PATH}"
    )
    mineral_rent = pd.read_csv(
        f"{project_root}/{MINERAL_RENT_WB_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code',
    # select only the countries with are not within the IO table
    ROW_gdp_worldbank = gdp_worldbank[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~gdp_worldbank[COUNTRY_CODE_COL].isin(IO_countries)
    ]
    ROW_mineral_rent_worldbank = mineral_rent[[COUNTRY_CODE_COL, str(mriot_year)]][
        ~mineral_rent[COUNTRY_CODE_COL].isin(IO_countries)
    ]

    # select only the countries which are in the countries list (and not in the IO from before)
    filtered_gdp_worldbank = ROW_gdp_worldbank[
        ROW_gdp_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]
    filtered_mineral_rent_worldbank = ROW_mineral_rent_worldbank[
        ROW_mineral_rent_worldbank[COUNTRY_CODE_COL].isin(countries)
    ]

    # Combine the GDP with the mineral rent to get back to a production
    merged_df = pd.merge(
        filtered_gdp_worldbank,
        filtered_mineral_rent_worldbank,
        on=COUNTRY_CODE_COL,
        suffixes=("_gdp", "_mineral_rent"),
    )

    # Divide the GDP by 100 and multiply it with the mineral rent (in % of GDP) to get back to a production in USD per country
    merged_df[f"{str(mriot_year)}{MINERAL_PRODUCTION_COL}"] = (
        merged_df[f"{str(mriot_year)}_gdp"] / 100
    ) * merged_df[f"{str(mriot_year)}_mineral_rent"]
    mineral_production_worldbank = merged_df[
        [COUNTRY_CODE_COL, f"{str(mriot_year)}{MINERAL_PRODUCTION_COL}"]
    ].copy()

    ROW_total_mineral_prodoction = mineral_production_worldbank[
        f"{str(mriot_year)}{MINERAL_PRODUCTION_COL}"
    ].sum()
    # Create a new column with normalized GDP values
    mineral_production_worldbank[NORMALIZED_PROD_COL] = (
        mineral_production_worldbank[f"{str(mriot_year)}{MINERAL_PRODUCTION_COL}"]
        / ROW_total_mineral_prodoction
    )
    return mineral_production_worldbank


###Using the World Mining Data (providing Total mineral production with adequate products such as in MRIO)
def get_ROW_factor_WorldMiningData(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the MP (mineral production) of countries
    WorldMiningData = pd.read_excel(
        f"{project_root}/{WORLD_MINING_DATA_PATH}"
    )

    # Select only the specified year column and filter rows based on the 'Country Code',
    # select only the countries with are not within the IO table
    ROW_MiningProd = WorldMiningData[
        [ISO3_COL, TOTAL_MINING_VALUE_COL]
    ][~WorldMiningData[ISO3_COL].isin(IO_countries)]
    # select only the countries which are in the countries list (and not in the IO from before)
    filtered_MiningProd = ROW_MiningProd[ROW_MiningProd[ISO3_COL].isin(countries)]

    ROW_total_prod = filtered_MiningProd[
        TOTAL_MINING_VALUE_COL
    ].sum()
    # Create a new column with normalized GDP values
    filtered_MiningProd[NORMALIZED_PROD_COL] = (
        filtered_MiningProd[TOTAL_MINING_VALUE_COL]
        / ROW_total_prod
    )
    return filtered_MiningProd


data = pd.read_hdf(
    f"{project_root}/{MINING_H5_PATH}"
)
countries = data[REGION_ID_COL].unique().tolist()
countries.sort()

# apply function that alters the value using MRIO
exp = get_mining_exp(
    countries=countries,
    mriot_type=MRIOT_TYPE_DEFAULT,
    mriot_year=MRIOT_YEAR_DEFAULT,
    repr_sectors=REPR_SECTORS_DEFAULT,
)

"""
Saving of file, first, locally and secondly also to the s3 Bukcet
"""

# Save a shape file to check it in QGIS
df_shape = exp.gdf.drop(columns=[AREA_COL, NORMALIZED_AREA_COL])
filename_shp = f"{project_root}/{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.shp"
s3_filename_shp = f"{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.shp"
df_shape.to_file(filename_shp, driver=SHAPEFILE_DRIVER)
# upload the file to the s3 Bucket
upload_to_s3_bucket(filename_shp, s3_filename_shp)
print(f"upload of {s3_filename_shp} to s3 bucket successful")


# Save final file to a climada available format h5
df = exp.gdf.drop(columns=["geometry", AREA_COL, NORMALIZED_AREA_COL])
filename_h5 = f"{project_root}/{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.h5"
s3_filename_h5 = f"{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.h5"
df.to_hdf(filename_h5, key=HDF_KEY, mode=HDF_MODE)
# upload the file to the s3 Bucket
upload_to_s3_bucket(filename_h5, s3_filename_h5)
print(f"upload of {s3_filename_h5} to s3 bucket successful")

# Save individual country files
for region_id in df[REGION_ID_COL].unique():
    subset_df = df[df[REGION_ID_COL] == region_id]
    filename_country = f"{project_root}/{COUNTRY_SPLIT_DIR}/{OUTPUT_FILE_BASE}_{region_id}.h5"
    s3_filename_country = f"{COUNTRY_SPLIT_DIR}/{OUTPUT_FILE_BASE}_{region_id}.h5"
    subset_df.to_hdf(filename_country, key=HDF_KEY, mode=HDF_MODE)
    # upload the individual country files to s3 bucket
    upload_to_s3_bucket(filename_country, s3_filename_country)
    print(f"upload of {s3_filename_country} to s3 bucket successful")


# count number of zeros
num_rows_with_zero = len(df[df[VALUE_COL] == 0])

# #### Some checkpoints:

# plot the exposre map
from matplotlib.colors import Normalize

fig, ax = plt.subplots(figsize=(30, 12))
worldmap.plot(color="lightgrey", ax=ax)
# Use scatter to plot the points with color based on the 'value' column
norm = Normalize(vmin=exp.gdf[VALUE_COL].min(), vmax=exp.gdf[VALUE_COL].mean())
sc = ax.scatter(
    exp.gdf["longitude"],
    exp.gdf["latitude"],
    c=exp.gdf[VALUE_COL],
    cmap="viridis",
    norm=norm,
    s=0.01,
)
# Add a colorbar
cbar = plt.colorbar(sc, ax=ax, label="Value")
# Set axis labels and title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Mining Exposure with MRIO values scaled by area of the mine in M.USD")
# Show the plot
plt.show()

# country sum of value
value_sum_per_country = df.groupby(REGION_ID_COL)[VALUE_COL].sum().reset_index()
print("value_sum_per_country for Mining", value_sum_per_country)
# plot a barblot with this
# Plot the bar chart for the sum of values per country
# Sort the values and select the top 30
sorted_value_sum_per_country = value_sum_per_country.sort_values(
    by=VALUE_COL, ascending=False
).head(30)
print("sorted_value_sum_per_country for Mining", sorted_value_sum_per_country)

# Number of points per country
num_points_per_country = exp.gdf.groupby(REGION_ID_COL).size()
# Sort the values and select the top 30
sorted_num_points_per_country = num_points_per_country.sort_values(
    ascending=False
).head(30)

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# Create a 2x2 grid of subplots
fig = plt.figure(figsize=(30, 18))
gs = GridSpec(2, 2, width_ratios=[2, 1])

# Plot the world map with scatter plot
ax0 = plt.subplot(gs[0, 0])
worldmap.plot(color="lightgrey", ax=ax0)
norm = Normalize(vmin=exp.gdf[VALUE_COL].min(), vmax=exp.gdf[VALUE_COL].mean())
sc = ax0.scatter(
    exp.gdf["longitude"],
    exp.gdf["latitude"],
    c=exp.gdf[VALUE_COL],
    cmap="cividis",
    norm=norm,
    s=0.01,
)
cbar = plt.colorbar(sc, ax=ax0, label="Value")
ax0.set_xlabel("Longitude")
ax0.set_ylabel("Latitude")
ax0.set_title("Mining Exposure with MRIO values scaled by area of the mine in M.USD")

# Plot the bar chart for the number of points per country (switched order)
ax2 = plt.subplot(gs[0, 1])
sorted_num_points_per_country.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Number of Points")
ax2.set_xlabel("Country Code")
ax2.set_title("Top 30 Countries by Number of Points")

# Plot the bar chart for the sum of values per country (switched order)
ax1 = plt.subplot(gs[1, :])
sorted_value_sum_per_country.plot(x=REGION_ID_COL, y=VALUE_COL, kind="bar", ax=ax1)
ax1.set_ylabel("Sum of Values")
ax1.set_xlabel("Country Code")
ax1.set_title("Top 30 Countries by Sum of Values")

plt.tight_layout()
plt.savefig(
    f"{project_root}/{OUTPUT_DIR}/intermediate_data_MAUS/{OUTPUT_FILE_BASE}.png",
    bbox_inches="tight",
)
plt.show()

##total area
sum_area = exp.gdf[AREA_COL].sum()
print("Total area within grid cells in sqkm", sum_area)
print("area covered in  apper is 101'583km2")

# country sum of value
value_sum_per_country = df.groupby(REGION_ID_COL)[VALUE_COL].sum().reset_index()
print(value_sum_per_country)

# country sum or normalized area
# should be 1 everwhere
norm_area_sum = exp.gdf.groupby(REGION_ID_COL)[NORMALIZED_AREA_COL].sum().reset_index()
print(norm_area_sum)

# Check the total sum of value that gets distributed to the ROW countries
# countries that are wthin WIOD 16
regions = WIOD16_REGIONS
# take out of the final matrix the total value that is assigned to the counrties
filtered_df = value_sum_per_country[[REGION_ID_COL, VALUE_COL]][
    ~value_sum_per_country[REGION_ID_COL].isin(regions)
]
# Sum all the ROW values
filtered_df[VALUE_COL].sum()