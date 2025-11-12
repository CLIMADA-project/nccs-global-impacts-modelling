import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from climada.entity import Exposures
from climada_petals.engine import get_mriot
import logging

from exposures.utils import root_dir
from nccs.utils.s3client import upload_to_s3_bucket

# --- CONSTANTS ---
MRIOT_TYPE_DEFAULT = "WIOD16"
MRIOT_YEAR_DEFAULT = 2011
EDGAR_YEAR = 2011

# File paths and S3 prefixes
DATA_H5_FILENAME = f"global_noxemissions_{EDGAR_YEAR}_above_100t_0.1deg_ISO3.h5"
DATA_H5_PATH = f"exposures/manufacturing/manufacturing_general_exposure/refinement_1/intermediate_data_EDGAR/{DATA_H5_FILENAME}"
WB_CSV_PATH = "exposures/manufacturing/manufacturing_general_exposure/refinement_1/WorldBank_Manufac_output_without_regions.csv"
OUTPUT_DIR = "exposures/manufacturing/manufacturing_general_exposure/refinement_1"
COUNTRY_SPLIT_DIR = f"{OUTPUT_DIR}/country_split"
OUTPUT_FILE_BASE = "general_manufacture_values"

# Column names
REGION_ID_COL = "region_id"
EMISSIONS_COL = "emi_nox"
NORMALIZED_EMISSIONS_COL = "normalized_emissions"
VALUE_COL = "value"
COUNTRY_CODE_COL = "Country Code"
NORMALIZED_PROD_COL = "Normalized_Prod"
ROW_REGION = "ROW"

# HDF and Shapefile settings
HDF_KEY = "data"
HDF_MODE = "w"
SHAPEFILE_DRIVER = "ESRI Shapefile"

# Manufacturing Sectors (WIOD16)
MANUFACTURING_SECTORS = [
    "Manufacture of food products, beverages and tobacco products",
    "Manufacture of textiles, wearing apparel and leather products",
    "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials",
    "Manufacture of paper and paper products",
    "Printing and reproduction of recorded media",
    "Manufacture of coke and refined petroleum products ",
    "Manufacture of chemicals and chemical products ",
    "Manufacture of basic pharmaceutical products and pharmaceutical preparations",
    "Manufacture of rubber and plastic products",
    "Manufacture of other non-metallic mineral products",
    "Manufacture of basic metals",
    "Manufacture of fabricated metal products, except machinery and equipment",
    "Manufacture of computer, electronic and optical products",
    "Manufacture of electrical equipment",
    "Manufacture of machinery and equipment n.e.c.",
    "Manufacture of motor vehicles, trailers and semi-trailers",
    "Manufacture of other transport equipment",
    "Manufacture of furniture; other manufacturing",
    "Repair and installation of machinery and equipment",
]
# --- END CONSTANTS ---

# Get the root directory
project_root = root_dir()

LOGGER = logging.getLogger()
worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


def get_manufacturing_exp(data, countries, mriot_type, mriot_year, repr_sectors):
    glob_prod, repr_sectors, IO_countries = get_prod_secs(
        mriot_type, mriot_year, repr_sectors
    )

    cnt_dfs = []
    for iso3_cnt in countries:
        cnt_df = data.loc[data[REGION_ID_COL] == iso3_cnt]

        # calculate total emissions per country (tons)
        country_sum_emissions = cnt_df[EMISSIONS_COL].sum()

        # normalize grid cell emission by the total country emission
        # Attempt to calculate total area and normalize if it's non-zero
        try:
            if country_sum_emissions != 0:
                # Normalize 'emissions' values by dividing by total area
                cnt_df[NORMALIZED_EMISSIONS_COL] = (
                    cnt_df[EMISSIONS_COL] / country_sum_emissions
                )
                print(f"Total emissions of {iso3_cnt} NOx is {country_sum_emissions}")
            else:
                cnt_df[NORMALIZED_EMISSIONS_COL] = cnt_df[EMISSIONS_COL]
                print(
                    f"Total area of {iso3_cnt} is zero. Cannot perform normalization of emissions."
                )
        except Exception:
            print(f"Emissions of {cnt_df} is not zero")

        try:
            # For countries that are explicitly in the MRIO  table:
            cnt_df[VALUE_COL] = cnt_df[NORMALIZED_EMISSIONS_COL] * (
                glob_prod.loc[iso3_cnt].loc[repr_sectors].sum().values[0]
            )
        except KeyError:
            # For Rest of the world countries:
            LOGGER.warning(
                "your are simulating a country for which there are no specific production data in the chose IO --> ROW country"
            )

            #### OPTION 3: Distribute value according to WorldBank manufacturing production
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


###Using the WorldBank Manufacturing data (providing Total manufacturing sector output MRIO)
def get_ROW_factor_WorldBank_manufac(mriot_year, IO_countries, countries):
    IO_countries = IO_countries

    # load the Manufacturing of countries
    WB_manufac = pd.read_csv(
        f"{project_root}/{WB_CSV_PATH}"
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
mriot_type = MRIOT_TYPE_DEFAULT
mriot_year = MRIOT_YEAR_DEFAULT
# all the manufacturing sectors of WIOD16, to be changed if another table would be used
repr_sectors = MANUFACTURING_SECTORS

data = pd.read_hdf(
    f"{project_root}/{DATA_H5_PATH}"
)
countries = data[REGION_ID_COL].unique().tolist()
countries.sort()

# apply function that alters the value using MRIO
exp = get_manufacturing_exp(
    data=data,
    countries=countries,
    mriot_type=mriot_type,
    mriot_year=mriot_year,
    repr_sectors=repr_sectors,
)

"""
Saving of file, first, locally and secondly also to the s3 Bukcet
"""

# Save a shape file to check it in QGIS
df_shape = exp.gdf.drop(columns=["emi_nox", "normalized_emissions"])
filename_shp = f"{project_root}/{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.shp"
s3_filename_shp = f"{OUTPUT_DIR}/{OUTPUT_FILE_BASE}.shp"
df_shape.to_file(filename_shp, driver=SHAPEFILE_DRIVER)
# upload the file to the s3 Bucket
upload_to_s3_bucket(filename_shp, s3_filename_shp)
print(f"upload of {s3_filename_shp} to s3 bucket successful")


# Save final file to a climada available format h5
df = exp.gdf.drop(columns=["geometry", "emi_nox", "normalized_emissions"])
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


"""Check points, not needed fot the final output, but create some credibility"""

# count number of zeros
num_rows_with_zero = len(df[df[VALUE_COL] == 0])

#
# ##total emissions
# sum_emissions= exp.gdf['emi_nox'].sum()
# print("Total emissions within grid cells in t/year", sum_emissions)
#
#

# #### Some checkpoints:
# country sum of value
value_sum_per_country = df.groupby(REGION_ID_COL)[VALUE_COL].sum().reset_index()
print("value_sum_per_country for Manufacturing", value_sum_per_country)
# plot a barblot with this
# Plot the bar chart for the sum of values per country
# Sort the values and select the top 30
sorted_value_sum_per_country = value_sum_per_country.sort_values(
    by=VALUE_COL, ascending=False
).head(30)

# Number of points per country
num_points_per_country = exp.gdf.groupby(REGION_ID_COL).size()
# Sort the values and select the top 30
sorted_num_points_per_country = num_points_per_country.sort_values(
    ascending=False
).head(30)

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
ax0.set_title(
    "Manufacturing Exposure with MRIO values scaled by total Manufacturing production in M.USD"
)

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
    f"{project_root}/{OUTPUT_DIR}/intermediate_data_EDGAR/{OUTPUT_FILE_BASE}.png",
    bbox_inches="tight",
)
plt.show()