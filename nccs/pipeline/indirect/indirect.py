import os

import country_converter as coco
import numpy as np
import pandas as pd
import pycountry
from climada_petals.engine.supplychain import (
    DirectShocksSet,
    StaticIOModel,
    BoARIOModel,
    get_mriot,
)

from exposures.utils import root_dir

cc = coco.CountryConverter()

project_root = root_dir()
GDP_WORLDBANK_FILE = f"{project_root}/resources/GDP_Worldbank_without_regions.csv"

# original
# SERVICE_SEC = {"service": range(26, 56)}
MRIOT_TYPE = {"WIOD16-2011": "WIOD"}

SUPER_SEC = {
    "agriculture": ["Crop and animal production, hunting and related service activities"],
    "forestry": ["Forestry and logging"],
    "mining": ["Mining and quarrying"],
    "manufacturing": [
        "Manufacture of basic metals",
        "Manufacture of basic pharmaceutical products and pharmaceutical preparations",
        "Manufacture of chemicals and chemical products ",
        "Manufacture of coke and refined petroleum products ",
        "Manufacture of computer, electronic and optical products",
        "Manufacture of electrical equipment",
        "Manufacture of fabricated metal products, except machinery and equipment",
        "Manufacture of food products, beverages and tobacco products",
        "Manufacture of furniture; other manufacturing",
        "Manufacture of machinery and equipment n.e.c.",
        "Manufacture of motor vehicles, trailers and semi-trailers",
        "Manufacture of other non-metallic mineral products",
        "Manufacture of other transport equipment",
        "Manufacture of paper and paper products",
        "Manufacture of rubber and plastic products",
        "Manufacture of textiles, wearing apparel and leather products",
        "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials",
        "Repair and installation of machinery and equipment",
        "Construction",
    ],
    # after interim, Construction (26) included, as well as repair and installation of machinery and equipment was
    # missing
    "food": ["Manufacture of food products, beverages and tobacco products"],
    "wood": [
        "Manufacture of food products, beverages and tobacco products",
    ],
    "refin_and_transform": [
        "Manufacture of coke and refined petroleum products ",
    ],
    "chemical": ["Manufacture of chemicals and chemical products "],
    "pharmaceutical": [
        "Manufacture of basic pharmaceutical products and pharmaceutical preparations"
    ],
    "rubber_and_plastic": ["Manufacture of rubber and plastic products"],
    "non_metallic_mineral": ["Manufacture of other non-metallic mineral products"],
    "basic_metals": ["Manufacture of basic metals"],
    # utilities
    "energy": ["Electricity, gas, steam and air conditioning supply"],
    "electricity": ["Electricity, gas, steam and air conditioning supply"],
    "water": ["Water collection, treatment and supply"],
    "waste": [
        "Sewerage; waste collection, treatment and disposal activities; materials recovery; remediation activities and other waste management services "
    ],
    # service
    "service": [
        "Wholesale and retail trade and repair of motor vehicles and motorcycles",
        "Wholesale trade, except of motor vehicles and motorcycles",
        "Retail trade, except of motor vehicles and motorcycles",
        "Land transport and transport via pipelines",
        "Water transport",
        "Air transport",
        "Warehousing and support activities for transportation",
        "Postal and courier activities",
        "Accommodation and food service activities",
        "Publishing activities",
        "Motion picture, video and television programme production, sound recording and music publishing activities; programming and broadcasting activities",
        "Telecommunications",
        "Computer programming, consultancy and related activities; information service activities",
        "Financial service activities, except insurance and pension funding",
        "Insurance, reinsurance and pension funding, except compulsory social security",
        "Activities auxiliary to financial services and insurance activities",
        "Real estate activities",
        "Legal and accounting activities; activities of head offices; management consultancy activities",
        "Architectural and engineering activities; technical testing and analysis",
        "Scientific research and development",
        "Advertising and market research",
        "Other professional, scientific and technical activities; veterinary activities",
        "Administrative and support service activities",
        "Public administration and defence; compulsory social security",
        "Education",
        "Human health and social work activities",
        "Other service activities",
        "Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use",
        "Activities of extraterritorial organizations and bodies",
    ],  # after interim, construction (26 excluded)
}


# SUPER_SEC = {
#     "agriculture": [0],
#     "forestry": [1],
#     "mining": [3],
#     "manufacturing": list(range(4, 23)) + [26],
#     # after interim, Construction (26) included, as well as repair and installation of machinery and equipment was
#     # missing
#     "food": [4],
#     "wood": [6],
#     "refin_and_transform": [9],
#     "chemical": [10],
#     "pharmaceutical": [11],
#     "rubber_and_plastic": [12],
#     "non_metallic_mineral": [13],
#     "basic_metals": [14],
#     # utilities
#     "energy": [23],
#     "electricity": [23],
#     "water": [24],
#     "waste": [25],
#     # service
#     "service": list(range(27, 54)),  # after interim, construction (26 excluded)
# }


def get_country_modifier(country_iso3alpha, io_countries, mriot_name="WIOD16", mriot_year=2011):
    """"""
    mrio_region = cc.convert(country_iso3alpha, to=MRIOT_TYPE[mriot_name]).upper()

    if mrio_region == "ROW":
        # Using the GDP values to create a factor
        row_gdp = get_gdp_modifier(io_countries=io_countries, mriot_year=mriot_year)
        gdp_factor = row_gdp.loc[
            row_gdp["Country Code"] == country_iso3alpha, "Normalized_GDP"
        ].values[0]
        if np.isnan(gdp_factor):
            gdp_factor = 0
    else:
        gdp_factor = 1

    return gdp_factor


def get_gdp_modifier(io_countries, mriot_year):
    # load the GDP of countries
    gdp_worldbank = pd.read_csv(GDP_WORLDBANK_FILE)

    # Select only the specified year column and filter rows based on the 'Country Code',
    # select only the countries with are not within the IO table
    row_gdp_worldbank = gdp_worldbank[["Country Code", str(mriot_year)]][
        ~gdp_worldbank["Country Code"].isin(io_countries)
    ]

    ROW_total_GDP = row_gdp_worldbank[str(mriot_year)].sum()
    # Create a new column with normalized GDP values
    row_gdp_worldbank["Normalized_GDP"] = row_gdp_worldbank[str(mriot_year)] / ROW_total_GDP
    return row_gdp_worldbank


def get_secs_prod(country_iso3alpha, impacted_secs, mriot, n_total=195):
    """
    Calculate the country modifier for a given country in a supply chain.
    If the country is listed in the mrio table then the modifier is 1.0.
    else the modifier is 1 / (n_total - (number of countries in the mrio table - 1)).

    :param supchain:
    :param country_iso3alpha:
    :param n_total:
    :return:
    """
    mrio_region = cc.convert(country_iso3alpha, to=MRIOT_TYPE[mriot.name]).upper()
    if mrio_region == "ROW":
        return (1 / (n_total - (len(set(r[0] for r in mriot.x.axes[0])) - 1))) * mriot.x.loc[
            ("ROW", impacted_secs), :
        ]
    return mriot.x.loc[(mrio_region, impacted_secs), :]


def get_secs_shock(direct_shock, country_iso3alpha, impacted_secs, mriot):
    """
    Calculate the country modifier for a given country in a supply chain.
    If the country is listed in the mrio table then the modifier is 1.0.
    else the modifier is according to the GDP

    :param supchain:
    :param country_iso3alpha:
    :param n_total:
    :return:

    Definitions:
    secs_imp : pd.DataFrame
        Impact dataframe for the directly affected countries/sectors for each event with
        impacts. Columns are the same as the chosen MRIOT and rows are the hazard events ids.

    secs_shock : pd.DataFrame
        Shocks (i.e. impact / exposure) dataframe for the directly affected countries/sectors
        for each event with impacts. Columns are the same as the chosen MRIOT and rows are the
        hazard events ids.
    """
    # I would argue to still use secs_shock as it accounts for the exposure impact ratio (The attribute
    # self.secs_shock is proportional to the ratio between self.secs_imp and self.secs_exp, so self.secs_shock
    # is a number between 0 and 1. self.secs_shock will be used in the indirect impact calculation to assses
    # how much production loss is experienced by each sector.)
    # if using secs_shock again, the value extractions would need to change again

    # Simplest scaling without any GDP connection
    mrio_region = cc.convert(country_iso3alpha, to=MRIOT_TYPE[mriot.name]).upper()
    io_countries = mriot.get_regions()
    if mrio_region == "ROW":
        # Using the GDP values to create a factor
        row_gdp = get_gdp_modifier(io_countries=io_countries, mriot_year=mriot.year)
        row_fract_per_county = row_gdp.loc[
            row_gdp["Country Code"] == country_iso3alpha, "Normalized_GDP"
        ].values[0]
        if np.isnan(row_fract_per_county):
            row_fract_per_county = 0

        return row_fract_per_county * direct_shock.impacted_assets.loc[:, ("ROW", impacted_secs)]
    return direct_shock.impacted_assets.loc[:, (country_iso3alpha, impacted_secs)]


# def get_supply_chain() -> SupplyChain:
#     return SupplyChain.from_mriot(mriot_type='WIOD16', mriot_year=2011)


def supply_chain_climada(
    exposure,
    direct_impact,
    io_approach,
    mriot_type="WIOD16",
    mriot_year=2011,
    impacted_sector="service",
    shock_factor=None,
):
    assert impacted_sector in SUPER_SEC.keys(), f"impacted_sector must be one of {SUPER_SEC.keys()}"
    impacted_secs = SUPER_SEC[impacted_sector]
    mriot = get_mriot(mriot_type, mriot_year)

    # Assign exposure and shock direct_impact to MRIOT country-sector

    #
    #impacted_secs = mriot.get_sectors()[sec_range].tolist()
    direct_shocks = DirectShocksSet.from_exp_and_imp(
        mriot=mriot,
        exposure=exposure,
        impact=direct_impact,
        affected_sectors=impacted_secs,
        impact_distribution=None,
        # shock_factor=shock_factor
    )  # renamed the function from

    # Calculate the propagation of production losses
    model = StaticIOModel(mriot, direct_shocks)
    res = model.calc_indirect_impacts()
    return res


def dump_direct_to_csv(
    direct_shocks,
    mriot,
    haz_type,
    sector,
    scenario,
    ref_year,
    country,
    n_sim=100,
    return_period=100,
    output_file=None,
):
    index_rp = np.floor(n_sim / return_period).astype(int) - 1
    impacted_secs = SUPER_SEC[sector]
    #impacted_secs = mriot.get_sectors()[sec_range].tolist()
    country_iso3alpha = pycountry.countries.get(name=country).alpha_3
    secs_prod = get_secs_prod(country_iso3alpha, impacted_secs, mriot, n_total=195)

    # create a lookup table for each sector and its total production
    lookup = {}
    for idx, row in secs_prod.iterrows():
        lookup[idx] = row["indout"]

    direct_impacts = []
    for sec, v in get_secs_shock(direct_shocks, country_iso3alpha, impacted_secs, mriot).items():
        # NOTE we are using the SHOCK TABLE instead of the  IMPACT_TABLE. The shock table tells us what fraction of
        # the sector is impacted. (impacted asset value / total asset value). If we'd use the impact table, we
        # would have to convert the currencies and units of the exposure and the mrio table to match.

        # First we extract the values from the shock table, these are only ratios, not the actual production loss
        # A 100rp ratio of outage
        rp_ratio = v.sort_values(ascending=False).iloc[index_rp]
        # A average annual outage ratio
        avg_ann_ratio = v.sum() / n_sim
        # The maximum outage ratio
        max_ratio = v.max()

        # Check if the denominator is non-zero before performing division
        total_production = lookup[sec]
        obj = {
            "sector": sec[1],
            "total_sectorial_production_mriot": lookup[sec],
            "maxPL": max_ratio * total_production,
            "rmaxPL": max_ratio * 100,
            "AAPL": avg_ann_ratio * total_production,
            "rAAPL": avg_ann_ratio * 100,
            f"PL{return_period}": rp_ratio * total_production,
            f"rPL{return_period}": rp_ratio * 100,
            "hazard_type": haz_type,
            "sector_of_impact": sector,
            "scenario": scenario,
            "ref_year": ref_year,
            "country_of_impact": country,
        }
        direct_impacts.append(obj)

    df_direct = pd.DataFrame(direct_impacts)
    # newly added to get ISO3 code

    df_direct.to_csv(output_file)
    return


def dump_supchain_events_to_csv(
    model,
    results,
    haz_type,
    sector,
    scenario,
    ref_year,
    country,
    io_approach,
    output_file=None,
):
    # total production of each sector for country in Millions

    secs_prod = model.mriot.x.loc[("CHE"), :]

    # country_iso3alpha = pycountry.countries.get(name=country).alpha_3
    # row_factor = get_country_modifier(supchain, country_iso3alpha)

    # create a lookup table for each sector and its total production
    lookup = {}
    for idx, row in secs_prod.iterrows():
        lookup[idx] = row["indout"]

    df = results.loc[
        (results["region"] == "CHE")
        & (results["metric"] == "absolute production change")
        & (results["method"] == io_approach).copy()
    ]
    df = df.rename(columns={"value": "impact", "method": "io_approach"})
    df = df.drop(columns=["region", "metric"])
    df["hazard_type"] = haz_type
    df["sector_of_impact"] = sector
    df["scenario"] = scenario
    df["ref_year"] = ref_year
    df["country_of_impact"] = country
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file)
    return


def dump_supchain_to_csv(
    model,
    results,
    haz_type,
    sector,
    scenario,
    ref_year,
    country,
    io_approach,
    n_sim=100,
    return_period=100,
    output_file=None,
):
    index_rp = np.floor(n_sim / return_period).astype(int) - 1
    indirect_impacts = []

    # total production of each sector for country in Millions

    secs_prod = model.mriot.x.loc[("CHE"), :]

    country_iso3alpha = pycountry.countries.get(name=country).alpha_3
    io_countries = model.mriot.get_regions()
    rotw_factor = get_country_modifier(
        country_iso3alpha,
        io_countries,
        mriot_name=model.mriot.name,
        mriot_year=model.mriot.year,
    )

    # create a lookup table for each sector and its total production
    lookup = {}
    for idx, row in secs_prod.iterrows():
        lookup[idx] = row["indout"]

    df = results.loc[
        (results["region"] == "CHE")
        & (results["metric"] == "absolute production change")
        & (results["method"] == io_approach)
    ]
    df = df.rename(columns={"value": "impact"})
    df = df.drop(columns=["region", "metric", "method"])
    df = df.set_index(["event_id", "sector"]).unstack()

    for sec, v in df.items():
        # We scale all values such that countries in the rest of the world category
        # are divided evenly by the number of countries in ROW. Countries explicitely in the MRIO
        # table have a rotw_factor of 1
        rp_value = v.sort_values(ascending=False).iloc[index_rp] * rotw_factor
        mean = (v.sum() / n_sim) * rotw_factor
        max_val = v.max() * rotw_factor

        total_production = lookup[
            sec[1]
        ]  # no multiply with the rotw factor, since we use the Swiss productions
        obj = {
            "sector": sec[1],
            "total_sectorial_production_mriot_CHE": total_production,
            "imaxPL": max_val,
            "irmaxPL": ((max_val / total_production) * 100 if total_production != 0 else 0),
            "iAAPL": mean,
            "irAAPL": (mean / total_production) * 100 if total_production != 0 else 0,
            f"iPL{return_period}": rp_value,
            f"irPL{return_period}": (
                (rp_value / total_production) * 100 if total_production != 0 else 0
            ),
            "hazard_type": haz_type,
            "sector_of_impact": sector,
            "scenario": scenario,
            "ref_year": ref_year,
            "country_of_impact": country,
            "io_approach": io_approach,
        }
        indirect_impacts.append(obj)

    df_indirect = pd.DataFrame(indirect_impacts)

    df_indirect.to_csv(output_file)
    return
