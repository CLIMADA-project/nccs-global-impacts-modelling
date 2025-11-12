# for the wilfire impact function:
# /climada_petals/blob/main/climada_petals/entity/impact_funcs/wildfire.py

import logging
from pathlib import Path
from typing import get_args, Union, Optional

import pandas as pd
import pycountry
from climada.engine.impact_calc import ImpactCalc, Impact
from climada.entity import Exposures
from climada.entity import ImpactFuncSet, ImpfSetTropCyclone, ImpfTropCyclone
from climada.entity.impact_funcs.storm_europe import ImpfStormEurope
from climada.hazard import Hazard
from climada.util.api_client import Client
from climada_petals.entity.impact_funcs.river_flood import (
    RIVER_FLOOD_REGIONS_CSV,
    flood_imp_func_set,
)

# for the wilfire impact function:
# https://github.com/CLIMADA-project/climada_petals/blob/main/climada_petals/entity/impact_funcs
from climada_petals.entity.impact_funcs.wildfire import ImpfWildfire

from exposures.utils import root_dir
from nccs.pipeline.direct import agriculture, stormeurope
from nccs.pipeline.direct.river_flood import ImpfFlood
from nccs.pipeline.direct.business_interruption import convert_impf_to_sectoral_bi_dry
from nccs.pipeline.direct.business_interruption import convert_impf_to_sectoral_bi_wet
from nccs.pipeline.direct.combine_impact_funcs import scale_impf
from nccs.utils.folder_naming import get_resources_dir
from nccs.utils.euler import load_hazard_from_storage

LOGGER = logging.getLogger(__name__)


project_root = root_dir()
# /wildfire.py

HAZ_TYPE_LOOKUP = {
    "tropical_cyclone",
    "river_flood",
    "wildfire",
    "storm_europe",
    "relative_crop_yield",
    "sea_level_rise",
}

SECTOR_FILE_MAP = {
    "manufacturing": "manufacturing/manufacturing_general_exposure/country_split/general_manufacture_values",
    "mining": "{sector}/country_split/mining_values",
    "forestry": "forestry/country_split/{sector}_values",
    "energy": "utilities/{sector}/country_split/{sector}_values",
    "waste": "utilities/{sector}/country_split/{sector}_values",
    "water": "utilities/{sector}/country_split/{sector}_values",
    "pharmaceutical": "manufacturing/manufacturing_sub_exposures/{sector}/country_split/{sector}_manufacture_values",
    "basic_metals": "manufacturing/manufacturing_sub_exposures/{sector}/country_split/{sector}_manufacture_values",
    "chemical": "manufacturing/manufacturing_sub_exposures/{sector}_process/country_split/{sector}_process_manufacture_values",
    "food": "manufacturing/manufacturing_sub_exposures/{sector}_and_paper/country_split/{sector}_and_paper_manufacture_values",
    "non_metallic_mineral": "manufacturing/manufacturing_sub_exposures/{sector}/country_split/{sector}_manufacture_values",
    "refin_and_transform": "manufacturing/manufacturing_sub_exposures/{sector}/country_split/{sector}_manufacture_values",
    "rubber_and_plastic": "manufacturing/manufacturing_sub_exposures/{sector}/country_split/{sector}_manufacture_values",
    "wood": "manufacturing/manufacturing_sub_exposures/{sector}/country_split/{sector}_manufacture_values",
}


def nccs_direct_impacts_simple(
    haz_type: str,
    sector: str,
    country: str,
    scenario: str,
    ref_year: Union[str, int],
    data_path: str,
    business_interruption: bool = True,
    calibrated: bool = True,
    use_sector_bi_scaling: bool = True,
) -> Impact:
    """Calculate direct impacts for a given hazard, sector, and country.

    Parameters
    ----------
    haz_type : str
        The type of hazard (e.g., 'tropical_cyclone', 'river_flood').
    sector : str
        The economic sector.
    country : str
        The name of the country.
    scenario : str
        The climate scenario (e.g., 'rcp60', 'historical').
    ref_year : Union[str, int]
        The reference year for the scenario.
    data_path : str
        The path to the data directory.
    business_interruption : bool, optional
        Whether to include business interruption in the impact function, by default True.
    calibrated : bool, optional
        Whether to use calibrated impact functions, by default True.
    use_sector_bi_scaling : bool, optional
        Whether to use country-specific scaling for business interruption, by default True.

    Returns
    -------
    Impact
        A CLIMADA Impact object containing the calculated impacts.

    Raises
    ------
    ValueError
        If the hazard or exposure object cannot be found.
    """
    # Country names can be checked here: https://github.com/flyingcircusio/pycountry/blob/main/src/pycountry
    # /databases/iso3166-1.json
    country_iso3alpha = pycountry.countries.get(name=country).alpha_3
    LOGGER.info(f"Computing direct impact for {haz_type}, {sector}, {country} ({country_iso3alpha}), {scenario}, {ref_year}")
    data_path = Path(data_path)
    haz = get_hazard(haz_type, country_iso3alpha, scenario, ref_year, data_path)
    if not haz:
        raise ValueError(
            f"Cound not find hazard object for {haz_type} {country_iso3alpha} {scenario} {ref_year}"
        )
    exp = get_sector_exposure(sector, country, data_path=data_path)  # was originally here
    if not exp:
        raise ValueError(
            f"Cound not find exposure object for {country_iso3alpha} {scenario} {ref_year}"
        )

    # exp = sectorial_exp_CI_MRIOT(country=country_iso3alpha, sector=sector) #replaces the command above
    impf_set = apply_sector_impf_set(
        haz_type,
        sector,
        country_iso3alpha,
        business_interruption,
        calibrated,
        use_sector_bi_scaling,
    )
    exp.gdf["impf_"] = impf_set.get_ids(impf_set.get_hazard_types()[0])[0]
    imp = ImpactCalc(exp, impf_set, haz).impact(save_mat=True)
    imp.event_name = [str(e) for e in imp.event_name]
    return imp


def load_exposure_from_storage(country: str, file_short: str, data_path: str) -> Exposures:
    """Load exposure data from a local HDF5 file.

    Parameters
    ----------
    country : str
        The name of the country.
    file_short : str
        The short name of the file, used to construct the full path.

    Returns
    -------
    Exposures
        A CLIMADA Exposures object.
    """
    LOGGER.info(f"Fetching exposure for {country} from {file_short}")
    country_iso3alpha = pycountry.countries.get(name=country).alpha_3
    outputfile = f"{data_path}/exposures/{file_short}_{country_iso3alpha}.h5"
    h5_file = pd.read_hdf(outputfile)
    # Generate an Exposures instance from DataFrame
    exp = Exposures(h5_file)
    exp.set_geometry_points()
    exp.gdf["value"] = exp.gdf.value
    exp.check()
    return exp


def get_sector_exposure(sector: str, country: str, data_path) -> Exposures:
    """Get the exposure data for a given sector and country.

    This function fetches the appropriate exposure data based on the sector.
    For many sectors, it constructs a file path and loads data from storage.
    For special cases like 'service' or 'agriculture', it uses specific logic.

    Parameters
    ----------
    sector : str
        The name of the economic sector.
    country : str
        The name of the country.

    Returns
    -------
    Exposures
        A CLIMADA Exposures object for the given sector and country.

    Raises
    ------
    ValueError
        If the specified sector is not recognized.
    """
    client = Client()

    if sector in ("service", "economic_assets"):
        exp = client.get_litpop(country)

    elif sector.startswith("agriculture_"):
        _, crop_type = agriculture.split_agriculture_sector(sector)
        exp = agriculture.get_exposure(
            country=country, crop_type=crop_type, scenario="histsoc", irr="firr"
        )

    elif sector == "agriculture":
        # Combine exposures for all crop types.
        exps = [
            agriculture.get_exposure(
                country=country, crop_type=crop_type, scenario="histsoc", irr="firr"
            )
            for crop_type in get_args(agriculture.CropType)
        ]
        # Sum up the exposure values at each location.
        exp = exps[0].copy()
        for other_exp in exps[1:]:
            exp.gdf["value"] += other_exp.gdf["value"]

    elif sector in SECTOR_FILE_MAP:
        file_template = SECTOR_FILE_MAP[sector]
        file_short = file_template.format(sector=sector)
        exp = load_exposure_from_storage(country, file_short, data_path=data_path)

    else:
        raise ValueError(f"Sector {sector} not recognised")

    exp.gdf.reset_index(inplace=True)

    # Some exposures have alpha3 codes as region IDs, some have numeric codes. Let's standardise
    if not all(exp.gdf.region_id):
        raise ValueError("There are missing region IDs")
    try:
        # convert numeric codes to alphanumeric
        exp.gdf.region_id = pd.to_numeric(exp.gdf.region_id)
        exp.gdf.region_id = [
            pycountry.countries.get(numeric=str(int(id)).zfill(3)).alpha_3
            for id in exp.gdf.region_id
        ]
    except ValueError as e:
        # Just check the countries are legit
        exp.gdf.region_id = [
            pycountry.countries.get(alpha_3=id).alpha_3 for id in exp.gdf.region_id
        ]
    return exp


def apply_sector_impf_set(
    hazard: str,
    sector: str,
    country_iso3alpha: str,
    business_interruption: bool = True,
    calibrated: bool = True,
    use_sector_bi_scaling: bool = True,
) -> ImpactFuncSet:
    """Apply a sector-specific impact function set for a given hazard.

    Parameters
    ----------
    hazard : str
        The type of hazard.
    sector : str
        The economic sector.
    country_iso3alpha : str
        The ISO 3166-1 alpha-3 code of the country.
    business_interruption : bool, optional
        Whether to include business interruption, by default True.
    calibrated : bool, optional
        Whether to use calibrated impact functions, by default True.
    use_sector_bi_scaling : bool, optional
        Whether to use country-specific scaling for business interruption, by default True.

    Returns
    -------
    ImpactFuncSet
        A CLIMADA ImpactFuncSet object.

    Raises
    ------
    ValueError
        If no impact functions are defined for the given hazard.
    """
    if not business_interruption or sector in ["agriculture", "economic_assets"]:
        sector_bi = None
    else:
        sector_bi = sector

    if hazard == "tropical_cyclone" and sector == "agriculture":
        return agriculture.get_impf_set_tc()
    if hazard == "tropical_cyclone":
        return ImpactFuncSet(
            [
                get_sector_impf_tc(
                    country_iso3alpha, sector_bi, calibrated, use_sector_bi_scaling
                )
            ]
        )
    if hazard == "river_flood" and sector.startswith("agriculture"):
        return agriculture.get_impf_set_rf(country_iso3alpha)
    if hazard == "river_flood":
        return ImpactFuncSet(
            [
                get_sector_impf_rf(
                    country_iso3alpha, sector_bi, calibrated, use_sector_bi_scaling
                )
            ]
        )
    # Use for sea level rise the same functions as for river flood
    if hazard == "sea_level_rise" and sector.startswith("agriculture"):
        return agriculture.get_impf_set_rf(country_iso3alpha, haz_type="TCSurgeBathtub")
    if hazard == "sea_level_rise":
        return ImpactFuncSet(
            [
                get_sector_impf_rf(
                    country_iso3alpha,
                    sector_bi,
                    calibrated,
                    use_sector_bi_scaling,
                    haz_type="TCSurgeBathtub",
                )
            ]
        )
    if hazard == "wildfire":
        return ImpactFuncSet(
            [get_sector_impf_wf(sector_bi, use_sector_bi_scaling=use_sector_bi_scaling, country_iso3alpha=country_iso3alpha)]
        )
    if hazard == "storm_europe" and sector.startswith("agriculture"):
        return agriculture.get_impf_set_tc(haz_type="WS")
    if hazard == "storm_europe":
        return ImpactFuncSet(
            [
                get_sector_impf_stormeurope(
                    country_iso3alpha, sector_bi, calibrated, use_sector_bi_scaling
                )
            ]
        )
    if hazard.startswith("relative_crop_yield"):
        _, crop_type = agriculture.split_agriculture_hazard(hazard)
        return agriculture.get_impf_set(crop_type)
    raise ValueError(f"No impact functions defined for hazard {hazard}")


def get_sector_impf_tc(
    country_iso3alpha: str,
    sector_bi: Optional[str],
    calibrated: Union[bool, int] = True,
    use_sector_bi_scaling: bool = True,
) -> Union[ImpfTropCyclone, "ImpactFuncComposable"]:
    """Get a sector-specific impact function for tropical cyclones.

    Parameters
    ----------
    country_iso3alpha : str
        The ISO 3166-1 alpha-3 code of the country.
    sector_bi : Optional[str]
        The business interruption sector. If None, BI is not applied.
    calibrated : Union[bool, int], optional
        If True or 1, use calibrated impact functions. Otherwise, use default.
        By default True.
    use_sector_bi_scaling : bool, optional
        Whether to use country-specific scaling for business interruption, by default True.

    Returns
    -------
    Union[ImpfTropCyclone, "ImpactFuncComposable"]
        The impact function for tropical cyclones, potentially composed with a
        business interruption function.

    Raises
    ------
    ValueError
        If a unique region cannot be found for the country.
    """
    impf_id, regions, _ = ImpfSetTropCyclone.get_impf_id_regions_per_countries([country_iso3alpha])
    # region = [
    #     region
    #     for region, country_list in region_mapping.items()
    #     if country_iso3alpha in country_list
    # ]
    # if len(region) != 1:
    #     raise ValueError(
    #         f"Could not find a unique region for ISO3 code {country_iso3alpha}. Results: {region}"
    #     )
    region = regions[0]

    if calibrated == 1:
        calibrated_impf_parameters_file = Path(
            get_resources_dir(),
            "impact_functions",
            "tropical_cyclone",
            "calibrated_emanuel_v2.csv",
        )
        calibrated_impf_parameters = pd.read_csv(
            calibrated_impf_parameters_file
        ).set_index(["region"])
        impf = ImpfTropCyclone.from_emanuel_usa(
            scale=calibrated_impf_parameters.loc[region, "scale"],
            v_thresh=calibrated_impf_parameters.loc[region, "v_thresh"],
            v_half=calibrated_impf_parameters.loc[region, "v_half"],
        )
    else:
        #fun_id = impf_ids[region]
        impf = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet().get_func(
            haz_type="TC", fun_id=impf_id
        )  # To use Eberenz functions

    impf.id = 1
    if not sector_bi:
        return impf
    return convert_impf_to_sectoral_bi_dry(
        impf,
        sector_bi,
        country_iso3alpha=country_iso3alpha,
        use_sector_bi_scaling=use_sector_bi_scaling,
    )


def get_sector_impf_rf(
    country_iso3alpha: str,
    sector_bi: Optional[str],
    calibrated: Union[bool, int] = True,
    use_sector_bi_scaling: bool = True,
    haz_type: str = "RF",
) -> Union[ImpfFlood, "ImpactFuncComposable"]:
    """Get a sector-specific impact function for river floods.

    Parameters
    ----------
    country_iso3alpha : str
        The ISO 3166-1 alpha-3 code of the country.
    sector_bi : Optional[str]
        The business interruption sector. If None, BI is not applied.
    calibrated : Union[bool, int], optional
        If True or 1, use calibrated impact functions. Otherwise, use default.
        By default True.
    use_sector_bi_scaling : bool, optional
        Whether to use country-specific scaling for business interruption, by default True.
    haz_type : str, optional
        The hazard type string, by default "RF".

    Returns
    -------
    Union[ImpfFlood, "ImpactFuncComposable"]
        The impact function for river floods, potentially composed with a
        business interruption function.

    Raises
    ------
    ValueError
        If the custom impact function file has unrecognized columns.
    """
    # Use the flood module's lookup to get the regional impact function for the country
    country_info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)

    if not calibrated:
        impf_id = country_info.loc[
            country_info["ISO"] == country_iso3alpha, "impf_RF"
        ].values[0]
        # Grab just that impact function from the flood set, and set its ID to 1
        impf_set = flood_imp_func_set()
        impf = impf_set.get_func(haz_type="RF", fun_id=impf_id)
    else:
        if calibrated == 1:
            calibrated_impf_parameters_file = Path(
                get_resources_dir(),
                "impact_functions",
                "river_flood",
                "calibrated_v1.csv",
            )
        else:
            calibrated_impf_parameters_file = Path(
                get_resources_dir(), "impact_functions", "river_flood", "custom.csv"
            )
        calibrated_impf_parameters = pd.read_csv(calibrated_impf_parameters_file)

        if set(calibrated_impf_parameters.columns) == {"v_half"}:
            assert calibrated_impf_parameters.shape[0] == 1
            impf = ImpfFlood.from_exp_sigmoid(
                v_half=calibrated_impf_parameters.loc[0, "v_half"]
            )
        elif set(calibrated_impf_parameters.columns) == {"v_half", "translate"}:
            assert calibrated_impf_parameters.shape[0] == 1
            impf = ImpfFlood.from_exp_sigmoid(
                v_half=calibrated_impf_parameters.loc[0, "v_half"],
                translate=calibrated_impf_parameters.loc[0, "translate"],
            )
        elif set(calibrated_impf_parameters.columns) == {
            "quantile",
            "v_half",
            "translate",
        }:
            vuln_country_data = pd.read_csv(
                Path(get_resources_dir(), "vuln_2022_nd_gain.csv")
            )
            n_vuln_quantiles = 5
            vuln_country_data["vuln_quantile"] = pd.qcut(
                vuln_country_data["vuln_nd_gain_2022"], n_vuln_quantiles, labels=False
            )
            quantile = vuln_country_data[
                vuln_country_data["country"] == country_iso3alpha
            ]["vuln_quantile"]
            # TODO move this impact function out of calibration andinto the pipeline
            impf = ImpfFlood.from_exp_sigmoid(
                v_half=calibrated_impf_parameters.loc[quantile, "v_half"].item(),
                translate=calibrated_impf_parameters.loc[quantile, "translate"].item(),
            )
        else:
            raise ValueError(
                f"Could not process a custom impact function file with these column names: {calibrated_impf_parameters.columns}"
            )

    if haz_type != "RF":
        impf.haz_type = haz_type
    impf.id = 1
    if not sector_bi:
        return impf
    return convert_impf_to_sectoral_bi_wet(
        impf,
        sector_bi,
        country_iso3alpha=country_iso3alpha,
        use_sector_bi_scaling=use_sector_bi_scaling,
    )


def get_sector_impf_stormeurope(
    country_iso3alpha: str,
    sector_bi: Optional[str],
    calibrated: Union[bool, int] = True,
    use_sector_bi_scaling: bool = True,
) -> Union[ImpfStormEurope, "ImpactFuncComposable"]:
    """Get a sector-specific impact function for European storms.

    Parameters
    ----------
    country_iso3alpha : str
        The ISO 3166-1 alpha-3 code of the country.
    sector_bi : Optional[str]
        The business interruption sector. If None, BI is not applied.
    calibrated : Union[bool, int], optional
        If True or 1, use calibrated impact functions. Otherwise, use default.
        By default True.
    use_sector_bi_scaling : bool, optional
        Whether to use country-specific scaling for business interruption, by default True.

    Returns
    -------
    Union[ImpfStormEurope, "ImpactFuncComposable"]
        The impact function for European storms, potentially composed with a
        business interruption function.
    """
    if not calibrated:
        impf = ImpfStormEurope.from_schwierz()
    else:
        if calibrated == 1:
            calibrated_impf_parameters_file = Path(
                get_resources_dir(),
                "impact_functions",
                "storm_europe",
                "calibrated_v1.csv",
            )
        else:
            calibrated_impf_parameters_file = Path(
                get_resources_dir(), "impact_functions", "storm_europe", "custom.csv"
            )

        calibrated_impf_parameters = pd.read_csv(calibrated_impf_parameters_file)
        assert calibrated_impf_parameters.shape[0] == 1
        impf = scale_impf(
            ImpfStormEurope.from_schwierz(),
            translate=calibrated_impf_parameters.loc[0, "translate"],
            scale=calibrated_impf_parameters.loc[0, "scale"],
        )

    if not sector_bi:
        return impf
    return convert_impf_to_sectoral_bi_dry(
        impf,
        sector_bi,
        country_iso3alpha=country_iso3alpha,
        use_sector_bi_scaling=use_sector_bi_scaling,
    )


# for wildfire, not sure if it is working
def get_sector_impf_wf(
    sector_bi: Optional[str],
    country_iso3alpha: Optional[str] = None,
    use_sector_bi_scaling: bool = True,
) -> Union[ImpfWildfire, "ImpactFuncComposable"]:
    """Get a sector-specific impact function for wildfires.

    Parameters
    ----------
    sector_bi : Optional[str]
        The business interruption sector. If None, BI is not applied.
    country_iso3alpha : Optional[str], optional
        The ISO 3166-1 alpha-3 code of the country, by default None.
    use_sector_bi_scaling : bool, optional
        Whether to use country-specific scaling for business interruption, by default True.

    Returns
    -------
    Union[ImpfWildfire, "ImpactFuncComposable"]
        The impact function for wildfires, potentially composed with a
        business interruption function.
    """
    impf = ImpfWildfire.from_default_FIRMS(
        i_half=409.4
    )  # adpated i_half according to hazard resolution of 4km: i_half=409.4
    impf.haz_type = "WFseason"  # TODO there is a warning when running the code that the haz_type is set to WFsingle,
    # but if I set it to WFsingle, the code does not work
    if not sector_bi:
        return impf
    return convert_impf_to_sectoral_bi_dry(
        impf,
        sector_bi,
        country_iso3alpha=country_iso3alpha,
        use_sector_bi_scaling=use_sector_bi_scaling,
    )


def get_hazard(
    haz_type: str,
    country_iso3alpha: str,
    scenario: str,
    ref_year: Union[str, int],
    data_path: str,
) -> Optional[Hazard]:
    """Get hazard data for a given type, country, and scenario.

    Parameters
    ----------
    haz_type : str
        The type of hazard.
    country_iso3alpha : str
        The ISO 3166-1 alpha-3 code of the country.
    scenario : str
        The climate scenario.
    ref_year : Union[str, int]
        The reference year.
    data_path : str
        The path to the data directory.

    Returns
    -------
    Optional[Hazard]
        A CLIMADA Hazard object, or None if not found.

    Raises
    ------
    ValueError
        If the hazard type is not recognized.
    """
    LOGGER.info(
        f"Trying to get {haz_type}, for {country_iso3alpha}, {scenario}, {ref_year} from {data_path}."
    )
    client = Client()
    if haz_type == "tropical_cyclone":
        if scenario == "None" and ref_year == "historical":
            storage_path = (
                f"hazard/tc_wind/historical/tropcyc_{country_iso3alpha}_historical.hdf5"
            )
        else:
            storage_path = (
                f"hazard/tc_wind/{scenario}_{ref_year}/tropcyc_150arcsec_25synth_"
                f"{country_iso3alpha}_1980_to_2023_{scenario}_{ref_year}.hdf5"
            )
        return load_hazard_from_storage(storage_path, data_path)

    elif haz_type == "river_flood":
        if scenario == "None" and ref_year == "historical":
            return client.get_hazard(
                haz_type,
                properties={
                    "country_iso3alpha": country_iso3alpha,
                    "climate_scenario": "historical",
                    "year_range": "1980_2000",
                },
            )
        else:
            if not isinstance(ref_year, int):
                raise ValueError(
                    f"ref_year must be an integer for river flood hazard, got {ref_year}"
                )
            year_range_midpoint = round(ref_year / 20) * 20
            year_range = (
                str(year_range_midpoint - 10) + "_" + str(year_range_midpoint + 10)
            )
            return client.get_hazard(
                haz_type,
                properties={
                    "country_iso3alpha": country_iso3alpha,
                    "climate_scenario": scenario,
                    "year_range": year_range,
                },
            )
    elif haz_type == "wildfire":
        year_range = "2001_2020"
        if scenario == "None" and ref_year == "historical":
            return client.get_hazard(
                haz_type,
                properties={
                    "country_iso3alpha": country_iso3alpha,
                    "climate_scenario": "historical",
                    "year_range": year_range,
                },
            )
    elif haz_type == "storm_europe":
        # country_iso3num = pycountry.countries.get(alpha_3=country_iso3alpha).numeric
        haz = stormeurope.get_hazard(
            scenario=scenario, country_iso3alpha=country_iso3alpha, data_path=data_path
        )
        return haz

    elif haz_type == "sea_level_rise":
        """
        Sea level rise has the following file configurations:

        "future climate" includes changes also to the TC surge

        - historical climate, no SLR: tc_surge/no_cc/no_slr/
        - future climate, no SLR:
            tc_surge/rcp26_2060/no_slr/
            tc_surge/rcp85_2060/no_slr/

        -historical climate, future SLR:
            tc_surge/no_cc/ssp126_2060slr/
            tc_surge/no_cc/ssp585_2060slr/
        future climate, future SLR:
            tc_surge/rcp26_2060/ssp126_2060slr/
            tc_surge/rcp26_2060/ssp585_2060slr/
        """
        if scenario == "None" and ref_year == "historical":
            storage_path = f"hazard/tc_surge/no_cc/no_slr/surge_28arcsec_25synth_{country_iso3alpha}_nossp_noslr_no_cc.hdf5"
        else:
            storage_path = (
                f"hazard/tc_surge/no_cc/{scenario}_{ref_year}slr/surge_28arcsec_25synth_{country_iso3alpha}"
                f"_{scenario}_{ref_year}slr_no_cc.hdf5"
            )
        return load_hazard_from_storage(storage_path, data_path)

    elif haz_type.startswith("relative_crop_yield"):
        _, crop_type = agriculture.split_agriculture_hazard(haz_type)
        if scenario == "None" and ref_year == "historical":
            # For soy, there is another historical period available (due to availability)
            if crop_type == "soy":
                return agriculture.get_hazard(
                    country=country_iso3alpha,
                    year_range="1980_2012",
                    scenario="historical",
                    crop_type=crop_type,
                )
            # For the other crop types we use this historical period (due to availability)
            else:
                return agriculture.get_hazard(
                    country=country_iso3alpha,
                    year_range="1971_2001",
                    scenario="historical",
                    crop_type=crop_type,
                )
        else:
            haz = agriculture.get_hazard(
                country=country_iso3alpha,
                year_range="2006_2099",
                scenario=scenario,
                crop_type=crop_type,
            )
            if not haz:
                return None
            return haz.select(date=("2045-01-01", "2074-12-31"))
    else:
        raise ValueError(
            f"Unrecognised haz_type variable: {haz_type}.\nPlease use one of: {list(HAZ_TYPE_LOOKUP)}"
        )
