import typing
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Union, get_args, cast
from climada.hazard import Hazard
from climada.entity import Exposures, ImpactFunc
from climada.entity import ImpactFuncSet
from climada.util.api_client import Client
from climada.util.constants import SYSTEM_DIR
from climada_petals.entity.impact_funcs.relative_cropyield import ImpfRelativeCropyield
from climada_petals.entity.impact_funcs.river_flood import RIVER_FLOOD_REGIONS_CSV
from pycountry import countries

LOGGER = logging.getLogger(__name__)

CropType = typing.Literal[
    "whe",
    "mai",
    "soy",
    "ric",
]
IrrigationType = typing.Literal["firr", "noirr"]


def split_agriculture_hazard(label: str) -> Tuple[str, CropType]:
    """Parse a label like 'relative_crop_yield_whe' into its components.

    Parameters
    ----------
    label : str
        The label to parse.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the hazard type and the crop type.
    """
    hazard = "relative_crop_yield"
    crop_type = label.replace("relative_crop_yield_", "")
    if crop_type not in get_args(CropType):
        raise ValueError(f"Unknown crop type '{crop_type}' in label '{label}'")
    return hazard, cast(CropType, crop_type)


def split_agriculture_sector(label: str) -> Tuple[str, CropType]:
    """Parse a label like 'agriculture_whe' into its components.

    Parameters
    ----------
    label : str
        The label to parse.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the sector and the crop type.
    """
    sector = "agriculture"
    crop_type = label.replace("agriculture_", "")
    if crop_type not in get_args(CropType):
        raise ValueError(f"Unknown crop type '{crop_type}' in label '{label}'")
    return sector, cast(CropType, crop_type)


def get_exposure(
    country: str,
    crop_type: CropType = "whe",
    scenario: str = "histsoc",
    irr: IrrigationType = "firr",
) -> Exposures:
    """Get agriculture exposure for a given country and crop type.

    Parameters
    ----------
    country : str
        The name of the country.
    crop_type : CropType, optional
        The type of crop, by default "whe".
    scenario : str, optional
        The scenario, by default "histsoc".
    irr : IrrigationType, optional
        The irrigation type, by default "firr".

    Returns
    -------
    Exposures
        A CLIMADA Exposures object for the specified country and crop.
    """
    client = Client()

    exp = client.get_exposures(
        "crop_production",
        properties={"irrigation_status": irr, "crop": crop_type, "unit": "USD"},
    )
    region_id = int(countries.get(name=country).numeric)
    return Exposures(data=exp.gdf[exp.gdf["region_id"] == region_id])


def get_impf_set(crop_type: Union[CropType, None] = None) -> ImpactFuncSet:
    """Get the impact function set for relative crop yield.

    Parameters
    ----------
    crop_type : Union[CropType, None], optional
        The crop type. This parameter is not currently used but is kept for
        future compatibility. By default None.

    Returns
    -------
    ImpactFuncSet
        A CLIMADA ImpactFuncSet object.
    """
    # TODO: check if other impact functions are needed
    impf_cp = ImpactFuncSet()
    impf_def = ImpfRelativeCropyield.impf_relativeyield()

    # Invert the impact function to match the expected behavior
    impf_def.mdd = -impf_def.mdd

    impf_cp.append(impf_def)
    impf_cp.check()
    return impf_cp


def get_impf_set_tc(haz_type: str = "TC") -> ImpactFuncSet:
    """Get the impact function set for tropical cyclone agriculture damage.

    Parameters
    ----------
    haz_type : str, optional
        The hazard type string, by default "TC".

    Returns
    -------
    ImpactFuncSet
        A CLIMADA ImpactFuncSet object.
    """
    imp_fun_maize = ImpactFunc(
        id=1,
        name=f"{haz_type} agriculture damage",
        intensity_unit="m/s",
        haz_type=haz_type,
        intensity=np.array([0, 11, 38, 60]),
        mdd=np.array([0, 0, 1, 1]),
        paa=np.array([1, 1, 1, 1]),
    )
    imp_fun_maize.check()
    imp_fun_set = ImpactFuncSet([imp_fun_maize])
    return imp_fun_set


def get_impf_set_rf(country_iso3alpha: str, haz_type: str = "RF") -> ImpactFuncSet:
    """Get new Impact function set for river flood and agriculture.

    Based on the JRC publication: https://publications.jrc.ec.europa.eu/repository/handle/JRC105688

    Regional functions available for Africa, Europe, North America, and Asia.
    Other regions (Oceania and South America will get the Global function assigned).

    In this study the damage fractions in the damage curves are intended to span from zero
    (no damage) to one (maximum damage).

    For agriculture the damage is related to a loss in output when the yield is destroyed by floods.

    Parameters
    ----------
    country_iso3alpha : str
        The ISO 3166-1 alpha-3 code of the country.
    haz_type : str, optional
        The hazard type string, by default "RF".

    Returns
    -------
    ImpactFuncSet
        A CLIMADA ImpactFuncSet object for the specified region.
    """
    # Use the flood module's lookup to get the regional impact function for the country
    country_info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
    impf_id = country_info.loc[
        country_info["ISO"] == country_iso3alpha, "impf_RF"
    ].values[0]

    if impf_id == 1:
        impf = ImpactFunc(
            id=1,
            name="Flood Africa JRC Agriculture",
            intensity_unit="m",
            haz_type=haz_type,
            intensity=np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]),
            mdd=np.array([0.00, 0.24, 0.47, 0.74, 0.92, 1.00, 1.00, 1.00, 1.00]),
            paa=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        )

    elif impf_id == 2:
        impf = ImpactFunc(
            id=1,
            name="Flood Asia JRC Agriculture",
            intensity_unit="m",
            haz_type=haz_type,
            intensity=np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]),
            mdd=np.array([0.00, 0.14, 0.37, 0.52, 0.56, 0.66, 0.83, 0.99, 1.00]),
            paa=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        )

    elif impf_id == 3:
        impf = ImpactFunc(
            id=1,
            name="Flood Europe JRC Agriculture",
            intensity_unit="m",
            haz_type=haz_type,
            intensity=np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]),
            mdd=np.array([0.00, 0.30, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00, 1.00]),
            paa=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        )

    elif impf_id == 4:
        impf = ImpactFunc(
            id=1,
            name="Flood North America JRC Agriculture",
            intensity_unit="m",
            haz_type=haz_type,
            intensity=np.array([0, 0.01, 0.5, 1, 1.5, 2, 3, 4, 5, 6]),
            mdd=np.array([0, 0.02, 0.27, 0.47, 0.55, 0.60, 0.76, 0.87, 0.95, 1.00]),
            paa=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        )

    else:
        impf = ImpactFunc(
            id=1,
            name="Flood Global JRC Agriculture",
            intensity_unit="m",
            haz_type=haz_type,
            intensity=np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]),
            mdd=np.array([0.00, 0.24, 0.47, 0.62, 0.71, 0.82, 0.91, 0.99, 1.00]),
            paa=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        )

    impf.check()

    return ImpactFuncSet([impf])


def get_hazard(
    country: str,
    year_range: str,
    scenario: typing.Literal["historical", "rcp60"] = "historical",
    crop_type: CropType = "whe",  # mai instead of wheat
    irr: IrrigationType = "firr",
) -> Hazard:
    """Get relative crop yield hazard for a country.

    This function downloads the hazard data if not already present, assigns
    region IDs to the centroids, saves the result, and then subsets the
    hazard to the specified country.

    Parameters
    ----------
    country : str
        The ISO 3166-1 alpha-3 code of the country.
    year_range : str
        The year range for the hazard data (e.g., "2010-2019").
    scenario : typing.Literal["historical", "rcp60"], optional
        The climate scenario, by default "historical".
    crop_type : CropType, optional
        The crop type, by default "whe".
    irr : IrrigationType, optional
        The irrigation type, by default "firr".

    Returns
    -------
    Hazard
        A CLIMADA Hazard object subsetted for the specified country.
    """
    # TODO how to map the year to the years in this model
    # TODO What about the firr and noirr?
    client = Client()
    properties = {
        "climate_scenario": scenario,
        "crop": crop_type,
        "irrigation_status": irr,
        "year_range": year_range,
    }

    # Bit of a hack: we don't want to assign centroids every time we load this hazard. So we'll get it from the
    # data API (if we haven't already), assign centroids amd then overwrite the download

    # Find where the Data API downloads files to
    haz_info = client.get_dataset_info(
        data_type="relative_cropyield", properties=properties
    )
    assert len(haz_info.files) == 1
    haz_path = Path(
        SYSTEM_DIR,
        haz_info.data_type.data_type_group,
        haz_info.data_type.data_type,
        haz_info.name,
        haz_info.version,
        haz_info.files[0].file_name,
    )

    # Get the hazard if it exists
    if os.path.exists(haz_path):
        hazard = Hazard.from_hdf5(haz_path)

    # Otherwise download, assign centroids, and overwrite
    else:
        hazard = client.get_hazard("relative_cropyield", properties=properties)

    # Check the region IDs are assigned
    # First, delete them if thethey're all 1 (happens in newer climada)
    if hasattr(hazard.centroids, "gdf") and np.all(hazard.centroids.region_id == 1):
        hazard.centroids.gdf.region_id = np.nan
    # And if any are missing, assign them
    if np.any(hazard.centroids.gdf.region_id.isna()):
        LOGGER.info(
            f"One-off region_id assignment to hazard at {haz_path}. Saving result"
        )
        hazard.centroids.set_region_id()
        hazard.write_hdf5(haz_path)

    # Subset to the region we want
    region_id = int(countries.get(alpha_3=country).numeric)
    if not hazard:
        LOGGER.warning(f"There was a problem for {properties}")
    hazard = hazard.select(reg_id=region_id)
    if not hazard:
        LOGGER.warning(f"Selection to {region_id} lead to empty hazard.")
    return hazard
