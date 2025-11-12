# Exposures scripts and data

This directory contains the scripts to generated the geographical distribution of exposed assets used for the direct impact calculations, as well as the necessary data.

Each folder aims at generating `.h5` files for each considered sectors, both at the global scope and split by country.

These generated files can be imported by CLIMADA as `Exposures` objects and are point-based representation of economic assets. 
Methological details for each exposure types are described in markdown files in each folder.

- `exposures/forestry/`: Refined exposure data for the forestry sector.
- `exposures/manufacturing/`: Refined exposure data for the manufacturing sector, including subexposures for individual subsectors.
- `exposures/mining/`: Refined exposure data for the mining sector.
- `exposures/utilities/`: Refined exposure data for the utilities sector, including subexposures for individual subsectors.

Output files contain:

| Name           | Type    | HDF5 Dtype | Description                                                                                     |
| :------------- | :------ | :--------- | :---------------------------------------------------------------------------------------------- |
| **latitude**   | Dataset | `float64`  | Coordinates of assets (typically part of the exposure point's location).                        |
| **longitude**  | Dataset | `float64`  | Coordinates of assets (typically part of the exposure point's location).                        |
| **region_id**  | Dataset | `object`   | **ISO ALPHA3** standard country code corresponding to the coordinates.                          |
| **value**      | Dataset | `float64`  | The estimated share of total economic sector production (WIOD 2011 baseline, **USD Millions**). |