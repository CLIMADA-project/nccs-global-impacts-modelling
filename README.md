# NCCS Global impacts of climate change on Switzerland Modelling Pipeline

## Introduction

This repository contains all the code that was developed and run to generate the results for the [NCCS Global Impacts project](https://www.nccs.admin.ch/nccs/en/home/climate-change-and-impacts/nccs-impacts/global-impacts.html).

It is associated with the [NCCS Global impacts of climate change on Switzerland Data Collection](http://hdl.handle.net/20.500.11850/785985) hosted by ETH research collection.

Contact person: Samuel Juhel, sjuhel[at]ethz.ch

## Repository Content

Here is a quick outlook at the repository structure (note that some low level folders were omitted):

```tree
├── nccs
│   ├── pipeline
│   │   ├── direct
│   │   └── indirect
│   ├── run_configurations
│   └── utils
├── exposures
│   ├── forestry
│   ├── manufacturing
│   │   ├── manufacturing_general_exposure
│   │   └── manufacturing_sub_exposures
│   ├── mining
│   └── utilities
├── resources
│   └── impact_functions
│       ├── business_interruption
│       ├── river_flood
│       ├── storm_europe
│       └── tropical_cyclone
├── additionnal_scripts
│   ├── bi_functions_scripts
│   ├── hazard
│   │   └── stormeurope
│   └── impacts
│       └── slr_intermediate
├── code_examples
└── other
```

- The `nccs` folder contains the code for the main modelling pipeline.
- The `exposure` folder contains the scripts used to generated exposure layers from open-access data for the different economic sectors.
- The `resources` folder contains some additional data files used in the pipeline, notably calibrated impact functions.
- The `additionnal_scripts` folder contains some additional scripts used for additional data treatment contigent to the main modeling pipeline.
- The `code_examples` folder contains examples of how to read and manipulate exposure layers.
- The `other` folder contains possibly helpfull code to setup a virtual machine or setup a dashboard for the results, or check errors of simulation runs.

## Environment requirements

The following describes how to setup an environment able to run a simulation pipeline.

Note however, that the simulation pipeline for the whole configuration used during the project requires significant computation power and space to store all results.
This pipeline was run on a computing server. Users wanting to reproduce the results or run the pipeline on their own configuration
are advised to get a deep understanding of the resources required and possibly to reach the contact person.

Also note that results from the final run are available in the the [NCCS Global impacts of climate change on Switzerland Data Collection](http://hdl.handle.net/20.500.11850/785985)

### Installing CLIMADA and CLIMADA Petals

We refer users without `conda` or `mamba` already installed to the [installation tutorial of CLIMADA](https://climada-python.readthedocs.io/en/stable/getting-started/install.html)

The code within this repository was succesfully run with CLIMADA and CLIMADA petals `v6.1.0`. The following lines should install these specific versions:

```bash
mamba create -n nccs -c conda-forge "climada==6.1.0"
mamba activate nccs
mamba install -c conda-forge "climada_petals==6.1.0"
```

While the CLIMADA team strive to keep backward compatibility, we cannot insure that future version will stay compatible with the code in this repository.

Also note that while the main pipeline was fully tested with this version of CLIMADA, it is not necessarily the case for all other scripts.

### Installing this repository

1. Clone this repository:

```bash
git clone https://github.com/CLIMADA-project/nccs-global-impacts-modelling
```

2. Activate the environment if not already done and place yourself in the repository:

```bash
mamba activate nccs
cd nccs-global-impacts-modelling
```

3. Install the repository

```bash
pip install .
```

#### A note on CLIMADA versions

Some parts of the NCCS repository rely on functionality from different 'periods' of CLIMADA's history.

The above instructions set up CLIMADA as it's required to run the modelling pipeline. This uses CLIMADA 5.0.0 or higher, which is required by the latest tropical cyclone event set (changed from 4.1.0 on 16 Sept 2024).

However, the code used to generate the European Windstorm data depends on version 4.1.0 or earlier. This is _not_ required to run the model pipeline, which uses precalculated windstorm data. It's only relevant if you want to (re)generate windstorm data.

## Usage

### Logging

There is extensive logging within the pipeline, to track the progression of the pipeline and warn users for possible errors.

We thus strongly advise users to [setup a logger](https://docs.python.org/3/howto/logging.html#logging-to-a-file).

### Inputs

Required inputs for the pipeline to run are available in the [NCCS Global impacts of climate change on Switzerland Data Collection](http://hdl.handle.net/20.500.11850/785985).

The folder structure of the archive should match the one expected by the code, thus users simply have to
setup the `'data_path'` value in their configuration.

### Configuration

A run of the pipeline can be configured from a dictionary with the following properties:

```python
{
 'data_path': '<root path for the input data>',
 'direct_output_dir': '<path to output results for direct impacts>',
 'indirect_output_dir': '<path to output results for indirect impacts>',
 'mriot_name': 'WIOD16',               # Only WIOD16 is available
 'mriot_year': 2011,                   # Any valid year for the WIOD16 MRIOT release
 'run_title': 'test_run',              # An identifying name for the run
 'n_sim_years': 300,                   # The number of years to simulate when generating year set of extreme events
 'io_approach': ['ghosh', 'leontief'], # The Input Output approaches to use for indirect impacts
 'force_recalculation': False,         #
 'log_level': 'INFO',
 'seed': 161,            # a seed for the year set simulation
 'do_direct': True,
 'do_yearsets': True,
 'do_multihazard': False,
 'do_indirect': True,
 'use_sector_bi_scaling': True,
 'business_interruption': True,
 'calibrated': True,
 'do_parallel': False,
 'ncpus': 4,
 'runs': [
    {
        'hazard': 'tropical_cyclone', # The list of hazard to be considered
        'sectors': ['manufacturing'], # The list of sectors to be considered 
        'countries': ['United States'], # The list of countries to be considered 
        'scenario_years': [{'scenario': 'None', 'ref_year': 'historical'}] # The list of scenarios to be considered 
    }
        ]
}
```

- Available hazards are: `['tropical_cyclone','river_flood', 'storm_europe', 'relative_crop_yield']`
- Available sectors are: `["agriculture", "forestry", "mining", "manufacturing", "service", "energy", "water", "waste", "basic_metals", "pharmaceutical", "food", "wood", "chemical", "rubber_and_plastic", "non_metallic_mineral", "refin_and_transform"]` (except for the `'relative_crop_yield'` hazard which only accepts `'agriculture'`)
- Available countries depend on hazards and sectors. Look at the different provided configurations for list of countries.
- Available scenarios are: 

  - `{"scenario": "None", "ref_year": "historical"}`
  - `{"scenario": "rcp26", "ref_year": "2060"}`
  - `{"scenario": "rcp85", "ref_year": "2060"}`

Except for agriculture which only has {"scenario": "rcp60", "ref_year": "2060"} for the future scenario.

### Test run

If you correctly installed the repository you should be able to import from the `nccs` module, for instance the test configuration shown before.

```python
from nccs.run_configurations.test_config import CONFIG
print(CONFIG)
```

You need to setup you own paths for the inputs and outputs:

```python
CONFIG['data_path'] = <your own path>
CONFIG['direct_output_dir'] = <your own path>
CONFIG['indirect_output_dir'] = <your own path>
```

You should then be able to run the pipeline on that configuration:

```python
from nccs.analysis import run_pipeline_from_config
run_pipeline_from_config(CONFIG)
```

If you configured a logger you should have the following logging:

```
06:19:30 PM nccs.analysis INFO: Direct output will be saved to <data_path>/test_run/direct
06:19:30 PM nccs.analysis INFO: Config:
{'data_path': '<data_path>/', 'direct_output_dir': '<data_path>/test_run/direct', 'indirect_output_dir': '<data_path>/test_run/indirect', 'mriot_name': 'WIOD16', 'mriot_year': 2011, 'run_title': 'test_run', 'n_sim_years': 300, 'io_approach': ['ghosh'], 'force_recalculation': False, 'use_s3': False, 'log_level': 'INFO', 'seed': 161, 'do_direct': True, 'do_yearsets': True, 'do_multihazard': False, 'do_indirect': True, 'use_sector_bi_scaling': True, 'business_interruption': True, 'calibrated': True, 'do_parallel': False, 'ncpus': 4, 'runs': [{'hazard': 'tropical_cyclone', 'sectors': ['manufacturing'], 'countries': ['United States'], 'scenario_years': [{'scenario': 'None', 'ref_year': 'historical'}]}], 'time_run': '2025-11-11 18:19:30.116809'}
06:19:30 PM nccs.analysis INFO: 

RUNNING DIRECT IMPACT CALCULATIONS
06:19:30 PM nccs.analysis INFO: There are 1 direct impacts to calculate. (0 exist already. Full analysis has 1 impacts.)
06:19:30 PM nccs.analysis INFO: Calculating direct impacts for for {'hazard': 'tropical_cyclone', 'sector': 'manufacturing', 'country': 'United States', 'scenario': 'None', 'ref_year': 'historical'}
06:19:30 PM nccs.pipeline.direct.direct INFO: Computing direct impact for tropical_cyclone, manufacturing, United States (USA), None, historical
06:19:30 PM nccs.pipeline.direct.direct INFO: Trying to get tropical_cyclone, for USA, None, historical from <data_path>.
06:19:30 PM nccs.utils.euler INFO: Fetching <data_path>/hazard/tc_wind/historical/tropcyc_USA_historical.hdf5
06:19:39 PM nccs.pipeline.direct.direct INFO: Fetching exposure for United States from manufacturing/manufacturing_general_exposure/country_split/general_manufacture_values
2025-11-11 18:19:39,342 - climada.entity.exposures.base - WARNING - There are no impact functions assigned to the exposures
06:19:39 PM nccs.pipeline.direct.business_interruption INFO: Fetching sector scaling for USA
06:19:41 PM nccs.analysis INFO: 

CREATING IMPACT YEARSETS
06:19:41 PM nccs.analysis INFO: There are 1 yearsets to create. (0 already exist, 0 of the remaining are missing direct impact data, full analysis has 1 yearsets.)
06:19:41 PM nccs.analysis INFO: yearsets will be saved in <data_path>/test_run/direct/yearsets
06:19:41 PM nccs.analysis INFO: Generating yearsets for {'hazard': 'tropical_cyclone', 'sector': 'manufacturing', 'country': 'United States', 'scenario': 'None', 'ref_year': 'historical'}
06:19:42 PM nccs.pipeline.direct.direct INFO: Fetching exposure for United States from manufacturing/manufacturing_general_exposure/country_split/general_manufacture_values
2025-11-11 18:19:42,049 - climada.entity.exposures.base - WARNING - There are no impact functions assigned to the exposures
06:19:42 PM nccs.pipeline.direct.calc_yearset INFO: Correcting TC event frequencies.
06:19:45 PM nccs.analysis INFO: Skipping multihazard impact calculations. Set do_multihazard: True in your config to change this
06:19:45 PM nccs.analysis INFO: 

MODELLING SUPPLY CHAINS
06:19:45 PM nccs.analysis INFO: There are 0 out of 1 leontief supply chains to calculate
06:19:45 PM nccs.analysis INFO: There are 1 out of 1 ghosh supply chains to calculate
06:19:45 PM root INFO: Read metadata from climada/data/MRIOT/WIOD16/2011/metadata.json
06:19:45 PM root INFO: 20251111 18:19:45 - FILEIO -  Loaded IO system from climada/data/MRIOT/WIOD16/2011
06:19:45 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/Z.parquet
06:19:45 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/Y.parquet
06:19:45 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/x.parquet
06:19:45 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/A.parquet
06:19:45 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/B.parquet
06:19:46 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/L.parquet
06:19:46 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/G.parquet
06:19:46 PM root INFO: Load data from climada/data/MRIOT/WIOD16/2011/unit.parquet
06:19:56 PM nccs.analysis INFO: Starting indirect impact calculations for 1 configurations.
06:19:56 PM nccs.analysis INFO: Processing exposure for country: 'United States', sector: 'manufacturing'
06:19:56 PM nccs.pipeline.direct.direct INFO: Fetching exposure for United States from manufacturing/manufacturing_general_exposure/country_split/general_manufacture_values
2025-11-11 18:19:56,905 - climada.entity.exposures.base - WARNING - There are no impact functions assigned to the exposures
06:19:56 PM nccs.analysis INFO: Loading hazard from: <data_path>/test_run/direct/yearsets/yearset_tropical_cyclone_manufacturing_None_historical_USA.hdf5
06:19:58 PM nccs.analysis INFO: Running calculation for: {'hazard': 'tropical_cyclone', 'sector': 'manufacturing', 'country': 'United States', 'scenario': 'None', 'ref_year': 'historical'}
06:20:18 PM nccs.pipeline.indirect.event_aggregations INFO: Reading event files from <data_path>/test_run/indirect/events:
['<data_path>/test_run/indirect/events/indirect_impacts_tropical_cyclone_manufacturing_None_historical_ghosh_USA.csv']
06:20:19 PM nccs.analysis INFO: 

Done!
```

## Contributions

The methodologies, implementation and data collection were done by Alina Mastai, Christopher Fairless, Gaudenz Halter and Kaspar Tobler.

The final curation of the repository and data was done by Samuel Juhel.
