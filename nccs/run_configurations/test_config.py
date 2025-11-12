"""
This file contains config dictionaries for some small example analyses.
Each object contains one run with a different hazard and a few countries and can be imported seperately in your analysis.
"""

import pathos as pa

ncpus = 3
ncpus = pa.helpers.cpu_count() - 1


CONFIG = {
    "data_path": "/cluster/project/climate/sjuhel/NCCS",
    "direct_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run/direct",
    "indirect_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run/indirect",
    "mriot_name":"WIOD16",
    "mriot_year": 2011,
    "run_title": "test_run",
    "n_sim_years": 300,  # Number of stochastic years of supply chain impacts to simulate
    "io_approach": [
        "ghosh"
    ],  # Supply chain IO to use. One or more of "leontief", "ghosh"
    "force_recalculation": False,  # If an intermediate file or output already exists should it be recalculated?
    "use_s3": False,  # Also load and save data from an S3 bucket
    "log_level": "INFO",
    "seed": 161,
    # Which parts of the model chain to run:
    "do_direct": True,  # Calculate direct impacts (that aren't already calculated)
    "do_yearsets": True,  # Calculate direct impact yearsets (that aren't already calculated)
    "do_multihazard": False,  # Also combine hazards to create multi-hazard supply chain shocks
    "do_indirect": True,  # Calculate any indirect supply chain impacts (that aren't already calculated)
    "use_sector_bi_scaling": True,  # Calculate sectoral business interruption scaling
    # Impact functions:
    "business_interruption": True,  # Turn off to assume % asset loss = % production loss. Mostly for debugging and reproducibility
    "calibrated": True,  # Turn off to use best guesstimate impact functions. Mostly for debugging and reproducibility
    # Parallisation:
    "do_parallel": False,  # Parallelise some operations
    "ncpus": ncpus,
    "runs": [
        {
            "hazard": "tropical_cyclone",
            "sectors": ["manufacturing"],
            "countries": ["United States"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
            ],
        }
    ],
}

CONFIG2 = {
    "data_path": "/cluster/project/climate/sjuhel/NCCS",
    "direct_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run_agri_fun/direct",
    "indirect_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run_agri_fun/indirect",
    "mriot_name":"WIOD16",
    "mriot_year": 2011,
    "run_title": "test_run_agri_fun",
    "n_sim_years": 300,  # Number of stochastic years of supply chain impacts to simulate
    "io_approach": [
        "ghosh"
    ],  # Supply chain IO to use. One or more of "leontief", "ghosh"
    "force_recalculation": False,  # If an intermediate file or output already exists should it be recalculated?
    "use_s3": False,  # Also load and save data from an S3 bucket
    "log_level": "INFO",
    "seed": 161,
    # Which parts of the model chain to run:
    "do_direct": True,  # Calculate direct impacts (that aren't already calculated)
    "do_yearsets": True,  # Calculate direct impact yearsets (that aren't already calculated)
    "do_multihazard": False,  # Also combine hazards to create multi-hazard supply chain shocks
    "do_indirect": True,  # Calculate any indirect supply chain impacts (that aren't already calculated)
    "use_sector_bi_scaling": True,  # Calculate sectoral business interruption scaling
    # Impact functions:
    "business_interruption": True,  # Turn off to assume % asset loss = % production loss. Mostly for debugging and reproducibility
    "calibrated": True,  # Turn off to use best guesstimate impact functions. Mostly for debugging and reproducibility
    # Parallisation:
    "do_parallel": False,  # Parallelise some operations
    "ncpus": ncpus,
    "runs": [
        {
            "hazard": "river_flood",
            "sectors": ["agriculture"],
            "countries": ["Nigeria", "United States", "Germany", "China", "Australia"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
            ],
        }
    ],
}

CONFIG3 = {
    "data_path": "/cluster/project/climate/sjuhel/NCCS",
    "direct_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run3/direct",
    "indirect_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run3/indirect",
    "mriot_name":"WIOD16",
    "mriot_year": 2011,
    "run_title": "test_run3",
    "n_sim_years": 300,  # Number of stochastic years of supply chain impacts to simulate
    "io_approach": [
        "ghosh"
    ],  # Supply chain IO to use. One or more of "leontief", "ghosh"
    "force_recalculation": False,  # If an intermediate file or output already exists should it be recalculated?
    "use_s3": False,  # Also load and save data from an S3 bucket
    "log_level": "INFO",
    "seed": 161,
    # Which parts of the model chain to run:
    "do_direct": True,  # Calculate direct impacts (that aren't already calculated)
    "do_yearsets": True,  # Calculate direct impact yearsets (that aren't already calculated)
    "do_multihazard": False,  # Also combine hazards to create multi-hazard supply chain shocks
    "do_indirect": True,  # Calculate any indirect supply chain impacts (that aren't already calculated)
    "use_sector_bi_scaling": True,  # Calculate sectoral business interruption scaling
    # Impact functions:
    "business_interruption": True,  # Turn off to assume % asset loss = % production loss. Mostly for debugging and reproducibility
    "calibrated": True,  # Turn off to use best guesstimate impact functions. Mostly for debugging and reproducibility
    # Parallisation:
    "do_parallel": False,  # Parallelise some operations
    "ncpus": ncpus,
    "runs": [
        {
            "hazard": "wildfire",
            "sectors": ["manufacturing"],
            "countries": ["Italy", "China", "Russian Federation"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
            ],
        }
    ],
}

CONFIG4 = {
    "data_path": "/cluster/project/climate/sjuhel/NCCS",
    "direct_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run4/direct",
    "indirect_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run4/indirect",
    "mriot_name":"WIOD16",
    "mriot_year": 2011,
    "run_title": "test_run4",
    "n_sim_years": 300,  # Number of stochastic years of supply chain impacts to simulate
    "io_approach": [
        "ghosh"
    ],  # Supply chain IO to use. One or more of "leontief", "ghosh"
    "force_recalculation": False,  # If an intermediate file or output already exists should it be recalculated?
    "use_s3": False,  # Also load and save data from an S3 bucket
    "log_level": "DEBUG",
    "seed": 161,
    # Which parts of the model chain to run:
    "do_direct": True,  # Calculate direct impacts (that aren't already calculated)
    "do_yearsets": True,  # Calculate direct impact yearsets (that aren't already calculated)
    "do_multihazard": False,  # Also combine hazards to create multi-hazard supply chain shocks
    "do_indirect": True,  # Calculate any indirect supply chain impacts (that aren't already calculated)
    "use_sector_bi_scaling": True,  # Calculate sectoral business interruption scaling
    # Impact functions:
    "business_interruption": True,  # Turn off to assume % asset loss = % production loss. Mostly for debugging and reproducibility
    "calibrated": True,  # Turn off to use best guesstimate impact functions. Mostly for debugging and reproducibility
    # Parallisation:
    "do_parallel": False,  # Parallelise some operations
    "ncpus": ncpus,
    "runs": [
        {
            "hazard": "storm_europe",
            "io_approach": ["ghosh"],
            "sectors": ["manufacturing"],
            "countries": ["Germany", "Ireland"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
            ],
        }
    ],
}

CONFIG5 = {
    "data_path": "/cluster/project/climate/sjuhel/NCCS",
    "direct_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run5/direct",
    "indirect_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_run5/indirect",
    "mriot_name":"WIOD16",
    "mriot_year": 2011,
    "run_title": "test_run5",
    "n_sim_years": 300,  # Number of stochastic years of supply chain impacts to simulate
    "io_approach": [
        "leontief",
        "ghosh",
    ],  # Supply chain IO to use. One or more of "leontief", "ghosh"
    "force_recalculation": False,  # If an intermediate file or output already exists should it be recalculated?
    "use_s3": False,  # Also load and save data from an S3 bucket
    "log_level": "INFO",
    "seed": 161,
    # Which parts of the model chain to run:
    "do_direct": True,  # Calculate direct impacts (that aren't already calculated)
    "do_yearsets": True,  # Calculate direct impact yearsets (that aren't already calculated)
    "do_multihazard": False,  # Also combine hazards to create multi-hazard supply chain shocks
    "do_indirect": True,  # Calculate any indirect supply chain impacts (that aren't already calculated)
    "use_sector_bi_scaling": True,  # Calculate sectoral business interruption scaling
    # Impact functions:
    "business_interruption": True,  # Turn off to assume % asset loss = % production loss. Mostly for debugging and reproducibility
    "calibrated": True,  # Turn off to use best guesstimate impact functions. Mostly for debugging and reproducibility
    # Parallisation:
    "do_parallel": False,  # Parallelise some operations
    "ncpus": ncpus,
    "runs": [
        {
            "hazard": "relative_crop_yield",
            "sectors": ["agriculture"],
            "countries": ["Germany", "United States"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
                {"scenario": "rcp60", "ref_year": 2060},
            ],
        },
    ],
}

CONFIG6 = {
    "data_path": "/cluster/project/climate/sjuhel/NCCS",
    "direct_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_sea/direct",
    "indirect_output_dir":"/cluster/project/climate/sjuhel/NCCS-euler/test_sea/indirect",
    "mriot_name":"WIOD16",
    "mriot_year": 2011,
    "run_title": "test_sea",
    "n_sim_years": 300,  # Number of stochastic years of supply chain impacts to simulate
    "io_approach": [
        "ghosh"
    ],  # Supply chain IO to use. One or more of "leontief", "ghosh"
    "force_recalculation": False,  # If an intermediate file or output already exists should it be recalculated?
    "use_s3": False,  # Also load and save data from an S3 bucket
    "log_level": "INFO",
    "seed": 161,
    # Which parts of the model chain to run:
    "do_direct": True,  # Calculate direct impacts (that aren't already calculated)
    "do_yearsets": True,  # Calculate direct impact yearsets (that aren't already calculated)
    "do_multihazard": True,  # Also combine hazards to create multi-hazard supply chain shocks
    "do_indirect": True,  # Calculate any indirect supply chain impacts (that aren't already calculated)
    "use_sector_bi_scaling": True,  # Calculate sectoral business interruption scaling
    # Impact functions:
    "business_interruption": True,  # Turn off to assume % asset loss = % production loss. Mostly for debugging and reproducibility
    "calibrated": True,  # Turn off to use best guesstimate impact functions. Mostly for debugging and reproducibility
    # Parallisation:
    "do_parallel": False,  # Parallelise some operations
    "ncpus": ncpus,
    "runs": [
        {
            "hazard": "tropical_cyclone",
            "sectors": ["agriculture", "manufacturing"],
            "countries": ["Germany", "United States"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
                # {"scenario": "rcp26", "ref_year": "2040"},
                {"scenario": "rcp26", "ref_year": "2060"},
                # {"scenario": "rcp26", "ref_year": "2080"},
                # {"scenario": "rcp45", "ref_year": "2040"},
                # {"scenario": "rcp45", "ref_year": "2060"},
                # {"scenario": "rcp45", "ref_year": "2080"},
                # {"scenario": "rcp60", "ref_year": "2040"},
                # {"scenario": "rcp60", "ref_year": "2060"},
                # {"scenario": "rcp60", "ref_year": "2080"},
                # {"scenario": "rcp85", "ref_year": "2040"},
                {"scenario": "rcp85", "ref_year": "2060"},
            ],
        },
        {
            "hazard": "river_flood",
            "sectors": ["agriculture", "manufacturing"],
            "countries": ["Germany", "United States"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
                # {"scenario": "rcp26", "ref_year": 2020},
                # {"scenario": "rcp26", "ref_year": 2040},
                {"scenario": "rcp26", "ref_year": 2060},
                # {"scenario": "rcp26", "ref_year": 2080},
                # {"scenario": "rcp60", "ref_year": 2020},
                # {"scenario": "rcp60", "ref_year": 2040},
                # {"scenario": "rcp60", "ref_year": 2060},
                # {"scenario": "rcp60", "ref_year": 2080},
                # {"scenario": "rcp85", "ref_year": 2020},
                # {"scenario": "rcp85", "ref_year": 2040},
                {"scenario": "rcp85", "ref_year": 2060},
                # {"scenario": "rcp85", "ref_year": 2080},
            ],
        },
        {
            "hazard": "wildfire",
            "sectors": ["agriculture", "manufacturing"],
            "countries": ["Germany", "United States"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
            ],
        },
        {
            "hazard": "storm_europe",
            "sectors": ["agriculture", "manufacturing"],
            "countries": ["Germany"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
                # These combinations are possible, but since the windstorm is not yet developed fully, we exclude them.
                {"scenario": "rcp26", "ref_year": "future"},
                # {"scenario": "ssp245", "ref_year": "present"},
                # {"scenario": "ssp370", "ref_year": "present"},
                {"scenario": "rcp85", "ref_year": "future"},
            ],
        },
        {
            "hazard": "relative_crop_yield",
            "sectors": ["agriculture"],
            "countries": ["Germany", "United States"],
            "scenario_years": [
                {"scenario": "None", "ref_year": "historical"},
                {"scenario": "rcp60", "ref_year": 2060},
                {"scenario": "rcp60", "ref_year": 2060},
            ],
        },
    ],
}
