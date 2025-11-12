import logging

from climada.hazard import Hazard


LOGGER = logging.getLogger(__name__)

def load_hazard_from_storage(path, data_path):
    LOGGER.info(f"Fetching {data_path / path}")
    haz = Hazard.from_hdf5(data_path / path)
    return haz
