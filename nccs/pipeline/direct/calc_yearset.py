import numpy as np
import copy
import logging
from scipy import sparse
from functools import reduce
from typing import List, Dict, Union, Optional, Callable

from climada.entity import Exposures
from climada.engine import Impact, ImpactCalc
from climada.util import yearsets

LOGGER = logging.getLogger(__name__)


def yearset_from_imp(
    imp: Impact,
    n_sim_years: int,
    poisson: bool = True,
    cap_exposure: Optional[Union[Exposures, float]] = None,
    seed: Optional[int] = None,
) -> Impact:
    """Generate a yearset of impacts from a single-hazard impact object.

    Parameters
    ----------
    imp : Impact
        The CLIMADA Impact object to generate a yearset from.
    n_sim_years : int
        The number of years in the simulation.
    poisson : bool, optional
        If True, use a Poisson distribution to sample events per year.
        If False, sample one event per year. By default True.
    cap_exposure : Optional[Union[Exposures, float]], optional
        Exposure object or value to cap the impacts at. If provided, the impact
        at each exposure point will not exceed its value. By default None.
    seed : Optional[int], optional
        Random seed for reproducibility. By default None.

    Returns
    -------
    Impact
        A new CLIMADA Impact object representing the yearset.
    """
    if poisson:
        lam = np.sum(imp.frequency)
        LOGGER.info(
            "Correcting TC event frequencies."
        )
        lam = lam * 25 / 26
        yimp, samp_vec = yearsets.impact_yearset(
            imp,
            lam=lam,
            sampled_years=list(range(1, n_sim_years + 1)),
            correction_fac=False,
            seed=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        samp_vec = np.array(
            [np.array([x]) for x in rng.integers(len(imp.at_event), size=n_sim_years)]
        )
        yimp = yearsets.impact_yearset_from_sampling_vect(
            imp,
            sampled_years=list(range(1, n_sim_years + 1)),
            sampling_vect=samp_vec,
            correction_fac=False,
        )

    # TODO remove this once it's added to yearsets core
    yimp.event_name = [str(y) for y in range(1, n_sim_years + 1)]

    yimp.coord_exp = imp.coord_exp

    # TODO extend CLIMADA's yearsets class with this: it should generate this matrix automatically!
    yimp.imp_mat = sparse.csr_matrix(
        np.vstack([imp.imp_mat[samp_vec[i]].sum(0) for i in range(len(samp_vec))])
    )

    # TODO extend CLIMADA's yearsets (or possibly Impact) class with this too!
    if cap_exposure is not None:
        yimp = cap_impact(yimp, cap_exposure)

    return yimp


# Adapted from Zélie's code:
# https://github.com/CLIMADA-project/climada_papers/blob/main/202403_multi_hazard_risk_assessment/python_scripts/multi_risk.py
def combine_yearsets(
    impact_list: Union[List[Impact], Dict[str, Impact]],
    how: str = "sum",
    occur_together: bool = False,
    cap_exposure: Optional[Union[Exposures, float]] = None,
) -> Impact:
    """Combine multiple impact yearsets.

    Parameters
    ----------
    impact_list : Union[List[Impact], Dict[str, Impact]]
        A list or dictionary of CLIMADA Impact objects to combine.
    how : str, optional
        How to combine the impacts. Options are 'sum', 'max', or 'min'.
        By default 'sum'.
    occur_together : bool, optional
        If True, only keep impacts where events occurred in all yearsets.
        By default False.
    cap_exposure : Optional[Union[Exposures, float]], optional
        Exposure object or value to cap the final combined impacts at.
        By default None.

    Returns
    -------
    Impact
        A new CLIMADA Impact object with the combined impacts.

    Raises
    ------
    ValueError
        If `how` is not one of 'sum', 'max', or 'min'.
    """

    if isinstance(impact_list, dict):
        impact_list = list(impact_list.values())

    if how == "sum":
        f: Callable[[sparse.csr_matrix, sparse.csr_matrix], sparse.csr_matrix] = (
            lambda m1, m2: m1 + m2
        )
    elif how == "min":
        f = lambda m1, m2: m1.minimum(m2)
    elif how == "max":
        f = lambda m1, m2: m1.maximum(m2)
    else:
        raise ValueError(
            f"'{how}' is not a valid method. The implemented methods are sum, max or min"
        )

    imp_mat = reduce(f, [imp.imp_mat for imp in impact_list])

    if occur_together:
        mask_list = [
            np.abs(impact.imp_mat.A[imp_mat.nonzero()]) == 0 for impact in impact_list
        ]
        for mask in mask_list:
            imp_mat.data[mask] = 0
        imp_mat.eliminate_zeros()

    imp0 = impact_list[0]
    freq = np.ones(len(imp0.event_id)) / len(imp0.event_id)
    eai_exp = ImpactCalc.eai_exp_from_mat(imp_mat, freq)
    at_event = ImpactCalc.at_event_from_mat(imp_mat)
    aai_agg = ImpactCalc.aai_agg_from_eai_exp(eai_exp)

    imp_combined = Impact(
        event_id=imp0.event_id,
        event_name=imp0.event_name,
        date=imp0.date,
        frequency=freq,
        frequency_unit=imp0.frequency_unit,
        coord_exp=imp0.coord_exp,
        crs=imp0.crs,
        eai_exp=eai_exp,
        at_event=at_event,
        tot_value=imp0.tot_value,
        aai_agg=aai_agg,
        unit=imp0.unit,
        imp_mat=imp_mat,
        haz_type="COMBINED",
    )
    if cap_exposure is not None:
        imp_combined = cap_impact(imp_combined, cap_exposure)
    return imp_combined


# TODO add this to CLIMADA in either Impact or Yearsets
# TODO then get yearsets.impact_yearset to use it!
def cap_impact(imp: Impact, cap_exposure: Union[Exposures, float]) -> Impact:
    """Cap impacts at the value of the exposure.

    Parameters
    ----------
    imp : Impact
        The CLIMADA Impact object to be capped.
    cap_exposure : Union[Exposures, float]
        An Exposures object or a single float value to use for capping.

    Returns
    -------
    Impact
        The capped CLIMADA Impact object.
    """
    imp_mat = imp.imp_mat
    shape = imp_mat.shape
    m1 = imp_mat.data
    if isinstance(cap_exposure, Exposures):
        m2 = cap_exposure.gdf.reset_index().value[imp_mat.nonzero()[1]]
    else:
        m2 = cap_exposure

    imp_mat = sparse.csr_matrix(
        (np.minimum(m1, m2), imp_mat.indices, imp_mat.indptr), shape=shape
    )
    imp_mat.eliminate_zeros()

    imp.imp_mat = imp_mat
    imp.at_event = ImpactCalc.at_event_from_mat(imp_mat)
    imp.eai_exp = ImpactCalc.eai_exp_from_mat(imp_mat, freq=imp.frequency)
    imp.aai_agg = ImpactCalc.aai_agg_from_eai_exp(imp.eai_exp)
    return imp
