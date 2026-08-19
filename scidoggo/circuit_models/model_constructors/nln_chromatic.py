"""
Nln chromatic encoding model
"""

import pyro.distributions as dist

from ..chromatic import (
    ChromaticEncodingModel,
    NoiseThresholdedPrAdaptation,
)
from ..chromatic_priors import (
    FlyStavenga1993InnerSensitivityModel,
    FlyStavenga1993SensitivityModel,
)
from ..gaussian import GaussianObsEncodingModel
from ..lnl import LnlModel
from ..pyro_components import identity
from ..weights import WeightFunc


def construct_nln_chromatic(
    wls, connectivity, nonlin=identity, weight_prior=dist.Normal(0, 1), data_normalize="l1-norm"
):
    """
    connectivity : (pre x post)
    """
    weight_func = WeightFunc(connectivity, prior=weight_prior)
    lnl_model = LnlModel(weight_func, nonlin=nonlin)
    e_model = GaussianObsEncodingModel(lnl_model, normalize=data_normalize)

    p_func = NoiseThresholdedPrAdaptation()

    if connectivity.shape[0] == 4:
        s_func = FlyStavenga1993InnerSensitivityModel(wls)
    else:
        s_func = FlyStavenga1993SensitivityModel(wls)

    model = ChromaticEncodingModel(wls, s_func, e_model, p_func=p_func)
    return model
