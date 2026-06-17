"""
Prior, constructor, and fixed-sensitivity helper classes for the chromatic
encoding models.

This module holds the concrete sensitivity priors (fly-specific Stavenga 1993
models with defined priors), the fixed / normal sensitivity prior models, and
the concrete measurement-conversion implementation. The abstract base classes
and the core encoding model live in :mod:`scidoggo.circuit_models.chromatic`.
"""

import pyro.distributions as dist
import torch
from pyro.nn import PyroSample

from .chromatic import (
    MeasurementConversion,
    SpectralSensitivityModel,
    Stavenga1993SensitivityModel,
)
from .interp1d import Interp1d
from .pyro_components import FLOAT_TYPE

interp1d = Interp1d()


class FlyStavenga1993SensitivityModel(Stavenga1993SensitivityModel):
    """
    Fly model with defined priors.
    """

    _alpha_max = dist.Uniform(
        torch.tensor([450, 300, 340, 450, 525], dtype=FLOAT_TYPE),
        torch.tensor([550, 340, 380, 500, 675], dtype=FLOAT_TYPE),
    )
    # _beta_max = dist.Uniform(
    #     torch.tensor([320, 345, 345, 345, 345], dtype=FLOAT_TYPE),
    #     torch.tensor([380, 355, 355, 355, 355], dtype=FLOAT_TYPE)
    # )
    _a_alpha = dist.Uniform(
        torch.tensor([140] * 5, dtype=FLOAT_TYPE), torch.tensor([700] * 5, dtype=FLOAT_TYPE)
    )
    _a_beta = dist.Uniform(
        torch.tensor([70] * 5, dtype=FLOAT_TYPE), torch.tensor([700] * 5, dtype=FLOAT_TYPE)
    )
    _A_beta = dist.Uniform(
        torch.tensor([1] + [0] * 2 + [0] * 2, dtype=FLOAT_TYPE),
        torch.tensor([4] + [0.01] * 2 + [0.6] * 2, dtype=FLOAT_TYPE),
    )

    def __init__(
        self,
        wls=torch.arange(300, 700, dtype=FLOAT_TYPE),
        alpha_max=None,
        # beta_max=None,
        a_alpha=None,
        a_beta=None,
        A_beta=None,
        **kwargs,
    ):
        if alpha_max is None:
            alpha_max = PyroSample(self._alpha_max)
        # if beta_max is None:
        #     beta_max = PyroSample(self._beta_max)
        if a_alpha is None:
            a_alpha = PyroSample(self._a_alpha)
        if a_beta is None:
            a_beta = PyroSample(self._a_beta)
        if A_beta is None:
            A_beta = PyroSample(self._A_beta)

        super().__init__(
            wls,
            alpha_max=alpha_max,
            # beta_max=beta_max,
            a_alpha=a_alpha,
            a_beta=a_beta,
            A_beta=A_beta,
            **kwargs,
        )


class FlyStavenga1993InnerSensitivityModel(FlyStavenga1993SensitivityModel):
    """
    Fly model with defined priors.
    """

    _alpha_max = dist.Uniform(
        torch.tensor([325, 350, 425, 525], dtype=FLOAT_TYPE),
        torch.tensor([340, 365, 450, 675], dtype=FLOAT_TYPE),
    )
    # _beta_max = torch.tensor([350, 350, 350, 350], dtype=FLOAT_TYPE)
    # dist.Uniform(
    #     torch.tensor([345, 345, 345, 345], dtype=FLOAT_TYPE),
    #     torch.tensor([355, 355, 355, 355], dtype=FLOAT_TYPE)
    # )
    # same alpha band width for all photoreceptors
    _a_alpha = dist.Uniform(
        torch.tensor(100, dtype=FLOAT_TYPE), torch.tensor(500, dtype=FLOAT_TYPE)
    )
    # same beta band width for all photoreceptors
    _a_beta = dist.Uniform(torch.tensor(200, dtype=FLOAT_TYPE), torch.tensor(300, dtype=FLOAT_TYPE))
    # same beta band contribution across photoreceptors
    _A_beta = dist.Uniform(
        # torch.tensor([0]*2+[0]*2, dtype=FLOAT_TYPE),
        # torch.tensor([0.01]*2+[0.4, 0.7], dtype=FLOAT_TYPE)
        torch.tensor(0, dtype=FLOAT_TYPE),
        torch.tensor(0.8, dtype=FLOAT_TYPE),
    )


class FixedSensitivity(SpectralSensitivityModel):
    def get_prior(self):
        pass

    def forward(self, *args):
        return self.mean


class StrongNormalSensitivityPrior(SpectralSensitivityModel):
    def get_prior(self):
        return PyroSample(
            dist.Normal(self.mean, self.sd / torch.sqrt(self.n)).to_event(self.mean.ndim)
        )

    def forward(self, *args):
        # clamp sensitivity at zero
        return torch.relu(self.prior)


class WeakNormalSensitivityPrior(SpectralSensitivityModel):
    def get_prior(self):
        return PyroSample(dist.Normal(self.mean, self.sd).to_event(self.mean.ndim))

    def forward(self, *args):
        # clamp sensitivity at zero
        return torch.relu(self.prior)


class InterpolateSumMeasurement(MeasurementConversion):
    def forward(self, outputs):
        assert outputs.ndim == 2
        n_stimuli, _ = outputs.shape
        # init spectrum
        spectrum = torch.zeros((n_stimuli, self.n_wls))
        # interpolate for each measurement
        for idx, name in enumerate(self.names):
            # set and reshape output
            output = outputs[..., idx].expand(self.n_wls, n_stimuli)
            # get output x array and measurement y array
            moutput = getattr(self, name + self.output_postfix)
            measurement = getattr(self, name)
            # (wavlengths x samples).T
            # clamp spectrum to zero
            # get new y array
            spectrum += torch.relu(interp1d(moutput, measurement, output)).T
        return spectrum
