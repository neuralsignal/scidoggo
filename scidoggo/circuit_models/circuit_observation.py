"""
Observation / likelihood models for the circuit dynamics models.

This module holds the observation-model component used to compare circuit
predictions (and optional latents) to data. The core circuit dynamics live in
:mod:`scidoggo.circuit_models.circuit`.
"""

from pyro.nn import PyroModule

from .gaussian import GaussianObsModel


class ObservationModel(PyroModule):
    """
    Classic Paccman observation model
    """

    def __init__(self, obs_scale=1, latent_scale=1, normalize=None, latent=False):
        super().__init__()
        self.obs_scale = obs_scale
        self.latent_scale = latent_scale
        self.latent = latent
        self.normalize = normalize

        self.obs_model = GaussianObsModel(self.obs_scale, normalize=self.normalize)
        if self.latent:
            self.latent_model = GaussianObsModel(self.latent_scale, normalize=None)

    def forward(self, xobs, xpred, xlatent=None, labels=None):
        self.obs_model(xobs, xpred, labels=labels)
        if self.latent and xlatent is not None:
            self.latent_model(xpred, xlatent)
