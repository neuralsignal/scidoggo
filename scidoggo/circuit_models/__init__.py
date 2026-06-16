"""Anatomically constrained circuit models (the ``[circuits]`` extra).

This subpackage provides Pyro-based probabilistic models for building
anatomically constrained circuit models, chromatic encoding models, and the
associated inference utilities.

The heavy dependencies (Pyro, torch, pandas, tqdm) ship as the optional
``[circuits]`` extra. Importing this subpackage without them installed raises
:class:`scidoggo.exceptions.MissingDependencyError` with an install hint rather
than an opaque :class:`ImportError`.

The network-generation helpers additionally require networkx (the ``[graph]``
extra). Their import is guarded separately so that a missing networkx does not
prevent the rest of the circuit models from importing.
"""

from ..exceptions import MissingDependencyError

try:
    from .chromatic import (
        ChromaticEncodingModel,
        MeasurementConversion,
        NoiseThresholdedPrAdaptation,
        PhotoreceptorAdaptation,
        SpectralSensitivityModel,
        Stavenga1993SensitivityModel,
        calc_capture,
        calc_log_excitation,
    )
    from .chromatic_priors import (
        FixedSensitivity,
        FlyStavenga1993InnerSensitivityModel,
        FlyStavenga1993SensitivityModel,
        InterpolateSumMeasurement,
        StrongNormalSensitivityPrior,
        WeakNormalSensitivityPrior,
    )
    from .circuit import (
        CircuitModel,
        ConductanceBasedIntegration,
        ObservedInputDynamicCircuitModel,
        StandardDynamicCircuitModel,
        UnobservedInputDynamicCircuitModel,
    )
    from .circuit_observation import ObservationModel
    from .deep import ANN, MonotonicNonlinearity
    from .gaussian import (
        FixedGaussianModel,
        FixedPriorModel,
        FixedScalePriorModel,
        FixedValueModel,
        GaussianObsEncodingModel,
        GaussianObsModel,
        SampleGaussianModel,
        UninformativePriorModel,
    )
    from .inference import fit_model, predict_model, summary
    from .lnl import (
        DeepLnlModel,
        FuncLnlModel,
        GaussianObsDeepLnlModel,
        GaussianObsFuncLnlModel,
        GaussianObsLnlModel,
        LnlModel,
        SimpleLnlModel,
    )
    from .model_constructors import construct_nln_chromatic
    from .pyro_components import (
        DirichletAdaptationPrior,
        FlyAdaptationPrior,
        HillEquation,
        Scalar,
        Tanh,
        TanhLike,
    )
    from .weights import (
        FixedWeightFunc,
        FuncWeight,
        InnerCircuitWeight,
        MonotonicWeight,
        WeightFunc,
    )
except ImportError as exc:
    raise MissingDependencyError(
        "circuit_models requires the [circuits] extra. "
        'Install with: pip install "scidoggo[circuits]"'
    ) from exc


__all__ = [
    # pyro_components
    "Tanh",
    "TanhLike",
    "HillEquation",
    "Scalar",
    "DirichletAdaptationPrior",
    "FlyAdaptationPrior",
    # deep
    "ANN",
    "MonotonicNonlinearity",
    # weights
    "WeightFunc",
    "FixedWeightFunc",
    "FuncWeight",
    "MonotonicWeight",
    "InnerCircuitWeight",
    # gaussian
    "FixedPriorModel",
    "FixedScalePriorModel",
    "UninformativePriorModel",
    "FixedValueModel",
    "SampleGaussianModel",
    "FixedGaussianModel",
    "GaussianObsModel",
    "GaussianObsEncodingModel",
    # lnl
    "LnlModel",
    "DeepLnlModel",
    "FuncLnlModel",
    "SimpleLnlModel",
    "GaussianObsLnlModel",
    "GaussianObsFuncLnlModel",
    "GaussianObsDeepLnlModel",
    # chromatic
    "calc_capture",
    "calc_log_excitation",
    "PhotoreceptorAdaptation",
    "NoiseThresholdedPrAdaptation",
    "ChromaticEncodingModel",
    "Stavenga1993SensitivityModel",
    "SpectralSensitivityModel",
    "MeasurementConversion",
    # chromatic_priors
    "FlyStavenga1993SensitivityModel",
    "FlyStavenga1993InnerSensitivityModel",
    "FixedSensitivity",
    "StrongNormalSensitivityPrior",
    "WeakNormalSensitivityPrior",
    "InterpolateSumMeasurement",
    # circuit
    "ConductanceBasedIntegration",
    "CircuitModel",
    "StandardDynamicCircuitModel",
    "ObservedInputDynamicCircuitModel",
    "UnobservedInputDynamicCircuitModel",
    "ObservationModel",
    # inference
    "fit_model",
    "summary",
    "predict_model",
    # model_constructors
    "construct_nln_chromatic",
    # network_generator (requires the [graph] extra; lazily loaded)
    "circuit_generator",
    "stack_ws",
    "create_nx_plot",
]


def __getattr__(name):
    """Lazily expose the networkx-dependent network-generation helpers.

    These live in :mod:`scidoggo.circuit_models.network_generator`, which
    imports networkx (the ``[graph]`` extra) at module load. Loading them
    lazily keeps a missing networkx from breaking the rest of the circuits
    import; the install hint surfaces only when one of these symbols is
    actually accessed.

    Parameters
    ----------
    name : str
        Attribute being accessed on the package.

    Returns
    -------
    object
        The requested symbol from :mod:`network_generator`.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the network-generation helpers.
    MissingDependencyError
        If the helper is requested but networkx is not installed.
    """
    if name in ("circuit_generator", "stack_ws", "create_nx_plot"):
        try:
            from . import network_generator
        except ImportError as exc:
            raise MissingDependencyError(
                "circuit_models.network_generator requires the [graph] extra. "
                'Install with: pip install "scidoggo[graph]"'
            ) from exc
        return getattr(network_generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
