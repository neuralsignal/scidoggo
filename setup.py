"""Build shim for the optional Pythran-accelerated RBF kernels.

All package metadata lives in ``pyproject.toml``. This file exists solely to
register the optional Pythran C++ extension. When Pythran is not installed (or
the extension source has not yet been relocated), the build silently falls back
to the pure-Python implementation in ``scidoggo.interpolation.rbf``.
"""
import os

from setuptools import setup

PYTHRAN_SOURCE = os.path.join("scidoggo", "interpolation", "_rbf_kernels_pythran.py")

setup_args = {}
if os.path.exists(PYTHRAN_SOURCE):
    try:
        from pythran.dist import PythranBuildExt, PythranExtension

        setup_args = {
            "cmdclass": {"build_ext": PythranBuildExt},
            "ext_modules": [
                PythranExtension(
                    "scidoggo.interpolation._rbf_kernels_pythran",
                    [PYTHRAN_SOURCE],
                ),
            ],
        }
    except ImportError:
        print(
            "not building Pythran extension - install pythran for more efficient code"
        )

setup(**setup_args)
