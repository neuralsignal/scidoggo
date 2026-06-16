"""Cross decomposition algorithms (Partial Least Squares)."""

from .pls import CCA, PLSCanonical, PLSRegression, PLSSVD

__all__ = ["PLSRegression", "PLSCanonical", "CCA", "PLSSVD"]
