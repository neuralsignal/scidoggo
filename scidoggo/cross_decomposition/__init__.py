"""Cross decomposition algorithms (Partial Least Squares)."""

from .pls import CCA, PLSSVD, PLSCanonical, PLSRegression

__all__ = ["PLSRegression", "PLSCanonical", "CCA", "PLSSVD"]
