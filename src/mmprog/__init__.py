"""mmprog: predicting investigator-defined gene programs from bone marrow morphology.

Pipeline stages (see ``mmprog.pipeline``):

    S0 configure      S1 ingest        S2 QC + targets     S2b reliability
    S3 features       S3b selection    S4 model            S5 cross-validation
    S6 clinical       S7 final fit + morphology-only inference   S8 reporting
"""
from .config import Config, load_config

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "__version__"]
