from .agnews import run_agnews
from .cola import run_cola
from . iris import run_iris
from .mrpc import run_mrpc
from .mscinli import run_mscinli
from .scierc import run_scierc
from .semeval import run_semeval
from .sst import run_sst
from .xnli import run_xnli
from .hcc import run_hcc

__all__ = [
    "run_agnews",
    "run_cola",
    "run_iris",
    "run_mrpc",
    "run_mscinli",
    "run_scierc",
    "run_semeval",
    "run_sst",
    "run_xnli",
    "run_hcc",
]