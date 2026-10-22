"""afo — loader, eksport COCO i benchmark dla zbioru AFO (Aerial dataset of Floating Objects).

Zbior kanoniczny: https://huggingface.co/datasets/jgasienica/afo-dataset
Praca: Gasienica-Jozkowy, Knapik, Cyganek, ICAE 28(3) 2021, doi:10.3233/ICA-210649
"""

from .classes import CLASS_NAMES, DISPLAY_NAMES, NUM_CLASSES, display_name, load_class_names
from .dataset import AFODataset, Annotation, Sample
from .download import HF_REPO_ID, dataset_root, default_root, ensure_dataset

__version__ = "0.1.0"

__all__ = [
    "AFODataset",
    "Annotation",
    "Sample",
    "CLASS_NAMES",
    "DISPLAY_NAMES",
    "NUM_CLASSES",
    "display_name",
    "load_class_names",
    "HF_REPO_ID",
    "dataset_root",
    "default_root",
    "ensure_dataset",
    "__version__",
]
