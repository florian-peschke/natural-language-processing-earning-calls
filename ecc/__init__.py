import datetime
import logging
import typing as t

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

tqdm.pandas()
sns.set()
pd.set_option("use_inf_as_na", True)

log_wandb: bool = True
time: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
wandb_project: str = f"seminar ({time})" if log_wandb else "test"
ENTITY: str = "flo0128"
FOLDER_NAME_ML_TUNING: str = f"hyper-parameters/ml/{time}"
FOLDER_NAME_NLP_TUNING: str = f"hyper-parameters/nlp/{time}"

SEED: int = 128


def set_seed(seed: t.Optional[int] = None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(seed=SEED)
