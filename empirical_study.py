import logging
import os

import torch

from ecc import SEED
from ecc.study import TextClassification

if __name__ == "__main__":
    logging.info(f"Is Cuda available: {str(torch.cuda.is_available())}")
    logging.info(f"Seed for numpy, optuna and pytorch is: {SEED}")
    run: TextClassification = TextClassification(
        calc_topic_models=True,
        calc_embeddings_models=True,
        calc_bert_models=True,
        nlp_tuning_trials=10,
    )
    run.process(
        max_epochs_tuning=5,
        max_epochs_best_model=20,
        optuna_ml_trials=10,
        batch_size=4,
        pre_pre_process_raw_data=True,
        working_directory=os.getcwd(),
    )
