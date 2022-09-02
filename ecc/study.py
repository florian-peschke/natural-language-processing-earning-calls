import os
import re
import shutil
import typing as t

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from attr import define, field
from matplotlib import pyplot as plt
from scipy.constants import golden_ratio
from typeguard import typechecked

from ecc import FOLDER_NAME_ML_TUNING, FOLDER_NAME_NLP_TUNING
from ecc.data.make_datasets import MakeDatasets
from ecc.nlp.models import EmbeddingsModels, TopicModels, Transformer, WrapperForTuning
from ecc.torch.lightning import NNRoutine
from ecc.utils import create_dir_if_missing, frame_to_latex, to_pickle


def gen_frames(path: str, find: str = "evaluation.csv") -> t.Generator[pd.DataFrame, None, None]:
    for root, dirs, files in os.walk(path):
        for file in files:
            if bool(re.search(find, file)):
                yield pd.read_csv(os.path.join(root, file))


NLP_DATA_PATH: str = os.path.join(os.getcwd(), "nlp_data")
create_dir_if_missing(path=NLP_DATA_PATH)


def zip_folders() -> None:
    shutil.make_archive(os.path.join(FOLDER_NAME_ML_TUNING, "ml"), "zip", FOLDER_NAME_ML_TUNING)
    shutil.make_archive(os.path.join(FOLDER_NAME_ML_TUNING, "nlp"), "zip", FOLDER_NAME_NLP_TUNING)


@define
class TextClassification:
    calc_bert_models: bool
    calc_topic_models: bool
    calc_embeddings_models: bool
    nlp_tuning_trials: t.Optional[int]
    splitting_proportion: str

    _topic_models_data: t.Dict[str, t.Dict[str, t.Dict[str, torch.Tensor]]] = field(init=False)
    _embeddings_models_data: t.Dict[str, t.Dict[str, t.Dict[str, torch.Tensor]]] = field(init=False)
    _transformer_modelling_data: t.Dict[str, t.Dict[str, t.Dict[str, torch.Tensor]]] = field(init=False)

    _max_epochs_tuning: int = field(init=False)
    _max_epochs_best_model: int = field(init=False)
    _optuna_ml_trials: int = field(init=False)
    _batch_size: int = field(init=False)
    _shuffle: bool = field(init=False)

    @typechecked
    def __init__(
        self,
        calc_bert_models: bool,
        calc_topic_models: bool,
        calc_embeddings_models: bool,
        nlp_tuning_trials: t.Optional[int] = None,
        splitting_proportion="70:15:15",
    ) -> None:
        self.__attrs_init__(
            calc_bert_models=calc_bert_models,
            calc_topic_models=calc_topic_models,
            calc_embeddings_models=calc_embeddings_models,
            nlp_tuning_trials=nlp_tuning_trials,
            splitting_proportion=splitting_proportion,
        )

    def process(
        self,
        max_epochs_tuning: int,
        max_epochs_best_model: int,
        optuna_ml_trials: int,
        batch_size: int,
        working_directory: str,
        shuffle: bool = False,
        pre_pre_process_raw_data: bool = False,
        label_studio_api: t.Optional[str] = None,
    ) -> None:

        self._max_epochs_tuning = max_epochs_tuning
        self._max_epochs_best_model = max_epochs_best_model
        self._optuna_ml_trials = optuna_ml_trials
        self._batch_size = batch_size
        self._shuffle = shuffle

        dataset: MakeDatasets = MakeDatasets(
            splitting_proportion=self.splitting_proportion,
            process_raw_data=pre_pre_process_raw_data,
            label_studio_api=label_studio_api,
            working_directory=working_directory,
        )
        data_for_tuning: WrapperForTuning = WrapperForTuning(dataset)

        if self.calc_embeddings_models:
            self._run_classification(
                inputs=self._init_nlp_data(
                    pickle_filename="embeddings_models_data.pkl",
                    nlp_modelling=EmbeddingsModels(tune_n_trials=self.nlp_tuning_trials, data=data_for_tuning),
                ),
                outputs=dataset.encoded_labels_binary,
            )
        if self.calc_bert_models:
            self._run_classification(
                inputs=self._init_nlp_data(
                    pickle_filename="transformer_modelling_data.pkl", nlp_modelling=Transformer(data=data_for_tuning)
                ),
                outputs=dataset.encoded_labels_binary,
            )
        if self.calc_topic_models:
            self._run_classification(
                inputs=self._init_nlp_data(
                    pickle_filename="topic_models_data.pkl",
                    nlp_modelling=TopicModels(tune_n_trials=self.nlp_tuning_trials, data=data_for_tuning),
                ),
                outputs=dataset.encoded_labels_binary,
            )
        self._post()

    def _run_classification(
        self,
        inputs: t.Dict[str, t.Dict[str, t.Dict[str, torch.Tensor]]],
        outputs: t.Dict[str, t.Dict[str, t.Union[t.Dict[str, np.ndarray], t.Dict[str, t.Dict[str, np.ndarray]]]]],
    ):
        for model_name, model_data in inputs.items():
            for qa_type, qa_type_data in model_data.items():
                nn_routine: NNRoutine = NNRoutine(
                    inputs=qa_type_data,
                    outputs=outputs[qa_type]["multiclass"],
                    shuffle=self._shuffle,
                    batch_size=self._batch_size,
                    name=model_name,
                    name_postfix=qa_type.lower(),
                    path=create_dir_if_missing(
                        os.path.join(os.getcwd(), FOLDER_NAME_ML_TUNING, qa_type, model_name.replace("/", "-"))
                    ),
                )
                nn_routine.tune(
                    max_epochs_tuning=self._max_epochs_tuning,
                    n_trials=self._optuna_ml_trials,
                    max_epochs_evaluation=self._max_epochs_best_model,
                )

    def _init_nlp_data(
        self, pickle_filename: str, nlp_modelling: t.Union[Transformer, EmbeddingsModels, TopicModels]
    ) -> t.Dict[str, t.Dict[str, t.Dict[str, torch.Tensor]]]:
        nlp_modelling_data: t.Dict[str, t.Dict[str, t.Dict[str, torch.Tensor]]] = nlp_modelling.x
        to_pickle(
            object_=nlp_modelling_data,
            filename=os.path.join(NLP_DATA_PATH, pickle_filename),
        )
        return nlp_modelling_data

    def _post(self) -> None:
        data_frames: t.Dict[str, pd.DataFrame] = {}
        for qa_type in ["Question", "Answer"]:
            results: pd.DataFrame = pd.concat(
                list(gen_frames(path=os.path.join(FOLDER_NAME_ML_TUNING, qa_type)))
            ).reset_index(drop=True)
            results.to_csv(os.path.join(FOLDER_NAME_ML_TUNING, f"summary_{qa_type.lower()}.csv"), index=False)
            frame_to_latex(frame=results, path=FOLDER_NAME_ML_TUNING, name=f"summary_{qa_type.lower()}")
            data_frames.update(
                {
                    qa_type.lower(): pd.melt(
                        results,
                        id_vars="model",
                        value_name="accuracy",
                        value_vars=results.columns,
                    )
                }
            )
            print(f"\n\n{qa_type}:\n{results}")
        self._plot(data_frames)
        zip_folders()

    @staticmethod
    def _plot(frames: t.Dict[str, pd.DataFrame]) -> None:
        qa_type: str
        data: pd.DataFrame
        for qa_type, data in frames.items():
            grid = sns.FacetGrid(data, col="variable", col_wrap=4, height=5, aspect=1 / golden_ratio)
            grid.map(sns.barplot, "accuracy", "model", order=np.sort(data["model"].unique()))
            grid.set_axis_labels(x_var="", y_var="")
            plt.tight_layout()
            plt.savefig(os.path.join(FOLDER_NAME_ML_TUNING, f"{qa_type.lower()}.pdf"))
