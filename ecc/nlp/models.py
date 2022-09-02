import os
import typing as t
from shutil import rmtree
from typing import Dict, List

import numpy as np
import torch
from gensim.corpora.dictionary import Dictionary
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from typeguard import typechecked

import ecc.utils as utils
from ecc.data.make_datasets import MakeDatasets
from ecc.nlp.tuning import ModelWithHyperParameter, NLPTuning
from ecc.utils import log


def process_gensim_return(document: t.List[str], model: t.Any) -> t.List[float]:
    data: t.List[t.Tuple] = model[document]
    distribution: t.List[float] = [0.0 for _ in range(model.num_topics)]
    id_: int
    value: float
    for id_, value in data:
        distribution[id_] = value
    return distribution


class WrapperForTuning:
    dataset: MakeDatasets

    def __init__(self, data: MakeDatasets) -> None:
        self.dataset = data

    def get_data_for_tuning(self, qa_type: str) -> t.Dict[str, np.ndarray]:
        """
        Prepare data for tuning.
        """
        container: t.Dict[str, np.ndarray] = {}

        container.update({"x_train_raw": self.dataset.x_raw_pre_processed[qa_type][self.dataset.NAME_TRAINING]})
        container.update({"x_val_raw": self.dataset.x_raw_pre_processed[qa_type][self.dataset.NAME_VALIDATION]})

        random_qa_label: str = self.dataset.random_number_generator.choice(a=self.dataset.MAIN_LABELS)
        container.update(
            {
                "y_train": self.dataset.encoded_labels_ordinal[qa_type][self.dataset.LABEL_MULTICLASS][random_qa_label][
                    self.dataset.NAME_TRAINING
                ]
            }
        )
        container.update(
            {
                "y_val": self.dataset.encoded_labels_ordinal[qa_type][self.dataset.LABEL_MULTICLASS][random_qa_label][
                    self.dataset.NAME_VALIDATION
                ]
            }
        )

        return container


class TopicModels:
    TOPIC_MODELLING_LABEL: str = "topic"
    TOPIC_MODELLING_SKLEARN: str = "scikit-learn"
    TOPIC_MODELLING_GENSIM: str = "gensim"

    TOPIC_MODEL_TRAINING_MODEL: Dict[str, dict] = {}
    PIPELINE_TOPIC_MODEL_NAMES_SKLEARN: List[str] = [
        "TFIDF",
        "BOW",
        "LDA",
        "LSA",
        "RP",
        "NMF",
    ]

    PIPELINE_TOPIC_MODEL_NAMES_GENSIM: List[str] = ["LDA", "LSI", "RP"]
    TOPIC_MODEL_HYPER_PARAMETER: Dict[str, dict] = {}

    _tune_n_trials: int
    _nlp_tuning: NLPTuning
    _data: WrapperForTuning

    x: t.Dict = {}

    @typechecked
    def __init__(self, data: WrapperForTuning, tune_n_trials: t.Optional[int] = 5) -> None:

        self._tune_n_trials = tune_n_trials
        self._data = data
        self.run_topic_models()

    @log
    def run_topic_models(self):
        """
        Apply various topic models:
        - Latent Dirichlet Allocation (LDA)
        - Latent Semantic Analysis (LSA)
        - Non-Negative Matrix Factorization (NMF)
        - Random projections (RP)
        - TFIDF|BOW (actually frequency models, but required by topic models).
        """
        for model_name in tqdm(self.PIPELINE_TOPIC_MODEL_NAMES_SKLEARN, desc="Topic modelling with sklearn"):
            self._sklearn_fit_transform(
                model_information=utils.NLPModelInfo(
                    model_name=model_name,
                    package_name=self.TOPIC_MODELLING_SKLEARN,
                    type=self.TOPIC_MODELLING_LABEL,
                )
            )
        for model_name in tqdm(self.PIPELINE_TOPIC_MODEL_NAMES_GENSIM, desc="Topic modelling with gensim"):
            self._gensim_train_and_infer(
                model_information=utils.NLPModelInfo(
                    model_name=model_name,
                    package_name=self.TOPIC_MODELLING_GENSIM,
                    type=self.TOPIC_MODELLING_LABEL,
                )
            )

    def _sklearn_fit_transform(self, model_information: utils.NLPModelInfo):
        """
        Wraps the general fitting and transformation process implemented in sklearn (scikit-learn).
        """
        tmp_dict: Dict = {}
        qa_type: str
        sub_dictionary: Dict
        self.TOPIC_MODEL_HYPER_PARAMETER.update({model_information.model_name: {}})

        for qa_type in self._data.dataset.QA_TYPES:
            model_with_hyper_parameter: ModelWithHyperParameter = self.tune(
                model_information=model_information, qa_type=qa_type
            )
            self.TOPIC_MODEL_HYPER_PARAMETER[model_information.model_name].update(
                {qa_type: model_with_hyper_parameter.hyperparameter}
            )
            tmp_dict.update({qa_type: {}})
            # Iterate across the partitions (training, validation and test)
            for partition, documents_by_type in self._data.dataset.x_raw_pre_processed[qa_type].items():
                # Transform the data of the specific data set ...
                data: torch.Tensor
                try:
                    data = torch.from_numpy(model_with_hyper_parameter.model.transform(documents_by_type).toarray())
                except AttributeError:
                    data = torch.from_numpy(model_with_hyper_parameter.model.transform(documents_by_type))
                # ... and store it inside the `store_in` dictionary.
                tmp_dict[qa_type].update({partition: data})
        # Update `x` with the newly created dictionary `tmp_dict`.
        self.x.update({f"{model_information.package_name}_{model_information.model_name}": tmp_dict})

    def _gensim_train_and_infer(self, model_information: utils.NLPModelInfo):
        """
        Wraps the general vocabulary building and fitting implemented in `tomotopy`.
        """
        tmp_dict: Dict = {}
        qa_type: str
        sub_dictionary: Dict
        self.TOPIC_MODEL_HYPER_PARAMETER.update({model_information.model_name: {}})
        for qa_type in self._data.dataset.QA_TYPES:
            hyper_parameters: Dict
            model_with_hyper_parameter: ModelWithHyperParameter = self.tune(
                model_information=model_information, qa_type=qa_type
            )
            self.TOPIC_MODEL_HYPER_PARAMETER[model_information.model_name].update(
                {qa_type: model_with_hyper_parameter.hyperparameter}
            )
            tmp_dict.update({qa_type: {}})
            data_by_type: dict = self._data.dataset.x_raw_pre_processed[qa_type]
            documents_for_training: np.ndarray = data_by_type[self._data.dataset.NAME_TRAINING]
            # Prepare the corpus and create it with the training data.
            tokens_for_training: List[str] = [doc.split() for doc in documents_for_training]
            dictionary_for_training: Dictionary = Dictionary(tokens_for_training)
            # Iterate across the partitions (training, validation and test).
            for partition, documents_by_type in self._data.dataset.x_raw_pre_processed[qa_type].items():
                # Prepare the corpus and create it with the partition data.
                tokens_partition: List[str] = [doc.split() for doc in documents_by_type]
                # noinspection PyTypeChecker
                corpus_partition: List = [dictionary_for_training.doc2bow(token) for token in tokens_partition]
                # Transform the data of the specific data set ...
                data_by_partition: np.ndarray = np.array(
                    [
                        process_gensim_return(document=doc, model=model_with_hyper_parameter.model)
                        for doc in corpus_partition
                    ]
                )
                data_by_partition: torch.Tensor = torch.from_numpy(data_by_partition)
                # ... and store it inside the `store_in` dictionary.
                tmp_dict[qa_type].update({partition: data_by_partition})
        # Update `x` with the newly created dictionary `tmp_dict`.
        self.x.update({f"{model_information.package_name}_{model_information.model_name}": tmp_dict})

    def tune(self, model_information: utils.NLPModelInfo, qa_type: str) -> ModelWithHyperParameter:
        tuning: NLPTuning = NLPTuning(
            **self._data.get_data_for_tuning(qa_type=qa_type),
            nlp_model_info=model_information,
            n_trials=self._tune_n_trials,
            just_get_back_default_model=self._tune_n_trials is None,
            description=qa_type,
        )
        return tuning.tune_hyper_parameters()


class EmbeddingsModels:
    EMBEDDING_MODEL_NAME_DOC2VEC: str = "Doc2Vec"
    EMBEDDING_MODEL_NAME_WORD2VEC: str = "Word2Vec"
    EMBEDDING_MODELLING_GENSIM_NAME: str = "gensim"
    EMBEDDING_MODEL_PIPELINE: t.List[Dict[str, t.Union[utils.NLPModelInfo, bool]]] = {}
    EMBEDDING_MODELS_HYPER_PARAMETER: Dict[str, dict] = {}
    _tune_n_trials: int
    _nlp_tuning: NLPTuning
    _data: WrapperForTuning

    x: t.Dict = {}

    @typechecked
    def __init__(
        self,
        data: WrapperForTuning,
        tune_n_trials: t.Optional[int] = 5,
    ) -> None:
        self._tune_n_trials = tune_n_trials
        self._data = data
        self._create_embedding_model_transformation_pipeline()
        self.run_embeddings_models()

    @log
    def run_embeddings_models(self):
        """
        Apply two embeddings models:
        - Word2Vec
        - Doc2Vec
        """
        for model_information in tqdm(self.EMBEDDING_MODEL_PIPELINE, desc="Embeddings modelling with gensim"):
            self.__gensim_train_and_infer(**model_information)

    def _create_embedding_model_transformation_pipeline(self):
        """
        Create the pipeline, i.e., define per nlp_model the transformer, the data to be transformed and where to
        store the transformed values in.
        """
        # transformer: dict, transform_data_type: dict, store_in
        self.EMBEDDING_MODEL_PIPELINE = [
            {
                "is_word_type_model": False,
                "model_information": utils.NLPModelInfo(
                    model_name=self.EMBEDDING_MODEL_NAME_DOC2VEC,
                    package_name=self.EMBEDDING_MODELLING_GENSIM_NAME,
                    type="embeddings",
                ),
            },
            {
                "is_word_type_model": True,
                "model_information": utils.NLPModelInfo(
                    model_name=self.EMBEDDING_MODEL_NAME_WORD2VEC,
                    package_name=self.EMBEDDING_MODELLING_GENSIM_NAME,
                    type="embeddings",
                ),
            },
        ]

    def __gensim_train_and_infer(self, is_word_type_model: bool, model_information: utils.NLPModelInfo):
        """
        Wraps the general vocabulary building and fitting implemented in `gensim`.
        """
        container: Dict = {}
        qa_type: str
        sub_dictionary: Dict
        self.EMBEDDING_MODELS_HYPER_PARAMETER.update({model_information.model_name: {}})
        for qa_type in self._data.dataset.QA_TYPES:
            tuning: NLPTuning = NLPTuning(
                **self._data.get_data_for_tuning(qa_type=qa_type),
                nlp_model_info=model_information,
                n_trials=self._tune_n_trials,
                is_word_type_model=is_word_type_model,
                just_get_back_default_model=self._tune_n_trials is None,
                description=qa_type,
            )
            model_with_hyper_parameter: ModelWithHyperParameter = tuning.tune_hyper_parameters()
            # Add an empty dictionary, that stores the transformed data of the training, validation and test
            self.EMBEDDING_MODELS_HYPER_PARAMETER[model_information.model_name].update(
                {qa_type: model_with_hyper_parameter.hyperparameter}
            )
            container.update({qa_type: {}})
            data_by_type: dict = self._data.dataset.x_raw_pre_processed[qa_type]
            if self._tune_n_trials is None:
                documents_for_training: np.ndarray = data_by_type[self._data.dataset.NAME_TRAINING]
                documents_with_tokens: t.List[t.List[str]] = [document.split() for document in documents_for_training]
                if is_word_type_model:
                    model_with_hyper_parameter.model.build_vocab(corpus_iterable=documents_with_tokens)
                    model_with_hyper_parameter.model.train(
                        documents_with_tokens,
                        total_examples=model_with_hyper_parameter.model.corpus_count,
                        epochs=model_with_hyper_parameter.model.epochs,
                    )
                else:
                    tagged_documents: t.List[TaggedDocument] = list(utils.tag_doc(documents=documents_for_training))
                    model_with_hyper_parameter.model.build_vocab(
                        corpus_iterable=tagged_documents,
                    )
                    model_with_hyper_parameter.model.train(
                        tagged_documents,
                        total_examples=model_with_hyper_parameter.model.corpus_count,
                        epochs=model_with_hyper_parameter.model.epochs,
                    )
            # Iterate across the partitions (training, validation and test).
            for partition, documents_by_type in data_by_type.items():
                # Transform the data of the specific data set ...
                data_by_partition: torch.Tensor
                if is_word_type_model:
                    data_by_partition = torch.from_numpy(
                        np.array(
                            [
                                utils.get_gensim_embedding(
                                    document=document.split(),
                                    gensim_model=model_with_hyper_parameter.model,
                                ).flatten()
                                for document in documents_by_type
                            ]
                        )
                    )
                else:
                    data_by_partition = torch.from_numpy(
                        np.array(
                            [
                                model_with_hyper_parameter.model.infer_vector(doc_words=document.split()).flatten()
                                for document in documents_by_type
                            ]
                        )
                    )
                # ... and store it inside the `store_in` dictionary.
                container[qa_type].update({partition: data_by_partition})
        # Update `x_nlp_processed` with the newly created dictionary `container`.
        self.x.update({model_information.model_name: container})


class Transformer:
    TRANSFORMER_MODEL_TRAINING_MODEL: Dict[str, t.Dict] = {}
    PIPELINE_TRANSFORMER_PRE_TRAINED_TOKENIZER: t.List[str] = [
        "bert-base-uncased",
        "ProsusAI/finbert",
    ]
    TRANSFORMER_HYPER_PARAMETER: Dict[str, t.Dict] = {}

    should_tune: bool
    tune_n_trials: int
    nlp_tuning: NLPTuning
    x: t.Dict = {}

    data: WrapperForTuning

    @typechecked
    def __init__(self, data: WrapperForTuning) -> None:
        self.data = data
        self.transform()

    @log
    def transform(self):
        """
        Tokenize the corpora according to the given pretrained transformer.
        """
        for model_name in tqdm(
            self.PIPELINE_TRANSFORMER_PRE_TRAINED_TOKENIZER,
            desc="Transformer modelling with huggingface",
        ):
            self.x.update({model_name: {}})
            for qa_type in self.data.dataset.QA_TYPES:
                self.x[model_name].update({qa_type: {}})
                split: np.ndarray
                for split_name, split in self.data.dataset.x_raw_pre_processed[qa_type].items():
                    self.x[model_name][qa_type].update(
                        {split_name: self.get_pooler_outputs(data=split, model_name=model_name)}
                    )

    @staticmethod
    def get_pooler_outputs(data: np.ndarray, model_name: str) -> torch.Tensor:
        temp_folder: str = utils.init_folder(os.path.join(os.getcwd(), "tmp"))
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        bert: BertModel = BertModel.from_pretrained(model_name)
        try:
            # concatenating the tensors directly somehow kills the python kernel after some time. Thus, saving the
            # tensors in a temporal folder first and then load them back to concatenate fixes the problem.
            for i, text in tqdm(enumerate(data), desc="Getting pooler output"):
                torch.save(
                    bert(
                        **tokenizer(
                            text,
                            max_length=512,
                            truncation=True,
                            return_tensors="pt",
                        )
                    ).pooler_output.detach(),
                    os.path.join(temp_folder, f"{i}_tensor.pt"),
                )
            return utils.load_and_concat_tensors(temp_folder)
        finally:
            rmtree(temp_folder, ignore_errors=True)
