import os
import typing as t
import warnings

import gensim
import gensim.models as gm
import numpy as np
import optuna
import wandb
from gensim.corpora.dictionary import Dictionary
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from optuna.integration import WeightsAndBiasesCallback
from optuna.samplers import TPESampler

# noinspection PyPep8Naming
from sklearn.decomposition import (
    LatentDirichletAllocation as LDA,
    NMF,
    TruncatedSVD as LSA,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from transformers import BertTokenizer, TFAutoModelForSequenceClassification

import ecc.utils as utils
from ecc.utils import ModelInfo, NLPModelInfo, TransformerPipeline
from .. import ENTITY, FOLDER_NAME_NLP_TUNING, log_wandb, SEED, wandb_project


class NLPTemplate:
    # term frequency-inverse document frequency
    TOPIC_MODEL_NAME_TFIDF: NLPModelInfo = NLPModelInfo(
        type="frequency", model_name="TFIDF", package_name="scikit-learn"
    )
    # Bag-of-words
    TOPIC_MODEL_NAME_BOW: NLPModelInfo = NLPModelInfo(type="frequency", model_name="BOW", package_name="scikit-learn")
    # Latent dirichlet allocation
    TOPIC_MODEL_NAME_LDA_SKLEARN: NLPModelInfo = NLPModelInfo(
        type="topic", model_name="LDA", package_name="scikit-learn"
    )
    # Latent semantic analysis
    TOPIC_MODEL_NAME_LSA_SKLEARN: NLPModelInfo = NLPModelInfo(
        type="topic", model_name="LSA", package_name="scikit-learn"
    )
    # Non negative matrix factorization
    TOPIC_MODEL_NAME_NMF_SKLEARN: NLPModelInfo = NLPModelInfo(
        type="topic", model_name="NMF", package_name="scikit-learn"
    )
    # Random projection
    TOPIC_MODEL_NAME_RP_SKLEARN: NLPModelInfo = NLPModelInfo(type="topic", model_name="RP", package_name="scikit-learn")

    # Latent dirichlet allocation
    TOPIC_MODEL_NAME_LDA_GENSIM: NLPModelInfo = NLPModelInfo(type="topic", model_name="LDA", package_name="gensim")
    # Latent semantic indexing
    TOPIC_MODEL_NAME_LSI_GENSIM: NLPModelInfo = NLPModelInfo(type="topic", model_name="LSI", package_name="gensim")
    # Random projections
    TOPIC_MODEL_NAME_RP_GENSIM: NLPModelInfo = NLPModelInfo(type="topic", model_name="RP", package_name="gensim")

    # BERT
    TRANSFORMERS_MODEL_NAME_BERT: NLPModelInfo = NLPModelInfo(
        type="transformer", model_name="BERT", package_name="huggingface"
    )

    # Document to Vector
    EMBEDDING_MODEL_NAME_DOC2VEC: NLPModelInfo = NLPModelInfo(
        type="embeddings", model_name="Doc2Vec", package_name="gensim"
    )
    # Word to Vector
    EMBEDDING_MODEL_NAME_WORD2VEC: NLPModelInfo = NLPModelInfo(
        type="embeddings", model_name="Word2Vec", package_name="gensim"
    )

    HUGGINGFACE_LABEL: str = "huggingface"
    GENSIM_LABEL: str = "gensim"
    SKLEARN_LABEL: str = "scikit-learn"

    MAX_N_GRAMS: int
    N_GRAMS: t.List[str]
    BOOLEANS: t.List[bool] = [True, False]
    NAME_N_GRAMS: str = "ngram_range"

    NAME_TOPIC_MODELLING: str = "topic"
    NAME_FREQUENCY_MODELLING: str = "frequency"
    NAME_EMBEDDINGS_MODELLING: str = "embeddings"
    NAME_TRANSFORMER_MODELLING: str = "transformer"

    y_train: np.ndarray
    y_val: np.ndarray
    x_train_raw: np.ndarray
    x_val_raw: np.ndarray

    x_train: np.ndarray
    x_val: np.ndarray

    x_train_transformer: t.List[t.Dict[str, list]]
    x_val_transformer: t.List[t.Dict[str, list]]

    is_freq_or_topic_model: bool = False
    is_embeddings_model: bool = False
    is_transformer_model: bool = False

    nlp_model_info: NLPModelInfo

    just_get_back_default_model: bool
    n_trials: int

    sklearn_classifier: t.Any

    is_word_type_model: bool
    study: optuna.Study
    path: str
    description: t.Optional[str]


class ModelWithHyperParameter(t.NamedTuple):
    hyperparameter: t.Dict
    model: t.Any


class NLPTuning(NLPTemplate):
    def __init__(
        self,
        nlp_model_info: NLPModelInfo,
        y_train: np.ndarray = None,
        y_val: np.ndarray = None,
        x_train_raw: np.ndarray = None,
        x_val_raw: np.ndarray = None,
        just_get_back_default_model: bool = False,
        is_word_type_model: bool = False,
        sklearn_classifier: t.Callable[..., t.Any] = BernoulliNB,
        n_trials: int = 50,
        max_n_grams: int = 5,
        description: t.Optional[str] = None,
    ) -> None:

        super().__init__()

        self.nlp_model_info = nlp_model_info

        self.x_train_raw = x_train_raw
        self.x_val_raw = x_val_raw
        self.y_train = y_train
        self.y_val = y_val

        self.n_trials = n_trials
        self.is_word_type_model = is_word_type_model

        self.just_get_back_default_model = just_get_back_default_model

        self.sklearn_classifier = sklearn_classifier
        self.MAX_N_GRAMS = max_n_grams

        self.description = description

        self.path = utils.create_dir_if_missing(
            path=os.path.join(
                os.getcwd(),
                FOLDER_NAME_NLP_TUNING,
                nlp_model_info.package_name,
                nlp_model_info.model_name,
                self.description,
            )
        )

        self.N_GRAMS = [
            f"{first} {second}"
            for first in np.arange(1, self.MAX_N_GRAMS + 1)
            for second in np.arange(first, self.MAX_N_GRAMS + 1)
        ]

    def _get_wandb_callback(self) -> WeightsAndBiasesCallback:
        return WeightsAndBiasesCallback(
            metric_name="accuracy",
            wandb_kwargs=dict(
                project=wandb_project,
                entity=ENTITY,
                name=f"nlp_tuning – {self.nlp_model_info.name} - {self.description}",
                settings=wandb.Settings(start_method="fork"),
            ),
        )

    def tune_hyper_parameters(self) -> ModelWithHyperParameter:
        self.study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            study_name=self.nlp_model_info.name,
            sampler=TPESampler(seed=SEED),
        )
        hyperparameter: t.Dict = {}
        model: t.Any = None

        if (self.nlp_model_info.type == self.NAME_FREQUENCY_MODELLING) | (
            self.nlp_model_info.type == self.NAME_TOPIC_MODELLING
        ):
            hyperparameter, model = self._tune_freq_topic_models(study=self.study)
        elif self.nlp_model_info.type == self.NAME_EMBEDDINGS_MODELLING:
            hyperparameter, model = self._tune_embeddings_models(study=self.study)

        self._save_best_parameters()

        if log_wandb:
            wandb.finish()

        return ModelWithHyperParameter(hyperparameter=hyperparameter, model=model)

    def _tune_freq_topic_models(self, study: optuna.study) -> t.Tuple[dict, t.Any]:
        if self.nlp_model_info.package_name == self.GENSIM_LABEL:
            return self._tune_and_return_model(
                fit_transform=self._fit_transform_gensim_topic, objective=self._gensim_topic_objective, study=study
            )
        elif self.nlp_model_info.package_name == self.SKLEARN_LABEL:
            return self._tune_and_return_model(
                fit_transform=self._fit_transform_sklearn, objective=self._sklearn_objective, study=study
            )

    def _tune_embeddings_models(self, study: optuna.study) -> t.Tuple[dict, t.Any]:
        return self._tune_and_return_model(
            fit_transform=self._fit_transform_gensim_embeddings,
            objective=self._gensim_embeddings_objective,
            study=study,
        )

    def _tune_and_return_model(
        self,
        fit_transform: t.Callable,
        objective: t.Callable,
        study: optuna.study,
    ) -> t.Tuple[t.Dict, t.Any]:

        if self.just_get_back_default_model:
            return {}, fit_transform(True, **{})
        else:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                callbacks=[self._get_wandb_callback()] if log_wandb else [],
            )
            best_parameters: dict = study.best_params
            if self.NAME_N_GRAMS in best_parameters.keys():
                best_parameters[self.NAME_N_GRAMS] = utils.n_grams_workaround(best_parameters[self.NAME_N_GRAMS])
            return best_parameters, fit_transform(True, **best_parameters)

    def _gensim_topic_objective(self, trial):
        classes = np.unique(self.y_train)
        hyper_parameters: t.Dict[str, t.Any] = {}
        if self.nlp_model_info.model_name == "LDA":
            hyper_parameters.update(
                {
                    "num_topics": trial.suggest_int("num_topics", 4, 20, log=True),
                    "decay": trial.suggest_float("decay", 0.5, 0.99, log=True),
                    "minimum_phi_value": trial.suggest_float("minimum_probability", 0.005, 0.05, log=True),
                    "minimum_probability": trial.suggest_float("minimum_probability", 0.005, 0.05, log=True),
                    "gamma_threshold": trial.suggest_float("gamma_threshold", 0.0005, 0.002, log=True),
                }
            )
        elif self.nlp_model_info.model_name == "LSI":
            hyper_parameters.update(
                {
                    "num_topics": trial.suggest_int("num_topics", 4, 20, log=True),
                    "chunksize": trial.suggest_int("chunksize", 1, 40000, log=True),
                    "power_iters": trial.suggest_int("power_iters", 1, 100, log=True),
                    "extra_samples": trial.suggest_int("extra_samples", 1, 200, log=True),
                    "decay": trial.suggest_float("decay", 0.1, 1, log=True),
                }
            )
        elif self.nlp_model_info.model_name == "RP":
            hyper_parameters.update({"num_topics": trial.suggest_int("num_topics", 4, 20, log=True)})

        self._fit_transform_gensim_topic(**hyper_parameters)

        return self._optimize(classifier=self.sklearn_classifier(), trial=trial, classes=classes)

    def _gensim_embeddings_objective(self, trial):
        classes = np.unique(self.y_train)
        hyper_parameters: t.Dict[str, t.Any] = {}
        if self.nlp_model_info.model_name == "Doc2Vec":
            hyper_parameters.update(
                {
                    "vector_size": trial.suggest_int("vector_size", 15, 300, log=True),
                    "dm": trial.suggest_categorical("dm", [0, 1]),
                    "dbow_words": trial.suggest_categorical("dbow_words", [0, 1]),
                    "ns_exponent": trial.suggest_float("ns_exponent", 0.01, 1, log=True),
                    "epochs": trial.suggest_int("epochs", 10, 500, log=True),
                    "alpha": trial.suggest_float("alpha", 0.01, 0.99, log=True),
                    "window": trial.suggest_int("window", 4, 20, log=True),
                }
            )
        elif self.nlp_model_info.model_name == "Word2Vec":
            hyper_parameters.update(
                {
                    "vector_size": trial.suggest_int("vector_size", 15, 300, log=True),
                    "sg": trial.suggest_categorical("sg", [0, 1]),
                    "ns_exponent": trial.suggest_float("ns_exponent", 0.01, 1, log=True),
                    "epochs": trial.suggest_int("epochs", 10, 500, log=True),
                    "alpha": trial.suggest_float("alpha", 0.01, 0.99, log=True),
                    "window": trial.suggest_int("window", 4, 20, log=True),
                }
            )

        self._fit_transform_gensim_embeddings(**hyper_parameters)
        classifier = self.sklearn_classifier()

        return self._optimize(classifier=classifier, trial=trial, classes=classes)

    def _optimize(
        self,
        classifier: t.Any,
        trial: optuna.trial,
        classes: t.Union[t.List[t.Union[str, int]], np.ndarray],
    ) -> float:
        for step in range(100):
            classifier.partial_fit(self.x_train, self.y_train, classes=classes)
            # Report intermediate objective value.
            intermediate_value = 1.0 - classifier.score(self.x_val, self.y_val)
            trial.report(intermediate_value, step)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()
        return classifier.score(self.x_val, self.y_val)

    def _sklearn_objective(self, trial):
        classes = np.unique(self.y_train)
        parameters: t.Dict[str, t.Any] = {}
        if self.nlp_model_info.model_name == "TFIDF":
            parameters.update(
                {
                    self.NAME_N_GRAMS: trial.suggest_categorical(self.NAME_N_GRAMS, self.N_GRAMS),
                    "norm": trial.suggest_categorical("norm", ["l1", "l2"]),
                    "use_idf": trial.suggest_categorical("use_idf", self.BOOLEANS),
                    "smooth_idf": trial.suggest_categorical("smooth_idf", self.BOOLEANS),
                    "sublinear_tf": trial.suggest_categorical("sublinear_tf", self.BOOLEANS),
                }
            )
            parameters[self.NAME_N_GRAMS] = utils.n_grams_workaround(parameters[self.NAME_N_GRAMS])

        elif self.nlp_model_info.model_name == "BOW":
            parameters.update(
                {
                    self.NAME_N_GRAMS: trial.suggest_categorical(self.NAME_N_GRAMS, self.N_GRAMS),
                }
            )
            parameters[self.NAME_N_GRAMS] = utils.n_grams_workaround(parameters[self.NAME_N_GRAMS])

        elif self.nlp_model_info.model_name == "LDA":
            parameters.update(
                {
                    "n_components": trial.suggest_int("n_components", 4, 20, log=True),
                    "learning_decay": trial.suggest_float("learning_decay", 0.5, 0.99, log=True),
                    "max_iter": trial.suggest_int("max_iter", 10, 100, log=True),
                    "batch_size": trial.suggest_int("batch_size", 4, 256, log=True),
                }
            )

        elif self.nlp_model_info.model_name == "RP":
            parameters.update(
                {
                    "n_components": trial.suggest_int("n_components", 4, 20, log=True),
                    "eps": trial.suggest_float("eps", 0.01, 0.2, log=True),
                    "random_state": np.random.RandomState(SEED),
                }
            )

        elif self.nlp_model_info.model_name == "LSA":
            parameters.update(
                {
                    "n_components": trial.suggest_int("n_components", 4, 20, log=True),
                    "algorithm": trial.suggest_categorical("algorithm", ["arpack", "randomized"]),
                    "n_iter": trial.suggest_int("n_iter", 4, 100, log=True),
                }
            )

        elif self.nlp_model_info.model_name == "NMF":
            parameters.update(
                {
                    "n_components": trial.suggest_int("n_components", 4, 20, log=True),
                    "init": trial.suggest_categorical("init", [None, "random", "nndsvd", "nndsvda", "nndsvdar"]),
                    "solver": trial.suggest_categorical("solver", ["cd", "mu"]),
                    "max_iter": trial.suggest_int("max_iter", 4, 400, log=True),
                }
            )

        self._fit_transform_sklearn(**parameters)
        classifier = self.sklearn_classifier()

        return self._optimize(classifier=classifier, trial=trial, classes=classes)

    def _fit_transform_gensim_topic(self, return_model: bool = False, **hyperparameter: t.Any) -> t.Optional[t.Any]:
        # tokens
        tokens_for_training: t.List[t.List[str]] = [doc.split() for doc in self.x_train_raw]
        tokens_for_validation: t.List[t.List[str]] = [doc.split() for doc in self.x_val_raw]
        # dictionary
        dictionary_for_training: Dictionary = Dictionary(tokens_for_training)
        dictionary_for_validation: Dictionary = Dictionary(tokens_for_validation)
        # corpus
        corpus_for_training: t.List = [dictionary_for_training.doc2bow(token) for token in tokens_for_training]
        corpus_for_validation: t.List = [dictionary_for_validation.doc2bow(token) for token in tokens_for_validation]
        hyperparameter.update(dict(id2word=dictionary_for_training))
        # get model and transform text
        model = self._get_new_model(corpus=corpus_for_training, **hyperparameter).model
        self.x_train = np.asarray(list(utils.infer_gensim(model=model, corpus=corpus_for_training)))
        self.x_val = np.asarray(list(utils.infer_gensim(model=model, corpus=corpus_for_validation)))
        if return_model:
            return model

    def _fit_transform_gensim_embeddings(
        self, return_model: bool = False, **hyperparameter: t.Any
    ) -> t.Optional[t.Any]:
        documents_for_training: np.ndarray = self.x_train_raw
        documents_for_validation: np.ndarray = self.x_val_raw
        documents_with_tokens: t.List[t.List[str]] = [document.split() for document in documents_for_training]
        model: gensim.models = self._get_new_model(**hyperparameter).model
        if self.is_word_type_model:
            model.build_vocab(corpus_iterable=documents_with_tokens)
            model.train(documents_with_tokens, total_examples=model.corpus_count, epochs=model.epochs)
            self.x_train = np.array(
                [
                    utils.get_gensim_embedding(document=document.split(), gensim_model=model).flatten()
                    for document in documents_for_training
                ]
            )
            self.x_val = np.array(
                [
                    utils.get_gensim_embedding(document=document.split(), gensim_model=model).flatten()
                    for document in documents_for_validation
                ]
            )
        else:
            tagged_documents: t.List[TaggedDocument] = list(utils.tag_doc(documents=documents_for_training))
            model.build_vocab(corpus_iterable=tagged_documents)
            model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
            self.x_train = np.array(
                [model.infer_vector(doc_words=document.split()).flatten() for document in documents_for_training]
            )
            self.x_val = np.array(
                [model.infer_vector(doc_words=document.split()).flatten() for document in documents_for_validation]
            )

        if return_model:
            return model

    def _fit_transform_sklearn(self, return_model: bool = False, **hyperparameter: t.Any) -> t.Optional[t.Any]:
        model = self._get_new_model(**hyperparameter).model
        model.fit(self.x_train_raw)
        data: np.ndarray
        try:
            self.x_train = model.transform(self.x_train_raw).toarray()
            self.x_val = model.transform(self.x_val_raw).toarray()
        except AttributeError:
            self.x_train = model.transform(self.x_train_raw)
            self.x_val = model.transform(self.x_val_raw)

        if return_model:
            return model

    def _transform_data_transformers(self, return_model: bool = False, **hyperparameter: t.Any) -> t.Optional[t.Any]:
        pipeline: TransformerPipeline = self._get_new_model()
        data: t.Dict[str, list]

        self.x_train_transformer = [
            utils.tokenize_function(tokenizer=pipeline.tokenizer, text=text) for text in self.x_train_raw
        ]

        self.x_val_transformer = [
            utils.tokenize_function(tokenizer=pipeline.tokenizer, text=text) for text in self.x_val_raw
        ]

        if return_model:
            return pipeline.model

    def _get_new_model(self, corpus: t.Optional[t.Any] = None, **hyper_parameters: t.Any):
        """
        Return the doc2vec object linked to the doc2vec's name.
        Objective: Ensure that each doc2vec is unique, i.e., has a unique memory address.
        """
        model_info: t.Tuple = (
            self.nlp_model_info.model_name,
            self.nlp_model_info.package_name,
        )
        if model_info == (
            self.TOPIC_MODEL_NAME_TFIDF.model_name,
            self.TOPIC_MODEL_NAME_TFIDF.package_name,
        ):
            return ModelInfo(package=self.SKLEARN_LABEL, model=TfidfVectorizer(**hyper_parameters))
        elif model_info == (
            self.TOPIC_MODEL_NAME_BOW.model_name,
            self.TOPIC_MODEL_NAME_BOW.package_name,
        ):
            return ModelInfo(package=self.SKLEARN_LABEL, model=CountVectorizer(**hyper_parameters))

        elif model_info == (
            self.TOPIC_MODEL_NAME_LDA_SKLEARN.model_name,
            self.TOPIC_MODEL_NAME_LDA_SKLEARN.package_name,
        ):
            return ModelInfo(
                package=self.SKLEARN_LABEL,
                model=Pipeline(
                    [
                        ("count", CountVectorizer(lowercase=False)),
                        ("lda", LDA(**hyper_parameters)),
                    ]
                ),
            )
        elif model_info == (
            self.TOPIC_MODEL_NAME_LSA_SKLEARN.model_name,
            self.TOPIC_MODEL_NAME_LSA_SKLEARN.package_name,
        ):
            return ModelInfo(
                package=self.SKLEARN_LABEL,
                model=Pipeline(
                    [
                        ("tfidf", TfidfVectorizer(lowercase=False)),
                        ("svd", LSA(**hyper_parameters)),
                    ]
                ),
            )
        elif model_info == (
            self.TOPIC_MODEL_NAME_RP_SKLEARN.model_name,
            self.TOPIC_MODEL_NAME_RP_SKLEARN.package_name,
        ):
            return ModelInfo(
                package=self.SKLEARN_LABEL,
                model=Pipeline(
                    [
                        ("bow", CountVectorizer()),
                        ("rp", GaussianRandomProjection(**hyper_parameters)),
                    ]
                ),
            )
        elif model_info == (
            self.TOPIC_MODEL_NAME_NMF_SKLEARN.model_name,
            self.TOPIC_MODEL_NAME_NMF_SKLEARN.package_name,
        ):
            return ModelInfo(
                package=self.SKLEARN_LABEL,
                model=Pipeline(
                    [
                        ("tfidf", TfidfVectorizer(lowercase=False)),
                        ("nmf", NMF(**hyper_parameters)),
                    ]
                ),
            )

        elif model_info == (
            self.TOPIC_MODEL_NAME_LDA_GENSIM.model_name,
            self.TOPIC_MODEL_NAME_LDA_GENSIM.package_name,
        ):
            return ModelInfo(
                package=self.GENSIM_LABEL,
                model=gm.LdaModel(corpus=corpus, **hyper_parameters),
            )
        elif model_info == (
            self.TOPIC_MODEL_NAME_LSI_GENSIM.model_name,
            self.TOPIC_MODEL_NAME_LSI_GENSIM.package_name,
        ):
            return ModelInfo(
                package=self.GENSIM_LABEL,
                model=gm.LsiModel(corpus=corpus, **hyper_parameters),
            )
        elif model_info == (
            self.TOPIC_MODEL_NAME_RP_GENSIM.model_name,
            self.TOPIC_MODEL_NAME_RP_GENSIM.package_name,
        ):
            return ModelInfo(
                package=self.GENSIM_LABEL,
                model=gm.RpModel(corpus=corpus, **hyper_parameters),
            )
        elif model_info == (
            self.EMBEDDING_MODEL_NAME_DOC2VEC.model_name,
            self.EMBEDDING_MODEL_NAME_DOC2VEC.package_name,
        ):
            return ModelInfo(package=self.GENSIM_LABEL, model=Doc2Vec(**hyper_parameters))
        elif model_info == (
            self.EMBEDDING_MODEL_NAME_WORD2VEC.model_name,
            self.EMBEDDING_MODEL_NAME_WORD2VEC.package_name,
        ):
            return ModelInfo(package=self.GENSIM_LABEL, model=Word2Vec(**hyper_parameters))

        elif model_info == (
            self.TRANSFORMERS_MODEL_NAME_BERT.model_name,
            self.TRANSFORMERS_MODEL_NAME_BERT.package_name,
        ):
            return TransformerPipeline(
                tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
                model=TFAutoModelForSequenceClassification.from_pretrained(
                    "bert-base-cased", num_labels=len(np.unique(self.y_train))
                ),
            )
        else:
            raise KeyError("The doc2vec does not exist.")

    def _save_best_parameters(self) -> None:
        try:

            # save beste hyperparameters as .yaml
            utils.to_yaml(
                dictionary=self.study.best_params,
                filename=os.path.join(
                    self.path,
                    "best_params.yaml",
                ),
            )

            # save study info as .csv
            self.study.trials_dataframe().to_csv(
                os.path.join(self.path, "study.csv"),
            )

            # save best trial info as .csv
            self.study.trials_dataframe().query(f"number == {self.study.best_trial.number}").to_csv(
                os.path.join(self.path, "best_trial.csv"),
            )

            # save the hyperparameter importance as .pdf
            utils.save_optuna_plot(
                path=self.path,
                name="hyperparameter_importance",
                study=self.study,
                plot_fun=optuna.visualization.plot_param_importances,
            )

            # save the slice plot as .pdf
            utils.save_optuna_plot(
                path=self.path,
                name="plot_slice",
                study=self.study,
                plot_fun=optuna.visualization.plot_slice,
            )

        except ValueError as e:
            warnings.warn(str(e))
