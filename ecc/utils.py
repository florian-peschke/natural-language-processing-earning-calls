import datetime
import inspect
import io
import os
import pickle
import re
import sys
import time
import typing as t
from functools import wraps
from shutil import rmtree

import numpy as np
import optuna
import pandas as pd
import tabulate
import torch
import yaml
from attr import define, field
from gensim.models.doc2vec import TaggedDocument
from typeguard import typechecked


def stop_time(func):
    def dec(*args, **kwargs):
        start: float = time.time()
        output: t.Any = func(*args, **kwargs)
        stop: float = time.time()
        print(f"Time: {Bcolors.BLUE}{datetime.timedelta(seconds=stop - start)}{Bcolors.RESET}\n")
        return output

    return dec


class Bcolors:
    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class ModelInfo(t.NamedTuple):
    package: t.AnyStr
    model: t.Any


class NLPModelInfo(t.NamedTuple):
    type: t.AnyStr
    model_name: t.AnyStr
    package_name: t.AnyStr

    @property
    def name(self) -> str:
        return "_".join(
            [
                self.type,
                self.model_name,
                self.package_name,
            ]
        )


class TransformerPipeline(t.NamedTuple):
    tokenizer: t.Any
    model: t.Any


def log(func):
    """
    Log the main functions performed to follow all the steps in the chain.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function:\n{Bcolors.BLUE}{func.__module__}.{func.__name__}{inspect.signature(func)}\n{Bcolors.RESET}")
        print(f"Description:\n{Bcolors.RESET}{Bcolors.BLUE}{str(func.__doc__).strip()}\n{Bcolors.RESET}")
        # Stop time to execute the function.
        start: float = time.time()
        returned_values: any = func(*args, **kwargs)
        stop: float = time.time()
        # Print duration.
        print(f"Time: {Bcolors.BLUE}{datetime.timedelta(seconds=(stop - start))}{Bcolors.RESET}\n")
        print(f"{Bcolors.RED}{'–' * 14}\n{Bcolors.RESET}")
        return returned_values

    return wrapper


def infer_gensim(model: t.Any, corpus: t.List) -> t.Generator[np.ndarray, None, None]:
    for document in corpus:
        ids: np.ndarray = np.arange(model.num_topics)
        dist_vec: t.Dict = {str(id_): 0 for id_ in ids}
        for id_, dist in model[document]:
            dist_vec.update({str(id_): dist})
        yield np.asarray([dist for dist in dist_vec.values()])


def get_gensim_embedding(document: t.List[t.AnyStr], gensim_model, row_wise_operation: t.AnyStr = "sum") -> np.ndarray:
    """
    Retrieve the word vectors of all words in that document and apply a row-wise operation.
    Implements a recursive behavior.
    """
    zero_vector: np.ndarray = np.zeros(shape=(gensim_model.vector_size, 1))
    # Stopping point for the recursion.
    if len(document) == 0:
        return zero_vector
    # Get the last word in the list.
    word: t.AnyStr = document.pop()
    # Create the word_vector storage variable.
    word_vector: np.ndarray
    # Try to get the vector representation of the word.
    try:
        word_vector = gensim_model.wv[word].reshape(-1, 1)
    except KeyError:
        word_vector = zero_vector
    # Apply recursion and last the row-wise operation.
    return getattr(np, row_wise_operation)(
        [
            word_vector,
            get_gensim_embedding(document=document[:-1], gensim_model=gensim_model),
        ],
        axis=0,
    )


def tag_doc(
    documents: t.List[str] or np.ndarray,
) -> t.Generator[TaggedDocument, None, None]:
    """
    Create tagged documents required by Doc2Vec.
    """
    for i, text in enumerate(documents):
        yield TaggedDocument(text, [i])


def n_grams_workaround(val: t.AnyStr) -> t.Tuple[int, ...]:
    return tuple(int(v) for v in val.split(" "))


def tokenize_function(tokenizer: t.Any, text: t.AnyStr, **kwargs: t.Dict[t.AnyStr, t.Any]) -> t.Dict:
    return tokenizer(text, **kwargs)


def catch_print(fun: t.Callable) -> str:
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    fun()
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    return output


def check_type(value: t.Any, type_: t.Any) -> t.Any:
    if type(value) == type_:
        return value
    else:
        raise TypeError(f"Called with type {type(value)}, but only type {type_} is accepted!")


def investigate(value: t.Any) -> t.Any:
    print(type(value))
    print(value)
    return value


def to_yaml(dictionary: t.Dict[str, t.Any], filename: str) -> None:
    with open(filename, "w") as file:
        file.write(yaml.dump(dictionary))


@typechecked
def from_yaml(filename: str) -> t.Dict[str, t.Any]:
    with open(filename, "r") as file:
        return yaml.safe_load(file)


@typechecked
def create_dir_if_missing(path: str, return_dir: bool = True) -> t.Optional[str]:
    if not os.path.exists(path):
        os.makedirs(path)
    if return_dir:
        return path


@typechecked
def init_folder(path: str, ignore_errors: bool = True) -> str:
    rmtree(path, ignore_errors=ignore_errors)
    return create_dir_if_missing(path=path)


@typechecked
def to_pickle(object_: t.Any, filename: str):
    with open(filename, "wb") as file:
        pickle.dump(object_, file, pickle.HIGHEST_PROTOCOL)


@typechecked
def from_pickle(filename: str):
    with open(filename, "rb") as file:
        # noinspection PickleLoad
        return pickle.load(file)


def cast_types_of_dictionary(dictionary: t.Dict) -> t.Dict:
    dictionary: t.Dict = {}
    for key, value in dictionary.items():
        if isinstance(value, int):
            dictionary.update({key: int(value)})
        elif isinstance(value, float):
            dictionary.update({key: float(value)})
        else:
            dictionary.update({key: value})
    return dictionary


@define
class TensorScaler:
    scaling_range: t.Tuple[int, int]
    min: torch.Tensor = field(init=False)
    max: torch.Tensor = field(init=False)

    @typechecked
    def __init__(self, scaling_range: t.Tuple[int, int] = (-1, 1)) -> None:
        self.__attrs_init__(scaling_range=scaling_range)

    @typechecked
    def fit(self, x: torch.Tensor) -> None:
        self.min = x.min()
        self.max = x.max()

    @typechecked
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return (x - self.min) / (self.max - self.min) * (
                self.scaling_range[-1] - self.scaling_range[0]
            ) + self.scaling_range[0]
        except AttributeError:
            raise AttributeError("Please fit the scaler before calling transform.")

    @typechecked
    def fit_and_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)


def load_and_concat_tensors(path: str, regex: str = r"(?P<number>.*)_.*.pt") -> torch.Tensor:
    tensor: t.Optional[torch.Tensor] = None
    for root, dirs, files in os.walk(path):
        for i, file in enumerate(sorted(files, key=lambda x: int(re.search(regex, x).groupdict()["number"]))):
            if i == 0:
                tensor = torch.load(os.path.join(root, file))
            else:
                tensor = torch.concat([tensor, torch.load(os.path.join(root, file))])
    return tensor


def save_optuna_plot(
    path: str,
    study: optuna.Study,
    name: str,
    plot_fun: t.Callable = optuna.visualization.plot_param_importances,
    plot_kwargs: t.Optional[dict] = None,
) -> None:
    if plot_kwargs is None:
        plot_kwargs = {}
    try:
        plot_fun(study, **plot_kwargs).write_image(
            os.path.join(
                path,
                f"{name}.pdf",
            )
        )
    except BaseException as e:
        with open(
            os.path.join(
                path,
                f"{name}.txt",
            ),
            "w",
        ) as f:
            print(str(e), file=f)


def frame_to_latex(frame: pd.DataFrame, path: str, name: str, **kwargs) -> None:
    with open(os.path.join(path, f"{name}.txt"), "w") as f:
        print(tabulate.tabulate(frame, headers="keys", tablefmt="latex", showindex=True, **kwargs), file=f)
