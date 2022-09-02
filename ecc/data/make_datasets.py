import copy
import re
import typing as t
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
)

from ecc import SEED
from ecc.data.earning_calls import EarningsCalls
from ecc.utils import log


class MakeDatasets(EarningsCalls):
    x_raw_pre_processed: t.Dict[str, t.Dict[str, np.ndarray]] = {}
    x_raw_text: t.Dict[str, t.Dict[str, np.ndarray]] = {}
    x_nlp_processed: t.Dict[str, t.Dict[str, t.Dict[str, torch.Tensor]]] = {}

    y_multiclass_multi_output: t.Dict[str, torch.Tensor] = {}
    y_multiclass: t.Dict[str, t.Dict[str, torch.Tensor]] = {}

    LABEL_MULTICLASS: str = "multiclass"
    LABEL_MULTILABEL_MULTICLASS: str = "multilabel-multiclass"

    LABEL_BINARY: str = "Binary"
    LABEL_ORDINAL: str = "Ordinal"

    REGEX_SPLITTING: str = r"^(?P<Training>\d{1,2}):(?P<Validation>\d{1,2}):(?P<Testing>\d{1,2})$"
    DIVIDE_PERCENTAGE_TO_MAKE_FLOAT: int = 100

    NAME_TRAINING: str = "Training"
    NAME_VALIDATION: str = "Validation"
    NAME_TESTING: str = "Testing"

    split_proportions: t.Dict[str, float] = {
        NAME_TRAINING: np.nan,
        NAME_VALIDATION: np.nan,
        NAME_TESTING: np.nan,
    }

    DATASET_IDS_PER_TYPE: t.Dict[str, t.Dict[str, float]]
    _DATASET_IDS_TEMPLATE: t.Dict[str, float] = {
        NAME_TRAINING: [],
        NAME_VALIDATION: [],
        NAME_TESTING: [],
    }

    random_number_generator: np.random.Generator

    label_encoder_ordinal: t.Dict[str, OrdinalEncoder or t.Dict[str, LabelEncoder]]
    label_encoder_binary: t.Dict[str, MultiLabelBinarizer or t.Dict[str, OneHotEncoder]]

    encoded_labels_ordinal: t.Dict[str, t.Dict[str, np.ndarray]]
    encoded_labels_binary: t.Dict[
        str, t.Dict[str, t.Union[t.Dict[str, np.ndarray], t.Dict[str, t.Dict[str, np.ndarray]]]]
    ]

    def __init__(
        self,
        working_directory: str,
        splitting_proportion: str = "80:10:10",
        requery: bool = False,
        process_raw_data: bool = False,
        label_studio_api: t.Optional[str] = None,
    ) -> None:

        super().__init__(
            requery=requery,
            process_raw_data=process_raw_data,
            label_studio_api=label_studio_api,
            working_directory=working_directory,
        )
        # Create the blueprint to store the ids for training, validation and testing.

        self.DATASET_IDS_PER_TYPE = {key: copy.deepcopy(self._DATASET_IDS_TEMPLATE) for key in self.QA_TYPES}
        # Initialize the random number generator.
        self.random_number_generator = np.random.default_rng(SEED)
        # Set the dataset proportions.
        self.__set_split_proportions(splitting_proportion=splitting_proportion)
        # Draw the ids randomly, according to the proportions assigned by `splitting_proportion`.
        self.__get_ids()
        # Create the dictionaries that will store the encoders and the encoded labels.
        self.__prepare_encoding_dictionaries()
        # Encode the labels and fill the dictionaries.
        self.__encode_labels()
        # Store and split the target labels Y.
        self.__store_and_split_y()
        # Partitioning X
        self.__partitioning()

    def __store_and_split_y(self) -> None:
        """
        Store all labels and also split them into separate data frames.
        """
        for type_ in self.QA_TYPES:
            # Get all labels at once.
            self.y_multiclass_multi_output.update(
                {type_: self.earning_calls_data.query(f"{self.QA_TYPE_NAME} == @type_")[self.MAIN_LABELS].to_numpy()}
            )
            self.y_multiclass.update({type_: {}})
            # Split the labels into separate arrays.
            for label in self.MAIN_LABELS:
                self.y_multiclass[type_][label] = self.earning_calls_data.query(f"{self.QA_TYPE_NAME} == @type_")[
                    label
                ].values

    def __partitioning(self) -> None:
        """
        Partition the X and Y values according to the ids given by the different partitions.
        """
        for type_, dictionary in self.DATASET_IDS_PER_TYPE.items():
            data_per_type: np.ndarray = self.earning_calls_data.query(f"{self.QA_TYPE_NAME} == @type_")[
                self.NAME_OF_TEXT_COLUMN
            ].to_numpy()
            self.x_raw_pre_processed.update({type_: {}})
            self.x_raw_text.update({type_: {}})
            for partition_name, ids in dictionary.items():
                self.x_raw_pre_processed[type_].update({partition_name: data_per_type[ids]})
                self.x_raw_text[type_].update({partition_name: data_per_type[ids]})

    @log
    def __set_split_proportions(self, splitting_proportion: str = "80:10:10") -> None:
        """
        Set the splitting proportion by using the format 'x:x:x'.
        Example: Using the proportions '80:10:10' (represents: 80%:10%:10%) results in the following proportions:
        - training:     0.8
        - validation:   0.1
        - testing:      0.1
        """
        regex_splitting: re.Match = re.search(self.REGEX_SPLITTING, splitting_proportion)
        if not bool(regex_splitting):
            warnings.warn("Split proportions are not valid. Use the format 'x:x:x'")
            return None
        if np.asarray(regex_splitting.groups()).astype(int).sum() != self.DIVIDE_PERCENTAGE_TO_MAKE_FLOAT:
            warnings.warn("Make sure that the proportions add up to 100%.")
            return None
        # Transform the respective value to float and store it inside the dictionary `split_proportions`
        # Example: Value 80 is divided by 100 to represent a floating point number => 0.8
        for key in self.split_proportions.keys():
            self.split_proportions[key] = int(regex_splitting.groupdict()[key]) / self.DIVIDE_PERCENTAGE_TO_MAKE_FLOAT

    @log
    def __get_ids(self) -> None:
        """
        Sample index values according to the specified proportion.
        """
        for type_ in self.QA_TYPES:
            data: pd.DataFrame = self.earning_calls_data.query(f"{self.QA_TYPE_NAME} == @type_").index
            # Get all the initial ids
            set_of_initial_ids = set(np.arange(len(data)))
            # Define, which ids are still available for sampling.
            set_of_available_ids: set = copy.deepcopy(set_of_initial_ids)
            for dataset in self.DATASET_IDS_PER_TYPE[type_].keys():
                size_ = int(np.ceil(len(set_of_initial_ids) * self.split_proportions[dataset]))
                # It ensures that no error occurs by taking more values than are available in the population.
                if size_ > len(set_of_available_ids):
                    ids_ = list(set_of_available_ids)
                else:
                    ids_ = self.random_number_generator.choice(list(set_of_available_ids), size=size_, replace=False)
                # Saves the sampled ids in the respective dictionary
                self.DATASET_IDS_PER_TYPE[type_][dataset] = ids_
                # The sampled ids are not available anymore for sampling.
                set_of_available_ids -= set(ids_)
            # Double-check: Ensures that all ids of `set_of_initial_ids` are used and distributed.
            if len(set_of_available_ids) > 0:
                warnings.warn(f"{len(set_of_available_ids)} set_of_initial_ids are left-over! Please re-check.")

    def __prepare_encoding_dictionaries(self):
        """
        Create all dictionaries that will store the encoders and the encoded labels.
        """
        self.label_encoder_ordinal = self.__create_encoder_dictionary(
            multilabel_multiclass_encoder=OrdinalEncoder(),
            multiclass_encoder=LabelEncoder(),
        )
        self.label_encoder_binary = self.__create_encoder_dictionary(
            multilabel_multiclass_encoder=MultiLabelBinarizer(),
            multiclass_encoder=OneHotEncoder(),
        )

        self.encoded_labels_ordinal = self.__create_encoded_labels_dictionary()
        self.encoded_labels_binary = self.__create_encoded_labels_dictionary()

    def __create_encoder_dictionary(self, multilabel_multiclass_encoder, multiclass_encoder) -> t.Dict:
        """
        Create the blueprint of the dictionary that will store the label encoders.
        Specify the encoders to use for the multiclass and multiclass-multi-output encoding.

        - 'multiclass' => One label with multiple classes
        - 'multilabel-multiclass' => Multiple labels with multiple classes
        """
        template_ = {
            self.LABEL_MULTILABEL_MULTICLASS: copy.deepcopy(multilabel_multiclass_encoder),
            self.LABEL_MULTICLASS: {label: copy.deepcopy(multiclass_encoder) for label in self.MAIN_LABELS},
        }
        return {type_: copy.deepcopy(template_) for type_ in self.QA_TYPES}

    def __create_encoded_labels_dictionary(self):
        """
        Create the blueprint of the dictionary that will store the encoded labels.
        """
        partition_template: dict = {partition: [] for partition in self.split_proportions.keys()}

        template_ = {
            self.LABEL_MULTILABEL_MULTICLASS: copy.deepcopy(partition_template),
            self.LABEL_MULTICLASS: {label: copy.deepcopy(partition_template) for label in self.MAIN_LABELS},
        }
        return {type_: copy.deepcopy(template_) for type_ in self.QA_TYPES}

    @log
    def __encode_labels(self):
        """
        Encode the labels and classes into a binary and ordinal representation.
        """
        # Iterates across the QA-type (Question/Answer).
        for type_ in self.QA_TYPES:
            self.__encode_multilabel_multiclass(type_=type_)
            # Iterate across all labels ("QA1", "QA2" and "QA3") and encode them separately.
            for label in self.label_encoder_binary[type_][self.LABEL_MULTICLASS].keys():
                self.__encode_multiclass(type_=type_, label=label)

    def __encode_multiclass(self, type_: str, label: str):
        """
        multiclass => One label with multiple classes.
        """
        # Get the classes of each label.
        data_multiclass: pd.Series = self.earning_calls_data.query(f"{self.QA_TYPE_NAME} == @type_").loc[:, label]

        binary: np.ndarray = (
            self.label_encoder_binary[type_][self.LABEL_MULTICLASS][label]
            .fit_transform(data_multiclass.to_numpy().reshape(-1, 1))
            .toarray()
        )
        ordinal: np.ndarray = self.label_encoder_ordinal[type_][self.LABEL_MULTICLASS][label].fit_transform(
            data_multiclass
        )

        for partition, ids in self.DATASET_IDS_PER_TYPE[type_].items():
            self.encoded_labels_ordinal[type_][self.LABEL_MULTICLASS][label][partition] = ordinal[ids]
            self.encoded_labels_binary[type_][self.LABEL_MULTICLASS][label][partition] = binary[ids]

    def __encode_multilabel_multiclass(self, type_: str):
        """
        multilabel-multiclass => Multiple labels with multiple classes.
        """
        # Get the columns that are going to be encoded.
        data_multilabel_multiclass: pd.DataFrame = self.earning_calls_data.query(f"{self.QA_TYPE_NAME} == @type_").loc[
            :, self.MAIN_LABELS
        ]

        binary: np.ndarray = self.label_encoder_binary[type_][self.LABEL_MULTILABEL_MULTICLASS].fit_transform(
            data_multilabel_multiclass.values
        )
        ordinal: np.ndarray = self.label_encoder_ordinal[type_][self.LABEL_MULTILABEL_MULTICLASS].fit_transform(
            data_multilabel_multiclass.values
        )

        for partition, ids in self.DATASET_IDS_PER_TYPE[type_].items():
            self.encoded_labels_ordinal[type_][self.LABEL_MULTILABEL_MULTICLASS][partition] = ordinal[ids]
            self.encoded_labels_binary[type_][self.LABEL_MULTILABEL_MULTICLASS][partition] = binary[ids]

    def inverse_transform_binary(
        self,
        data: t.List or np.ndarray,
        type_enc: str,
        type_qa: str,
        type_lc,
        label: t.Optional[str] = None,
    ) -> t.Optional[pd.DataFrame or pd.Series]:
        """
        Convert the data back to the original representation.
        -----------------------------------------------------
        The type_enc (encoding type) can only be:
        - 'Binary'
        - 'Ordinal'
        The type_qa can only be:
        - 'Question'
        - 'Answer'
        The type_lc can only be:
        - 'multiclass' => One label with multiple classes
        - 'multilabel-multiclass' => Multiple labels with multiple classes
        """
        encoder: dict
        # Assigns the respective dictionary with the encoders (either binary or ordinal encoding) stored in to the
        # dictionary `encoder`.
        if type_enc == self.LABEL_ORDINAL:
            encoder = self.label_encoder_ordinal[type_qa]
        elif type_enc == self.LABEL_BINARY:
            encoder = self.label_encoder_binary[type_qa]
        else:
            warnings.warn(
                f"Encoding type {type_enc} is not supported. Either use {self.LABEL_BINARY} or "
                f""
                f"{self.LABEL_ORDINAL}"
            )
            return None
        # Use the encoders stored in the dictionary `encoder` to re-transform the encoded labels back to the original
        # representation.
        if type_lc == self.LABEL_MULTILABEL_MULTICLASS:
            return np.asarray(encoder[type_lc].inverse_transform(data))
        elif type_lc == self.LABEL_MULTICLASS:
            if label is None:
                warnings.warn("Please specify a label!")
                return None
            return encoder[type_lc][label].inverse_transform(data).flatten()
        warnings.warn(
            f"Type {type_lc} is not supported. Either use {self.LABEL_MULTILABEL_MULTICLASS} or "
            f"{self.LABEL_MULTICLASS}"
        )
