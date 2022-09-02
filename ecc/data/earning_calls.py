import os.path
import pickle
import re
import typing as t

import contractions
import numpy as np
import pandas as pd
import spacy
from label_studio_sdk import Client, project
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from tqdm import tqdm

from ecc.utils import log

nlp = spacy.load("en_core_web_trf")


class EarningsCalls:
    _LABEL_STUDIO_API: str
    _LABEL_STUDIO_URL: str = "http://132.231.59.226:8080"

    _PROCESSED_DATA_FILE_NAME: str = "earning_calls_data.csv"
    _RAW_DATA_FILE_NAME: str = "earning_calls_raw.pkl"

    ID_NAME: str = "ID"
    QA_TYPE_NAME: str = "Type"
    NAME_OF_TEXT_COLUMN: str

    LABEL_PROCESSED_TEXT: str = "Processed Text"
    LABEL_RAW_TEXT: str = "Raw Text"

    MAIN_LABELS: t.List[str] = ["QA1", "QA2", "QA3"]
    NUMBER_OF_CORRECT_LABELS: int = 4
    LABEL_NAMES_WITH_IDENTIFIER: t.Dict[str, str] = {
        "QA_ID": "ID",
        "QA1": "_1_",
        "QA2": "_2_",
        "QA3": "_3_",
    }
    LABEL_IDENTIFIER: t.Dict[str, t.List[str]] = {
        "Answer": ["AID", "A"],
        "Question": ["QID", "Q"],
    }
    QA_TYPES: t.List[str] = ["Question", "Answer"]
    # Only use tokens that are an alpha character and not a stop word.
    # Additional Part-of-Speech (POS) and Named Entity Recognition (NER) filtering: see
    # https://spacy.io/usage/spacy-101#annotations
    _PATTERN: t.List[dict] = [
        {
            "IS_ALPHA": True,
            "IS_STOP": False,
            "POS": {
                "NOT_IN": [
                    # Determiner, e.g. a, an, the
                    "DET",
                    # Interjection, e.g. psst, ouch, bravo, hello
                    "INTJ",
                    # Particle, e.g. ’s, not,
                    "PART",
                    # Pronoun, e.g I, you, he, she, myself, themselves, somebody
                    "PRON",
                    # Subordinating conjunction, e.g. if, while, that
                    "SCONJ",
                ]
            },
            "ENT_TYPE": {"NOT_IN": ["DATE", "TIME", "PERCENT", "PERSON", "CARDINAL"]},
        }
    ]

    earning_calls_data: pd.DataFrame = pd.DataFrame()
    raw_data: t.List[dict]
    working_directory: str
    _matcher: Matcher = Matcher(nlp.vocab)
    _matcher.add(key="pre_proc", patterns=[_PATTERN])

    def __init__(
        self,
        working_directory: str,
        requery: bool = False,
        process_raw_data: bool = False,
        label_studio_api: t.Optional[str] = None,
    ) -> None:
        self.NAME_OF_TEXT_COLUMN = self.LABEL_PROCESSED_TEXT
        self._LABEL_STUDIO_API = label_studio_api
        self.working_directory = working_directory
        if requery:
            self._get_data()
        else:
            with open(os.path.join(self.working_directory, self._RAW_DATA_FILE_NAME), "rb") as f:
                # noinspection PickleLoad
                self.raw_data = pickle.load(f)
        if process_raw_data:
            self._process_raw_data()
        else:
            self.earning_calls_data = pd.read_csv(
                os.path.join(self.working_directory, self._PROCESSED_DATA_FILE_NAME), index_col=self.ID_NAME
            )

    @log
    def _get_data(self):
        """
        Query the data from label studio.
        """
        ls = Client(url=self._LABEL_STUDIO_URL, api_key=self._LABEL_STUDIO_API)
        ls.check_connection()
        pro = project.Project.get_from_id(ls, 1)

        self.raw_data = project.Project.get_labeled_tasks(pro)

        with open("../../earning_calls_raw.pkl", "wb") as f:
            pickle.dump(self.raw_data, f)

    @staticmethod
    def _find_keyword_in_list(data: list or np.ndarray or pd.Series, keyword: str) -> np.ndarray:
        """
        Returns an array with booleans, whether the respective element at that position contains the given keyword.
        """
        return np.array([bool(re.search(keyword, element)) for element in data])

    def _are_classes_correct(
        self,
        data: list or np.ndarray or pd.Series,
        unique_identifier: t.Optional[t.List[str]] = None,
    ) -> bool:
        """
        Checks, whether the labels are complete, i.e., only one element of the list contains one of the values
        in LABEL_NAMES_WITH_IDENTIFIER.
        Example: A list with the elements
            - QID_1
            - Question_1_Market_related
            - Question_3_support
            - Question_3_neutral
        contains two times the keyword <_3_>, which indicates wrongly labelled data.
        """
        if unique_identifier is None:
            unique_identifier = self.LABEL_NAMES_WITH_IDENTIFIER.values()

        for unique_identifier in unique_identifier:
            if np.sum(self._find_keyword_in_list(data=data, keyword=unique_identifier)) != 1:
                return False
        return True

    def _are_labels_correct(
        self,
        data: list or np.ndarray or pd.Series,
        unique_identifier: t.Optional[t.List[str]] = None,
    ) -> bool:
        """
        Checks if labels are correct.
        """
        if unique_identifier is None:
            unique_identifier = self.LABEL_NAMES_WITH_IDENTIFIER.values()

        labels_per_type: dict = self._get_label_counts_per_type(data=data, unique_identifier=unique_identifier)
        count_labels_per_type_with_most_labels: np.ndarray[int] = np.max(list(labels_per_type.values()))
        return count_labels_per_type_with_most_labels == len(unique_identifier)

    def _are_labels_and_classes_valid(
        self,
        data: list or np.ndarray or pd.Series,
        unique_identifier: t.Optional[t.List[str]] = None,
    ) -> bool:
        return self._are_classes_correct(data=data, unique_identifier=unique_identifier) and self._are_labels_correct(
            data=data, unique_identifier=unique_identifier
        )

    def _get_label_counts_per_type(
        self,
        data: pd.Series or np.ndarray or list,
        unique_identifier: t.Optional[t.List[str]] = None,
    ) -> t.Dict[str, int]:
        """
        Counts the labels by type (question or answer).
        """
        if unique_identifier is None:
            unique_identifier = self.LABEL_NAMES_WITH_IDENTIFIER.values()

        count_labels_per_type: t.Dict[str, int] = {_type: 0 for _type in self.LABEL_IDENTIFIER.keys()}
        for _type, identifier in self.LABEL_IDENTIFIER.items():
            if self._are_classes_correct(data=data, unique_identifier=unique_identifier):
                count_labels_per_type[_type] = self._find_keyword_in_list(data=data, keyword="|".join(identifier)).sum()
        return count_labels_per_type

    def _extract_labels(self, data: pd.Series) -> t.Dict[str, str]:
        """
        Extract the labels from the data and assign to the respective key of LABEL_NAMES_WITH_IDENTIFIER.
        """
        return {
            key: data[self._find_keyword_in_list(data=data, keyword=value)].to_list().pop()
            for key, value in self.LABEL_NAMES_WITH_IDENTIFIER.items()
        }

    def _get_qa_type(self, data: pd.Series) -> str:
        """
        Returns the type of the labellings (either question or answer).
        """
        count_labels_per_type: t.Dict[str, int] = self._get_label_counts_per_type(data=data)
        return list(count_labels_per_type.keys())[np.argmax(list(count_labels_per_type.values()))]

    def _get_processed_text(self, text: str) -> t.Generator[str, None, None]:
        """
        Filters the tokens using SpaCy's _matcher and the defined pattern. Everything that is not matched is ignored.
        """
        document: Doc = nlp(contractions.fix(text))
        matched_tokens: t.List[t.Tuple[int, int, int]] = self._matcher(document)
        for match_id, start, end in matched_tokens:
            yield Span(document, start, end, label=match_id).lemma_

    @log
    def _process_raw_data(self):
        """
        Take the raw data (list given by label studio) and process it.
        Includes quality checks and text preprocessing.
        """
        # Iterates over the given label studio dictionary
        for section in tqdm(self.raw_data, desc="Pre-Processing Earning-Calls"):
            _result: list = section["annotations"][0]["result"]
            if len(_result) == 0:
                continue
            # Dataframe is initialized with the first entry
            annotation: pd.DataFrame = pd.DataFrame(_result[0]["value"])
            company: str = section["data"]["Company"]
            # Append all the other entries
            for subsequent_result in section["annotations"][0]["result"][1:]:
                annotation = pd.concat([annotation, pd.DataFrame(subsequent_result["value"])])
            # Remove all leading and trailing whitespaces of the text
            annotation["text"] = annotation["text"].apply(lambda x: x.strip())
            # Round the start value to nearest 10
            annotation["start"] = annotation["start"].apply(lambda x: round(x, -1))
            # Iterate over all unique text blocks that should contain for unique and type-related labels
            for start_value in sorted(annotation["start"].unique()):
                # Derive one data entry for a unique start value
                _single_block: pd.DataFrame = annotation.loc[annotation["start"] == start_value, :]
                # Extract the labels
                _single_block_labels: pd.Series = _single_block.loc[:, "labels"]
                # Check if the labelling is done correctly, i.e., all labels are related to the same type (
                # question or answer) and shows four and only four labels.
                # Everything else is ignored.
                if len(_single_block_labels) == self.NUMBER_OF_CORRECT_LABELS and self._are_labels_and_classes_valid(
                    data=_single_block_labels
                ):
                    # Extract the raw text
                    raw_text = annotation.loc[annotation["start"] == start_value, ["text"]].iloc[0]["text"]
                    text_to_use: pd.Series or np.ndarray
                    self.NAME_OF_TEXT_COLUMN = self.LABEL_PROCESSED_TEXT
                    text_to_use = " ".join(list(self._get_processed_text(raw_text)))
                    # Continue if there are no words left after preprocessing.
                    if len(text_to_use) == 0:
                        continue
                    # Fill the dictionary
                    values: t.Dict[str, str] = {
                        "Company": company,
                        self.NAME_OF_TEXT_COLUMN: text_to_use,
                        self.QA_TYPE_NAME: self._get_qa_type(data=_single_block_labels),
                        self.LABEL_RAW_TEXT: raw_text,
                    }
                    # Insert the labels
                    values.update(self._extract_labels(data=_single_block_labels))
                    # Create a dataframe from the dictionary and append it to the global dataframe
                    # <earning_calls_data>
                    self.earning_calls_data: pd.DataFrame = pd.concat(
                        [self.earning_calls_data, pd.Series(values).to_frame().T]
                    )
        # Save the dataframe as .csv
        self.earning_calls_data = self.earning_calls_data.reset_index(drop=True)
        self.earning_calls_data.to_csv(
            os.path.join(self.working_directory, self._PROCESSED_DATA_FILE_NAME), index_label=self.ID_NAME
        )
