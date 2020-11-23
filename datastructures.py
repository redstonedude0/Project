# This file contains the data structures used to represent data throughout the program, including the datasets, processeddata, and output data

from typing import List

"""Represents a candidate for a mention"""


class Candidate:
    id: str = None  # Unique ID
    text: str = None  # Textual label for this candidate
    initial_prob: float = None  # Initial probability for this candidate

    def __init__(self, id, initial_prob, text):
        self.id = id
        self.initial_prob = initial_prob
        self.text = text


"""Represents a mention in a document"""


class Mention:
    # Represents a mention in a document
    text: str = None  # The text of the mention
    left_context: str = None  # Context to the left of the mention
    right_context: str = None  # Context to the right of the mention
    candidates: List[Candidate] = None  # Candidates for the mention
    gold_id: str = None  # Gold truth - actual candidate (id) for this mention


class Document:
    id: str = None  # Document ID
    mentions: List[Mention] = None


"""
Represents a dataset from the generated data
"""


class Dataset:
    # We can determine the structure by looking at the data files and how the original implementation uses them
    documents: List[Document] = None
