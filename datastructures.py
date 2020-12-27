# This file contains the data structures used to represent data throughout the program, including the datasets, processeddata, and output data

from typing import List

import numpy as np
import torch

import processeddata

"""Represents a candidate for a mention"""


class Candidate:
    id: int = None  # Unique ID
    text: str = None  # Textual label for this candidate
    # TODO - I understand initial_prob to be p_e_m scores
    initial_prob: float = None  # Initial probability for this candidate
    MAXID = 0  # TODO - unused?

    def __init__(self, id, initial_prob, text):
        self.id = id
        self.initial_prob = initial_prob
        self.text = text

    def __repr__(self):
        return f"Candidate({self.id},{self.initial_prob},\"{self.text}\")"

    def entEmbedding(self) -> np.ndarray:
        # TODO what happens if an ent isn't seen before?
        return processeddata.entid2embedding[processeddata.ent2entid.get(self.text)]

    def entEmbeddingTorch(self) -> torch.Tensor:
        # TODO what happens if an ent isn't seen before?
        return torch.from_numpy(self.entEmbedding()).type(
            torch.Tensor)

"""Represents a mention in a document"""


class Mention:
    # Represents a mention in a document
    id: int = None  # Document-Unique identifier
    text: str = None  # The text of the mention
    left_context: str = None  # Context to the left of the mention
    right_context: str = None  # Context to the right of the mention
    candidates: List[Candidate] = None  # Candidates for the mention
    gold_id: str = None  # Gold truth - actual candidate (id) for this mention

    def FromData(id, text, left_context, right_context, candidates, gold_id):
        self = Mention()
        self.id = id
        self.text = text
        self.left_context = left_context
        self.right_context = right_context
        self.candidates = candidates
        self.gold_id = gold_id
        return self

    def goldCand(self):
        for cand in self.candidates:
            if cand.id == self.gold_id:
                return cand
        return None

    def wordEmbedding(self):
        return processeddata.wordid2embedding[processeddata.word2wordid.get(self.text, processeddata.unkwordid)]

    def __repr__(self):
        return f"Mention.FromData({self.id},\"{self.text}\",\"{self.left_context}\",\"{self.right_context}\",{self.candidates},\"{self.gold_id}\")"


class Document:
    id: str = None  # Document ID
    mentions: List[Mention] = None

    def FromData(id, mentions):
        self = Document()
        self.id = id
        self.mentions = mentions
        return self

    def __repr__(self):
        # TODO - if any string contains " then it will cause errors recreating
        return f"Document.FromData(\"{self.id}\",{self.mentions})"


"""
Represents a dataset from the generated data
"""


class Dataset:
    # We can determine the structure by looking at the data files and how the original implementation uses them
    documents: List[Document] = None


"""Represents a model (nn parameters, etc)"""


class Model:
    neuralNet = None
    pass


"""Represents the evaluation of a model on a dataset (F1, etc)"""


class EvaluationMetrics:
    precision: float = 0
    recall: float = 0
    f1: float = 0
