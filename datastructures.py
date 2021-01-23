# This file contains the data structures used to represent data throughout the program, including the datasets, processeddata, and output data
import json
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
        # If unseen then give average embedding (https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/utils.py line 104)
        # TODO discuss this
        return processeddata.entid2embedding[processeddata.ent2entid.get(self.text, processeddata.unkentid)]

    def entEmbeddingTorch(self) -> torch.Tensor:
        # TODO what happens if an ent isn't seen before?
        return torch.from_numpy(self.entEmbedding()).to(SETTINGS.device).to(
            torch.float)

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

    def goldCandIndex(self):
        for cand_idx, cand in enumerate(self.candidates):
            if cand.id == self.gold_id:
                return cand_idx
        return -1

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


from hyperparameters import SETTINGS
"""Represents a model (nn parameters, etc)"""


class Model:
    neuralNet = None
    evals: 'EvalHistory' = None

    def save(self, name):
        print(f"Saving checkpoint '{name}'...")
        torch.save(self.neuralNet, SETTINGS.dataDir_checkpoints + name + ".pt")
        self.evals.save(SETTINGS.dataDir_checkpoints + name + ".evals")
        print("Saved.")

    @classmethod
    def load(cls, name):
        inst = cls()
        inst.neuralNet = torch.load(SETTINGS.dataDir_checkpoints + name + ".pt")
        inst.evals = EvalHistory.load(SETTINGS.dataDir_checkpoints + name + ".evals")
        print("Loaded.")
        return inst


"""Represents the evaluation of a model on a dataset (F1, etc)"""


class EvaluationMetrics:
    # Evaluation data
    loss: float = 0  # Total loss
    accuracy: float = 0  # Accuracy=MicroF1
    accuracy_possible: float = 0  # Maximum possible
    #    correctRatio: float = 0#Ratio of mentions correct
    #    microF1:float = 0
    #    correctRatio_possible:float = 0#Best possible value
    #    microF1_possible:float = 0#Best possible value
    # Metadata
    step: int = 0  # Step this occurred at (0+)
    time: float = 0  # Time since model start

    def print(self):
        print(f"Evaluation {self.step} [{self.time}]:")
        print(f" Loss     | {self.loss}")
        print(f" Accuracy | {self.accuracy}")
        print(f" Poss.Acc | {self.accuracy_possible}")


class EvalHistory:
    metrics: List[EvaluationMetrics] = []

    def __init__(self):
        self.metrics = []  # new array

    def print(self):
        print("EvalHistory:")
        for metric in self.metrics:
            metric.print()

    def save(self, f):
        def serialise(obj):
            return obj.__dict__

        with open(f, "w") as fp:
            json.dump(self, fp, default=serialise)

    @classmethod
    def load(cls, f) -> 'EvalHistory':
        def as_evalhistory(dct):
            if "metrics" in dct:
                evals = EvalHistory()
                evals.metrics = []
                for metric in dct["metrics"]:
                    evalmetrics = EvaluationMetrics()
                    evalmetrics.accuracy = metric["accuracy"]
                    evalmetrics.accuracy_possible = metric["accuracy_possible"]
                    evalmetrics.loss = metric["loss"]
                    evalmetrics.time = metric["time"]
                    evalmetrics.step = metric["step"]
                    evals.metrics.append(evalmetrics)
                return evals
            return dct

        with open(f) as fp:
            return json.load(fp, object_hook=as_evalhistory)
