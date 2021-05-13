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

    def __init__(self, id, initial_prob, text):
        self.id = id
        self.initial_prob = initial_prob
        self.text = text

    def __repr__(self):
        return f"Candidate({self.id},{self.initial_prob},\"{self.text}\")"

    def ent_embedding(self) -> np.ndarray:
        # If unseen then give average embedding (https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/utils.py line 104)
        # TODO discuss this
        return processeddata.ent_id_to_embedding[processeddata.ent_to_ent_id.get(self.text, processeddata.unk_ent_id)]

    def ent_embedding_torch(self) -> torch.Tensor:
        # TODO what happens if an ent isn't seen before?
        return torch.from_numpy(self.ent_embedding()).to(SETTINGS.device).to(
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
    conll_lctx: List[str] = None# Optional conll data
    conll_mctx: List[str] = None# Optional conll data
    conll_rctx: List[str] = None# Optional conll data

    def from_data(id, text, left_context, right_context, candidates, gold_id):
        self = Mention()
        self.id = id
        self.text = text
        self.left_context = left_context
        self.right_context = right_context
        self.candidates = candidates
        self.gold_id = gold_id
        self.conll_lctx = []
        self.conll_mctx = []
        self.conll_rctx = []
        return self

    def gold_cand(self):
        for cand in self.candidates:
            if cand.id == self.gold_id:
                return cand
        return None

    def gold_cand_index(self):
        for cand_idx, cand in enumerate(self.candidates):
            if cand.id == self.gold_id:
                return cand_idx
        return -1

    def word_embedding(self):
        return processeddata.word_id_to_embedding[processeddata.word_to_word_id.get(self.text, processeddata.unk_word_id)]

    def __repr__(self):
        return f"Mention.from_data({self.id},\"{self.text}\",\"{self.left_context}\",\"{self.right_context}\",{self.candidates},\"{self.gold_id}\")"


class Document:
    id: str = None  # Document ID
    mentions: List[Mention] = None

    def from_data(id, mentions):
        self = Document()
        self.id = id
        self.mentions = mentions
        return self

    def __repr__(self):
        # TODO - if any string contains " then it will cause errors recreating
        return f"Document.from_data(\"{self.id}\",{self.mentions})"


"""
Represents a dataset from the generated data
"""


class Dataset:
    # We can determine the structure by looking at the data files and how the original implementation uses them
    documents: List[Document] = None


from hyperparameters import SETTINGS
"""Represents a model (nn parameters, etc)"""


class Model:
    neural_net = None
    evals: 'EvalHistory' = None

    def save(self, name):
        print(f"Saving checkpoint '{name}'...")
        torch.save(self.neural_net, SETTINGS.data_dir_checkpoints + name + ".pt")
        self.evals.save(SETTINGS.data_dir_checkpoints + name + ".evals")
        print("Saved.")

    @classmethod
    def load(cls, name):
        inst = cls()
        inst.neural_net = torch.load(SETTINGS.data_dir_checkpoints + name + ".pt")
        inst.evals = EvalHistory.load(SETTINGS.data_dir_checkpoints + name + ".evals")
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
                    eval_metrics = EvaluationMetrics()
                    eval_metrics.accuracy = metric["accuracy"]
                    eval_metrics.accuracy_possible = metric["accuracy_possible"]
                    eval_metrics.loss = metric["loss"]
                    eval_metrics.time = metric["time"]
                    eval_metrics.step = metric["step"]
                    evals.metrics.append(eval_metrics)
                return evals
            return dct

        with open(f) as fp:
            return json.load(fp, object_hook=as_evalhistory)
