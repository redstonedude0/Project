# This file contains the data structures used to represent data throughout the program, including the datasets, processeddata, and output data
import json
from typing import List

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
