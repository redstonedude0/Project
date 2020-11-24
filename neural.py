# This file is for setting up and interfacing directly with the neural model

"""Evaluate the F1 score (and other metrics) of a neural model"""
from datastructures import Model
from hyperparameters import SETTINGS


def evaluate():  # TODO - params
    # TODO - return an EvaluationMetric object (create this class)
    pass


'''Do 1 round of training on a specific dataset and neural model'''


def train(model: Model):  # TODO - add params
    print(vars(SETTINGS))
    for doc in SETTINGS.dataset.documents:
        aggCands = 0
        minCands = 100000
        maxCands = 0
        totCount = 0
        for mention in doc.mentions:
            # print(len(mention.candidates))
            cands = len(mention.candidates)
            aggCands += cands
            totCount += 1
            if cands > maxCands:
                maxCands = cands
            if cands < minCands:
                minCands = cands
        print(f"Document: {doc.id}")
        if totCount != 0:
            avgCands = aggCands / totCount
        else:
            avgCands = "NaN"
        print(f"  {avgCands} [{minCands}-{maxCands}]")

    # TODO - fill in
    pass
