# This file is for handling the model - it's resonsible for applying most hyperparameters and performing training

"""Do all the training on a specific dataset and neural model"""
import neural
from datastructures import Model


def trainToCompletion():  # TODO - add params
    # TODO - fill in
    # TODO - checkpoint along the way
    # TODO - return EvaluationMetric object as well as final model?
    model = Model()
    neural.train(model)
    return model, None
