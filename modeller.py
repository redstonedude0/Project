# This file is for handling the model - it's resonsible for applying most hyperparameters and performing training

"""Do all the training on a specific dataset and neural model"""
import neural
from datastructures import Model
from hyperparameters import SETTINGS


def trainToCompletion():  # TODO - add params
    # TODO - checkpoint along the way

    model = Model()
    # TODO - select top 30 candidates per mention
    # TODO - keep top 4 p_e_m and top 3 et
    for doc in SETTINGS.dataset:
        pass



    neural.train(model)
    # TODO - return EvaluationMetric object as well as final model?
    return model, None
