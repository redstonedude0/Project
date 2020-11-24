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
    for doc in SETTINGS.dataset.documents:
        for mention in doc.mentions:
            cands = mention.candidates
            # Sort p_e_m high to low
            cands.sort(key=lambda cand: -cand.initial_prob)
            if len(cands) > 30:
                # Need to trim to top 30 p_e_m
                cands = cands[0:30]  # Select top 30
            keptCands = cands[0:4]  # Keep top 4 always
            # TODO - assuming no duplicates appear, but duplicates take top 3 spots
            # Keep top 3 eT(Sigw)
            # TODO move some of this to GPU??





    neural.train(model)
    # TODO - return EvaluationMetric object as well as final model?
    return model, None
