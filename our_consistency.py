# This file contains the data structures used to represent data throughout the program, including the datasets, processeddata, and output data
import torch
import os.path

TESTING = True
SAVED = {}

""" Save tensor details """
def save(tensor, ident):
    if TESTING:#Only save if testing
        SAVED[ident] = tensor
