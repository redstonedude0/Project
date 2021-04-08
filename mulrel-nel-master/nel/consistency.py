# This file contains the data structures used to represent data throughout the program, including the datasets, processeddata, and output data
import torch
import os.path

""" Save tensor details """
def save(tensor, ident):
    if False:#Bypass for testing
        return
    path = f"/rds/user/hrjh2/hpc-work/consistency/mrn_{ident}.pt"
    if not os.path.isfile(path):
        torch.save(tensor,path)
    else:
        print(f"Tensor '{ident}' already saved.")