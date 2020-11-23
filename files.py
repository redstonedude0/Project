# all code for working with file I/O
import os

from hyperparameters import SETTINGS

"""
Checks if the datadir is a valid data directory.
Does not check for exact files - only checks for approximate structure.
If folder is not present, will raise error.
If folder does not conform to expected structure, will attempt to download.
"""


def checkDataDir():
    datadir = SETTINGS.dataDir
    if not os.path.isdir(datadir):
        raise NotADirectoryError("Datadir did not point to an existing directory. Please create it before continuing.")
    else:
        contents = os.listdir(datadir)
        if "basic_data" in contents and "generated" in contents:
            return  # valid
        else:
            # TODO - empty the folder
            # TODO - download
            downloadurl = "https://drive.google.com/file/d/1IDjXFnNnHf__MO5j_onw4YwR97oS8lAy/view"
