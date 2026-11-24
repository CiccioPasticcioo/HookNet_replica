import sys
import os
from pathlib import Path

utils_fma = os.path.abspath(r"/content/fma-master")

if utils_fma not in sys.path:
    sys.path.append(utils_fma)

import utils

fma_master_location = "INSERT FMA LOCATION"

# Dataset spli in training set, validation set and test set
def datasets_creation():

    tracks = utils.load(f'/{fma_master_location}/fma-master/data/fma_metadata/tracks.csv')

    small_dataset_metadata = tracks.loc[tracks[("set" , "subset")] == "small"].copy()

    train_set = small_dataset_metadata.loc[small_dataset_metadata[('set','split')] == "training"].copy() #provare senza loc

    test_set = small_dataset_metadata.loc[small_dataset_metadata[('set' , 'split')] == 'test'].copy()

    validation_set = small_dataset_metadata.loc[small_dataset_metadata[('set' , 'split')] == 'validation'].copy()

    return train_set, test_set, validation_set

train, test, validation = datasets_creation()

def get_audio_path(track_id):
    return Path(utils.get_audio_path(str(f"/{fma_master_location}/fma-master/data/fma_small"), track_id))
