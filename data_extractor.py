import torch
from torch.utils.data import Dataset
import numpy as np
from click_applier_and_store import CACHE_DIR

# Creation of the cache where the information about the waveform audio and label will be stored
class click_dataset(Dataset):

    def __init__(self, track_set):

        super().__init__()

        self.data = []

        for track_id in track_set.index:

            audio_npy = CACHE_DIR / f"{track_id}_audio.npy"
            labels_path = CACHE_DIR / f"{track_id}_labels.npy"

            if not audio_npy.exists() or not labels_path.exists():
                print(f"AUDIO {track_id} DOESN'T EXIST")
                continue

            X = np.load(audio_npy, mmap_mode = 'r', allow_pickle = False)
            Y = np.load(labels_path, mmap_mode = 'r', allow_pickle = False)

            n_segments = len(X)

            for i in range(n_segments):
                self.data.append((track_id, i))

            del X, Y

        if len(self.data) == 0:
            
            raise ValueError("EMPTY DATASET")            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        track_id, segment_n = self.data[index]

        X = np.load(CACHE_DIR/f"{track_id}_audio.npy", mmap_mode = 'r', allow_pickle = False)
        Y = np.load(CACHE_DIR/f"{track_id}_labels.npy", mmap_mode = 'r', allow_pickle = False)

        x = X[segment_n].copy()
        y = Y[segment_n].copy()

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y

