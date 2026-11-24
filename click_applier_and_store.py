from pathlib import Path
import numpy as np
from click_generator import click_adder, ground_truth, SEGMENT_LENGTH
import librosa
import fma_data_estraction as de
import soundfile as sf

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Click injection in tracks and creation of labels. Every information about the waveform saved in cache
def cache(tracks):

    CACHE_DIR.mkdir(parents = True, exist_ok = True)

    for track_id in tracks.index:

        track_path = de.get_audio_path(track_id)

        print(f"Downloading: {track_id}: {track_path}", flush=True)

        try:
            audio, sr = librosa.load(track_path, sr = 44100, mono = True)
        except sf.LibsndfileError:
            print(f"Unable to load: track {track_id} skipped")
            continue

        np.random.seed(42 + int(track_id))

        degraded_signal, click_positions = click_adder(audio)

        n_segments = len(degraded_signal) // SEGMENT_LENGTH
            
        degraded = degraded_signal[:n_segments * SEGMENT_LENGTH]

        
        X = degraded.reshape(n_segments, SEGMENT_LENGTH) 
        X = X.astype(np.float32) 
    
        Y = ground_truth(click_positions, n_segments) 

        np.save(CACHE_DIR / f"{track_id}_audio.npy", X, allow_pickle = False)
        np.save(CACHE_DIR / f"{track_id}_labels.npy", Y, allow_pickle = False)

        del audio, degraded_signal, degraded, X, Y, click_positions
