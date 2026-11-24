# HookNet Replica

A PyTorch implementation inspired by [Audio Defect Detection in Music with Deep Networks](https://archives.ismir.net/ismir2021/paper/000095.pdf) by D. Wolff, R. Mignot, and A. Roebel. The model detects and localizes synthetic clicks in music from FMA Small. For each 16,384-sample segment, it predicts a click score for each of 128 time intervals.

## Code

| File | Purpose |
| --- | --- |
| `fma_data_estraction.py` | Loads the FMA Small splits and resolves audio file paths. |
| `click_generator.py` | Injects synthetic clicks and generates binary labels for each time interval. |
| `click_applier_and_store.py` | Splits tracks into segments and saves the audio and labels as `.npy` files. |
| `data_extractor.py` | Loads cached segments and labels as a PyTorch `Dataset`. |
| `dataloaders.py` | Creates training, validation, and test batches of eight segments. |
| `HookNet.py` | Defines the 1D encoder–decoder network and its skip connections. |
| `train.py` | Trains and validates HookNet, saves model weights, and plots loss curves. |
| `test_nn.py` | Evaluates saved weights at a threshold of 0.5 and reports precision, recall, F1-score, and a confusion matrix. |

`click_generator.py` contains two definitions of `click_adder`. Python uses the second one, which generates clicks lasting 1–3 samples.
