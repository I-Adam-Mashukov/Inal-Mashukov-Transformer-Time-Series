### Written by Dr. Inal Adam Mashukov.
# MIT LICENSE
### General Documentation:
Transformer model suitable for classification, regression, other tasks.  
- `config/config.json`: contains parameters for training and model tuning.
- `data/`: directory where `.npy` data must reside, where `data` has dimensions `(num_seq, seq_len, n_features)`.
- `experiments/`: directory where model outpus (logs, weights) will be saved.
- `src/`: source directory containing:
  - `models/`: Transformer model source code.
  - `utils/`: modularized utils for loading, preprocessing data, evaluating the model.
- `train.py`: modularized training script.
