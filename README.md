# BANDNN_mirror
REF: J. Comp. Chem. 2020, 41, 790-799

Model architecture: bandnn_model.py

Model weights: BANDNN-weights-200425.pth
```
from bandnn_model import BANDNN
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BANDNN(..)
model.load_state_dict(torch.load("BANDNN-weights-200425.pth", map_location=device))
model.to(device)
model.eval()
```

DATASET source: https://drive.google.com/drive/folders/1YiR_p7yZ5POTznWfYADCPmi3UEh0LhNv?usp=drive_link

Training pipeline: BANDNN_mirror.ipynb

Optuna Hyperparameter Tuning: hpt-BANDNN.ipynb (sample code, not executed)
