from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import lib.pyanitools as pya
import os
import copy 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using', device)

import h5py
PATH = 'molecules.h5'
features_list = []
with h5py.File(PATH, 'r') as h5f:
    for mol_key in h5f.keys():
        group = h5f[mol_key]
        mol_data = {key: group[key][()] for key in group}
        for k, v in mol_data.items():
            if isinstance(v, bytes):
                mol_data[k] = v.decode('utf-8')
        features_list.append(mol_data)

print(len(features_list))

y = pd.read_csv('energy_list.csv').values

class CustomDataset(Dataset):

  def __init__(self, features, targets):
    self.features = features
    self.targets = torch.tensor(targets, dtype=torch.float32)

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, index):
    return self.features[index], self.targets[index]
  

def collate_Fn(batch):
    feature_batch, target_batch = zip(*batch)
    return list(feature_batch), torch.tensor(target_batch, dtype=torch.float32)


from torch.utils.data import DataLoader, random_split, Dataset

full_dataset = CustomDataset(features_list, y)

# Split sizes
total_len = len(full_dataset)
train_len = int(0.8 * total_len)
val_len   = int(0.1 * total_len)
test_len  = total_len - train_len - val_len

# Random split
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)  # for reproducibility
)


# Dataloaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

BONDS_DIM, ANGLES_DIM, NONBONDS_DIM, DIHEDRALS_DIM = 17, 27, 17, 38

import torch
import torch.nn as nn
from typing import List                # just for clearer type hints

import copy
from tqdm import tqdm
import torch

# --- Early‑stopping utility (your class, unchanged) ---------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, restore_best_weights=True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.restore    = restore_best_weights

        self.best_loss  = None
        self.best_state = None
        self.counter    = 0

    def __call__(self, model, val_loss) -> bool:
        """
        Returns True  ➜  stop training.
        Returns False ➜  continue.
        """
        if self.best_loss is None or self.best_loss - val_loss >= self.min_delta:
            # improvement
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            # no improvement
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore:
                    model.load_state_dict(self.best_state)
                return True
        return False

class BANDNN(nn.Module):
    """
    Each intra‑molecular interaction type (bond / angle / non‑bond / dihedral)
    is mapped → (N_i, F) → (N_i, 1).  Summing over rows gives the energy
    contribution from that interaction.  The total molecular energy is the sum
    of the four contributions.
    """

    def __init__(
        self,
        bonds_in_dim: int,
        angles_in_dim: int,
        nonbonds_in_dim: int,
        dihedrals_in_dim: int,
        hidden: int = 128,
    ):
        super().__init__()

        def mlp(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),          # ⇒ scalar per row
            )

        self.bonds_model     = mlp(bonds_in_dim)
        self.angles_model    = mlp(angles_in_dim)
        self.nonbonds_model  = mlp(nonbonds_in_dim)
        self.dihedrals_model = mlp(dihedrals_in_dim)

    def _energy_per_type(self, mat: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """
        mat : (N_i, F)  for one molecule & one interaction type
        returns a 0‑D tensor (scalar)
        """
        return net(mat).sum()          # ∑ over rows → scalar

    def forward(
        self,
        bonds_batch:     List[torch.Tensor],
        angles_batch:    List[torch.Tensor],
        nonbonds_batch:  List[torch.Tensor],
        dihedrals_batch: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        All four *_batch arguments are lists of length B, where element i is a
        (N_i, F) tensor.  Returns a tensor of shape (B, 1).
        """
        energies = []
        for bonds, angles, nonbonds, dihedrals in zip(
            bonds_batch, angles_batch, nonbonds_batch, dihedrals_batch
        ):
            e_total = (
                self._energy_per_type(bonds,     self.bonds_model)     +
                self._energy_per_type(angles,    self.angles_model)    +
                self._energy_per_type(nonbonds,  self.nonbonds_model)  +
                self._energy_per_type(dihedrals, self.dihedrals_model)
            )
            energies.append(e_total)

        # list of 0‑D tensors → (B, 1)
        return torch.stack(energies).unsqueeze(1)


# from torchinfo import summary
model = BANDNN(BONDS_DIM, ANGLES_DIM, NONBONDS_DIM, DIHEDRALS_DIM)
model = model.to(device)
# model.summary()
# summary(model)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, items = 0.0, 0
    for feats, targets in loader:
        batch_bonds     = [torch.as_tensor(d["bonds"]).to(device)     for d in feats]
        batch_angles    = [torch.as_tensor(d["angles"]).to(device)    for d in feats]
        batch_nonbonds  = [torch.as_tensor(d["nonbonds"]).to(device)  for d in feats]
        batch_dihedrals = [torch.as_tensor(d["dihedrals"]).to(device) for d in feats]

        y_true = torch.as_tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
        y_pred = model(batch_bonds, batch_angles, batch_nonbonds, batch_dihedrals)

        loss   = criterion(y_pred, y_true)
        bs     = y_true.size(0)
        total += loss.item() * bs
        items += bs
    return total / items


model     = BANDNN(BONDS_DIM, ANGLES_DIM, NONBONDS_DIM, DIHEDRALS_DIM).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

early_stop = EarlyStopping(patience=7, min_delta=0.0001)

epochs = 1000
learning_rate = 0.01

for epoch in range(1, epochs + 1):
    # ─── training ────────────────────────────────────────────────────────────
    model.train()
    train_loss_sum, train_items = 0.0, 0

    for feats, targets in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
      
        batch_bonds     = [torch.as_tensor(d["bonds"]).to(device)     for d in feats]
        batch_angles    = [torch.as_tensor(d["angles"]).to(device)    for d in feats]
        batch_nonbonds  = [torch.as_tensor(d["nonbonds"]).to(device)  for d in feats]
        batch_dihedrals = [torch.as_tensor(d["dihedrals"]).to(device) for d in feats]

        y_true = torch.as_tensor(targets, dtype=torch.float32, device=device).view(-1, 1)

        optimizer.zero_grad()
        y_pred = model(batch_bonds, batch_angles, batch_nonbonds, batch_dihedrals)
        loss   = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        bs              = y_true.size(0)
        train_loss_sum += loss.item() * bs
        train_items    += bs

    train_loss = train_loss_sum / train_items

    # ─── validation ─────────────────────────────────────────────────────────
    val_loss = eval_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch:3d} | train = {train_loss:.4f} | val = {val_loss:.4f}")

    # ─── early stopping check ───────────────────────────────────────────────
    if early_stop(model, val_loss):
        print(f">>> Early-stopping triggered (val-loss = {early_stop.best_loss:.4f}) <<<")
        break

# model already contains the best weights (restored by EarlyStopping)
torch.save(model.state_dict(), "/home2/prathit.chatterjee/Adithya/BANDNN-best.pth")


import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in data_loader:
            feature_dicts, targets = batch

            # Convert each feature set to a list of tensors and send to device
            bond_feat_list     = [torch.tensor(d["bonds"], dtype=torch.float32).to(device) for d in feature_dicts]
            angle_feat_list    = [torch.tensor(d["angles"], dtype=torch.float32).to(device) for d in feature_dicts]
            nonbond_feat_list  = [torch.tensor(d["nonbonds"], dtype=torch.float32).to(device) for d in feature_dicts]
            dihedral_feat_list = [torch.tensor(d["dihedrals"], dtype=torch.float32).to(device) for d in feature_dicts]

            targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1).to(device)

            # Predict
            preds = model(bond_feat_list, angle_feat_list, nonbond_feat_list, dihedral_feat_list)
            loss = criterion(preds, targets)

            total_loss += loss.item() * len(targets)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all predictions and true values
    all_preds = torch.cat(all_preds).squeeze().numpy()
    all_targets = torch.cat(all_targets).squeeze().numpy()
    avg_loss = total_loss / len(data_loader.dataset)

    return all_preds, all_targets, avg_loss

# Evaluate on test set
model.to(device)
preds, true_vals, test_loss = evaluate_model(model, test_loader, device)

# Metrics
r2 = r2_score(true_vals, preds)
mae = mean_absolute_error(true_vals, preds)
rmse = mean_squared_error(true_vals, preds, squared=False)

print(f"Test Loss: {test_loss:.4f} | R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(true_vals, preds, alpha=0.6, color="teal", edgecolors="k")
plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], color="red", linestyle="--")
plt.xlabel("DFT Energy")
plt.ylabel("Predicted Energy")
plt.title("BANDNN Predictions vs Ground Truth")
plt.grid(True)
plt.tight_layout()
plt.savefig('/home2/prathit.chatterjee/Adithya/parityplot.png', dpi=300)
plt.show()
