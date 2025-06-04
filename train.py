from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(
    features_list, y, test_size=0.2, random_state=42
)



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
    
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32,collate_fn=collate_Fn, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_Fn, shuffle=False, pin_memory=True)

BONDS_DIM, ANGLES_DIM, NONBONDS_DIM, DIHEDRALS_DIM = 17, 27, 17, 38

torch.manual_seed(42)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False

class BANDNN(nn.Module):

    def __init__(self, bonds_input_dim, angles_input_dim, nonbonds_input_dim, dihedral_input_dim):
        super().__init__()
        self.bonds_model = nn.Sequential(
            nn.Linear(bonds_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.angles_model = nn.Sequential(
            nn.Linear(angles_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.nonbonds_model = nn.Sequential(
            nn.Linear(nonbonds_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.dihedrals_model = nn.Sequential(
            nn.Linear(dihedral_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, bonds_input, angles_input, non_bonds_input, dihedrals_input):
        bonds_energy = self.bonds_model(bonds_input).sum()
        angles_energy = self.angles_model(angles_input).sum()
        nonbonds_energy = self.nonbonds_model(non_bonds_input).sum()
        dihedrals_energy = self.dihedrals_model(dihedrals_input).sum()

        total_energy = bonds_energy + angles_energy + nonbonds_energy + dihedrals_energy
        return total_energy

# from torchinfo import summary
model = BANDNN(BONDS_DIM, ANGLES_DIM, NONBONDS_DIM, DIHEDRALS_DIM)
model = model.to(device)
# model.summary()
# summary(model)

epochs = 9
learning_rate = 0.01

# loss func
criterion = nn.MSELoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

check_point_epochs = [10, 20, 30, 40, 50, 60]
done = False
es = EarlyStopping()


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()  # Set model to eval mode
    total_loss = 0
    total_samples = 0

    predictions = []
    targets_list = []

    criterion = torch.nn.MSELoss()

    with torch.no_grad():  # No gradients needed
        for batch in test_loader:
            features_list, targets = batch

            for feature, target in zip(features_list, targets):
                # Convert and move feature components to device
                bonds = torch.stack([torch.tensor(b, dtype=torch.float32) for b in feature['bonds']]).to(device)
                angles = torch.stack([torch.tensor(a, dtype=torch.float32) for a in feature['angles']]).to(device)
                nonbonds = torch.stack([torch.tensor(n, dtype=torch.float32) for n in feature['nonbonds']]).to(device)
                dihedrals = torch.stack([torch.tensor(d, dtype=torch.float32) for d in feature['dihedrals']]).to(device)

                target = torch.tensor(target, dtype=torch.float32).to(device)

                # Get model output
                output = model(bonds, angles, nonbonds, dihedrals)

                # Compute loss
                loss = criterion(output, target)
                total_loss += loss.item()
                total_samples += 1

                predictions.append(output.item())
                targets_list.append(target.item())

    avg_loss = total_loss / total_samples

    print(f"Evaluation MSE Loss: {avg_loss:.4f}")
    print(f"Evaluation R2 Score: {r2_score(targets_list, predictions):.4f}")
    return predictions, targets_list, avg_loss


while epoch < 1000 and not done:
    epoch += 1
    steps = list(enumerate(train_loader))
    pbar = tqdm(steps)
# for epoch in range(0, 60):
#     print(f'Epoch {epoch + 1}')
#     total_epoch_loss = 0
#     num_samples = 0
#     tq_loader = tqdm(train_loader)

    for i, (batch) in pbar:

        for feature_dict, target in zip(*batch):

            bond_feat = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in feature_dict['bonds']]).to(device)
            angle_feat = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in feature_dict['angles']]).to(device)
            nonbond_feat = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in feature_dict['nonbonds']]).to(device)
            dihedral_feat = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in feature_dict['dihedrals']]).to(device)
            energy_feat = torch.tensor([target], dtype=torch.float32).to(device)

            optimizer.zero_grad()

            outputs = model(bond_feat, angle_feat, nonbond_feat, dihedral_feat)

            loss = criterion(outputs, energy_feat)
            loss.backward()

            optimizer.step()
            loss, current = loss.item(), (i + 1) * len(batch)
            if i == len(steps) - 1:
                predictions, targets_list, avg_loss = evaluate_model(model=model, test_loader=test_loader)
                model.eval()
                # pred = model(test_loader).flatten()
                # vloss = criterion(pred, y_test)
                if es(model, avg_loss):
                    done = True
                pbar.set_description(
                    f"Epoch: {epoch}, tloss: {loss}, vloss: {avg_loss:>7f}, EStop:[{es.status}]"
                )
            else:
                pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")

            num_samples+=1
            total_epoch_loss += loss.item()


        avg_loss = total_epoch_loss / num_samples

        if epoch in check_point_epochs:

            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,

                    }, f'BANDNN-chekpoint_epoch_{epoch}.pth')
    tq_loader.set_description(
        "Epoch: " + str(epoch + 1) + "  Training loss: " + str(avg_loss))
    print(f'Average epoch Loss: {avg_loss:.4f}')
        



torch.save(model.state_dict(), 'BANDNN-weights-260425-1.pth')


model.eval()

from sklearn.metrics import r2_score

model.to(device)
preds, true_vals, test_loss = evaluate_model(model, test_loader, device)

import matplotlib.pyplot as plt

plt.scatter(true_vals, preds)
plt.xlabel("DFT Energy")
plt.ylabel("Predicted Energy")
plt.title("BANDNN Predictions vs Ground Truth")
plt.savefig('parityplot.png')
plt.show()