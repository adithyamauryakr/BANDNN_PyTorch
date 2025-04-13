from lib import pyanitools as pya
from rdkit import Chem
from rdkit.Chem import AllChem
from featurizer import *

import h5py
import numpy as np
from tqdm import tqdm

hdf5file = f'/content/drive/MyDrive/BANDNN_proj/ANI-1_release/ani_gdb_s08.h5'
adl = pya.anidataloader(hdf5file)

# get bond connectivity list from smiles
def get_bond_connectivity_list(mol):
    mol = Chem.MolFromSmiles(mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    bond_connectivity_list = [[] for _ in range(mol.GetNumAtoms())]

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bond_connectivity_list[a1].append(a2)
        bond_connectivity_list[a2].append(a1)

    return bond_connectivity_list


elements = {'C', 'H', 'N', 'O', 'c', 'n', 'o'}
features_list = []
smiles_list = []
energy_list = []
coordinates_list = []
species_list = []
whole_bond_connectivity_list = []
counter = 0

for data in tqdm(adl):

  # Extract the data
  P = data['path']
  X = data['coordinates']
  E = data['energies']
  sm = data['smiles']

  sm = str.join('', sm)

  bond_connectivity_list = get_bond_connectivity_list(sm)
  species = [atom.upper() for atom in sm if atom in elements]

  for energy, coordinates in zip(E, X):
    if counter == 100001:
      done = True
      break
    try:
      coordinates_list.append(coordinates)
      species_list.append(species)
      energy_list.append(energy)
      smiles_list.append(sm)
      whole_bond_connectivity_list.append(bond_connectivity_list)
      counter+=1
    except:
      with open('probmoles.txt', 'a') as prob:
        prob.write(f'problem with molecule {counter} \n')
      counter+=1

      continue
  if done:
    break
# Closes the H5 data file
adl.cleanup()


for coordinates, bond_connectivity_list, species in tqdm(zip(coordinates_list, whole_bond_connectivity_list, species_list)):
  features = get_features(conformer=coordinates, bond_connectivity_list=bond_connectivity_list,S=species)
  features_list.append(features)

# save features to h5 file
with h5py.File('data/molecules.h5', 'w') as h5f:
    for i, entry in enumerate(features_list):
        group = h5f.create_group(f"mol_{i}")
        for key, value in entry.items():
            # Save strings as fixed-length byte strings
            if isinstance(value, str):
                dt = h5py.string_dtype(encoding='utf-8')
                group.create_dataset(key, data=value, dtype=dt)
            else:
                group.create_dataset(key, data=value)
