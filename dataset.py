import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


# def read_smiles(data_path):
#     smiles_data = []
#     with open(data_path) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for i, row in enumerate(csv_reader):
#             smiles = row[-1]
#             smiles_data.append(smiles)
#             #
#         smiles_data.pop(0)
#     return smiles_data

def read_smiles(data_path):
    df = pd.read_csv(data_path, sep=',')
    smiles_data = df['smiles']
    labels_1 = df['k_100']
    labels_2 = df['k_1000']
    labels_3 = df['k_10000']
    # smiles_data = []
    # labels_1 = []
    # labels_2 = []
    # labels_3 = []
    # with open(data_path) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     for i, row in enumerate(csv_reader):
    #         smiles = row[-1]
    #         smiles_data.append(smiles)
    #         label1 = row[1]
    #         label2 = row[2]
    #         label3 = row[3]
    #
    #         labels_1.append(label1)
    #         labels_2.append(label2)
    #         labels_3.append(label3)
    #     smiles_data.pop(0)
    #     labels_1.pop(0)
    #     labels_2.pop(0)
    #     labels_3.pop(0)
    return smiles_data, labels_1,labels_2, labels_3


class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data, self.label1, self.label2, self.label3 = read_smiles(data_path)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        y1 = torch.tensor(self.label1[index], dtype=torch.int64)
        y2 = torch.tensor(self.label2[index], dtype=torch.int64)
        y3 = torch.tensor(self.label3[index], dtype=torch.int64)

        data = Data(x=x, y1=y1, y2=y2, y3=y3, edge_index=edge_index, edge_attr=edge_attr)
        return data

        # random mask a subgraph of the molecule
        # num_mask_nodes = max([1, math.floor(0.25*N)])
        # num_mask_edges = max([0, math.floor(0.25*M)])
        # mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        # mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        # mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        # mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        # mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        # mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]
        #
        # x_i = deepcopy(x)
        # for atom_idx in mask_nodes_i:
        #     x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        # edge_index_i = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        # edge_attr_i = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        # count = 0
        # for bond_idx in range(2*M):
        #     if bond_idx not in mask_edges_i:
        #         edge_index_i[:,count] = edge_index[:,bond_idx]
        #         edge_attr_i[count,:] = edge_attr[bond_idx,:]
        #         count += 1
        # data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)
        #
        # x_j = deepcopy(x)
        # for atom_idx in mask_nodes_j:
        #     x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        # edge_index_j = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        # edge_attr_j = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        # count = 0
        # for bond_idx in range(2*M):
        #     if bond_idx not in mask_edges_j:
        #         edge_index_j[:,count] = edge_index[:,bond_idx]
        #         edge_attr_j[count,:] = edge_attr[bond_idx,:]
        #         count += 1
        # data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        #
        # return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader
