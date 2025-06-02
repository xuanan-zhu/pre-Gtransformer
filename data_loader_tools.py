
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset

from collections import defaultdict
import torch


def one_hot_encode0(snp):
    encoding_table = {
        'AA0': 0, 'AT0': 0, 'AC0': 0, 'AG0': 0,
        'TA0': 0, 'TT0': 0, 'TC0': 0, 'TG0': 0,
        'CA0': 0, 'CT0': 0, 'CC0': 0, 'CG0': 0,
        'GA0': 0, 'GT0': 0, 'GC0': 0, 'GG0': 0,
        'AT1': 1, 'AC1': 1, 'AG1': 1,
        'TA1': 1, 'TC1': 1, 'TG1': 1,
        'CA1': 1, 'CT1': 1, 'CG1': 1,
        'GA1': 1, 'GT1': 1, 'GC1': 1,
        'AT2': 2, 'AC2': 2, 'AG2': 2,
        'TA2': 2, 'TC2': 2, 'TG2': 2,
        'CA2': 2, 'CT2': 2, 'CG2': 2,
        'GA2': 2, 'GT2': 2, 'GC2': 2
    }

    num_classes = 3
    encoded_snp = np.zeros(num_classes)
    if snp in encoding_table:
        encoded_snp[encoding_table[snp]] = 1
    return encoded_snp

def one_hot_encode2(snp):
    encoding_table = {
        'AA0': 0, 'AT0': 1, 'AC0': 2, 'AG0': 3,
        'TA0': 4, 'TT0': 5, 'TC0': 6, 'TG0': 7,
        'CA0': 8, 'CT0': 9, 'CC0': 10, 'CG0': 11,
        'GA0': 12, 'GT0': 13, 'GC0': 14, 'GG0': 15,
        'AT1': 16, 'AC1': 17, 'AG1': 18,
        'TA1': 19, 'TC1': 20, 'TG1': 21,
        'CA1': 22, 'CT1': 23, 'CG1': 24,
        'GA1': 25, 'GT1': 26, 'GC1': 27,
        'AT2': 28, 'AC2': 29, 'AG2': 30,
        'TA2': 31, 'TC2': 32, 'TG2': 33,
        'CA2': 34, 'CT2': 35, 'CG2': 36,
        'GA2': 37, 'GT2': 38, 'GC2': 39
    }

    num_classes = len(encoding_table)
    encoded_snp = np.zeros(num_classes)
    if snp in encoding_table:
        encoded_snp[encoding_table[snp]] = 1
    return encoded_snp


def rename_numbers(input_list):
    unique_numbers = list(set(input_list))  # Get the unique number in the list
    unique_numbers.sort()  # Sorting unique numbers

    number_to_index = {num: index for index, num in enumerate(unique_numbers)}  # Creating a number-to-index mapping

    renamed_list = [number_to_index[num] for num in input_list]  # Renaming numbers using mapping
    return renamed_list


def data_xy_transform_to_tensor(data, chr_num=19, per_chromosome=2816):
    # Extract string data and labels
    snp_data = data[:, :-1]
    index = (chr_num - 1) * per_chromosome
    snp_data = snp_data[:, index:(index + per_chromosome)]
    snp_data = snp_data.tolist()
    labels = data[:, -1].astype(int)
    labels = labels.tolist()
    labels = rename_numbers(labels)
    # Create a new NumPy array to store the encoded data
    encoded_snp_data = []
    # Encodes the entire SNP data array and stores the result in encoded_snp_data
    for i in range(len(snp_data)):
        encoded_snp_data.append([one_hot_encode2(snp) for snp in snp_data[i]])
    encoded_snp_data = np.array(encoded_snp_data)
    x_data = torch.from_numpy(encoded_snp_data)
    y_data = torch.tensor(labels)
    return x_data, y_data


def get_indices(chr_num, indices):
    row = indices[indices.Chromosome == chr_num].iloc[0]
    return row.Start_Index, row.End_Index

def data_xy_transform_to_tensor_with_indices(data, start_index, end_index):
    # Extract string data and labels
    snp_data = data[:, :-1]
    snp_data = snp_data[:, start_index:end_index+1]
    snp_data = snp_data.tolist()
    labels = data[:, -1].astype(int)
    labels = labels.tolist()
    labels = rename_numbers(labels)
    # Create a new NumPy array to store the encoded data
    encoded_snp_data = []
    # Encodes the entire SNP data array and stores the result in encoded_snp_data
    for i in range(len(snp_data)):
        encoded_snp_data.append([one_hot_encode2(snp) for snp in snp_data[i]])
    encoded_snp_data = np.array(encoded_snp_data)
    x_data = torch.from_numpy(encoded_snp_data)
    y_data = torch.tensor(labels)
    return x_data, y_data

def data_xy_transform_to_tensor_all(data):
    # Extract string data and labels
    snp_data = data[:, :-1]
    snp_data = snp_data.tolist()
    labels = data[:, -1].astype(int)
    labels = labels.tolist()
    labels = rename_numbers(labels)
    # Create a new NumPy array to store the encoded data
    encoded_snp_data = []
    # Encodes the entire SNP data array and stores the result in encoded_snp_data
    for i in range(len(snp_data)):
        encoded_snp_data.append([one_hot_encode2(snp) for snp in snp_data[i]])
    encoded_snp_data = np.array(encoded_snp_data)
    x_data = torch.from_numpy(encoded_snp_data)
    y_data = torch.tensor(labels)
    return x_data, y_data

def data_xy_transform_to_tensor_all0(data):
    # Extract string data and labels
    snp_data = data[:, :-1]
    snp_data = snp_data.tolist()
    labels = data[:, -1].astype(int)
    labels = labels.tolist()
    labels = rename_numbers(labels)
    # Create a new NumPy array to store the encoded data
    encoded_snp_data = []
    # Encodes the entire SNP data array and stores the result in encoded_snp_data
    for i in range(len(snp_data)):
        encoded_snp_data.append([one_hot_encode0(snp) for snp in snp_data[i]])
    encoded_snp_data = np.array(encoded_snp_data)
    x_data = torch.from_numpy(encoded_snp_data)
    y_data = torch.tensor(labels)
    return x_data, y_data


def data_pos_transform_to_tensor(pos, chr_num=19, per_chromosome=2816):
    print("pos:", chr_num)
    pos_data = pos[:, 1].astype(int)

    index = (chr_num - 1) * per_chromosome
    pos_data = pos_data[index:(index + per_chromosome)]

    pos_data = torch.from_numpy(pos_data)
    pos_data = pos_data.unsqueeze(1)

    return pos_data

def data_pos_transform_to_tensor_with_indices(pos, start_index, end_index):

    pos_data = pos[:, 1].astype(int)

    pos_data = pos_data[start_index:end_index+1]

    pos_data = torch.from_numpy(pos_data)
    pos_data = pos_data.unsqueeze(1)

    return pos_data

def data_chr_transform_to_tensor_with_indices(pos, start_index, end_index):

    chr_data = pos[:, 0].astype(int)

    chr_data = chr_data[start_index:end_index+1]

    chr_data = torch.from_numpy(chr_data)
    chr_data = chr_data.unsqueeze(1)

    return chr_data

def data_pos_transform_to_tensor_all(pos):

    pos_data = pos[:, 1].astype(int)

    pos_data = torch.from_numpy(pos_data)
    pos_data = pos_data.unsqueeze(1)

    return pos_data

def data_chr_transform_to_tensor_all(pos):

    chr_data = pos[:, 0].astype(int)

    chr_data = torch.from_numpy(chr_data)
    chr_data = chr_data.unsqueeze(1)

    return chr_data


import torch
import numpy as np
from torch.utils.data import TensorDataset, Subset, DataLoader
from collections import defaultdict

def createValTestDataset(x_data, y_data, batch_size=42, seed=1, val_ratio=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(x_data, y_data)
    labels = y_data.numpy()
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1

    val_samples_per_class = {}
    test_samples_per_class = {}
    for label, count in label_counts.items():
        val_samples = int(count * val_ratio)
        test_samples = count - val_samples
        val_samples_per_class[label] = val_samples
        test_samples_per_class[label] = test_samples

    val_indices = []
    test_indices = []
    for label in label_counts:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
       
        val_count = val_samples_per_class[label]
        val_indice = label_indices[:val_count]
        test_indice = label_indices[val_count:]
    
        val_indices.extend(val_indice.tolist())
        test_indices.extend(test_indice.tolist())

 
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    print("val_indices:", val_indices)
    print("test_indices:", test_indices)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return val_loader, test_loader

def createValTestDataset_dy(x_data, y_data, batch_size=42, seed=1, val_ratio=0.5, collate_fn=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(x_data, y_data)
    labels = y_data.numpy()
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1


    val_samples_per_class = {}
    test_samples_per_class = {}
    for label, count in label_counts.items():
        val_samples = int(count * val_ratio)
        test_samples = count - val_samples
        val_samples_per_class[label] = val_samples
        test_samples_per_class[label] = test_samples

    val_indices = []
    test_indices = []
    for label in label_counts:

        label_indices = np.where(labels == label)[0]

        np.random.shuffle(label_indices)

        val_count = val_samples_per_class[label]
        val_indice = label_indices[:val_count]
        test_indice = label_indices[val_count:]

        val_indices.extend(val_indice.tolist())
        test_indices.extend(test_indice.tolist())


    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    print("val_indices:", val_indices)
    print("test_indices:", test_indices)


    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return val_loader, test_loader

def createXDataset(x_data, y_data, batch_size=42, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(x_data, y_data)
    labels = y_data.numpy()


    label1_indices = np.where(labels == 1)[0]
    label0_indices = np.where(labels == 0)[0]


    label1_test_size = len(label1_indices) // 2
    label1_test_indices = np.random.choice(label1_indices, label1_test_size, replace=False)
    label1_remaining = np.setdiff1d(label1_indices, label1_test_indices)


    train_val_indices = np.concatenate([label1_remaining, label0_indices])
    train_val_labels = labels[train_val_indices]


    label1_train_val = train_val_indices[train_val_labels == 1]
    label0_train_val = train_val_indices[train_val_labels == 0]


    label1_train_size = int(len(label1_train_val) * 0.8)
    label0_train_size = int(len(label0_train_val) * 0.8)


    label1_train = np.random.choice(label1_train_val, label1_train_size, replace=False)
    label0_train = np.random.choice(label0_train_val, label0_train_size, replace=False)


    label1_val = np.setdiff1d(label1_train_val, label1_train)
    label0_val = np.setdiff1d(label0_train_val, label0_train)


    train_indices = np.concatenate([label1_train, label0_train]).tolist()
    val_indices = np.concatenate([label1_val, label0_val]).tolist()
    test_indices = label1_test_indices.tolist()


    def print_distribution(name, indices):
        class0 = (y_data[indices] == 0).sum().item()
        class1 = (y_data[indices] == 1).sum().item()
        print(f"{name.ljust(8)}: 0: {class0}  | 1: {class1} ")

    print_distribution("train set", train_indices)
    print_distribution("val set", val_indices)
    print_distribution("test set", test_indices)


    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from collections import defaultdict


def create2Dataset(x_data, y_data, batch_size=42, seed=1, train_ratio=0.8, val_ratio=0.2,
                     datasetX=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(x_data, y_data)
    labels = y_data.numpy()
    # Counting the number of samples in each category
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1

    # Calculate the number of samples for each category in the training and validation sets
    # train_ratio = 0.6
    # val_ratio = 0.2
    # test_ratio = 0.2
    train_samples_per_class = {label: int(count * train_ratio) for label, count in label_counts.items()}
    val_samples_per_class = {label: int(count * val_ratio) for label, count in label_counts.items()}

    # Sample indexes in each category
    train_indices = []
    val_indices = []
    for label in label_counts:
        label_indices = np.where(labels == label)[0]
        i = datasetX
        n = train_samples_per_class[label] + val_samples_per_class[label]
        label_indices = np.random.choice(label_indices, n, replace=False)
        val_indice = label_indices[int(i * val_ratio * n):int((i + 1) * val_ratio * n)]
        train_indice1 = label_indices[:int(i * val_ratio * n)]
        train_indice2 = label_indices[int((i + 1) * val_ratio * n):]

        train_indices.extend(train_indice1)
        train_indices.extend(train_indice2)
        val_indices.extend(val_indice)

    # Creating subdatasets and data loaders
    print("train_indices:",train_indices)
    print("val_indices:", val_indices)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def createTestLoader(test_x_data, test_y_data, batch_size=42):

    test_dataset = TensorDataset(test_x_data, test_y_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader








