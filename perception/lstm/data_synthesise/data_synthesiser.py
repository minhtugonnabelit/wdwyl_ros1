import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):

    DATA_DIR = '/root/aifr/wdwyl_ros1/perception/lstm/data_synthesise'

    def __init__(self, sequence_length, num_sequences = 1000):
        """
        Args:
            sequence_length (int): Length of each sequence.
            num_sequences (int): Number of sequences in the dataset.
        """
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.save_path = SyntheticDataset.DATA_DIR + f'/all_synthetic_data_{self.sequence_length}_x{self.num_sequences}.npz'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_synthetic_data()


    def load_synthetic_data(self):
        if not os.path.exists(self.save_path):
            print('Synthetic data does not exist. Generate...')
            self.generate_synthetic_data()

        data = np.load(self.save_path)
        self.data = torch.tensor(data['synthetic_data'], dtype=torch.float32).to('cuda:0')
        self.labels = torch.tensor(data['labels'], dtype=torch.float32).to('cuda:0')


    def generate_synthetic_data(self):
        """
        Generates synthetic data in the shape [num_sequences, sequence_length].
        Each element in the sequence is 0 or 1. The label is 1 if there is at least one '1' in the sequence, otherwise 0.
        """

        if os.path.exists(self.save_path):
            print('Synthetic data already exists.')

        num_false_detection = 5
        num_each_class = self.num_sequences // 2

        synthetic_data = []
        labels = np.concatenate((np.ones(num_each_class), np.zeros(num_each_class)))

        positive_data = np.ones(self.sequence_length)        
        for _ in range(num_each_class):
            num_zeros = np.random.randint(0, num_false_detection)
            data = positive_data.copy()
            for _ in range(num_zeros):
                idx = np.random.randint(self.sequence_length)
                data[idx] = 0
            synthetic_data.append(data)

        negative_data = np.zeros(self.sequence_length)
        for _ in range(num_each_class):
            num_ones = np.random.randint(0, num_false_detection)
            data = negative_data.copy()
            for _ in range(num_ones):
                idx = np.random.randint(self.sequence_length)
                data[idx] = 1
            synthetic_data.append(data)

        np.savez(self.save_path, synthetic_data=synthetic_data, labels=labels)


    def __len__(self):
        return self.num_sequences - 1
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]
        return sequence, label

def inspect_dataloader(dataloader, num_batches=5):
    for i, (sequences, labels) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print("Sequences:", sequences)
        print("Labels:", labels)
        print("Labels shape:", labels.shape)
        if i >= num_batches - 1:  # Stop after inspecting the desired number of batches
            break

if __name__ == '__main__':

    sequence_length = 20
    num_sequences = 1000
    batch_size = 1

    dataset = SyntheticDataset(sequence_length, num_sequences)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    inspect_dataloader(data_loader, num_batches=5)

