import sys
import os
import argparse
from pathlib import Path
import pandas as pd
from .processing import normalize_sequences, one_hot_encode
from .NueralNetwork import ProteinStructurePredictor, train_model, evaluate_model
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim as optim
from .Output import save_results_to_csv, plot_training_metrics, plot_confusion_matrix

# Map DSSP8 characters to numeric labels
DSSP8_Map = {
    'H': 0,  # alpha helix
    'B': 1,  # beta bridge
    'E': 2,  # beta strand
    'G': 3,  # 310 helix
    'I': 4,  # pi helix
    'T': 5,  # turn
    'S': 6,  # bend
    'C': 7,  # coil
    'P': 8,  # unassigned/unknown structure
}

def load_csv_data(csv_file):
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    if 'input' not in df.columns or 'dssp8' not in df.columns:
        raise ValueError("CSV file must contain columns 'input' and 'dssp8'")
    sequences = df['input'].tolist()
    structures = df['dssp8'].tolist()
    return sequences, structures

def pad_or_truncate(labels, target_length):
    if len(labels) > target_length:
        return labels[:target_length]
    return labels + [DSSP8_Map['C']] * (target_length - len(labels))

def map_dssp8_structure(dssp8_string):
    return [DSSP8_Map.get(c, DSSP8_Map['C']) for c in dssp8_string]  

def collate_fn(batch):
    inputs, labels = zip(*batch)

    # Process inputs
    inputs = [torch.tensor(one_hot_encode(normalize_sequences(seq, length=20))).T.float() for seq in inputs]

    # Process labels (convert to lists and ensure consistent length)
    labels = [pad_or_truncate(map_dssp8_structure(label), target_length=20) for label in labels]

    # Debugging: Print the shape and type of processed inputs and labels
    for idx, (input_tensor, label_tensor) in enumerate(zip(inputs, labels)):
        print(f"Input {idx} shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"Label {idx} length: {len(label_tensor)}, data: {label_tensor}")

    # Convert to tensors
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in inputs])

    return inputs, labels, lengths



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict secondary structure of proteins from amino acid sequences")
    parser.add_argument('csv_file', help='Path to input CSV file containing sequences and DSSP8 structures')
    parser.add_argument('output_file', help='Path to output file')
    args = parser.parse_args()
    sequences, raw_structures = load_csv_data(args.csv_file)
    dataset = [(seq, label) for seq, label in zip(sequences, raw_structures)]
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = ProteinStructurePredictor(input_dim=20, hidden_dim=256, num_layers=3, output_dim=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(model=model, dataloader=dataloader, optimizer=optimizer, criterion=criterion, epochs=10)
    test_loss, test_accuracy, predictions = evaluate_model(model, dataloader, criterion)
    save_results_to_csv(predictions, args.output_file, {"test loss": test_loss, "test accuracy": test_accuracy})
   