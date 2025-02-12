#Build the neural network model using PyTorch
import torch
import torch.nn as nn
import os
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ProteinStructurePredictor(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, num_layers=3, output_dim=9, dropout=0.3):
        super(ProteinStructurePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.fc(out)
        return out

def masked_cross_entropy_loss(outputs, labels, lengths):
    outputs = outputs.view(-1, outputs.size(-1))
    labels = labels.view(-1)
    mask = torch.arange(labels.size(1), device = labels.device).expand(len(lengths), -1) < lengths.unsqueeze(1)
    mask = mask.view(-1)

    outputs = outputs[mask]
    labels = labels[mask]

    return nn.functional.cross_entropy(outputs, labels)

def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    all_losses = []
    all_accuracies = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels, lengths in dataloader:
            optimizer.zero_grad()

            # Debugging
            print(f"Debug: Inputs shape: {inputs.shape}")
            print(f"Debug: Labels shape: {labels.shape}")
            print(f"Debug: Lengths: {lengths}")

            outputs = model(inputs, lengths)
            loss = masked_cross_entropy_loss(outputs, labels, lengths)
            print(f"Debug: Model outputs shape: {outputs.shape}")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.view(-1, outputs.size(-1)).max(1)
            valid_labels = labels.view(-1)[:predicted.size(0)]

            # Debugging
            print(f"Debug: Predicted shape: {predicted.shape}, Valid labels shape: {valid_labels.shape}")

            correct += (predicted == valid_labels).sum().item()
            total += valid_labels.size(0)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {correct / total:.4f}")
        all_losses.append(total_loss)
        all_accuracies.append(correct / total)

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            outputs = model(inputs, lengths)
            loss = masked_cross_entropy_loss(outputs, labels, lengths)
            total_loss += loss.item()

            _, predicted = outputs.view(-1, outputs.size(-1)).max(1)
            valid_labels = labels.view(-1)[:predicted.size(0)]
            correct += (predicted == valid_labels).sum().item()
            total += valid_labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())

            torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy()) 

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    print(f"Test Loss = ({avg_loss}), Accuracy = ({accuracy})")
    return avg_loss, accuracy, all_predictions  


