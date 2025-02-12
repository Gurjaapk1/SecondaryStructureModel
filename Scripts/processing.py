from Bio import SeqIO
import os
import numpy as np
import torch
import torch.nn as nn

def normalize_sequences(sequences, length=20, pad_char='X'):
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    normalized_sequences = []
    for seq in sequences:
        if not isinstance(seq, str):
            seq = ''.join(seq)
        seq = ''.join([aa if aa in valid_chars else pad_char for aa in seq])
        if len(seq) < length:
            seq = seq + pad_char * (length - len(seq))
        else:
            seq = seq[:length]
        normalized_sequences.append(seq)
    return normalized_sequences

def one_hot_encode(sequence, length=20):
    Amino_Acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = {aa: i for i, aa in enumerate(Amino_Acids)}
    one_hot = []

    for aa in sequence:
        vector = [0] * len(Amino_Acids)  # Default to zeros for invalid characters
        if aa in encoding:
            vector[encoding[aa]] = 1
        one_hot.append(vector)

    # Ensure the sequence is truncated or padded to the required length
    if len(one_hot) < length:
        one_hot.extend([[0] * len(Amino_Acids)] * (length - len(one_hot)))
    else:
        one_hot = one_hot[:length]

    return one_hot



