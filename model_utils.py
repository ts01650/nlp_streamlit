import torch
import numpy as np
from model_architecture import BiGRUCRFClass  # This should match your model file name

def load_model(model_path):
    embedding_matrix = np.load("embedding_matrix.npy")
    hidden_dim = 256  # Use the same as in training
    output_dim = 4    # Based on your label2idx.json (can also be len(label2idx))

    model = BiGRUCRFClass(embedding_matrix, hidden_dim, output_dim)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, sentence, label_map):
    # Dummy prediction logic – replace with your actual preprocessing + prediction
    tokens = sentence.split()
    prediction = ["O"] * len(tokens)  # Simulate predictions
    return list(zip(tokens, prediction))
