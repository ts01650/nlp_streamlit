import torch
import json
import os

def load_model(model_path):
    # Placeholder load function – customize based on actual model class
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

def predict(model, sentence, label_map):
    # Dummy prediction logic – replace with your actual preprocessing + prediction
    tokens = sentence.split()
    prediction = ["O"] * len(tokens)  # Simulate predictions
    return list(zip(tokens, prediction))
