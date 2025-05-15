import torch
import numpy as np
from model_architecture import BiGRUCRFClass  # This should match your model file name

def load_model(model_2.2.3_bigru_crf_biowordvec.pth):
    embedding_matrix = np.load("embedding_matrix.npy")
    hidden_dim = 256  # Use the same as in training
    output_dim = 4    # Based on your label2idx.json (can also be len(label2idx))

    model = BiGRUCRFClass(embedding_matrix, hidden_dim, output_dim)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, sentence, label_map):
    # Load actual token2idx mapping
    with open("token2idx.json") as f:
        token2idx = json.load(f)

    # Tokenize and index the sentence
    words = sentence.strip().split()
    input_ids = [token2idx.get(w.lower(), token2idx.get("<PAD>", 0)) for w in words]

    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    mask = torch.tensor([[1]*len(input_ids)], dtype=torch.uint8)

    with torch.no_grad():
        outputs = model(input_tensor, mask=mask)

    idx2label = {v: k for k, v in label_map.items()}
    tags = [idx2label.get(tag, "O") for tag in outputs[0]]

    return list(zip(words, tags))
