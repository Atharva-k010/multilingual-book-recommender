import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load dataset
def load_data():
    df = pd.read_csv("data/books.csv")
    return df

# Create embeddings and FAISS index
def create_index(df):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(df["description"].tolist())

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return model, index
