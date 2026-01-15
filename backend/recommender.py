import numpy as np
import faiss
from langdetect import detect

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"


def recommend_books(query, model, index, df, genre=None, language=None, top_k=10):

    filtered_df = df

    # Filter by language
    if language:
        filtered_df = filtered_df[filtered_df["language"] == language]
        if filtered_df.empty:
            return [], f"No books available in selected language ({language})"

    # Filter by genre
    if genre:
        filtered_df = filtered_df[
            filtered_df["genre"].str.lower() == genre.lower()
        ]
        if filtered_df.empty:
            return [], f"No books available for genre '{genre}'"

    # Browse mode
    if not query:
        results = []
        seen = set()

        for _, book in filtered_df.head(top_k).iterrows():
            if book["title"] in seen:
                continue
            seen.add(book["title"])
            results.append({
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "language": book["language"],
                "reason": "Browsing books based on selected filters."
            })

        return results, None

    # Semantic search
    descriptions = filtered_df["description"].tolist()
    embeddings = model.encode(descriptions)

    dim = embeddings.shape[1]
    temp_index = faiss.IndexFlatL2(dim)
    temp_index.add(np.array(embeddings))

    query_embedding = model.encode([query])
    distances, indices = temp_index.search(query_embedding, top_k)

    results = []
    seen_titles = set()

    for idx in indices[0]:
        book = filtered_df.iloc[idx]

        if book["title"] in seen_titles:
            continue

        seen_titles.add(book["title"])

        results.append({
            "title": book["title"],
            "author": book["author"],
            "genre": book["genre"],
            "language": book["language"],
            "reason": f"This book matches your interest in {query.lower()}."
        })

    return results, None
