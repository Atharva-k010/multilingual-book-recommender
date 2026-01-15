from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import load_data, create_index
from recommender import recommend_books, detect_language

# ---------------------------------
# CREATE FASTAPI APP
# ---------------------------------
app = FastAPI(
    title="Multilingual Semantic Book Recommendation System",
    description="Semantic and genre-based book recommendation using embeddings and FAISS",
    version="1.0"
)

# ---------------------------------
# ENABLE CORS
# ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# LOAD DATA & MODEL
# ---------------------------------
df = load_data()
model, index = create_index(df)

# ---------------------------------
# REQUEST SCHEMA
# ---------------------------------
class UserInput(BaseModel):
    query: str | None = ""
    language: str | None = None
    genre: str | None = None

# ---------------------------------
# RECOMMEND ENDPOINT
# ---------------------------------
@app.post("/recommend")
def recommend(user_input: UserInput):
    query = user_input.query.strip() if user_input.query else ""

    # Weak query protection (semantic search only)
    if query and len(query.split()) < 2:
        return {
            "error": "Please provide a more descriptive interest (minimum 2 words).",
            "recommendations": []
        }

    results, error = recommend_books(
        query=query,
        model=model,
        index=index,
        df=df,
        genre=user_input.genre,
        language=user_input.language,
        top_k=10
    )

    if error:
        return {
            "error": error,
            "recommendations": []
        }

    return {
        "selected_language": user_input.language,
        "recommendations": results
    }

# ---------------------------------
# TOP BOOKS ENDPOINT
# ---------------------------------
@app.get("/top-books")
def top_books():
    top = {}
    genres = df["genre"].unique()

    for g in genres:
        top[g] = (
            df[df["genre"] == g]
            .head(10)
            .to_dict(orient="records")
        )

    return top
