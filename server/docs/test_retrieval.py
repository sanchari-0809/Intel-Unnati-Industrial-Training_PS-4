import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Load env variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ----- ENTER YOUR TEST QUERY HERE -----
query = "What is diabetes?"
# --------------------------------------

# Create query embedding
print("Embedding query...")
query_embedding = embed_model.embed_query(query)

# Search Pinecone
print("Querying Pinecone...")
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

print("\n--- RESULTS ---\n")

if not results.matches:
    print("❌ No matches found. (Vectors may not be stored)")
else:
    for match in results.matches:
        print(f"Score: {match.score}")
        print(f"Source File: {match.metadata.get('source')}")
        print(f"Role: {match.metadata.get('role')}")
        print(f"Doc ID: {match.metadata.get('doc_id')}")
        print("\nExtracted Text:\n", match.metadata.get("text", "")[:400], "...")
        print("-" * 80)
