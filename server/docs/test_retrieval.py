import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Test query
query = "What is diabetes?"

query_embedding = embed_model.embed_query(query)

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

print("\n--- RESULTS ---\n")

if not results.matches:
    print("❌ No matches found")
else:
    for match in results.matches:
        print(f"Score: {match.score}")
        print(f"Source File: {match.metadata.get('source')}")
        print(f"Doc ID: {match.metadata.get('doc_id')}")
        print("\nText:\n", match.metadata.get("text", "")[:400], "...")
        print("-" * 60)
