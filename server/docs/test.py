from retriever import retrieve_relevant_chunks

query = "What is the document about?"
matches = retrieve_relevant_chunks(query)

for m in matches:
    print("\nScore:", m.score)
    print("Text Source:", m.metadata["source"])
    print("Doc Role:", m.metadata["role"])
    print("Doc ID:", m.metadata["doc_id"])
