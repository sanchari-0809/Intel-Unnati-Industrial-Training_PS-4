from retriever import retrieve_relevant_chunks

query = "What is the document about?"
matches = retrieve_relevant_chunks(query)

for m in matches:
    print("\nScore:", m.score)
    print("Text Source:", m.metadata.get("source"))
    print("Doc ID:", m.metadata.get("doc_id"))
