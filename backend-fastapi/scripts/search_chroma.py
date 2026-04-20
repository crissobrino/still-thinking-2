import chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="papers")


def search(query: str, k: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    output = []

    if not results["ids"] or not results["ids"][0]:
        return output

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i] or {}
        document = results["documents"][0][i] or ""

        output.append({
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", ""),
            "categories": metadata.get("categories", ""),
            "update_date": metadata.get("update_date", ""),
            "document": document,
            "abstract_snippet": document[:500]
        })

    return output


if __name__ == "__main__":
    query = "machine learning for genomics"
    results = search(query)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}]")
        print("ID:", result["id"])
        print("Distance:", result["distance"])
        print("Title:", result["title"])
        print("Authors:", result["authors"])
        print("Categories:", result["categories"])
        print("Update date:", result["update_date"])
        print("Document:", result["abstract_snippet"])