import pandas as pd
import chromadb

INPUT_PATH = "backend-fastapi/data/papers_final.jsonl"
CHROMA_PATH = "backend-fastapi/chroma_db"
COLLECTION_NAME = "papers"
BATCH_SIZE = 500


def format_authors(authors):
    if isinstance(authors, list):
        return ", ".join(str(a) for a in authors)
    return str(authors or "").strip()


def main():
    df = pd.read_json(INPUT_PATH, lines=True)

    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    # Clean again lightly just in case
    for col in ["id", "title", "abstract", "authors", "categories", "update_date"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["id"] = df["id"].astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
    df["authors"] = df["authors"].apply(format_authors)
    df["categories"] = df["categories"].fillna("").astype(str).str.strip()
    df["update_date"] = df["update_date"].fillna("").astype(str).str.strip()

    # Remove bad rows
    df = df[(df["id"] != "") & (df["title"] != "") & (df["abstract"] != "")]
    df = df[df["abstract"].str.len() >= 50]

    # Final dedup safety
    df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=["title", "abstract"])

    print(f"Rows after cleanup: {len(df)}")

    # Build text for embedding/search
    df["document"] = df.apply(
        lambda row: f"Title: {row['title']}\nAbstract: {row['abstract']}",
        axis=1
    )

    ids = df["id"].tolist()
    documents = df["document"].tolist()
    metadatas = [
        {
            "title": row["title"],
            "authors": row["authors"],
            "categories": row["categories"],
            "update_date": row["update_date"],
        }
        for _, row in df.iterrows()
    ]

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Recreate collection from scratch
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        print(f"No existing collection to delete: {COLLECTION_NAME}")

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Insert in batches
    total = len(ids)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Inserted {end}/{total}")

    print(f"\nDone. Inserted {total} papers into Chroma.")


if __name__ == "__main__":
    main()