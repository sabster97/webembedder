import chromadb
import json
import os
from pathlib import Path

def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="getclientell_content")
    
    raw_data_path = Path("raw_data")
    count = 0
    for json_file in raw_data_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                collection.add(
                    documents=[data['content']],
                    ids=[f"doc_{count}"],
                    metadatas=[{
                        "url": data["url"],
                        "title": data["title"],
                        "timestamp": data["timestamp"],
                        "filename": json_file.name,
                        "intent": data["intent"],
                        "keywords": str(data["keywords"]),
                        "internal_links": str(data["internal_links"]),
                        "meta_description": data["meta_description"]
                    }]
                )
                count += 1
                print(f"Added document {count}: {json_file.name}")
                
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
                continue
    
    print(f"\nTotal documents added: {count}")

if __name__ == "__main__":
    main()