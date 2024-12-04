import chromadb
import json
import os
from pathlib import Path

def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Define the collections and their respective raw data directories
    data_sources=["metrics","pages", "outlines","sitemaps"]
    paths=["","","",""]
    collections_info = dict(zip(data_sources,paths))

    total_documents_added = 0
    
    for collection_name, data_path in collections_info.items():
        collection = client.get_or_create_collection(name=collection_name)
        raw_data_path = Path(data_path)
        count = 0
        
        for json_file in raw_data_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    collection.add(
                        documents=[data['content']],
                        ids=[f"{collection_name}_doc_{count}"],
                        metadatas=[{
                            "url": data.get("url", "N/A"),
                            "title": data.get("title", "N/A"),
                            "timestamp": data.get("timestamp", "N/A"),
                            "filename": json_file.name
                        }]
                    )
                    count += 1
                    print(f"Added document {count} to collection '{collection_name}': {json_file.name}")
                
                except json.JSONDecodeError as e:
                    print(f"Error reading {json_file}: {e}")
                    continue
        
        print(f"Total documents added to collection '{collection_name}': {count}\n")
        total_documents_added += count

    print(f"\nOverall total documents added: {total_documents_added}")

if __name__ == "__main__":
    main()
