import openai
import chromadb
from urllib.parse import urlparse
from typing import Dict, List, Any
import os
import json
from pathlib import Path

# Define the page structure schema - this is just for documentation
page_struct_collection = {
    "title": "The main title of the webpage",
    "url": "The URL of the webpage being scraped",
    "meta_title": "The SEO meta title of the webpage, which appears in search engines",
    "meta_description": "The SEO meta description that provides a summary of the page content",
    "original_sitemap": "The sitemap URL from which this page URL was retrieved",
    "bounce_rate": "The percentage of visitors who leave the page without interacting further",
    "visitors": "The number of unique visitors to the page",
    "page_intent": "The purpose of the page, e.g., Informational or Conversion-focused",
    "persona": "The target audience of the page, e.g., Salesforce Admin, Developer, etc.",
    "top_keywords": "The primary keywords relevant to the page content, separated by commas",
    "internal_links": "Comma-separated URLs of internal links present on the page",
    "keywords_in_links": "Keywords that appear in the anchor text of internal or external links",
    "pillar_keyword": "The core keyword that represents the overall theme of the page",
    "target_keyword": "The specific keyword the page aims to rank for in search engines",
    "word_count": "The total number of words in the content of the page",
    "title_length": "The character count of the main title of the page",
    "meta_title_length": "The character count of the SEO meta title",
    "meta_description_length": "The character count of the SEO meta description",
    "content_type": "The type of content, e.g., Blog, Product Page, FAQ",
    "h1_h6_headings": "A structured list of the H1 to H6 headings on the page",
    "keyword_density": "The percentage of the target keyword in relation to the total word count",
    "user_path": "The breadcrumb or user navigation path that leads to the page"
}

class ChromaDBLoader:
    def __init__(self, db_path: str = "./webpage_data_chroma_db"):
        """Initialize ChromaDB client with the specified path"""
        self.client = chromadb.PersistentClient(path=db_path)
        
    def create_collection(self, collection_name: str = "clientell_content") -> Any:
        """Create a new collection with metadata"""
        collection_metadata = {
            "description": "This collection contains documents focused on Salesforce best practices, implementation strategies, SEO optimization tips, content structure improvements, and industry-specific actionable insights.",
            "schema_version": "1.0"
        }
        
        return self.client.create_collection(
            name=collection_name,
            metadata=collection_metadata
        )
    
    def add_document(self, 
                    collection: Any,
                    content_text: str,
                    embedding: List[float],
                    metadata: Dict[str, Any]) -> None:
        """Add a document to the collection with its embedding and metadata"""
        # Generate a unique ID from the URL
        doc_id = urlparse(metadata['url']).path.replace('/', '_')
        if not doc_id:
            doc_id = "root"
            
        # Add the document to the collection
        collection.add(
            embeddings=[embedding],
            documents=[content_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def get_or_create_collection(self, collection_name: str = "clientell_content") -> Any:
        """Get existing collection or create new one if it doesn't exist"""
        try:
            return self.client.get_collection(collection_name)
        except ValueError:
            return self.create_collection(collection_name)

    def process_webpage_data(self, webpage_data_dir: str = "scraped_web_data"):
        """Process webpage data from the scraped_web_data directory"""
        collection = self.get_or_create_collection()
        if not collection:
            raise Exception("Failed to create or retrieve the collection.")
        webpage_data_path = Path(webpage_data_dir)
        
        # Look for pages_data.json in the scraped_web_data directory
        json_file = webpage_data_path / "pages_data.json"
        
        if not json_file.exists():
            print(f"Error: {json_file} not found")
            return
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                pages_data = json.load(f)
                
            print(f"Found {len(pages_data)} pages to process")
            
            for page_data in pages_data:
                # Extract content and metadata
                content = page_data.pop('raw_content', '')  # Remove and get content
                
                if not content:
                    print(f"Skipping page {page_data.get('url', 'unknown')}: No content found")
                    continue
                
                try:
                    # Generate embedding for the content
                    response = openai.embeddings.create(
                        input=content,
                        model="text-embedding-3-small"
                    )
                    embedding = response.data[0].embedding
                    
                    # Add document to collection
                    self.add_document(
                        collection=collection,
                        content_text=content,
                        embedding=embedding,
                        metadata=page_data
                    )
                    print(f"Processed and added document: {page_data.get('title', 'Untitled')}")
                    
                except Exception as e:
                    print(f"Error processing page {page_data.get('url', 'unknown')}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error reading {json_file}: {str(e)}")
            return


# Example usage:
if __name__ == "__main__":
    # Initialize the loader
    loader = ChromaDBLoader()
    
    # Set up OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Process all webpage data
    loader.process_webpage_data()
    
    print("All documents have been processed and added to ChromaDB collection.")