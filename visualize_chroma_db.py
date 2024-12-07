import streamlit as st
import chromadb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

class ChromaDBVisualizer:
    def __init__(self, persist_directory="./webpage_data_chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def get_collections(self):
        return self.client.list_collections()
    
    def get_collection_stats(self, collection):
        count = collection.count()
        peek = collection.peek()
        
        # Extract metadata fields
        metadata_fields = set()
        for item in peek['metadatas']:
            metadata_fields.update(item.keys())
            
        return {
            'document_count': count,
            'metadata_fields': list(metadata_fields),
            'sample_metadata': peek['metadatas'][0] if peek['metadatas'] else {}
        }
    
    def get_collection_data(self, collection):
        # Get all documents with their metadata
        results = collection.get()
        
        # Convert to DataFrame for easier visualization
        df = pd.DataFrame({
            'id': results['ids'],
            'content': results['documents'],
            **{f'metadata_{k}': [m.get(k) for m in results['metadatas']]
            for k in results['metadatas'][0].keys() if isinstance(results['metadatas'][0].get(k), (str, int, float))}
        })
        return df

def main():
    st.set_page_config(page_title="ChromaDB Visualizer", layout="wide")
    st.title("ChromaDB Data Visualization")
    
    visualizer = ChromaDBVisualizer()
    collections = visualizer.get_collections()
    
    if not collections:
        st.warning("No collections found in the database.")
        return
    
    # Sidebar for collection selection
    selected_collection_name = st.sidebar.selectbox(
        "Select Collection",
        options=[c.name for c in collections]
    )
    
    selected_collection = visualizer.client.get_collection(selected_collection_name)
    stats = visualizer.get_collection_stats(selected_collection)
    
    # Display collection overview
    st.header("Collection Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Documents", stats['document_count'])
        st.write("### Metadata Fields")
        st.write(", ".join(stats['metadata_fields']))
    
    with col2:
        st.write("### Sample Metadata")
        st.json(stats['sample_metadata'])
    
    # Get detailed collection data
    df = visualizer.get_collection_data(selected_collection)
    
    # Content Type Distribution
    if 'metadata_content_type' in df.columns:
        st.header("Content Type Distribution")
        fig = px.pie(df, names='metadata_content_type')
        st.plotly_chart(fig)
    
    # Word Count Distribution
    if 'metadata_word_count' in df.columns:
        st.header("Word Count Distribution")
        fig = px.histogram(df, x='metadata_word_count', nbins=20)
        st.plotly_chart(fig)
    
    # Document Explorer
    st.header("Document Explorer")
    selected_doc = st.selectbox("Select Document", options=df['id'].tolist())
    if selected_doc:
        doc_data = df[df['id'] == selected_doc].iloc[0]
        
        st.write("### Content")
        st.write(doc_data['content'])
        
        st.write("### Metadata")
        metadata = {k.replace('metadata_', ''): v for k, v in doc_data.items() 
                   if k.startswith('metadata_')}
        st.json(metadata)

if __name__ == "__main__":
    main()