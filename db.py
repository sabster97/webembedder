import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from openai import OpenAI
from typing import List, Dict, Any
import tiktoken
import json
import time
from dotenv import load_dotenv
from sys_prompt import get_sys_prompt

class DocumentQA:
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Get database path from environment variable or use default
        db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="getclientell_content")
        self.prompt_library=get_sys_prompt()
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm_client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_tokens = 4000

    def search_documents(self, query: str) -> List[Dict]:
        """Modified to return results instead of printing them"""
        results = self._get_relevant_chunks(query, n_results=10)
        formatted_results = []
        
        for i, doc in enumerate(results):
            result = {
                "rank": i + 1,
                "title": doc['metadata']['title'],
                "url": doc['metadata']['url'],
                "distance_score": doc['distance'],
                "occurrences": []
            }
            
            content = doc['content']
            lower_content = content.lower()
            lower_query = query.lower()
            
            start_pos = 0
            while True:
                pos = lower_content.find(lower_query, start_pos)
                if pos == -1:
                    break
                    
                context_start = max(0, pos - 50)
                context_end = min(len(content), pos + len(query) + 50)
                context = content[context_start:context_end]
                
                if context_start > 0:
                    context = "..." + context
                if context_end < len(content):
                    context = context + "..."
                
                result["occurrences"].append({"context": context})
                start_pos = pos + 1
            
            formatted_results.append(result)
        
        return formatted_results

   


    def classify_query(self, query: str) -> str:
        """Moved from __main__ to class method"""
        classification_prompt = f"""You are a helpful assistant specialized in understanding user queries. Based on the user's query, classify it into one of the following intents:
        
        1. "optimize_page": For queries related to optimizing content, keywords or link structure for SEO on an existing page.
        2. "research_question": For questions requiring research into a new topic or page.
        3. "verification": For queries asking to evaluate a page basis an SEO checklist.
        4. "analytics_question": For questions requiring analysis of website analytics data or suggestions for seo optimization
        5. "other": For general questions that do not fit into the other categories.
        
        Query: "{query}"

        Provide only the intent name (e.g., "optimize_page", "research_question", "verification", "analytics_question", "other")."""
        
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an seo intent classification assistant."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        classification= response.choices[0].message.content.strip().lower()
        return self.prompt_library[classification]
    

    def _get_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks from the vector database"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return documents

    def _truncate_context(self, documents: List[Dict], query: str) -> str:
        """Truncate context to fit within token limits"""
        # Calculate tokens for the template and query
        template = """Based on the following documents, please answer this question: {query}

Context documents:
{context}

Please provide a clear and concise answer based only on the information provided in the documents above. If the documents don't contain relevant information to answer the question, please say so."""
        
        base_tokens = self._count_tokens(template.format(query=query, context=""))
        available_tokens = self.max_tokens - base_tokens
        
        context_parts = []
        current_tokens = 0
        
        for i, doc in enumerate(documents):
            doc_text = f"Document {i+1}:\n{doc['content']}\nSource: {doc['metadata']['url']}\n\n"
            doc_tokens = self._count_tokens(doc_text)
            
            if current_tokens + doc_tokens > available_tokens:
                # If adding this document would exceed the limit, try to add a truncated version
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 100:  # Only add if we can include meaningful content
                    truncated_text = self.encoding.decode(
                        self.encoding.encode(doc_text)[:remaining_tokens]
                    )
                    context_parts.append(truncated_text + "...\n")
                break
            
            context_parts.append(doc_text)
            current_tokens += doc_tokens
        
        return "".join(context_parts)

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))

    def ai_magic(self, sys_prompt, query: str) -> Dict[str, Any]:
        """Enhanced RAG implementation with better context handling and structured output"""
        print("ENTERED AI MAGIC")
        relevant_docs = self._get_relevant_chunks(query, n_results=10)
        context = self._truncate_context(relevant_docs, query)
        
        # Enhanced prompt template for better RAG performance
        
        user_prompt=query
        # Count input tokens
        input_tokens = self._count_tokens(sys_prompt+user_prompt+context)
        
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        # Count output tokens
        output_tokens = self._count_tokens(response.choices[0].message.content)

        # Calculate confidence score based on document relevance
        confidence_score = self._calculate_confidence(relevant_docs, query)
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "url": doc["metadata"]["url"],
                    "title": doc["metadata"]["title"],
                    "relevance_score": 1 - doc["distance"]  # Convert distance to relevance
                } for doc in relevant_docs
            ],
            "confidence_score": confidence_score,
            "metadata": {
                "num_sources": len(relevant_docs),
                "query_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }
        }

    def _calculate_confidence(self, documents: List[Dict], query: str) -> float:
        """Calculate confidence score based on document relevance"""
        if not documents:
            return 0.0
        
        # Average distance score (lower is better)
        avg_distance = sum(doc["distance"] for doc in documents) / len(documents)
        
        # Convert distance to confidence (1 - distance), normalize to 0-1 range
        base_confidence = 1 - min(avg_distance, 1.0)
        
        # Adjust confidence based on number of relevant documents
        doc_count_factor = min(len(documents) / 5, 1.0)  # Normalize to max of 5 documents
        
        # Calculate final confidence score
        confidence = base_confidence * doc_count_factor
        
        return round(confidence, 2)

    def _enhance_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced retrieval with query expansion and reranking"""
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Perform initial retrieval
        initial_results = self._get_relevant_chunks(query, n_results=10)
        
        # Rerank results based on semantic similarity and relevance
        reranked_results = []
        for doc in initial_results:
            # Calculate semantic similarity score
            semantic_score = self._calculate_semantic_similarity(
                query_embedding, 
                doc['content']
            )
            
            # Combine with original distance score
            final_score = (semantic_score + (1 - doc['distance'])) / 2
            doc['final_score'] = final_score
            reranked_results.append(doc)
        
        # Sort by final score and return top 5
        reranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        return reranked_results[:5]

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using OpenAI's embedding model"""
        response = self.llm_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response.data[0].embedding

    def _calculate_semantic_similarity(self, query_embedding: List[float], text: str) -> float:
        """Calculate semantic similarity between query and text"""
        # Get text embedding
        text_embedding_response = self.llm_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        text_embedding = text_embedding_response.data[0].embedding
        
        # Calculate cosine similarity
        dot_product = sum(q * t for q, t in zip(query_embedding, text_embedding))
        query_norm = sum(q * q for q in query_embedding) ** 0.5
        text_norm = sum(t * t for t in text_embedding) ** 0.5
        
        return dot_product / (query_norm * text_norm)
    