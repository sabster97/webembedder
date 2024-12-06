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
import numpy as np

class DocumentQA:
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Get database path from environment variable or use default
        db_path = os.getenv('CHROMA_DB_PATH', './webpage_data_chroma_db')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="clientell_content")
        print("self.collection",self.collection)

        self.prompt_library = get_sys_prompt()
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm_client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
        self.max_tokens = 50000

    def search_documents(self, query: str) -> List[Dict]:
        """Modified to return results instead of printing them"""
        results = self._enhance_retrieval(query, n_results=10)
        formatted_results = []
        
        for i, doc in enumerate(results):
            result = {
                "rank": i + 1,
                "title": doc['metadata']['title'],
                "url": doc['metadata']['url'],
                "distance_score": doc['final_score'],
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
        print("Inside Classify Query")
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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an seo intent classification assistant."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        print("Chat GPT response classification:", response)
        classification= response.choices[0].message.content.strip().lower()
        print("classification ---> ", classification)
        return self.prompt_library[classification]
    
    def _get_relevant_chunks(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks from the vector database"""
        print("Inside _get_relevant_chunks", query)
        print("n_results", n_results)
        
        try:
            # First get the query embedding using the same model as document embeddings
            query_embedding = self._get_query_embedding(query)
            print(f"Query embedding dimensions: {len(query_embedding)}")  # Should print 1536
            
            # Query the collection with the embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],  # Use query_embeddings instead of query_texts
                n_results=n_results
            )
            
        except Exception as e:
            print("Error in _get_relevant_chunks:", e)
            results = {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
        
        # print("results", results)
        
        documents = []
        for i in range(len(results['documents'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        # print("documents", documents)
        return documents

    def _truncate_context(self, final_user_prompt) -> str:
        print("Inside _truncate_context")
        """Adapt context to fit within GPT-4's larger token limits"""
        print("Inside _truncate_context")
    # Calculate tokens for the template and query
        try:
            print("truncating context")
            base_tokens = self._count_tokens(final_user_prompt)
            print("base tokens", base_tokens);
        except Exception as e:
            print("Error in _truncate_context:", e)
            return final_user_prompt
    
    # If the combined text exceeds the maximum token limit, truncate it
        print("Checking if token count exceeds limit")
        if base_tokens > self.max_tokens:
            print("Token count exceeds limit, truncating")
            try:
                truncated_text = self.encoding.decode(self.encoding.encode(final_user_prompt)[:self.max_tokens])
                print("truncated_text",truncated_text)
                return truncated_text + "..."
            except Exception as e:
                print("Error in _truncate_context (truncation):", e)
                return final_user_prompt
    
    # If the token count is within the limit, return as is
        print("Token count is within limit, returning as is")
        print("Received final_user_prompt:", final_user_prompt)
        return final_user_prompt  

    def _count_tokens(self, text: str) -> int:
        print("Inside _count_tokens")
        """Count the number of tokens in a text string"""
        try:
            tokens = len(self.encoding.encode(text))
            print("Tokens in text:", tokens)
            return tokens
        except Exception as e:
            print("Error in _count_tokens:", e)
            return 0

    def ai_magic(self, sys_prompt, query: str) -> Dict[str, Any]:
        """Enhanced RAG implementation with better context handling and structured output"""
        print("Inside Ai Magic")
        relevant_docs = self._enhance_retrieval(query, n_results=10)
        # print("relevant_docs",relevant_docs)
        #context = self._truncate_context(relevant_docs, query)
        user_prompt=query
        context=relevant_docs
        print("before context")
        print("context",context)
        print("user_prompt",user_prompt)
        final_user_prompt = self._truncate_context(user_prompt + context)
        print("final_user_prompt",final_user_prompt)
        
        # Enhanced prompt template for better RAG performance
        
        
        # Count input tokens
        input_tokens = self._count_tokens(final_user_prompt+sys_prompt)

        print("input_tokens", input_tokens)
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": final_user_prompt}
                ],
                temperature=0.3,
                #for genereative and creative classifcation increase using switch
                max_tokens=16384
            )
        except Exception as e:
            print("Error in ai_magic:", e)
            return {"error": str(e)}

        print("Chat GPT response output:", response)

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

    def _enhance_retrieval(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Enhanced retrieval with query expansion and reranking"""
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        # print("query_embedding",query_embedding)
        
        # Perform initial retrieval
        initial_results = self._get_relevant_chunks(query, n_results=n_results)
        # print("initial_results",initial_results)
        
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
        
        # print("reranked_results",reranked_results)
        # Sort by final score and return top results
        reranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        return reranked_results[:n_results]

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
        return self._calculate_cosine_similarity(query_embedding, text_embedding)
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(q * t for q, t in zip(vec1, vec2))
        vec1_norm = sum(q * q for q in vec1) ** 0.5
        vec2_norm = sum(t * t for t in vec2) ** 0.5
        
        return dot_product / (vec1_norm * vec2_norm)
