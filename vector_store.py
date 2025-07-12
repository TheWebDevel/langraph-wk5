import os
import re
import shutil
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path

class VectorStore:
    def __init__(self, persist_directory: str = "vector_db"):
        self.persist_directory = Path(persist_directory)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.categories = []

    def clear_database(self):
        """Clear the existing vector database."""
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        # Always create the directory, whether it existed before or not
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        print(f"Cleared vector database at {self.persist_directory}")

    def chunk_document(self, content: str) -> List[str]:
        """Split document into Q&A chunks."""
        # Chunk by Q&A pairs (lines starting with Q: or Q.)
        qa_chunks = re.split(r'(?=^Q[:\.])', content, flags=re.MULTILINE)
        return [chunk.strip() for chunk in qa_chunks if chunk.strip()]

    def load_and_process_documents(self):
        """Load documents, chunk them, and create embeddings."""
        print("Loading and processing documents...")

        # Define document sources
        documents = {
            "IT": "data/it_faq.txt",
            "Finance": "data/finance_faq.txt"
        }

        all_chunks = []
        all_categories = []

        for category, file_path in documents.items():
            if os.path.exists(file_path):
                print(f"Processing {category} document: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                chunks = self.chunk_document(content)
                all_chunks.extend(chunks)
                all_categories.extend([category] * len(chunks))
                print(f"  - Created {len(chunks)} chunks for {category}")
            else:
                print(f"Warning: Document not found: {file_path}")

        if not all_chunks:
            print("No documents found to process!")
            return

        print(f"Total chunks created: {len(all_chunks)}")

        # Create embeddings
        print("Creating embeddings...")
        embeddings = self.model.encode(all_chunks)

        # Create FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        # Store chunks and categories
        self.chunks = all_chunks
        self.categories = all_categories

        # Ensure directory exists before saving
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Save to disk
        self.save_to_disk()
        print("Vector database initialized successfully!")

    def save_to_disk(self):
        """Save the vector database to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.persist_directory / "faiss_index.bin"))

        # Save chunks and categories
        with open(self.persist_directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        with open(self.persist_directory / "categories.pkl", "wb") as f:
            pickle.dump(self.categories, f)

        print(f"Vector database saved to {self.persist_directory}")

    def load_from_disk(self) -> bool:
        """Load the vector database from disk."""
        try:
            # Check if database exists
            if not (self.persist_directory / "faiss_index.bin").exists():
                return False

            # Load FAISS index
            self.index = faiss.read_index(str(self.persist_directory / "faiss_index.bin"))

            # Load chunks and categories
            with open(self.persist_directory / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)

            with open(self.persist_directory / "categories.pkl", "rb") as f:
                self.categories = pickle.load(f)

            print(f"Vector database loaded from {self.persist_directory}")
            print(f"  - {len(self.chunks)} chunks available")
            print(f"  - Categories: {set(self.categories)}")
            return True

        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False

    def search(self, query: str, category: str | None = None, top_k: int = 3) -> List[str]:
        """Search for relevant chunks."""
        if self.index is None:
            return ["Vector database not initialized"]

        try:
            # Encode query
            query_embedding = self.model.encode([query])
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Search
            k = top_k * 2  # Get more results for filtering
            D, I = self.index.search(query_embedding.astype('float32'), k)

            # Filter by category if specified
            results = []
            for idx in I[0]:
                if idx < len(self.chunks):
                    if category is None or self.categories[idx] == category:
                        results.append(self.chunks[idx])
                        if len(results) >= top_k:
                            break

            return results if results else ["No relevant information found."]

        except Exception as e:
            return [f"Error in vector search: {str(e)}"]

    def initialize_database(self, force_rebuild: bool = False):
        """Initialize the vector database, rebuilding if necessary."""
        if force_rebuild:
            print("Force rebuild requested - clearing existing database...")
            self.clear_database()
            self.load_and_process_documents()
        else:
            # Try to load existing database
            if not self.load_from_disk():
                print("No existing database found - creating new one...")
                self.load_and_process_documents()
            else:
                print("Database loaded successfully from disk")

        # Verify the database is properly initialized
        if self.index is None or len(self.chunks) == 0:
            print("Database verification failed - rebuilding...")
            self.clear_database()
            self.load_and_process_documents()

# Global vector store instance
vector_store = VectorStore()

def initialize_vector_store(force_rebuild: bool = False):
    """Initialize the global vector store."""
    vector_store.initialize_database(force_rebuild)

def vector_search_impl(query: str, category: str) -> str:
    """Vector search implementation using the persistent store."""
    # Ensure database is initialized
    if vector_store.index is None or len(vector_store.chunks) == 0:
        print("Vector database not initialized, attempting to load...")
        try:
            vector_store.initialize_database(force_rebuild=False)
        except Exception as e:
            print(f"Failed to initialize vector database: {e}")
            return "Vector database not available. Please run 'python initialize_db.py' to set up the database."

    results = vector_store.search(query, category, top_k=3)
    return "\n\n".join(results)