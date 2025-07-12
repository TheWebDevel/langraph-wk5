#!/usr/bin/env python3
"""
Vector Database Initialization Script

This script initializes the persistent vector database by:
1. Clearing any existing database
2. Loading and chunking documents from data/
3. Creating embeddings and FAISS index
4. Saving everything to disk for persistent storage

Run this script once on application startup or when you want to rebuild the database.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from vector_store import initialize_vector_store

def main():
    """Initialize the vector database."""
    print("🚀 Initializing Vector Database...")
    print("=" * 50)

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Error: data/ directory not found!")
        print("Please ensure you have the following files:")
        print("  - data/it_faq.txt")
        print("  - data/finance_faq.txt")
        return False

    # Check if required files exist
    required_files = ["data/it_faq.txt", "data/finance_faq.txt"]
    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    try:
        # Initialize vector store (force rebuild to ensure clean state)
        initialize_vector_store(force_rebuild=True)

        print("=" * 50)
        print("✅ Vector Database initialized successfully!")
        print("📁 Database location: vector_db/")
        print("🔍 Ready for queries!")
        return True

    except Exception as e:
        print(f"❌ Error initializing vector database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)