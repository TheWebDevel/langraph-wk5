#!/usr/bin/env python3
"""
Test script for the vector store functionality.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from vector_store import vector_store, initialize_vector_store

def test_vector_store():
    """Test the vector store functionality."""
    print("🧪 Testing Vector Store...")
    print("=" * 40)

    try:
        # Initialize the database
        print("1. Initializing database...")
        initialize_vector_store(force_rebuild=True)

        # Test IT search
        print("\n2. Testing IT search...")
        it_query = "How do I set up VPN?"
        it_results = vector_store.search(it_query, category="IT", top_k=2)
        print(f"Query: {it_query}")
        print(f"Results: {len(it_results)} found")
        for i, result in enumerate(it_results, 1):
            print(f"  {i}. {result[:100]}...")

        # Test Finance search
        print("\n3. Testing Finance search...")
        finance_query = "How do I file a reimbursement request?"
        finance_results = vector_store.search(finance_query, category="Finance", top_k=2)
        print(f"Query: {finance_query}")
        print(f"Results: {len(finance_results)} found")
        for i, result in enumerate(finance_results, 1):
            print(f"  {i}. {result[:100]}...")

        # Test cross-category search
        print("\n4. Testing cross-category search...")
        general_query = "What is the process for requesting something?"
        general_results = vector_store.search(general_query, category=None, top_k=3)
        print(f"Query: {general_query}")
        print(f"Results: {len(general_results)} found")
        for i, result in enumerate(general_results, 1):
            print(f"  {i}. {result[:100]}...")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_store()
    sys.exit(0 if success else 1)