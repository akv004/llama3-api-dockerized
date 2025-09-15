#!/usr/bin/env python3
"""
RAG Index Creation Utility
==========================

Reads text files from a source directory, splits them into chunks,
embeds them using a sentence-transformer model, and saves them
to a FAISS vector index for use by the Llama3 API.

Instructions:
1. Create a folder (e.g., `rag_source_docs`).
2. Add your .txt files to this folder.
3. Run this script:
   python create_rag_index.py

It will create a new folder (default: `my_rag_data`) containing the
FAISS index, which the main API can then load.
"""
import os
import glob
import logging

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
# This should match the model used in your main API script
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Directory where your .txt files are located
SOURCE_DOCS_DIR = "rag_source_docs"

# Directory where the FAISS index will be saved
FAISS_INDEX_DIR = "/home/amit/projects/my_rag_data"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_index():
    """Loads docs, creates embeddings, and saves a FAISS index."""
    if not os.path.isdir(SOURCE_DOCS_DIR) or not os.listdir(SOURCE_DOCS_DIR):
        logging.error(f"Source directory '{SOURCE_DOCS_DIR}' is empty or does not exist.")
        logging.error("Please create it and add your .txt files to it.")
        return

    logging.info(f"Loading documents from: {SOURCE_DOCS_DIR}")
    doc_files = glob.glob(os.path.join(SOURCE_DOCS_DIR, "*.txt"))
    documents = [TextLoader(f).load()[0] for f in doc_files]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(docs)} chunks.")

    logging.info(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})

    logging.info("Creating FAISS vector store... (this may take a moment)")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    logging.info(f"FAISS index successfully created and saved to: {FAISS_INDEX_DIR}")

if __name__ == "__main__":
    create_index()