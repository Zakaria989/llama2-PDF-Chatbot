from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import shutil
import os

# Data path to get information from
DATA_PATH = "data/"
# Path to store the embeddings
DB_FAISS_PATH = "vectorstores/db_faiss"

# Create a vector database
def create_vector_db():
    # Check if user wants to override vectorstores with new data or not
    if os.path.exists(DB_FAISS_PATH):
        overwrite = input("Vector store already exists do you want to override? (y/n): ")
        if overwrite.lower() == 'y':
            shutil.rmtree(DB_FAISS_PATH)
        else:
            print("Vector store were not overridden. Exiting...")
            return
    
    # Load PDF files as chunks of text using PyPDFLoader
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()  # Load documents as text chunks
        
    # Split text chunks into smaller segments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    text = text_splitter.split_documents(documents)
    
    # Create text embeddings; numerical vectors that represent the semantics of the text
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
    
    # Create a vector store using the embeddings and save it locally
    db = FAISS.from_documents(text, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

    
    
    
    