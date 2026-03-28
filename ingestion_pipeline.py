import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Changed to HuggingFace
from dotenv import load_dotenv
import shutil

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory with UTF-8 encoding"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # List all files for debugging
    all_files = os.listdir(docs_path)
    txt_files = [f for f in all_files if f.endswith('.txt')]
    print(f"Found text files: {txt_files}")
    
    # Load all .txt files with UTF-8 encoding
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        recursive=True
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    print(f"\n{'='*60}")
    print(f"Loaded {len(documents)} document(s):")
    print(f"{'='*60}")
    
    for i, doc in enumerate(documents):
        file_size = len(doc.page_content)
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {file_size:,} characters")
        print(f"  Content preview: {doc.page_content[:150]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks with overlap"""
    print("\n" + "="*60)
    print("Splitting documents into chunks...")
    print("="*60)
    
    # Using RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"\nCreated {len(chunks)} total chunks")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")
    
    # Show sample chunks
    if chunks:
        print("\n" + "="*60)
        print("Sample chunks (first 3):")
        print("="*60)
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {os.path.basename(chunk.metadata['source'])}")
            print(f"Length: {len(chunk.page_content):,} characters")
            if 'start_index' in chunk.metadata:
                print(f"Start index: {chunk.metadata['start_index']}")
            print("Content preview:")
            print(chunk.page_content[:250] + "..." if len(chunk.page_content) > 250 else chunk.page_content)
            print("-" * 60)
        
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using free HuggingFace embeddings"""
    print("\n" + "="*60)
    print("Creating embeddings and storing in ChromaDB...")
    print("="*60)
    
    # Use free HuggingFace embeddings (runs locally)
    print("Loading HuggingFace embedding model (this may take a moment on first run)...")
    print("Model: sentence-transformers/all-MiniLM-L6-v2")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True},
        show_progress=True  # Show progress bar for embeddings
    )
    
    print("✓ Embedding model loaded successfully")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
    
    # Create ChromaDB vector store
    print("\nCreating vector store and generating embeddings (this may take a few minutes)...")
    print(f"Processing {len(chunks)} chunks...")
    
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        print(f"\n✓ Vector store created and saved to: {persist_directory}")
        print(f"✓ Total vectors stored: {vectorstore._collection.count()}")
        
        return vectorstore
        
    except Exception as e:
        print(f"\n❌ Error creating vector store: {e}")
        raise

def test_vector_store(vectorstore, query="What is Google?"):
    """Test the vector store with a sample query"""
    print("\n" + "="*60)
    print("Testing Vector Store")
    print("="*60)
    
    print(f"\nQuery: '{query}'")
    print("Retrieving top 3 relevant chunks...\n")
    
    try:
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Source: {os.path.basename(doc.metadata['source'])}")
            print(f"  Content: {doc.page_content[:200]}...")
            print()
        
        print("✓ Vector store test successful!")
        
    except Exception as e:
        print(f"❌ Error testing vector store: {e}")

def main():
    """Main ingestion pipeline"""
    print("="*60)
    print("RAG Document Ingestion Pipeline (with Free HuggingFace Embeddings)")
    print("="*60)
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print(f"\n⚠️  Vector store already exists at: {persistent_directory}")
        response = input("Do you want to recreate it? (y/n): ")
        
        if response.lower() != 'y':
            print("\nLoading existing vector store...")
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vectorstore = Chroma(
                persist_directory=persistent_directory,
                embedding_function=embedding_model
            )
            print(f"✓ Loaded existing vector store with {vectorstore._collection.count()} documents")
            
            # Test the loaded vector store
            test_vector_store(vectorstore)
            return vectorstore
        else:
            print("\n🗑️  Removing existing vector store...")
            shutil.rmtree(persistent_directory)
    
    print("\n🚀 Initializing new vector store with free embeddings...\n")
    
    try:
        # Step 1: Load documents
        documents = load_documents(docs_path)
        
        # Step 2: Split into chunks
        chunks = split_documents(documents)
        
        # Step 3: Create vector store
        vectorstore = create_vector_store(chunks, persistent_directory)
        
        # Step 4: Test the vector store
        test_vector_store(vectorstore)
        
        # Step 5: Summary
        print("\n" + "="*60)
        print("✅ INGESTION COMPLETE!")
        print("="*60)
        print("📊 Final Statistics:")
        print(f"   • Files processed: {len([f for f in os.listdir(docs_path) if f.endswith('.txt')])}")
        print(f"   • Documents loaded: {len(documents)}")
        print(f"   • Chunks created: {len(chunks)}")
        print("   • Embedding model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"   • Vector store: {persistent_directory}")
        print("="*60)
        
        print("\n💡 You can now query your documents using:")
        print("   from langchain_chroma import Chroma")
        print("   from langchain_huggingface import HuggingFaceEmbeddings")
        print("   embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')")
        print("   vectorstore = Chroma(persist_directory='db/chroma_db', embedding_function=embeddings)")
        print("   results = vectorstore.similarity_search('your question here')")
        
        return vectorstore
        
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return None

if __name__ == "__main__":
    main()