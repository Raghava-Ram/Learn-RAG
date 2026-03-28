from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Create prompt with context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

prompt = f"""Based on the following documents, answer this question: {query}

Documents:
{context}

Instructions:
- Answer based ONLY on the information in these documents
- If the answer is not in the documents, say "I cannot find this information in the provided documents"
- Be concise and accurate

Answer:"""

# Generate answer with Ollama
print("--- Generating Answer ---")
model = ChatOllama(
    model="phi3",  # Change to "llama3.2" or your preferred model
    temperature=0.7,
)

response = model.invoke(prompt)

print(f"\nAnswer: {response.content}")
