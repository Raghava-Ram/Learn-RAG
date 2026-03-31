from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Connect to your document database (using local embeddings)
persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up local AI model with Ollama
model = ChatOllama(
    model="phi3",  # or "tinyllama", "llama3.2"
    temperature=0.7,
)


# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be a standalone search query. Return ONLY the rewritten question, no explanation."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview[:100]}...")
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{"\n".join([f"- {doc.page_content}" for doc in docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"\nAnswer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("="*60)
    print("📚 RAG Chat Assistant - Ask questions about your documents!")
    print("="*60)
    print("Type 'quit' to exit.")
    print("Type 'clear' to clear conversation history.")
    print("="*60)
    
    while True:
        question = input("\n💬 Your question: ").strip()
        
        if question.lower() == 'quit':
            print("\nGoodbye! 👋")
            break
        
        if question.lower() == 'clear':
            global chat_history
            chat_history = []
            print("\n✨ Conversation history cleared!")
            continue
            
        if not question:
            print("Please enter a question.")
            continue
            
        try:
            ask_question(question)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please make sure Ollama is running with 'ollama serve'")

if __name__ == "__main__":
    start_chat()