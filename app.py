import os
import glob
import faiss
import chromadb
import pinecone
import ollama
import logging
from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Pinecone, Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    filename="chatbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)

logging.info("Starting the Chatbot API...")

# Load Llama 3 from Ollama
logging.info("Loading Llama 3 model from Ollama...")
llm = Ollama(model="llama3")

# Load and process multiple documents
def load_documents():
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # Load PDFs
    pdf_files = glob.glob("knowledge_base/*.pdf")
    for pdf_file in pdf_files:
        logging.info(f"Loading PDF: {pdf_file}")
        pdf_loader = PyPDFLoader(pdf_file)
        docs.extend(pdf_loader.load())
    
    # Load text files
    txt_files = glob.glob("knowledge_base/*.txt")
    for txt_file in txt_files:
        logging.info(f"Loading Text File: {txt_file}")
        txt_loader = TextLoader(txt_file)
        docs.extend(txt_loader.load())
    
    # Split documents
    split_docs = text_splitter.split_documents(docs)
    logging.info(f"Loaded {len(split_docs)} document chunks.")
    return split_docs

documents = load_documents()

# Choose vector store
VECTOR_DB = "faiss"  # Change to "pinecone" or "chroma" if needed

# Create embeddings using Hugging Face (Free)
logging.info("Initializing Hugging Face Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize vector database
if VECTOR_DB == "faiss":
    logging.info("Using FAISS as the vector store.")
    vector_db = FAISS.from_documents(documents, embeddings)
elif VECTOR_DB == "pinecone":
    logging.info("Using Pinecone as the vector store.")
    pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-east1-gcp")
    vector_db = Pinecone.from_documents(documents, embeddings, index_name="rag-chatbot")
elif VECTOR_DB == "chroma":
    logging.info("Using Chroma as the vector store.")
    vector_db = Chroma.from_documents(documents, embeddings)

retriever = vector_db.as_retriever()

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define dynamic prompt template for reasoning
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    {question}

    **Relevant Knowledge Retrieved:**
    {context}

    **Analysis & Logical Reasoning:**
    - What are the key objectives?
    - What are the best practices or guidelines?
    - Are there challenges, and how can they be mitigated?
    - What are the technical, business, or operational factors?
    - Are there relevant case studies or examples?

    **Final Response:**
    Here’s the recommended answer:
    - [Key insight 1]
    - [Key insight 2]
    - [Key insight 3]

    Let me know if you need further details!
    """
)

# Define QA chain with RAG and dynamic reasoning
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

# Define tools for agent reasoning
tools = [
    Tool(name="RAG Retriever", func=qa_chain.run, description="Retrieves relevant knowledge, applies reasoning, and generates a structured response dynamically.")
]

# Initialize agent
logging.info("Initializing agent...")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory
)

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    API Endpoint to process user queries.
    Expects a JSON payload: {"query": "your question here"}
    Returns a JSON response with the answer.
    """
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            logging.warning("Received an empty query.")
            return jsonify({"error": "Query is missing"}), 400

        logging.info(f"Received query: {query}")

        response = agent.run(query)

        logging.info(f"Generated response: {response}")

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the request."}), 500

if __name__ == "__main__":
    logging.info("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)