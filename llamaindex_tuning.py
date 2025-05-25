from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import os, getpass
import time

os.environ['OPENAI_API_KEY'] = getpass.getpass('Open AI API Key:')

folder_path = r"./Data"

# Step 1 - Load and read all files within a folder 
docs = SimpleDirectoryReader(folder_path).load_data()

model_path = r"E:\RAG_Models\mistral-7b-instruct-v0.1.Q5_K_M.gguf"
llm = LlamaCPP(
    model_path=model_path, 
    temperature=0.35,
    max_new_tokens=512,
    context_window=4096,
    verbose=True,
)  
Settings.llm = llm

# Step 2 - To experiment with chunk size and overlap
text_splitter = SentenceSplitter(chunk_size = 300, chunk_overlap = 25)
Settings.node_parser = text_splitter

# Step 3 - Initialize the vector database
print(Settings.embed_model)
Settings.embed_model = HuggingFaceEmbedding(model_name = "all-MiniLM-L6-v2")
dimensions = len(Settings.embed_model.get_text_embedding("sample text"))

faiss_index = faiss.IndexFlatL2(dimensions)
faiss_db = FaissVectorStore(faiss_index = faiss_index)
storage_context = StorageContext.from_defaults(vector_store = faiss_db)

# Step 4 - Pass the docs to the VectorStoreIndex
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

# Step 5 - Adding memory 
memory = ChatMemoryBuffer.from_defaults(token_limit = 3000)
query_engine = index.as_query_engine(similarity_top_k = 5)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine = query_engine,
    memory = memory
)

response = chat_engine.chat('What is the triplet format for REBEL?')
print(response)

type(response)

# Checking the performance of FAISS
retriever = index.as_retriever(similarity_top_k = 5)
start = time.time()
nodes = retriever.retrieve("What is the triplet format for REBEL?")
print(f"FAISS took {time.time() - start:.2f} seconds")

first_node = nodes[0]

first_node.metadata
first_node.score
first_node.get_content()
