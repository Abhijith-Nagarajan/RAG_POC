from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_cpp import Llama
from llama_index.llms.llama_cpp import LlamaCPP
import getpass
import os

os.environ['OPENAI_API_KEY'] = getpass.getpass('Open AI API Key:')
data_dir = r"./Data"

embed_model = HuggingFaceEmbedding(model_name='BAAI/bge-small-en')
# service_context = ServiceContext.from_defaults(embed_model = embed_model) - Deprecated
Settings.embed_model = embed_model # Set globally

#model_path = r'./Models/mistral-7b-instruct-v0.1.Q2_K.gguf'
model_path = 'GIVE MODEL PATH'
llm = LlamaCPP(
    model_path=model_path, 
    temperature=0.7,
    max_new_tokens=512,
    context_window=4096,
    verbose=True,
)  
Settings.llm = llm

documents = SimpleDirectoryReader(data_dir).load_data()
len(documents) 

print(documents[0].__dict__)

print(documents[1])

text_str = ''
for index, doc in enumerate(documents[:3]):
    print(f'Index {index}\nText {doc.text}\n\n')
    text_str += doc.text

#print(text_str)
# 63 - 51 pages from the first paper and 12 pages from the second paper. 
# It seems to be that each page has become a document.

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k = 3)

response = query_engine.query('What is the triplet format for REBEL?')

print(response)