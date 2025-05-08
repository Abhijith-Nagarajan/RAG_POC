from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
data_dir = r"./Data"

import getpass
import os

os.environ['OPENAI_API_KEY'] = getpass.getpass('Open AI API Key:')

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
query_engine = index.as_query_engine()

response = query_engine.query('What is the triplet format for REBEL?')

print(response)