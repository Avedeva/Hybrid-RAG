# #storing the chunks in db's
import pickle
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENROUTER_FREE_RAG")
with open('chunks.pkl','rb') as f:
    chunks = pickle.load(f)
chunks = filter_complex_metadata(chunks) # as was throwing error for the metadata
from hashlib import md5

def store_in_dbs(api_key,directory_loc,chunks):

    vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    ),persist_directory=directory_loc,
    collection_name='sample'
    )

    ids = []
    for chunk in chunks:
            text = chunk.page_content
            source = chunk.metadata.get("source","")
            ids.append(md5((source + text).encode()).hexdigest())

    existing = set(vector_store.get()['ids'])

    new_ids = []
    new_chunks = []

    for chunk,id in zip(chunks,ids):
        if id not in existing:
            new_ids.append(id)
            new_chunks.append(chunk)

    if new_chunks:
        vector_store.add_documents(documents = new_chunks,
                                   ids = new_ids)
        print(f"Added {len(new_chunks)} new chunks.")
    else:
        print("No new documents found.")


    return vector_store

vectore_store = store_in_dbs(api_key,"my_db",chunks)
print(vectore_store)