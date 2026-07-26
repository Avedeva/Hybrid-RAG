
# importing all the docs file name
from pathlib import Path
files = Path("PDF's").rglob("*.pdf")
paths = []
for file in files:
    paths.append(file)



# important iports and adding 1 url in paths
from dotenv import load_dotenv
from langchain_docling import DoclingLoader
import os

load_dotenv()
key = os.getenv("HF_TOKEN")

url = "https://nlp.seas.harvard.edu/annotated-transformer/"

paths.append(url)



# loading the DOcs
print("🔁 Loading the Docs")

loader = DoclingLoader(file_path=paths)
docs = loader.lazy_load()
content = []
for i,doc in enumerate(docs,start=1):
    try:
        print(f"📃{i} Loading Paper")
        print(f"🪙 Loading metadata {doc.metadata}\n {"-"*80}")
        content.append({"page_content":doc.page_content,"metadata":doc.metadata})
    except Exception as e:
        print(f"❌ FAILED due to exception {e}")


# printing the docs (seeing the format is correct or not)
for i in content:
    print(i)


# seeing it in a format
from langchain_core.documents import Document
def to_docs(content):
    docs = []
    for i in content:
        docs.append(
        Document(page_content=i['page_content'],metadata=i['metadata'])
        )
    return docs



#  creating chunks for my document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
docs  = to_docs(content)
def get_chunks(docs):
    chunker = RecursiveCharacterTextSplitter(
    separators="\n\n",
    chunk_overlap  = 100)

    chunks = chunker.split_documents(docs)

    return chunks


chunks  = get_chunks(docs)

# how chunks be looking
for i in range(len(chunks)):
    print("CHUNK",{i},)
    print("Content length of chunk",len(chunks[i].page_content),"\n",chunks[i])
    print("\n")
    print("-"*80)



#storing the chunks in db's
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENROUTER_FREE_RAG")
chunks = filter_complex_metadata(chunks) # as was throwing error for the metadata

def store_in_dbs(api_key,directory_loc,chunks):

    vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(
        model='openai/text-embedding-3-small',
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    ),persist_directory=directory_loc,
    collection_name='sample'
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

vectore_store = store_in_dbs(api_key,"my_db",chunks)
print(vectore_store.get())