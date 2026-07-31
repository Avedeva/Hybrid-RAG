import streamlit as st
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import os



st.title("RAG",text_alignment="center")

load_dotenv()
api_key = os.getenv("OPENROUTER_FREE_RAG")

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    ),persist_directory="my_db",
    collection_name='sample'
    )



raw = vector_store.get(include=['documents','metadatas'])

docs = []
for i in raw['documents']:
    docs.append(Document(page_content=i,))






Query = st.chat_input("Ask anything from tokeizer,self attention..")
if not Query:
    st.stop()

with st.spinner("🔁 Retrieving Doc's .... "):
# retrieving using the bm25
    retriever_bm25 = BM25Retriever.from_documents(documents=docs)
    retriever_bm25.k = 5
    docs_retrieve_bm25 = retriever_bm25.invoke(Query)
# retrieving from using cosine similarity
    docs_retrieve_similarity = vector_store.similarity_search(Query,k=5)

with st.spinner("🤖Cross-Encoder Reranker"):
    pass




with st.spinner(" 📚 Getting the best match for the query ...."):
#fusing both result to get besult result
    def rrf(ranked_list,k=60):

        scores = {}
        doc_map = {}

        for i in ranked_list:
            for rank,doc in enumerate(i):
                key = doc.page_content
                scores[key] = scores.get(key,0)+1/(k+rank+1)
                doc_map[key] = doc


        sorted_keys = sorted(scores ,key=lambda x:scores[x],reverse = True)

        return [doc_map[keys] for keys in sorted_keys]


    fused = rrf([docs_retrieve_bm25,docs_retrieve_similarity],k=60)


with st.spinner('🤖 calling an llm... '):
# calling llm to answer the query using docs
    prompt = PromptTemplate(
        template="""You are a helpful assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer based on the context, just say that you don't know.
    Keep the answer concise and accurate. Do not make up information.

    Context:
    {fused}

    Question: {Query}""",
    input_variables=['fused','Query']
    )

    llm = ChatOpenAI(
        model='nvidia/nemotron-3-ultra-550b-a55b:free',
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.4
    )

    parser = StrOutputParser()

    chain = prompt|llm|parser

    result = chain.invoke({'fused':fused,'Query':Query})

st.write(result)