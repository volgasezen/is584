# %%
import argparse

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

parser = argparse.ArgumentParser(prog='Langchain_RAG')
parser.add_argument('-q', '--question') 
args = parser.parse_args()
# %%
embedder = HuggingFaceEmbeddings(model_name='WhereIsAI/UAE-Large-V1')

vector_store = FAISS.load_local(
    "faiss_index", embedder, allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# %%
llm = ChatOpenAI(
    model="LLaMA_CPP", 
    base_url="http://localhost:8081/v1",
    api_key = "sk-no-key-required",
    temperature=0.1
)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three to four sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""

prompt = ChatPromptTemplate.from_template(template) 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever_chain = retriever | format_docs

rag_chain = (
    {"context": retriever_chain, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%

print(f'Retrieved context: \n {retriever_chain.invoke(args.question)}')

print(f'LLM Response: \n{rag_chain.invoke(args.question)}')
