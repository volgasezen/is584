
import argparse
from tqdm.auto import tqdm

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from trulens.providers.litellm import LiteLLM
from trulens.apps.langchain import TruChain
from trulens.core import Feedback
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

parser = argparse.ArgumentParser(prog='Langchain_RAG_Trulens')
parser.add_argument('-o', '--output', nargs='?', const='trulens_results.json') 
args = parser.parse_args()

embedder = HuggingFaceEmbeddings(model_name='WhereIsAI/UAE-Large-V1')

vector_store = FAISS.load_local(
    "faiss_index", embedder, allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

llm = ChatOllama(model="llama3.1-8b-q6", temperature=0.1)

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

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

session = TruSession()
session.reset_database()

ollama_provider = LiteLLM(
    model_engine="ollama/llama3.1-8b-q6", api_base="http://localhost:11434"
)

f_qa_relevance = Feedback(
    ollama_provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

context = TruChain.select_context(rag_chain)

f_qs_relevance = (
    Feedback(ollama_provider.context_relevance_with_cot_reasons,
             name="Context Relevance"
            )
    .on_input()
    .on(context.collect())
)

f_groundedness = (
    Feedback(ollama_provider.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context.collect())
    .on_output()
)

tru_recorder = TruChain(
    rag_chain, 
    app_name="Llama3.1 RAG", 
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)

with open('questions.txt') as f:
    eval_questions = f.read().splitlines() 

for question in tqdm(eval_questions):
    with tru_recorder as recording:
        llm_response = rag_chain.invoke(question)

records, feedback = session.get_records_and_feedback()

if args.output:
    records.to_json(args.output)

run_dashboard(session)