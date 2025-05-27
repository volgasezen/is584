from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader("dlbook.pdf")
docs = loader.load()

print('1/4: Document loaded')

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap=50)

documents = text_splitter.split_documents(docs)

print('2/4: Recursive chunking done')

# Load the random page content
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings(model_name='WhereIsAI/UAE-Large-V1')

# Create the vector store 
vector = FAISS.from_documents(documents, embedder)

print('3/4: Vector store created')

vector.save_local("faiss_index")

print('4/4: Vector store saved')