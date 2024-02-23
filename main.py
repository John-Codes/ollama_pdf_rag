import os
import ollama
import bs4
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
import fitz

pdf ="/Resume 2024.pdf"
# Get the current working directory
current_directory = os.getcwd()

pathpdf= current_directory+pdf
# loader = WebBaseLoader(
#     web_paths=("https://en.wikipedia.org/wiki/Bitcoin",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# Create the retriever
def load_pdf_with_fitz(file_path):
    print("path to pdf:",pathpdf)
    doc = fitz.open(file_path)
    text_contents = []
    for page in doc:
        text = page.get_text()
        text_contents.append(text)
    return text_contents

async def load_pdf_and_split(pathpdf):

    try:
        
        # Load the PDF documents using asyncio.to_thread
        #document = await asyncio.to_thread(UnstructuredPDFLoader(pathpdf).load)
        #print(document)
        document =load_pdf_with_fitz(pathpdf)
        # text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
        
        # Initialize the text splitter
        
        # Create Document instances from loaded PDF documents
        docs = [Document(page_content=page, metadata={'source': pathpdf}) for page in document]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Split the documents
        splits = text_splitter.split_documents(docs)
        
        # Create Ollama embeddings and vector store
        embeddings = OllamaEmbeddings(model="mistral")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        return splits
    except Exception as loade:
        print("Load error:", loade)
        

asyncio.run(load_pdf_and_split(pathpdf)) 






def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    pass
    #retrieved_docs = retriever.invoke(question)
    #formatted_context = format_docs(retrieved_docs)
    #return ollama_llm(question, formatted_context)

# Use the RAG chain
#result = rag_chain("What is Task Decomposition?")
#print(result)