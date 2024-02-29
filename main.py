import os
import ollama
#import bs4
import asyncio
#Ollama Rag Youtube
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community import embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

model_local =ChatOllama(model="mistral")
  

from langchain.schema import Document
import fitz
from semantic_text_splitter import CharacterTextSplitter, HuggingFaceTextSplitter
from tokenizers import Tokenizer

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

def replace_newlines_with_space(text):
    # Replace '\n' with a space
    return text.replace('\n', ' ')

def extract_text_from_pdf(file_path):
    print("Path to PDF:", file_path)
    doc = fitz.open(file_path)
    text_contents = ""  # Initialize an empty string to store text
    for page in doc:
        text = page.get_text()
        text_contents += text  # Append the text from each page
    return replace_newlines_with_space(text_contents)

def semantic_text_split_no_model(content, maxCharacters):
    try:
        max_characters = 200
        splitter = CharacterTextSplitter(trim_chunks=False)
        chunks_no_model = splitter.chunks(content, max_characters)
        return chunks_no_model
    except Exception as sts:
        print(semantic_text_split_no_model.__name__,sts)

def semantic_text_split_bert(content, max_tokens):
    try:
        
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)
        splitter = CharacterTextSplitter(trim_chunks=False)
        chunks = splitter.chunks(content, max_tokens)
        return chunks
    except Exception as sts:
        print(semantic_text_split_no_model.__name__,sts)



def string_list_to_hf_documents(text_list, pathinfo):
    documents = []
    for text in text_list:
        # Create a Document instance for each text string
        doc = Document(page_content=text, metadata={'source': pathinfo})
        documents.append(doc)
    return documents

def text_spliter_for_vectordbs(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(text)
    
def print_splits(splits):
     
    for i, split in enumerate(splits, start=1):
        print(f"Split {i}:")
        print(split)
        print("------")  # Separator for readability

def init_chromaDB_and_embeddings(doc_splits):
    vectorstore =Chroma.from_documents(
    documents=doc_splits,
    embedding = embeddings.ollama.OllamaEmbeddings(model='mistral'),
    collection_name="rag_chroma",
    )

    
async def process_pdf(pathpdf):

    try:
        
        # Load the PDF documents using asyncio.to_thread
        #document = await asyncio.to_thread(UnstructuredPDFLoader(pathpdf).load)

        text =extract_text_from_pdf(pathpdf)
        splits = semantic_text_split_bert(text,500)
        doc_splits = string_list_to_hf_documents(splits,pathpdf)
        doc_splits=text_spliter_for_vectordbs(doc_splits)
        #print_splits(splits)
        retriever= vectorstore.as_retriever()

       
    except Exception as loade:
        print(process_pdf.__name__, loade)
        

asyncio.run(process_pdf(pathpdf)) 






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