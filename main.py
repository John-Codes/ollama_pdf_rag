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
from langchain.schema import Document
import fitz
from semantic_text_splitter import CharacterTextSplitter, HuggingFaceTextSplitter
from tokenizers import Tokenizer
import re


model_local =ChatOllama(model="mistral")
pdf ="/Resume 2024.pdf"
vectorDBDir="UsersVectorDbFiles"
# Get the current working directory
current_directory = os.getcwd()
pathpdf= current_directory+pdf


class OllamaRag:
    def __init__(self):
        pass
    
    def replace_newlines_with_space(self,text):
        # Replace '\n' with a space
        return text.replace('\n', ' ')

    def extract_text_from_pdf(self,file_path):
        print("Path to PDF:", file_path)
        doc = fitz.open(file_path)
        text_contents = ""  # Initialize an empty string to store text
        for page in doc:
            text = page.get_text()
            text_contents += text  # Append the text from each page
        return self.replace_newlines_with_space(text_contents)

    def semantic_text_split_no_model(self,content, maxCharacters):
        try:
            max_characters = 200
            splitter = CharacterTextSplitter(trim_chunks=False)
            chunks_no_model = splitter.chunks(content, max_characters)
            return chunks_no_model
        except Exception as sts:
            print(semantic_text_split_no_model.__name__,sts)

    def semantic_text_split_bert(self,content, max_tokens):
        try:
            
            tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
            splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)
            splitter = CharacterTextSplitter(trim_chunks=False)
            chunks = splitter.chunks(content, max_tokens)
            id=0
            for  chunk in chunks:
                id= id + 1
                print("Chunk:",id,chunk)
                print()

            return chunks
        except Exception as sts:
            print(self.semantic_text_split_no_model.__name__,sts)

    def string_list_to_hf_documents(self,text_list, pathinfo):
        documents = []
        for text in text_list:
            # Create a Document instance for each text string
            doc = Document(page_content=text, metadata={'source': pathinfo})
            documents.append(doc)
        return documents

    def text_spliter_for_vectordbs(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(text)
        
    def print_splits(self,splits):
        
        for i, split in enumerate(splits, start=1):
            print(f"Split {i}:")
            print(split)
            print("------")  # Separator for readability

    def new_temp_chromaDB_and_retriever(self,doc_splits):
        try:
            self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding = embeddings.ollama.OllamaEmbeddings(model='mistral'),
            collection_name="rag_chroma",
            )
            
            self.retriever= self.vectorstore.as_retriever()
            print("New Temp vector Store Created and Initialized")
        except Exception as cmdbe:
            print(self.new_temp_chromaDB_and_retriever.__name__, cmdbe)

    def clean_string_for_file_name(self,txt):
        try:
            match = re.search(r"([^.]*)(\.[^.]*)*$", txt)  # Search for the pattern in the string
            if match:
                txt = match.group(1)  # Extract the part before the last dot
                print(txt)  # Output: "/Resume 2024"
            txt = txt.replace(".","")
            txt = txt.replace(" ","")
            txt = txt.replace("/","")
            txt = txt.replace("@","")
            return txt
        except Exception as csf:
            print(self.clean_string_for_file_name.__name__,csf)

    def set_or_Update_multy_User_file_Structure(self,x_file_name,useremail):
        try:
            # Base directory where all vector DB files will be stored
                base_directory = os.path.join(os.getcwd(), vectorDBDir)
                
                
                clean_file_name=self.clean_string_for_file_name(x_file_name)
                clean_user_email=self.clean_string_for_file_name(useremail)
                # Directory for specific user vectordb
                x_file_vectors_directory = os.path.join(base_directory, f"{clean_user_email}")

                
                # Final directory for the specific vector DB
                persist_directory = os.path.join(x_file_vectors_directory, f"{clean_file_name}Vdb")

                # Ensure the final persist directory exists or create it
                os.makedirs(persist_directory, exist_ok=True)
                print("New directory Craeted: ",persist_directory)
                return persist_directory
        
        except Exception as sou:
            print(self.set_or_Update_multy_User_file_Structure.__name__,)

    def get_user_vectorDB_directory(self,user_email):
        
        try:
            email = self.clean_string_for_file_name(user_email)
            # Construct the path to the VectorDbFiles directory
            base_directory = current_directory = os.getcwd()
            vector_db_directory = os.path.join(base_directory, vectorDBDir)
            
            # Check if the VectorDbFiles directory exists
            if os.path.exists(vector_db_directory) and os.path.isdir(vector_db_directory):
                # Iterate over all directories and subdirectories in the VectorDbFiles directory
                for root, dirs, files in os.walk(vector_db_directory):
                    
                    
                    if email in dirs:
                        # If found, return the full path to the directory
                        return os.path.join(root, email)
        except Exception as gud:
            print(self.get_user_directory,gud)
        
        # If the user's directory is not found, return None
        return None
    
    
    def new_Persisted_chromadb_and_retriever_with_file_structure_for_multiple_users(self, doc_splits, x_file_name,useremail):
        try:
                dir =self.set_or_Update_multy_User_file_Structure(x_file_name,useremail)

                self.vectorstore = Chroma.from_documents(
                    documents=doc_splits,
                    embedding=embeddings.ollama.OllamaEmbeddings(model='mistral'),
                    collection_name="rag_chroma",
                    persist_directory=dir  # Pass the newly formed directory to persist the data
                )
                
                self.retriever = self.vectorstore.as_retriever()
                print(f"New Persisted Vector Store Created and Initialized at {dir}")
        except Exception as cmdbe:
                print(self.new_Persisted_chromadb_and_retriever_with_file_structure_for_multiple_users.__name__, cmdbe)

    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_persisted_vector_DB(self,email, doc_splits):
        try:
                dir =self.get_user_vectorDB_directory(email)

                self.vectorstore = Chroma.from_documents(
                    documents=doc_splits,
                    embedding=embeddings.ollama.OllamaEmbeddings(model='mistral'),
                    collection_name="rag_chroma",
                    persist_directory=dir  # Pass the newly formed directory to persist the data
                )
                
                self.retriever = self.vectorstore.as_retriever()
                print(f"Got Persisted Vector Store in {dir}")
        except Exception as cmdbe:
                print(self.get_persisted_vector_DB.__name__, cmdbe)
 
    def ollama_llm(self,question, context):
        try:    
            formatted_prompt = f"Question: {question}\n\nContext: {context}"
            response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
            return response['message']['content']
        except Exception as oll:
            print(self.ollama_llm.__name__,oll)
    
    def format_docs(self,docs):

        try:    
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as fm:
            print(self.format_docs.__name__,fm)

    def query_temp_rag(self,question):
        try:
            retrieved_docs = self.retriever.invoke(question)
            formatted_context = self.format_docs(retrieved_docs)
            return self.ollama_llm(question, formatted_context)
        except Exception as rg:
            print(self.query_temp_rag.__name__,rg)

    def add_pdf_to_new_temp_rag(self, pathpdf):
            text = self.extract_text_from_pdf(pathpdf)
            splits = self.semantic_text_split_bert(text, 500)
            doc_splits = self.string_list_to_hf_documents(splits, pathpdf)
            doc_splits = self.text_spliter_for_vectordbs(doc_splits)
            o.new_temp_chromaDB_and_retriever(doc_splits)
    
    def load_persisted_PDF_Chromadb_and_retriever(self,pathpdf):
       self.vectorstore= Chroma(persist_directory=pathpdf,embeddings=embeddings.ollama.OllamaEmbeddings(model='mistral'))
       
    def new_persisted_chromadb_single_user_file_structure(self,doc_splits,x_file_name):
        
        try:
            # # Base directory where all vector DB files will be stored
            # base_directory = os.path.join(os.getcwd(), "VectorDbFiles")
            
            # filename = x_file_name.replace(".","")
            # # Directory for specific file vectors
            # x_file_vectors_directory = os.path.join(base_directory, f"{filename}Vectors")

            # self.vectorstore = Chroma.from_documents(
            #         documents=doc_splits,
            #         embedding=embeddings.ollama.OllamaEmbeddings(model='mistral'),
            #         collection_name="rag_chroma",
            #         persist_directory=x_file_vectors_directory  # Pass the newly formed directory to persist the data
            #     )
            
                
            # self.retriever = self.vectorstore.as_retriever()
            # print(f"New Persisted Vector Store Created and Initialized at {x_file_vectors_directory}")
            pass
        except Exception as pcsue:
            print(self.new_Persisted_chromadb_and_retriever_with_file_structure_for_multiple_users.__name__,pcsue)

    


    




if __name__ == "__main__":

       try:
            o = OllamaRag()
            text = o.extract_text_from_pdf(pathpdf)
            splits = o.semantic_text_split_bert(text,200)
            doc_splits = o.string_list_to_hf_documents(splits, pathpdf)
            doc_splits = o.text_spliter_for_vectordbs(doc_splits)
            o.get_persisted_vector_DB("efexzium@gmail.com",doc_splits)
            #o.new_Persisted_chromadb_and_retriever_with_file_structure_for_multiple_users(doc_splits,pdf,"efexzium@gmail.com")
            
            
            #o.add_pdf_to_new_temp_rag(pathpdf)
            #result = o.query_temp_rag("Who is Johnny?")
            #print(result)

        
       except Exception as loade:
            print( loade)