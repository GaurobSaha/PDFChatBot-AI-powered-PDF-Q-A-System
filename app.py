#Importing Dependencies

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#Configuring API Keys
#Loads environment variables and sets up Google API keys for AI model access.
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Extracting Text from PDFs
## get a pdf, go through each of the pages of the pdf and extract the text
def get_pdf_text(pdf_files):
    text = ""
    if not pdf_files:
        return text
    for pdf in pdf_files:
        pdf.seek(0)  # Reset file pointer to the beginning
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

##Splitting Text into Chunks
## Splits the extracted text into smaller segments for better processing.

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Creating Vector Embeddings and Storing in FAISS
# Converts text chunks into vector embeddings and stores them using FAISS.
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

##Defining the Conversational QA Chain
## Creates an AI-powered question-answering chain that ensures accurate responses.
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details. If the answer is not in the provided context,
    just say, "The answer is not available in the context." Do not provide the wrong answer.
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#Processing User Input & Running AI Model
#Loads the FAISS vector index, searches for relevant documents, and queries the AI model for answers.
async def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # loading FAISS index from the local
    # FAISS_index consists of the resume PDFs which are converted into vectors
    # So, for similarity search of user_question with the resume database, loading Faiss_Index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = await chain.acall({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])

#Building the Streamlit Web UI
#Creates a simple Streamlit UI for:
#Uploading multiple PDFs
#Asking questions
#Processing and storing the PDF contents

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Gaurob's TOOL")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        asyncio.run(user_input(user_question))

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Submit and Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")

#Running the Application
if __name__ == "__main__":
    main()
