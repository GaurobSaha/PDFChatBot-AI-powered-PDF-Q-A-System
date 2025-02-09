"# PDFChatBot-AI-powered-PDF-Q-A-System" 

Project Name: PDFChatBot - AI-powered PDF Q&A System

Project Description:
PDFChatBot is a Streamlit-based web application that allows users to upload multiple PDF documents and interactively ask questions about their content. The system leverages Google's Gemini AI and FAISS vector storage for efficient retrieval-based answering.

The workflow involves:
Text Extraction: Extracts text from uploaded PDFs.
Chunking: Splits text into manageable segments.
Embedding & Indexing: Converts text chunks into vector representations and stores them in a FAISS index.
Question Answering: Uses Gemini AI to answer user queries based on vector similarity search.