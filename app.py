import json
from fastapi import Body, FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
from dotenv import load_dotenv
from models import DocModel, QueryModel, DeleteSession
from database import FileDB, create_db_and_tables, engine, create_engine
from vector_database import vector_database, db_conversation_chain
from data import load_n_split
from chat_session import ChatSession
from fastapi.staticfiles import StaticFiles
from utils import count_tokens
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import mimetypes
from docx import Document
import csv
import pandas as pd
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select
from uuid import uuid4
from rerank import rank_chunks_with_bm25
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from urllib.parse import quote
from prompts import generate_follow_up_questions
from langchain.chat_models import ChatOpenAI

app = FastAPI()

# Get the absolute path of the 'data' folder
data_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Print the path for debugging purposes
print(f"Serving static files from: {data_folder_path}")

# Mount the 'data' folder
app.mount("/static", StaticFiles(directory=data_folder_path), name="static")

# Get the absolute path of the 'data' folder
data_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Print the path for debugging purposes
print(f"Serving static files from: {data_folder_path}")

# Mount the 'data' folder
app.mount("/static", StaticFiles(directory=data_folder_path), name="static")
load_dotenv()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat-frontend1.vercel.app/login", "https://chat-backend1-wxxi.onrender.com/api/auth/userinfo"],  # Allows all origins, for development. You can specify allowed origins.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Set up logging configuration
logging.basicConfig(
    filename='query_log.log',  # Log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Get the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

chat_session = ChatSession()

# Define the directory path where files should be saved
dir_path: str = "../data"
CONVERTED_FILES_DIR: str = "../converted_files"

# Serve static files from the directory
app.mount("/static", StaticFiles(directory=dir_path), name="static")

@app.get("/files/{file_id}")
async def get_file(file_id: int):
    with Session(engine) as session:
        file_record = session.get(FileDB, file_id)
        if file_record:
            chunks = json.loads(file_record.chunks)  # Convert back to list
            return {"file_name": file_record.file_name, "chunks": chunks}
        else:
            return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/files/{filename}")
async def get_file(filename: str):
    file_path = os.path.join(CONVERTED_FILES_DIR, filename)
    if os.path.isfile(file_path):
        return FileResponse(
            path=file_path,
            media_type='application/octet-stream',
            headers={"Content-Disposition": f'inline; filename="{filename}"'}
        )
    return JSONResponse(status_code=404)

@app.get("/files")
async def list_files():
    try:
        if not os.path.exists(dir_path):
            return JSONResponse(status_code=404, content={"error": "Directory not found"})

        files_by_folder = {}
        
        # Original files stored in dir_path
        for root, dirs, files in os.walk(dir_path):
            relative_folder = os.path.relpath(root, dir_path)
            if files:
                files_by_folder[relative_folder] = [
                    {
                        "file": file,
                        "url": f"http://127.0.0.1:8000/static/{relative_folder}/{file}",  # Original file URL
                        "converted_url": f"http://127.0.0.1:8000/static/converted_files/{file}.html"  # Converted file URL
                    }
                    for file in files
                ]
        
        if not files_by_folder:
            return JSONResponse(status_code=200, content={"message": "No files found"})

        return {"files": files_by_folder}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload")
async def upload_file(
    request: Request,  # Move this to the beginning
    background_tasks: BackgroundTasks, 
    file: UploadFile, 
    folder: str = Form(...), 
    create_new_folder: bool = Form(False)
):
    try:
        # Define folder path and create it if necessary
        folder_path = os.path.join(dir_path, folder)
        if create_new_folder:
            os.makedirs(folder_path, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Generate dynamic static URL using the request object
        base_url = request.base_url
        static_url = f"{base_url}static/{folder}/{file.filename}"

        # Send an immediate response to the frontend
        response = {"message": "File uploaded successfully", "static_url": static_url}

        # Add the document ingestion task to background
        background_tasks.add_task(ingest_document, file_path, file.filename, static_url)

        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
def ingest_document(file_path: str, filename: str, static_url: str):
    try:
        # Load and split the document into chunks
        chunks = load_n_split(file_path)

        # Store file information and chunks in the database
        with Session(engine) as session:
            new_file = FileDB(
                file_name=filename,
                static_url=static_url,
                chunks=json.dumps([chunk.page_content for chunk in chunks])  # Store chunks as JSON string
            )
            session.add(new_file)
            session.commit()

        # Ingest the document chunks into the vector database
        vector_database(
            doc_text=chunks,  # Pass the chunks to be indexed
            collection_name="betaCollection",  # Replace with your actual collection name
            embeddings_name="openai"   # Replace with the embeddings model you are using
        )

        print(f"File '{filename}' has been ingested successfully.")

    except Exception as e:
        print(f"Error during ingestion: {e}")

async def add_documents(doc: DocModel):
    # doc.dir_path should be a string path to the directory containing documents
    if isinstance(doc.dir_path, str):
        docs = load_n_split(doc.dir_path)  # Use the directory path directly
    else:
        return {"message": "Invalid directory path"}
    
    vector_database(
        doc_text=docs,  # This should be appropriate type for vector_database
        collection_name=doc.collection_name,
        embeddings_name=doc.embeddings_name
    )
    return {"message": "Documents added successfully"}



@app.post("/doc_ingestion")
async def doc_ingestion(doc: DocModel):
    return await add_documents(doc)



def convert_file(file_path: str, file_extension: str):
    """
    Converts various document types to text format.
    
    Args:
        file_path (str): The path of the file to be converted.
        file_extension (str): The file extension (type) of the document.
    """
    try:
        # Detect file type and handle accordingly
        if file_extension == ".docx":
            # Convert DOCX to TXT
            doc = Document(file_path)
            content = ''
            for para in doc.paragraphs:
                content += para.text + "\n"
            
            # Save the converted text to a .txt file
            txt_file_path = os.path.join(CONVERTED_FILES_DIR, os.path.basename(file_path).replace(".docx", ".txt"))
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(content)

        elif file_extension == ".csv":
            # Convert CSV to TXT
            with open(file_path, newline='', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                content = ''
                for row in reader:
                    content += ','.join(row) + '\n'
            
            # Save the converted text to a .txt file
            txt_file_path = os.path.join(CONVERTED_FILES_DIR, os.path.basename(file_path).replace(".csv", ".txt"))
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(content)

        elif file_extension == ".xlsx":
            # Convert XLSX to TXT
            df = pd.read_excel(file_path)
            content = df.to_string(index=False)
            
            # Save the converted text to a .txt file
            txt_file_path = os.path.join(CONVERTED_FILES_DIR, os.path.basename(file_path).replace(".xlsx", ".txt"))
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(content)

        elif file_extension == ".epub":
            # Convert EPUB to TXT
            book = epub.read_epub(file_path)
            content = ''
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    content += soup.get_text() + "\n"

            # Save the converted text to a .txt file
            txt_file_path = os.path.join(CONVERTED_FILES_DIR, os.path.basename(file_path).replace(".epub", ".txt"))
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(content)

    except Exception as e:
        print(f"Error converting file {file_path}: {str(e)}")

def convert_existing_files():
    """
    Convert all existing files in the data directory upon startup.
    """
    os.makedirs(CONVERTED_FILES_DIR, exist_ok=True)

    # Iterate through files in the original data directory
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = Path(file).suffix.lower()

            # Only convert files with supported extensions
            if file_extension in ['.docx', '.csv', '.xlsx', '.epub']:
                convert_file(file_path, file_extension)

@app.on_event("startup")
def on_startup():
    """
    Event handler called when the application starts up.
    """
    # Ensure the data directory exists when the app starts
    os.makedirs(dir_path, exist_ok=True)
    create_db_and_tables()
    # Convert existing files to the supported formats
    convert_existing_files()
    

    
@app.post("/query")
def query_response(query: QueryModel):
    """
    Endpoint to process user queries.
    Automatically generates a session_id if not provided.
    """
    if not query.session_id:
        query.session_id = str(uuid.uuid4())

    # Load previous conversation history if it exists
    stored_memory = chat_session.load_history(query.session_id)

    # Get conversation chain with stored memory
    chain = db_conversation_chain(
        stored_memory=stored_memory,
        llm_name=query.llm_name,  # Use the llm_name from the query
        collection_name=query.collection_name
    )

    # Call LLM and calculate token cost
    result, cost = count_tokens(chain, query.text)

    # Check if any relevant information was found
    if not result.get('source_documents'):
        return {
            "answer": "I'm sorry, but I couldn't find any relevant information in my knowledge base to answer your question. Could you please rephrase your question or ask about a different topic?",
            "cost": cost,
            "ranked_chunks": [],
            "follow_up_questions": []
        }

    # Extract sources and chunks from the result
    source_documents = result.get('source_documents', [])
    sources = list(set([
        doc.metadata.get('source', '').replace('\\', '/').replace('data/', '')
        for doc in source_documents if isinstance(doc.metadata.get('source', ''), str)
    ]))
    chunks = [doc for doc in source_documents]

    # Use BM25 to rerank the chunks based on relevance to the query
    reranked_chunks = rank_chunks_with_bm25(chunks, query.text)

    # Extract chunk text and BM25 score
    ranked_chunks = [
        {"text": chunk.page_content, "bm25_score": score}
        for chunk, score in reranked_chunks
    ]

    # Prepare the final response
    answer = result.get('answer', '')

    # List of out-of-context queries to avoid attaching sources
    out_of_context_queries = ['hi', 'hello', 'hey', 'thank you', 'sorry']

    # Handle unknown or insufficient knowledge queries
    if not answer or "I don't know" in answer.lower() or 'sorry' in answer.lower():
        final_answer_with_sources = answer  # No sources for unknown queries
    elif query.text.lower().strip() not in out_of_context_queries and len(query.text.split()) > 1:
        # Format the sources properly for rendering as cards
        formatted_sources = [
            {
                "file_name": source.split("/")[-1],  # Extract the file name from the URL
                "static_url": f"http://127.0.0.1:8000/static/{quote(source.split('/')[-1])}"  # Correct static URL here
            }
            for source in sources
        ]
        # Attach the formatted sources to the answer once, avoiding duplication
        final_answer_with_sources = f"{answer}\n\n### Sources:\n" + "\n".join(
            [f'- <a href="{source["static_url"]}" target="_blank">Source {i+1}: {source["file_name"]}</a>' for i, source in enumerate(formatted_sources)]
        )
    else:
        # No sources for out-of-context queries
        final_answer_with_sources = answer

    # Generate follow-up questions
    follow_up_questions = generate_follow_up_questions(query.text, final_answer_with_sources)

    # Don't append follow-up questions to the response
    final_response = final_answer_with_sources

    # Save the session information in the database
    chat_session.save_sess_db(query.session_id, query.text, final_response)

    # Log the query, response, and ranked chunks with BM25 score
    log_data = {
        "session_id": query.session_id,
        "query": query.text,
        "response": final_response,
        "ranked_chunks": ranked_chunks,
        "bm25_scores": [chunk['bm25_score'] for chunk in ranked_chunks],
        "sources": formatted_sources if query.text.lower().strip() not in out_of_context_queries else []
    }

    return {
        "answer": final_response,
        "cost": cost,
        "ranked_chunks": ranked_chunks,
        "follow_up_questions": follow_up_questions  # Add this line
    }



@app.delete("/delete")
async def delete_file(folder: str = Body(...), fileName: str = Body(...)):
    try:
        file_path = os.path.join(dir_path, folder, fileName)

        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "File not found"})

        os.remove(file_path)
        return {"message": f"File '{fileName}' deleted successfully from {folder}."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Endpoint to fetch existing folders
@app.get("/folders")
async def get_folders():
    try:
        # Check if the data directory exists
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Get all folders in the directory
        folders = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
        return {"folders": folders}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})