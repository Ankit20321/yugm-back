from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from data import load_n_split
from utils import get_settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi import HTTPException
import pymilvus
from langchain.llms import GPT4All
import os
from langchain.schema import messages_from_dict
from prompts import prompt_doc, prompt_chat
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
# Directory containing the files
directory_path = '../data'


def vector_database(collection_name, doc_text=None, drop_existing_embeddings=False, embeddings_name='sentence'):
    """
    Creates and returns a Milvus database based on the specified parameters.
    
    Args:
        doc_text: The document text.
        collection_name: The name of the collection.
        drop_existing_embeddings: Whether to drop existing embeddings.
        embeddings_name: The name of the embeddings ('openai' or 'sentence').
        chunk_size: Maximum number of characters in a chunk.
        chunk_overlap: Number of characters of overlap between chunks.
    
    Returns:
        The Milvus database.
    """
    # Determine the embeddings model
    if embeddings_name == 'openai':
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading OpenAI embeddings: {e}")
    # elif embeddings_name == 'sentence':
    #     try:
    #         embeddings = HuggingFaceEmbeddings()
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=f"Error loading HuggingFace embeddings: {e}")
    # else:
        # raise ValueError('Invalid embeddings option. Choose either "openai" or "sentence".')

    if doc_text:
        try:
            vector_db = Milvus.from_documents(
                doc_text,  
                embeddings,
                collection_name=collection_name,
                drop_old=drop_existing_embeddings,
                connection_args={"host": "localhost", "port": "19530", "timeout": 60},
                # change localhost to get.settings().milvus_host when using docker
                
            )
        except pymilvus.exceptions.ParamError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{collection_name}' already exists. Set drop_existing_embeddings=True to overwrite. Error: {e}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create Milvus database: {e}")
    else:
        try:
            vector_db = Milvus(
                embeddings,
                collection_name=collection_name,
                connection_args={"host": "localhost", "port": "19530"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error initializing Milvus: {e}")

    return vector_db


def get_chat_history(inputs):
    """
    Get human input only
    """
    inputs = [i.content for i in inputs]
    return '\n'.join(inputs)

def db_conversation_chain(llm_name, stored_memory, collection_name):
    """
    Creates and returns a ConversationalRetrievalChain based on the specified parameters.
    
    Args:
        llm_name: The name of the language model ('openai', 'gpt4all', or 'llamacpp').
        stored_memory: Existing conversation.
        collection_name: The name of the collection (optional).
    
    Returns:
        The ConversationalRetrievalChain.
    """
    if llm_name == 'openai':
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            openai_api_key=get_settings().openai_api_key,
            temperature=0.3,
            verbose=False
        )
        embeddings_name = 'openai'

    else:
        raise ValueError('Invalid LLM name.')

    # Initialize the vector database
    vector_db = vector_database(
        collection_name=collection_name,
        embeddings_name=embeddings_name
    )

    # Initialize chat history memory
    if stored_memory:
        try:
            retrieved_messages = messages_from_dict(stored_memory)
            chat_history = ChatMessageHistory(messages=retrieved_messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing chat history: {e}")
    else:
        chat_history = ChatMessageHistory()

    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer',
        chat_memory=chat_history
    )

    # Create and return the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),  # Adjust k as needed
        memory=memory,
        chain_type="stuff",
        return_source_documents=True,
        verbose=True,
        condense_question_prompt=prompt_chat,
        return_generated_question=True,
        get_chat_history=get_chat_history,
        combine_docs_chain_kwargs={
            "prompt": prompt_doc,
            "document_variable_name": "context"
        }
    )
    return chain
