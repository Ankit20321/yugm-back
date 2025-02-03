import uuid
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, List
import os
import re
from fastapi import HTTPException
from utils import get_settings
from typing import Optional, Literal
from dotenv import load_dotenv


load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

def validate_uuid(uuid_string):
    """
    Regex pattern to validate uuid
    """
    pattern = re.compile(
        r'^[0-9a-fA-F]{8}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{12}$'
    )
    return re.match(pattern, uuid_string)


class DocModel(BaseModel):
    """
    Model for document processing.
    """
    dir_path: str = Field("../data", const=True)
    embeddings_name: Optional[Literal['openai', 'sentence']] = 'openai'
    collection_name: Optional[str] = 'betaCollection'
    drop_existing_embeddings: Optional[bool] = False
    chunk_size: Optional[int] = 1000  # Optimized chunk size for precision
    chunk_overlap: Optional[int] = 200  # Small overlap to maintain context without redundancy

    @validator('dir_path')
    def validate_dir_path(cls, dir_path):
        """
        Validates the directory path.
        """
        if not os.path.exists(dir_path):
            raise ValueError('Directory path must start with a forward slash')
        return dir_path

    @validator('embeddings_name')
    def validate_embeddings(cls, embeddings_name):
        """
        Validates the embeddings name.
        """
        if embeddings_name == 'openai':
            key = openai_api_key
            if not key.startswith('sk'):
                raise ValueError('The API is not valid or not provided')
        return embeddings_name


class QueryModel(BaseModel):
    """
    Represents a query model with text, session_id, and optional parameters.
    """

    text: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    llm_name: Optional[Literal['openai', 'llamacpp', 'gpt4all']] = 'openai'
    collection_name: Optional[str] = 'betaCollection'
    temperature: Optional[float] = 0.3  # Set to zero to prevent creative generation

    @validator('text')
    def validate_text(cls, text):
        """
        Validates the 'text' field to ensure it is not empty.
        """
        if not text:
            raise ValueError('Text must be provided')
        return text

    # This function can limit the chat history to the last 10 interactions
    def add_to_history(self, query: str, response: str):
        self.history.append(f"Q: {query}\nA: {response}")
        if len(self.history) > 10:
            self.history = self.history[-10:]

# manage session id
    @validator('session_id')
    def validate_session_id(cls, session_id):
        """
        Validates the 'session_id' field to ensure it is a valid uuid4 format.
        """
        if not validate_uuid(session_id):
            raise ValueError('Session ID must be in uuid4 format')
        return session_id

    @validator('llm_name')
    def validate_llm_name(cls, llm_name):
        """
        Validates the 'llm_name' field based on specific conditions for different values.
        """
        if llm_name == 'openai':
            key = openai_api_key
            # if not key.startswith('sk'):
            #     raise ValueError('The API is not valid or not provided')

        if llm_name == 'gpt4all':
            if not os.path.isfile('llms/ggml-gpt4all-j.bin'):
                raise HTTPException(status_code=404, detail="Model weights are not found")

        if llm_name == 'llamacpp':
            if not os.path.isfile('llms/ggml-gpt4all-l13b-snoozy.bin'):
                raise HTTPException(status_code=404, detail="Model weights are not found")    
  
        return llm_name


class DeleteSession(BaseModel):
    """
    Represents a request to delete a session.
    """
    session_id: str

    @validator('session_id')
    def validate_session_id(cls, session_id):
        """
        Validates the 'session_id' field to ensure it is a valid uuid4 format.
        """
        if not validate_uuid(session_id):
            raise ValueError('Session ID must be in uuid4 format')
        return session_id