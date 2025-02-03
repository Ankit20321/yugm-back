import json
import os
from fastapi import HTTPException
from sqlmodel import Session, delete
from database import engine, QueryDB

class ChatSession:
    """
    Class for loading and saving chat session history to a database and a JSON file.
    """
    json_file_path = 'chat_sessions.json'  # Path to the JSON file

    @staticmethod
    def _load_from_file():
        """Load chat sessions from a JSON file."""
        if not os.path.exists(ChatSession.json_file_path):
            return {}  # Return empty dictionary if file doesn't exist
        with open(ChatSession.json_file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _save_to_file(data):
        """Save chat sessions to a JSON file."""
        with open(ChatSession.json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_history(session_id):
        """
        Loads a chat session from the database and the JSON file and returns a list
        of the conversations.

        :param session_id: ID of the chat session
        :return: List containing queries and responses from the database and JSON file
        """
        # Load from database
        history_from_db = ChatSession.load_history_db(session_id)

        # Load from JSON file
        history_from_json = ChatSession.load_history_json(session_id)

        # Combine both histories
        combined_history = history_from_db + history_from_json

        return combined_history

    @staticmethod
    def load_history_db(session_id):
        """Load history from the database."""
        with Session(engine) as session:
            statement = f"SELECT * FROM querydb WHERE session_id = '{session_id}'"
            results = session.exec(statement)
            results = [i for i in results]

        result = [
            {'type': 'human', 'data': {'content': tup[1]}}
            for tup in results
        ] + [
            {'type': 'ai', 'data': {'content': tup[2]}}
            for tup in results
        ]

        return result

    @staticmethod
    def load_history_json(session_id):
        """Load history from the JSON file."""
        data = ChatSession._load_from_file()
        return data.get(session_id, [])

    @staticmethod
    def save_sess_db(session_id, query, answer):
        """
        Saves a chat session to the database and JSON file.

        :param session_id: ID of the chat session
        :param query: Query string from the user
        :param answer: Response string from the AI
        """
        # Save to database
        db = QueryDB(query=query, answer=answer, session_id=session_id)
        with Session(engine) as session:
            session.add(db)
            session.commit()
            session.refresh(db)

        # Save to JSON file
        data = ChatSession._load_from_file()
        
        # Create a new session entry if it doesn't exist
        if session_id not in data:
            data[session_id] = []

        # Append the new query and answer to the session
        data[session_id].append({
            'type': 'human',
            'data': {
                'content': query
            }
        })
        data[session_id].append({
            'type': 'ai',
            'data': {
                'content': answer
            }
        })
        
        ChatSession._save_to_file(data)

    @staticmethod
    def delete_sess_db(session_id):
        """
        Delete session from the database and JSON file.

        :param session_id: ID of the chat session
        """
        # Delete from database
        with Session(engine) as session:
            query = delete(QueryDB).where(QueryDB.session_id == session_id)
            result = session.execute(query)
            if result.rowcount == 0:
                raise HTTPException(status_code=404,
                                    detail=f"Session id {session_id} not found")
            session.commit()
        
        # Delete from JSON file
        data = ChatSession._load_from_file()
        if session_id in data:
            del data[session_id]  # Delete the specified session
            ChatSession._save_to_file(data)

        return {'message': f"Session id {session_id} deleted"}
