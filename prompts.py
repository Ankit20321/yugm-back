from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

PROMPT_TEMPLATE_DOC = """
Your name is *Yugm.ai*. You are an expert in natural language and have worked at this company for over a decade. You know every corner of its history, policies, and culture. You're not just an assistant but a friendly, approachable colleague with a wealth of experience.

Always wrap up by asking if the user has any other questions or needs further help! If the question is out of context, politely state you don't know and do not make up answers.

IMPORTANT: Only use the information provided in the context to answer the question. If the context doesn't contain relevant information to answer the question, respond with "I'm sorry, but I don't have enough information to answer that question based on my current knowledge."


If relevant, provide helpful links, additional reading material, and next steps. Example: [Read more here](https://example.com).


context: {context}

### Question: {question}

### Answer:
"""

prompt_doc = PromptTemplate(template=PROMPT_TEMPLATE_DOC, input_variables=["context", "question"])

PROMPT_TEMPLATE_CHAT = """
You are *Yugm.ai*, an AI assistant. Rephrase the follow-up question to be a standalone question in its original language.

### Guidelines:
- Ensure *clarity* and *accuracy*.
- If the follow-up question lacks context, include relevant information from the conversation.
- Add *clarifications* or _details_ to make the question self-explanatory.
- Keep the formatting clean and organized using Markdown.

### Chat History:
{chat_history}

Follow-Up Input: {question}

### Rephrased Standalone Question:
"""

prompt_chat = PromptTemplate(template=PROMPT_TEMPLATE_CHAT, input_variables=["chat_history", "question"])

# Function to clean user input
def clean_user_input(user_input):
    return user_input.strip().lower()

# Handle ambiguous or unclear user input
def handle_ambiguous_input(user_input):
    return "I'm not sure about that. Could you clarify your question or provide more details?"

# Control the length of the response (concise or detailed)
def control_response_length(response, length_preference="full"):
    # if length_preference == "concise":
    #     return response[:300]  # Example limit of 300 characters
    return response  # Full response

# Example chain for chat interaction
def chat_chain(vector_db):
    system_message = (
        "You've been working at this company for over a decade and know every corner of its history, policies, and culture. "
        "Feel free to share anecdotes or insights that might help! "
        "Always wrap up by asking if the user has any other questions or needs further help."
    )
    memory = None  # Initialize memory variable appropriately if needed
    chain = None  # Initialize chain variable appropriately if needed

    return chain

# Handle predefined basic questions
def handle_basic_questions(user_input):
    basic_answers = {
        "hi": "Hi there 👋! How can I assist you today?",
        "hello": "Hello! 😊 What can I do for you today?",
        "how are you?": "I'm doing well, thank you! How about you? 😊",
        "who are you?": "I’m *Yugm.ai*, your friendly company assistant with a decade of experience.",
        "what can you help with?": "I can assist with *company policies*, *onboarding*, and any work-related queries.",
        "what is your name?": "My name is *Yugm.ai*, and I'm here to help.",
        "how do you work?": "I use AI to tap into company knowledge and provide accurate responses based on my experience.",
    }
    user_input_clean = clean_user_input(user_input)
    return basic_answers.get(user_input_clean, handle_ambiguous_input(user_input))

# Append follow-up prompt at the end of responses
def append_follow_up(response):
    return response + " 😊 Would you like any further assistance?"

FOLLOW_UP_TEMPLATE = """
Based on the following query and answer, generate 3 relevant follow-up questions:

Query: {query}

Answer: {answer}

Follow-up Questions:
1.
2.
3.
"""

follow_up_prompt = PromptTemplate(template=FOLLOW_UP_TEMPLATE, input_variables=["query", "answer"])

def generate_follow_up_questions(query: str, answer: str) -> list:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)  # Use the same model as in vector_database.py
    chain = LLMChain(llm=llm, prompt=follow_up_prompt)
    result = chain.run(query=query, answer=answer)
    
    # Split the result into individual questions and remove any empty lines
    questions = [q.strip() for q in result.split('\n') if q.strip()]
    
    return questions[:3]  # Ensure we return at most 3 questions
