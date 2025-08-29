
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
load_dotenv()
import os
PROJECT_ID="langchain-be5bf"

SESSION_ID="user_session_new"
COLLECTION_NAME="chat_history"

# print("initialising Firestore CLient ✔️")
client=firestore.Client(project=PROJECT_ID)
chat_history=FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
    temperature=1,
)

while True:
    query=input("You:")
    if query.lower()=="exit":
        break
    chat_history.add_user_message(HumanMessage(content=query))

    result=model.invoke(chat_history.messages)
    response = result.content
    chat_history.add_ai_message(AIMessage(content=response))

    print(f"AI: {response}  ")

print("Chat ended.")    