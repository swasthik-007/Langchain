from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
    temperature=1,
)
chat_history = []

system_message=SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)

while True:
    query=input("You:")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query))

    result=model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}  ")

print("Chat ended.")    