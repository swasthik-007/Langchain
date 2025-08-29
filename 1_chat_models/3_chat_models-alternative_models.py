from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_groq import ChatGroq
load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  
print("Groq key loaded:", bool(os.getenv("GROQ_API_KEY")))
print("Google key loaded:", bool(os.getenv("GOOGLE_API_KEY")))

messages=[
    SystemMessage(content="solve the following problem"),
    HumanMessage(content="what is under root of 50 in decimal, only provide the final answer without any explanation")
]

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.01,  
)


result=llm.invoke(messages)
print(f"answer from groq model 1:{result.content}")

llm2=ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=1,
)
# llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result2=llm2.invoke(messages)
print(f"answer from llm model 2:{result2.content}")