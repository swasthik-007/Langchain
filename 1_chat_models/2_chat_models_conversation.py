from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


messages=[
    SystemMessage("you are an expert in social media content strategy "),
    HumanMessage("Give a short tip to create engaging post on Instagram"),
    
]


llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.01,  
)

result = llm.invoke(messages)
print(result.content)