from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
import os
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","you are a facts expert who knows facts about {animals}."),
        ("human","Tell me {fact_count} facts .")
    ]
)

chain=prompt_template | llm | StrOutputParser()

result=chain.invoke({
    "animals":"elephant",
    
    "fact_count":5
})

console=Console()
console.print(Markdown(result))