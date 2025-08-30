from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
import os
from rich.console import Console
from rich.markdown import Markdown
from langchain.schema.runnable import RunnableLambda,RunnablePassthrough, RunnableSequence


load_dotenv()

model=ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","you are a facts expert who knows facts about {animals}."),
        ("human","Tell me {fact_count} facts .")
    ]
)


format_prompt=RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model=RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output=RunnableLambda(lambda x:x.content)

chain=RunnableSequence(first=format_prompt,middle=[invoke_model],last=parse_output)

response=chain.invoke({
    "animals": "cats,dogs",
    "fact_count": 5
})
console=Console()


console.print(Markdown(response))
