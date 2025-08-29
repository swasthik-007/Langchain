from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.markdown import Markdown


load_dotenv()

llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-2.5-flash",
    temperature=1,
)

# template = "Write a {tone} email to {company} expressing interest in the {position} ,mentioning" \
# "{skill} as a key strength. keep it to 4 lines max"


# prompt_template=ChatPromptTemplate.from_template(template)

# prompt=prompt_template.invoke({
#     "tone":"energetic",
#     "company":"GOOGLE",
#     "position":"SDE",
#     "skill":"DSA"
# })


messages=[
    ("system","You are a comedian who tells joke about {topic}."),
    ("human","Tell me {Joke_count} jokes."),
]


prompt_template=ChatPromptTemplate.from_messages(messages)
prompt=prompt_template.invoke({
    "topic":"programming",
    "Joke_count":5
})
# print(prompt)
result=llm.invoke(prompt)
# print(result.content)


console = Console()
md = Markdown(result.content)
console.print(md)