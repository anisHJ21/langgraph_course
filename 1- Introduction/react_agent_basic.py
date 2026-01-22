from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool
import os

load_dotenv()
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")


search_tool = TavilySearch(max_results=2 , topic="general")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

tools = [search_tool, get_system_time]

agent = create_agent(model,tools=tools)

# agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")

user_input = "When was SpaceX's last launch and how many days ago was that from this instant"

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()