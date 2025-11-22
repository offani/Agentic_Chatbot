from deepagents import create_deep_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# Also add current dir to be safe
sys.path.append(os.getcwd())

from src.agents.web_search import web_search_subagent
from src.agents.multimodal import multimodal_subagent

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

print("Testing Web Search Subagent...")
try:
    agent = create_deep_agent(model=llm, subagents=[web_search_subagent])
    print("Web Search Success!")
except Exception as e:
    print(f"Web Search Failed: {e}")

print("\nTesting Multimodal Subagent...")
try:
    agent = create_deep_agent(model=llm, subagents=[multimodal_subagent])
    print("Multimodal Success!")
except Exception as e:
    print(f"Multimodal Failed: {e}")
