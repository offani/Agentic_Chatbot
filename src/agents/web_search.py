from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

def perform_web_search(query: str) -> str:
    """Performs a web search."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

def get_web_search_tool():
    return Tool(
        name="web_search",
        func=perform_web_search,
        description="Useful for answering questions about current events, facts, or general information available on the internet."
    )

web_search_subagent = {
    "name": "web_search_agent",
    "description": "A specialized agent for finding information on the internet. Use this for questions about current events, facts, weather, or general knowledge.",
    "system_prompt": "You are a web search expert. Your goal is to find accurate information on the internet to answer the user's query. Use the web_search tool.",
    "tools": [get_web_search_tool()]
}
