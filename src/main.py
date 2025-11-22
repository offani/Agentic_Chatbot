import os
import sys
import argparse
import io
from dotenv import load_dotenv

# # Force UTF-8 for stdout to handle special characters on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# # Add the project root directory to sys.path to allow imports from 'src'
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from agents.web_search import web_search_subagent
from agents.multimodal import multimodal_subagent

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Intelligent Assistant (Deepagents)")
    parser.add_argument("query", type=str, help="The user's query")
    parser.add_argument("--file", type=str, help="Path to an attached file", default=None)
    
    args = parser.parse_args()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return

    # Initialize LLM
    # Using gemini-1.5-flash-001
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)

    # Create Deep Agent
    agent = create_deep_agent(
        system_prompt=(
            "You are a helpful assistant with access to multiple subagents to answer user queries. "
            "You MUST use the `task` tool to delegate work to sub-agents. "
            "Use `subagent_type='web_search_agent'` for questions about current events, facts, or general information. "
            "Use `subagent_type='multimodal_agent'` for questions about files (txt, PDF, Excel, CSV, Images). "
            "Do NOT try to call subagents directly as tools. Do NOT try to read files yourself."
        ),
        model=llm,
        subagents=[web_search_subagent, multimodal_subagent],
        debug=True # Enable debug to see routing
    )

    # Prepare input
    user_input = args.query
    if args.file:
        user_input += f"\n\n[System Note: The user has attached a file at path: {args.file}. Use the multimodal_agent to analyze it if needed.]"

    print(f"Processing Query: {args.query}")
    if args.file:
        print(f"Attached File: {args.file}")
    print("-" * 40)

    # Run the agent
    try:
        # Initial state
        initial_state = {"messages": [("user", user_input)]}
        

        # Stream output to see steps
        for event in agent.stream(initial_state):
         
            for key, value in event.items():
                print(f"\n[{key}]: {value}")
                # Handle different types of values from LangGraph stream
                if isinstance(value, dict) and 'messages' in value:
                    messages = value['messages']
                    # Check if messages is a list (it might be an Overwrite object or similar)
                    if isinstance(messages, list) and len(messages) > 0:
                        last_msg = messages[-1]
                        if hasattr(last_msg, 'content'):
                            print(f"Msg: {last_msg.content}")
                        else:
                            print(f"Msg: {str(last_msg)}")
                    else:
                        print(f"Update: {str(messages)}")
                else:
                    print(f"Event Data: {str(value)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
