import os
import pandas as pd
from pypdf import PdfReader
from PIL import Image
from langchain_core.tools import tool
import mimetypes

@tool
def read_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

@tool
def read_structured_data(file_path: str) -> str:
    """Reads Excel or CSV files and returns markdown representation."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df.to_markdown(index=False)
    except Exception as e:
        return f"Error reading structured data: {str(e)}"

@tool
def analyze_image_placeholder(file_path: str, query: str) -> str:
    """Analyzes an image based on a query (Placeholder for VLM)."""
    return f"[Image Analysis of {file_path}] based on query: {query}. (Requires VLM integration)"

@tool
def read_text_file(file_path: str) -> str:
    """Reads a plain text file."""
    print(f"DEBUG: read_text_file called with {file_path}")
    try:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"DEBUG: read_text_file read {len(content)} chars")
            return content
    except Exception as e:
        print(f"DEBUG: read_text_file error: {e}")
        return f"Error reading text file: {str(e)}"

multimodal_subagent = {
    "name": "multimodal_agent",
    "description": "A specialized agent for analyzing files (PDFs, Excel, CSV, Images, Text). Use this when the user provides a file path or asks to analyze a document.",
    "system_prompt": "You are a multi-modal analysis expert. Your goal is to extract information from the provided files. You HAVE tools to read files. You MUST use them. Use `read_text_file` for plain text files. Do not say you cannot read the file. Just use the tool.",
    "tools": [read_pdf, read_structured_data, read_text_file, analyze_image_placeholder]
}
