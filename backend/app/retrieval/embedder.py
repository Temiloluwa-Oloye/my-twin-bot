import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# find_dotenv() explicitly hunts down the .env file in your folder tree
load_dotenv(find_dotenv())

class LocalEmbedder(GoogleGenerativeAIEmbeddings):
    """
    A drop-in replacement that hijacks the old local embedder class 
    and routes all embedding math to Google's blazing-fast Gemini API.
    """
    def __init__(self, model_name: str = None, **kwargs):
        # We ignore the old local model_name and force Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing! Python still can't find your .env file.")
            
        super().__init__(
            model="models/gemini-embedding-001", 
            google_api_key=api_key,
            **kwargs
        )