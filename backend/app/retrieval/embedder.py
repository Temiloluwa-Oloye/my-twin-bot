import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class LocalEmbedder(GoogleGenerativeAIEmbeddings):
    """
    A drop-in replacement that hijacks the old local embedder class 
    and routes all embedding math to Google's blazing-fast Gemini API.
    """
    def __init__(self, model_name: str = None, **kwargs):
        # We ignore the old local model_name and force Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables!")
            
        super().__init__(
            model="models/text-embedding-004", 
            google_api_key=api_key,
            **kwargs
        )