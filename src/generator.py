import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.config import LLM_MODEL

load_dotenv()


class Generator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to the .env file before "
                "running the application."
            )

        self.client = ChatGroq(
            model=LLM_MODEL,
            api_key=api_key,
            temperature=0,
        )

    def generate(self, prompt):
        response = self.client.invoke(prompt)
        return response.content
