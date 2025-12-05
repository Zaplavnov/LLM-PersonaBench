import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.models.base import BaseLLM

load_dotenv()

class CloudAPIModel(BaseLLM):
    """
    Wrapper для Cloud API 
    """
    def __init__(self, model_name: str, temperature: float = 0.7):
        api_key = os.environ["CLOUD_API_KEY"]
        base_url = "https://foundation-models.api.cloud.ru/v1"

        super().__init__(model_name)
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        response = prompt | self.llm
        return response.invoke({})

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(p) for p in prompts]
