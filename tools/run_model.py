import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.registry import get_model
from src.utils.config_loader import load_config
from langchain_core.prompts import ChatPromptTemplate

def main():
    if len(sys.argv) < 2:
        print("Использование: python run_model.py <путь_к_конфигу>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    model_config = config["model"]
    prompt_config = config["prompt"]

    # Создаём объект модели через Registry
    model = get_model(model_config)

    # Создаём LangChain prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt_config["system"]),
        ("human", prompt_config["user"])
    ])


    response = model.generate(prompt_template)
    print(f"Ввод: {prompt_config['user']}")
    print(f"Ответ модели: {response.content}")

if __name__ == "__main__":
    main()

