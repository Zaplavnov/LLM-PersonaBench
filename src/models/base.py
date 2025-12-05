class BaseLLM:
    """
    Абстрактный класс для всех моделей.
    Симулятор обращается только к нему.
    """
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Генерирует ответ на промпт.
        """
        raise NotImplementedError()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Генерирует ответы на несколько промптов.
        """
        raise NotImplementedError()
    
    def info(self) -> str:
        return f"LLM Model: {self.model_name}"
