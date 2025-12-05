from src.models.providers import cloud_api
def get_model(config: dict):
    provider = config["provider"]
    model_name = config["model_name"]

    if provider == "cloud":
        return cloud_api.CloudAPIModel(
            model_name=model_name,
            temperature=config.get("temperature", 0.7)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
