from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./project44.db"
    fast2sms_api_key: str = ""
    demo_sms_mode: bool = True
    ml_model_path: str = "app/ml/artifacts/model.pkl"
    frontend_url: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    opensky_client_id: str = ""
    opensky_client_secret: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
