from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str = "changeme"
    api_title: str = "Messages Classification API"
    api_version: str = "1.0.0"
    
    model_path: str = "model_output"
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    cors_origins: str = "*"
    
    @property
    def cors_origins_list(self) -> List[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
