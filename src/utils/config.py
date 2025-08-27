# Configuration management
from dataclasses import dataclass
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    name: str
    host: str
    database: str
    username: str
    password: str
    port: int = 5432
    ssl: bool = True

@dataclass
class LLMConfig:
    api_key: str
    model: str = "openai/gpt-4-turbo-preview"
    base_url: str = "https://openrouter.ai/api/v1"

class Config:
    # LLM Configuration
    LLM = LLMConfig(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "openai/gpt-4-turbo-preview")
    )
    
    # Database Configurations
    DATABASES = {}
    
    # Load DB1 if configured
    if all(os.getenv(f"DB1_{key}") for key in ["HOST", "NAME", "USER", "PASSWORD"]):
        DATABASES["users_db"] = DatabaseConfig(
            name="users_db",
            host=os.getenv("DB1_HOST"),
            database=os.getenv("DB1_NAME"),
            username=os.getenv("DB1_USER"),
            password=os.getenv("DB1_PASSWORD"),
            port=int(os.getenv("DB1_PORT", 5432))
        )
    
    # Load DB2 if configured
    if all(os.getenv(f"DB2_{key}") for key in ["HOST", "NAME", "USER", "PASSWORD"]):
        DATABASES["transactions_db"] = DatabaseConfig(
            name="transactions_db",
            host=os.getenv("DB2_HOST"),
            database=os.getenv("DB2_NAME"),
            username=os.getenv("DB2_USER"),
            password=os.getenv("DB2_PASSWORD"),
            port=int(os.getenv("DB2_PORT", 5432))
        )
    
    # Development settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")