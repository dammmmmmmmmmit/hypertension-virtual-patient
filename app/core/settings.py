"""Central settings object, read from .env via pydantic-settings. Every
module that needs DATABASE_URL/REDIS_URL/QDRANT_URL/ANTHROPIC_API_KEY should
import `settings` from here rather than reading os.environ directly."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/vps"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    api_base_url: str = "http://localhost:8000"

    # Local-model pivot (see DECISIONS.md #7). generate_report runs
    # against local Ollama (never fine-tuned — a strong off-the-shelf
    # instruct model is the right tool for open-ended prose). parse_query
    # does NOT use Ollama at all — see app/agent/local_finetuned_model.py
    # for why (GGUF export crashed the host machine repeatedly; direct
    # in-process 4-bit inference on the already-merged model was verified
    # safe instead) and always runs through the fine-tuned model, no base
    # model needed.
    ollama_base_url: str = "http://localhost:11434"
    generate_report_model: str = "qwen2.5:14b-instruct"


settings = Settings()
