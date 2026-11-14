from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"

    # Models
    target_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    draft_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    num_speculative_tokens: int = 4

    # PagedAttention
    page_size: int = 16
    max_num_pages: int = 2048
    kv_cache_mem_fraction: float = 0.90

    # Scheduler
    max_num_seqs: int = 256
    max_model_len: int = 8192

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
