"""Application settings, loaded from environment / .env file."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=("model_validate", "model_dump"),
    )

    # --- OpenAI (or any OpenAI-compatible provider: Gemini, Ollama, Groq...) ---
    openai_api_key: str = Field(default="", repr=False)
    openai_base_url: str | None = None  # set to use a non-OpenAI compatible endpoint
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Optional chat-only override; embeddings/native rerank keep the shared provider.
    deepseek_api_key: str = Field(default="", repr=False)
    deepseek_base_url: str | None = None

    # SiliconFlow is explicit opt-in; old OpenAI defaults remain reproducible.
    model_provider: Literal["openai", "siliconflow"] = "openai"
    llm_rerank_model: str | None = None  # falls back to openai_chat_model
    # SiliconFlow rerank only: None omits the wire field (historical provider default).
    llm_rerank_enable_thinking: bool | None = None
    native_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Native reranking is NOT wired into search: LLM rerank remains the baseline.
    siliconflow_pricing_file: str = "configs/providers/siliconflow.pricing.json"
    siliconflow_cost_log: str = "runs/siliconflow-costs.jsonl"
    # Deprecated: used only by historical standalone smoke scripts, not runtime/new capture.
    siliconflow_max_cost_cny: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    siliconflow_llm_max_tokens: int = Field(default=1024, gt=0)
    siliconflow_generation_max_tokens: int = Field(default=2048, gt=0)

    @model_validator(mode="after")
    def _provider_roles(self) -> Settings:
        if not self.openai_api_key.strip():
            raise ValueError("Set OPENAI_API_KEY in the local .env file")
        if self.deepseek_base_url and self.deepseek_base_url != "https://api.deepseek.com/v1":
            raise ValueError("DeepSeek requires https://api.deepseek.com/v1")
        if self.llm_rerank_enable_thinking is not None and self.model_provider != "siliconflow":
            raise ValueError("LLM_RERANK_ENABLE_THINKING is supported only by SiliconFlow")
        if self.model_provider == "siliconflow":
            if self.openai_base_url != "https://api.siliconflow.cn/v1":
                raise ValueError("SiliconFlow requires the explicit official HTTPS v1 URL")
            if self.openai_embedding_model != "BAAI/bge-m3" or self.embedding_dim != 1024:
                raise ValueError("SiliconFlow BGE-M3 requires 1024 dense dimensions")
            for model in (self.openai_chat_model, self.llm_rerank_model):
                if model and ("reranker" in model.lower() or "bge-" in model.lower()):
                    raise ValueError("Embedding/native reranker models cannot serve chat roles")
        return self

    # --- Database (Postgres + pgvector) ---
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    # --- RAG tuning ---
    chunk_size: int = 800  # tokens per chunk
    chunk_overlap: int = 120  # token overlap between chunks
    top_k: int = 4  # chunks passed to the LLM per query
    embedding_dim: int = 1536  # 1536 for text-embedding-3-small

    # --- Retrieval pipeline ---
    retrieval_candidates: int = 10  # candidates pulled per retriever before fusion/rerank
    use_hybrid: bool = True  # combine vector + keyword search via RRF
    use_rerank: bool = True  # LLM re-ranking of the fused candidates
    # Opt-in experiment; keep the historical prefix baseline as the default.
    rerank_passage_mode: Literal["prefix_600", "full_chunk"] = "prefix_600"
    # Explicit prompt experiments; historical scoring contract remains the default.
    rerank_prompt_version: Literal["baseline", "keyed_v1", "relevance_v1"] = "baseline"
    min_relevance: float = 0.12  # if the top match's cosine sim is below this,
    # decline instead of answering off-topic questions

    # --- Server / deploy ---
    run_db_init: bool = True  # bootstrap schema on startup; set false on serverless
    application_startup_check: bool = False  # Compose forces true; no implicit DDL there
    database_connect_timeout_seconds: int = Field(default=3, ge=1, le=10)
    session_cookie_secure: bool = True  # explicitly disable only for local HTTP demos
    session_ttl_seconds: int = Field(default=43200, ge=60, le=604800)

    # --- Application admission and provider resilience (not a spend budget) ---
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=10, ge=1, le=10000)  # per account
    application_pricing_file: str = "configs/providers/application-budget.example.json"
    application_concurrency: int = Field(default=4, ge=1, le=32)
    account_concurrency: int = Field(default=2, ge=1, le=16)
    application_generation_max_tokens: int = Field(default=2048, ge=1, le=16384)
    application_rerank_max_tokens: int = Field(default=1024, ge=1, le=16384)
    provider_timeout_seconds: float = Field(default=20, ge=0.1, le=120)
    provider_max_attempts: int = Field(default=2, ge=1, le=3)
    provider_retry_backoff_seconds: float = Field(default=0.25, ge=0, le=2)

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, value: str) -> str:
        """Accept plain Postgres URLs (Neon, Supabase, Heroku, etc.) and route
        them through the psycopg3 driver SQLAlchemy expects."""
        for prefix in ("postgresql://", "postgres://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value[len(prefix) :]
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
