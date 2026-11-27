import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- EMBEDDING MODEL (pinned) ---
    # Ingestion and queries MUST use the same model, so there is no automatic fallback.
    #   "gemini" -> gemini-embedding-2-preview, 3072-dim
    #   "local"  -> sentence-transformers all-mpnet-base-v2, 768-dim
    # If you change this, re-run ingestion with --wipe.
    # "local" is the default because it runs on this machine: no API key, quota or outage
    # can break retrieval at query time.
    EMBEDDING_MODEL = "local"

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "openai/gpt-oss-120b"
    GROQ_GUARD_MODEL = "openai/gpt-oss-20b"  # small, fast model for the guardrails gate (llama-3.1-8b-instant isn't available on this account)
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # --- PROMPT-INJECTION CLASSIFIER (GROQ) ---
    # Llama Prompt Guard 2 returns a jailbreak probability (0..1) instead of chat text.
    PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
    PROMPT_GUARD_THRESHOLD = 0.5  # block when score > this

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "rag"     # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = "brag"  # fallback: @brag/llama-3.1-8b-instant

    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
