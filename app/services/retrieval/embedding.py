import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_DIMS = {"gemini": _GEMINI_DIM, "local": _FALLBACK_DIM}

_active_model = None
_model_type: str | None = None  # "gemini" or "local", always equal to settings.EMBEDDING_MODEL


# ── Model initialisation ───────────────────────────────────────────────────────
# The model is PINNED by settings.EMBEDDING_MODEL. There is deliberately no automatic
# fallback: if ingestion used one model and queries used another, the vectors would
# not match and Qdrant search would fail (e.g. "expected dim: 768, got 3072").

def _load_gemini():
    """Load Gemini and verify it with one call. Raises if Gemini is unreachable."""
    model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        google_api_key=settings.GEMINI_API_KEY,
    )
    try:
        model.embed_query("probe")
    except Exception as e:
        raise RuntimeError(
            f"Gemini embeddings unavailable: {e}. EMBEDDING_MODEL is pinned to 'gemini', "
            "so there is no fallback. Fix the key/quota, or switch EMBEDDING_MODEL to "
            "'local' and re-run ingestion with --wipe."
        ) from e
    logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
    return model


def _load_local():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading local sentence-transformers model (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Load the pinned embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    choice = settings.EMBEDDING_MODEL
    if choice not in _DIMS:
        raise ValueError(f"EMBEDDING_MODEL must be 'gemini' or 'local', got {choice!r}.")

    _active_model = _load_gemini() if choice == "gemini" else _load_local()
    _model_type = choice


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension of the pinned model (no API call needed)."""
    return _DIMS[settings.EMBEDDING_MODEL]


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        # Exponential backoff: 1 s → 2 s → 4 s → 8 s (4 attempts total)
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings
