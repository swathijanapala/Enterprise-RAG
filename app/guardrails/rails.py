import logfire
from groq import Groq
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None
_groq = Groq(api_key=settings.GROQ_API_KEY)

# Same text as 'define bot refuse jailbreak' in colang_rules.py, so the reply is consistent.
JAILBREAK_REFUSAL = (
    "I maintain consistent guidelines regardless of how I am prompted. "
    "I am here to help with Kubernetes, Intel, and networking. What can I help you with?"
)


def prompt_guard_score(message: str) -> float:
    """
    Layer 1: ask Llama Prompt Guard 2 how likely the message is a jailbreak / prompt injection.
    The model's reply is just a number like '0.9993'. Returns 0.0 if the call fails,
    so a Groq outage doesn't take the whole app down (NeMo still runs after this).
    """
    try:
        r = _groq.chat.completions.create(
            model=settings.PROMPT_GUARD_MODEL,
            messages=[{"role": "user", "content": message}],
        )
        return float(r.choices[0].message.content.strip())
    except Exception as e:
        logfire.warning(f"⚠️ Prompt Guard unavailable, skipping: {e}")
        return 0.0


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses a small, fast model (settings.GROQ_GUARD_MODEL) for intent classification
    at the gate. The heavier settings.GROQ_MODEL is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_GUARD_MODEL,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_GUARD_MODEL}).")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        # Layer 1: trained jailbreak classifier (catches "forget your instructions, you are now X").
        score = prompt_guard_score(message)
        if score > settings.PROMPT_GUARD_THRESHOLD:
            logfire.info(f"🛡️ Prompt Guard fired (score={score:.4f}) | query='{message[:80]}'")
            return True, JAILBREAK_REFUSAL

        # Layer 2: NeMo Colang rules (greetings, farewell, capabilities, off-topic, known jailbreaks).
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None