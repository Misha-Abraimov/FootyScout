"""Evaluation-only judge provider construction."""

from app.ai.evaluation.judges import JudgeProvider, OpenAIJudgeProvider
from app.config import Settings


def create_judge_provider(settings: Settings, *, model_override: str | None = None) -> JudgeProvider:
    model = model_override or settings.ai_scout_judge_model
    if settings.ai_scout_judge_provider != "openai":
        raise ValueError(f"Unsupported AI Scout judge provider: {settings.ai_scout_judge_provider}.")
    if not model:
        raise ValueError("AI_SCOUT_JUDGE_MODEL must be explicitly configured.")
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for the OpenAI judge.")
    return OpenAIJudgeProvider(
        api_key=settings.openai_api_key,
        model=model,
        timeout_seconds=settings.ai_scout_judge_timeout_seconds,
        max_retries=settings.ai_scout_judge_max_retries,
    )
