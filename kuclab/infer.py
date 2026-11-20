from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .evaluate import load_checkpoint, predict_full, predict_proba_chunked
from .langdetect import detect_lang, must_catch_hit, is_support_context, SUPPORT_LINK


@dataclass
class Verdict:
    label: str
    probability: float
    threshold: float
    categories: list[str] | None = None
    chunks: int = 1

    @property
    def is_unsafe(self) -> bool:
        return self.label in ("1", "unsafe")

    def as_dict(self) -> dict:
        d = {"label": self.label, "unsafe": self.is_unsafe, "score": round(self.probability, 4)}
        if self.categories:
            d["categories"] = self.categories
        if self.chunks > 1:
            d["chunks"] = self.chunks
        return d


class SafetyModel:
    """Inference wrapper: text in, '0' or '1' out."""

    def __init__(self, checkpoint: str | Path = "checkpoints/best.pt", threshold: float = 0.5, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.tokenizer, self.model_config, self.state = load_checkpoint(checkpoint, self.device)
        self.max_chars = int(self.state.get("max_chars", 500))
        self.max_len = self.model_config.max_seq_len
        self.threshold = threshold
        self.temperature = float(self.state.get("temperature", 1.0) or 1.0)
        self.must_catch_override = dict(self.state.get("must_catch_override") or {})

    def _cats(self) -> list[str]:
        from .categories import CATEGORIES

        return CATEGORIES if len(self.model_config.answer_ids) > 2 else []

    def predict(self, text: str) -> Verdict:
        probs = predict_proba_chunked(self.model, self.tokenizer, [text[: self.max_chars * 8]], self.max_len, self.device, temperature=self.temperature)[0]
        cats = self._cats()
        row = predict_full(self.model, self.tokenizer, [text[: self.max_chars]], self.max_len, self.device, temperature=self.temperature)[0]
        lang_thr = self.state.get("lang_thresholds") or {}
        thr = float(lang_thr.get(detect_lang(text), lang_thr.get("default", self.threshold)))
        is_unsafe = probs >= thr
        hit = must_catch_hit(text)  # Barist.1 pattern pre-filter (CSAM / self-harm / DIY / jailbreak)
        if hit:
            is_unsafe = True
        support = is_support_context(text) or "self_harm" in hit
        flagged = None
        if cats and is_unsafe:
            flagged = sorted(set([c for c, p in zip(cats, row[2:]) if p >= self.threshold]) | set(hit))
        v = Verdict(label="unsafe" if is_unsafe else "safe", probability=probs,
                    threshold=thr, categories=flagged)
        v.override = "must_catch" if hit else None
        v.needs_support = SUPPORT_LINK if support else None
        return v

    def predict_batch(self, texts: list[str]) -> list[Verdict]:
        trimmed = [t[: self.max_chars * 8] for t in texts]
        probs = predict_proba_chunked(self.model, self.tokenizer, trimmed, self.max_len, self.device, temperature=self.temperature)
        return [
            Verdict(label="unsafe" if p >= self.threshold else "safe", probability=p, threshold=self.threshold)
            for p in probs
        ]
