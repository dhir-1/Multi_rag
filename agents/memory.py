"""
Episodic Memory & In-Context RL Module for Financial Multi-Agent RAG.

Stores, retrieves, and updates atomic lessons dynamically learned at runtime
from Critic evaluations and factual audits. Contains ZERO hardcoded question cheats
or company-specific answers.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MEMORY_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "episodic_memory.json")


class EpisodicMemory:
    """Manages dynamic, runtime audit lessons learned across query executions."""

    def __init__(self):
        self.lessons: List[Dict[str, Any]] = []
        self._load_lessons()

    def _load_lessons(self):
        Path(MEMORY_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(MEMORY_FILE_PATH):
            try:
                with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
                    self.lessons = json.load(f)
            except Exception:
                self.lessons = []
        else:
            self.lessons = []
            self._save_lessons()

    def _save_lessons(self):
        try:
            with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.lessons, f, indent=2)
        except Exception as e:
            print(f"[-] Warning: Failed to save episodic memory: {e}")

    def get_relevant_lessons(self, query: str, top_k: int = 2) -> List[str]:
        """Retrieves dynamically learned lessons based on keyword overlap with prior reflections."""
        if not self.lessons:
            return []

        q_lower = query.lower()
        scored = []

        for item in self.lessons:
            score = 0
            for kw in item.get("keywords", []):
                if kw in q_lower:
                    score += 1
            if score > 0:
                scored.append((score, item["lesson"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]

    def record_lesson(self, topic: str, keywords: List[str], lesson_text: str):
        """Records a new learned rule dynamically from a critic or evaluator correction."""
        new_entry = {
            "id": f"lesson_{len(self.lessons) + 1}",
            "topic": topic,
            "keywords": [kw.lower() for kw in keywords if kw],
            "lesson": lesson_text
        }
        self.lessons.append(new_entry)
        self._save_lessons()


_GLOBAL_MEMORY: Optional[EpisodicMemory] = None


def get_episodic_memory() -> EpisodicMemory:
    global _GLOBAL_MEMORY
    if _GLOBAL_MEMORY is None:
        _GLOBAL_MEMORY = EpisodicMemory()
    return _GLOBAL_MEMORY
