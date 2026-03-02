"""
File-based memory store with semantic retrieval.

Workspace layout:
workspace/
├── MEMORY.md
├── memory/
│   ├── YYYY-MM-DD.md
│   └── projects.md
└── agents/
    └── main.json
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .chunking import chunk_text_semantic
from .embedding import VoyageEmbedder

logger = logging.getLogger("memory_chat.file_memory")


@dataclass
class MemoryChunk:
    """Chunked memory segment used for retrieval."""

    file_path: str
    text: str
    score: float = 0.0


class FileMemoryStore:
    """
    File-based long-term memory manager.

    Supports:
    - append-only daily logs
    - curated memory file
    - evergreen project memory file
    - semantic retrieval (Voyage embeddings only)
    """

    def __init__(
        self,
        base_dir: str = "workspace",
        assistant_id: str = "main",
        embedder: Optional[VoyageEmbedder] = None,
    ):
        self.base_dir = Path(base_dir)
        self.memory_dir = self.base_dir / "memory"
        self.agents_dir = self.base_dir / "agents"
        self.assistant_id = assistant_id
        self.embedder = embedder

        self.curated_file = self.base_dir / "MEMORY.md"
        self.projects_file = self.memory_dir / "projects.md"
        self.agent_meta_file = self.agents_dir / f"{assistant_id}.json"

        self._lock = asyncio.Lock()
        self._chunk_cache: Dict[str, List[MemoryChunk]] = {"startup": [], "all": []}
        self._embedding_cache: Dict[str, List[List[float]]] = {"startup": [], "all": []}
        self._signature_cache: Dict[str, Tuple[Tuple[str, int, int], ...]] = {
            "startup": tuple(),
            "all": tuple(),
        }

    async def ensure_workspace(self) -> None:
        """Create workspace directories and core files if missing."""
        logger.info(
            "[MEMORY_EVENT] ensure_workspace start base_dir=%s memory_dir=%s agents_dir=%s",
            self.base_dir,
            self.memory_dir,
            self.agents_dir,
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)

        if not self.curated_file.exists():
            self.curated_file.write_text(
                "# Curated Long-Term Memory\n\n"
                "Use this file for persistent facts, rules, and preferences.\n",
                encoding="utf-8",
            )
            logger.info("[MEMORY_EVENT] created curated memory file path=%s", self.curated_file)
        if not self.projects_file.exists():
            self.projects_file.write_text(
                "# Projects Memory\n\n"
                "Evergreen project details are stored here.\n",
                encoding="utf-8",
            )
            logger.info("[MEMORY_EVENT] created projects memory file path=%s", self.projects_file)
        if not self.agent_meta_file.exists():
            self.agent_meta_file.write_text(
                json.dumps(
                    {
                        "agent_id": self.assistant_id,
                        "name": "Main Agent",
                        "memory_backend": "file",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("[MEMORY_EVENT] created agent metadata file path=%s", self.agent_meta_file)

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        self._ensure_daily_file(today)
        self._ensure_daily_file(yesterday)
        logger.info("[MEMORY_EVENT] ensure_workspace complete today=%s yesterday=%s", today.isoformat(), yesterday.isoformat())

    def _ensure_daily_file(self, day) -> Path:
        """Ensure a daily markdown file exists."""
        daily_path = self.memory_dir / f"{day.isoformat()}.md"
        if not daily_path.exists():
            daily_path.write_text(
                f"# Daily Log {day.isoformat()}\n\n",
                encoding="utf-8",
            )
            logger.info("[MEMORY_EVENT] created daily log path=%s", daily_path)
        return daily_path

    async def append_chat_log(self, user_id: str, role: str, content: str) -> None:
        """Append a chat turn into today's daily log."""
        safe = " ".join(content.strip().split())
        if not safe:
            logger.info("[MEMORY_EVENT] append_chat_log skipped empty payload user_id=%s role=%s", user_id, role)
            return
        topic = f"chat:{role}"
        logger.info(
            "[MEMORY_EVENT] append_chat_log user_id=%s role=%s topic=%s content=%s",
            user_id,
            role,
            topic,
            safe,
        )
        await self.write_memory(user_id=user_id, content=safe, memory_type="daily", topic=topic)

    async def write_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "daily",
        topic: str = "general",
    ) -> str:
        """Write memory content into the target file."""
        if not content or not content.strip():
            raise ValueError("content must not be empty")

        await self.ensure_workspace()
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        today_path = self._ensure_daily_file(now.date())

        entry = f"- [{timestamp}] ({user_id}/{topic}) {content.strip()}\n"

        async with self._lock:
            logger.info(
                "[MEMORY_WRITE] requested user_id=%s memory_type=%s topic=%s timestamp=%s content=%s",
                user_id,
                memory_type,
                topic,
                timestamp,
                content.strip(),
            )
            if memory_type == "daily":
                target = today_path
                with target.open("a", encoding="utf-8") as f:
                    f.write(entry)
            elif memory_type == "projects":
                target = self.projects_file
                with target.open("a", encoding="utf-8") as f:
                    f.write(f"\n## {topic}\n{entry}")
            elif memory_type == "curated":
                target = self.curated_file
                with target.open("a", encoding="utf-8") as f:
                    f.write(entry)
            else:
                raise ValueError("memory_type must be one of: daily, projects, curated")

            self._invalidate_cache()
            logger.info(
                "[MEMORY_WRITE] stored target=%s entry=%s cache_invalidated=true",
                target,
                entry.strip(),
            )

        return str(target)

    def _invalidate_cache(self) -> None:
        self._signature_cache = {"startup": tuple(), "all": tuple()}
        logger.info("[MEMORY_EVENT] index cache invalidated")

    async def retrieve(
        self,
        query: str,
        scope: str = "all",
        top_k: int = 5,
    ) -> List[MemoryChunk]:
        """
        Retrieve relevant memory chunks.

        scope:
        - startup: MEMORY.md + today + yesterday
        - all: MEMORY.md + all markdown files under memory/
        """
        q = query.strip()
        if not q:
            return []

        scope = scope if scope in {"startup", "all"} else "all"
        top_k = max(1, min(int(top_k), 10))
        logger.info("[MEMORY_RETRIEVE] start query=%s scope=%s top_k=%s", q, scope, top_k)

        await self.ensure_workspace()
        chunks, embeddings = await self._ensure_index(scope)
        if not chunks:
            logger.info("[MEMORY_RETRIEVE] no chunks available scope=%s", scope)
            return []

        if not self.embedder:
            logger.warning("Semantic retrieval unavailable: embedder is not initialized")
            return []

        if not embeddings:
            logger.warning("Semantic retrieval unavailable: no embeddings were generated")
            return []

        logger.info(
            "[MEMORY_RETRIEVE] semantic search query=%s chunk_count=%s embedding_count=%s",
            q,
            len(chunks),
            len(embeddings),
        )
        query_embedding = await self.embedder.embed_query(q)
        scored = [
            MemoryChunk(file_path=chunk.file_path, text=chunk.text, score=self._cosine(query_embedding, emb))
            for chunk, emb in zip(chunks, embeddings)
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        logger.info("[MEMORY_RETRIEVE] complete scope=%s top_results=%s", scope, min(top_k, len(scored)))
        for idx, hit in enumerate(scored[:top_k], start=1):
            logger.info(
                "[MEMORY_RETRIEVE_HIT] rank=%s score=%.4f file=%s text=%s",
                idx,
                hit.score,
                hit.file_path,
                hit.text.replace("\n", " "),
            )
        return scored[:top_k]

    async def _ensure_index(self, scope: str) -> Tuple[List[MemoryChunk], List[List[float]]]:
        """Build or reuse cached chunk index + embeddings."""
        files = self._files_for_scope(scope)
        logger.info("[MEMORY_INDEX] building scope=%s files=%s", scope, [str(f) for f in files])
        signature = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in files
            if path.exists()
        )

        if signature == self._signature_cache.get(scope, tuple()):
            logger.info("[MEMORY_INDEX] cache hit scope=%s chunk_count=%s", scope, len(self._chunk_cache[scope]))
            return self._chunk_cache[scope], self._embedding_cache[scope]

        chunks = self._build_chunks(files)
        embeddings: List[List[float]] = []
        if chunks and self.embedder:
            texts = [c.text for c in chunks]
            embeddings = await self.embedder.embed_documents(texts)
            logger.info("[MEMORY_INDEX] embedded chunks scope=%s chunk_count=%s", scope, len(chunks))
        else:
            logger.info("[MEMORY_INDEX] no embeddings generated scope=%s chunk_count=%s", scope, len(chunks))

        self._chunk_cache[scope] = chunks
        self._embedding_cache[scope] = embeddings
        self._signature_cache[scope] = signature
        logger.info("[MEMORY_INDEX] cache updated scope=%s", scope)
        return chunks, embeddings

    def _files_for_scope(self, scope: str) -> List[Path]:
        """Return ordered markdown files for the given retrieval scope."""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        files: List[Path] = [self.curated_file]
        if scope == "startup":
            files.extend(
                [
                    self.memory_dir / f"{today.isoformat()}.md",
                    self.memory_dir / f"{yesterday.isoformat()}.md",
                ]
            )
        else:
            md_files = sorted(self.memory_dir.glob("*.md"))
            files.extend(md_files)
        return files

    def _build_chunks(self, files: List[Path]) -> List[MemoryChunk]:
        """Read markdown files and split into semantic chunks."""
        chunks: List[MemoryChunk] = []
        for path in files:
            if not path.exists():
                logger.info("[MEMORY_CHUNKS] skipping missing file=%s", path)
                continue
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                logger.info("[MEMORY_CHUNKS] skipping empty file=%s", path)
                continue

            split_chunks = chunk_text_semantic(text, max_chars=1200)
            if not split_chunks:
                split_chunks = [text]
            logger.info("[MEMORY_CHUNKS] file=%s chunk_count=%s", path, len(split_chunks))

            for chunk_text in split_chunks:
                chunks.append(
                    MemoryChunk(
                        file_path=str(path),
                        text=chunk_text.strip(),
                    )
                )
        return chunks

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Cosine similarity for two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
