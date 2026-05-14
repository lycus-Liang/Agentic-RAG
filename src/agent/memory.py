import json
import os
import re
import sqlite3
import time
import uuid
from typing import Any, Dict, List

import yaml

from src.agent.state import GraphState


with open("./configs/config.yaml", "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", "r", encoding="utf-8") as f:
    prompts_dict = yaml.safe_load(f)


memory_config = config_dict.get("online", {}).get("agent_memory", {})
memory_prompts = prompts_dict.get("agent_memory_reflector", {})
_memory_store = None
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from src.utils.llm_client import LLMClient
        _llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))
    return _llm_client


def _now() -> int:
    return int(time.time())


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", str(text or "").lower()))


def _compact_text(value: Any, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _quality_from_reward(reward: float) -> str:
    if reward >= 0.7:
        return "success"
    if reward <= -0.2:
        return "failure"
    return "mixed"


class AgentMemoryStore:
    """SQLite-backed HERA-inspired experience memory store."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        directory = os.path.dirname(storage_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.has_fts = False
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.storage_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    utility TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    tool_trace_json TEXT NOT NULL,
                    answer_summary TEXT NOT NULL,
                    reward REAL NOT NULL,
                    reward_source TEXT NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    experience_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    comment TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
                        id UNINDEXED,
                        question,
                        profile,
                        insight,
                        utility,
                        tags
                    )
                """)
                self.has_fts = True
            except sqlite3.OperationalError:
                self.has_fts = False

    def _row_to_dict(self, row) -> Dict[str, Any]:
        item = dict(row)
        try:
            item["tags"] = json.loads(item.get("tags_json", "[]"))
        except json.JSONDecodeError:
            item["tags"] = []
        return item

    def _candidate_text(self, item: Dict[str, Any]) -> str:
        return " ".join([
            item.get("question", ""),
            item.get("profile", ""),
            item.get("insight", ""),
            item.get("utility", ""),
            item.get("tags_json", ""),
        ])

    def _enforce_limit(self, conn):
        max_experiences = int(memory_config.get("max_experiences", 1000) or 1000)
        if max_experiences <= 0:
            return
        count = conn.execute("SELECT COUNT(*) AS count FROM experiences").fetchone()["count"]
        overflow = max(0, int(count) - max_experiences)
        if overflow <= 0:
            return
        rows = conn.execute(
            """
            SELECT id FROM experiences
            ORDER BY reward ASC, updated_at ASC
            LIMIT ?
            """,
            (overflow,),
        ).fetchall()
        for row in rows:
            experience_id = row["id"]
            conn.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
            conn.execute("DELETE FROM feedback WHERE experience_id = ?", (experience_id,))
            if self.has_fts:
                try:
                    conn.execute("DELETE FROM experiences_fts WHERE id = ?", (experience_id,))
                except sqlite3.OperationalError:
                    pass

    def _find_similar(self, record: Dict[str, Any], session_id: str) -> Dict[str, Any] | None:
        query_tokens = _tokenize(" ".join([
            record.get("question", ""),
            record.get("profile", ""),
            record.get("insight", ""),
            record.get("utility", ""),
            " ".join([str(tag) for tag in record.get("tags", []) if tag]),
        ]))
        if not query_tokens:
            return None

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM experiences
                WHERE session_id = ? OR session_id = 'global'
                ORDER BY updated_at DESC
                LIMIT 100
                """,
                (session_id,),
            ).fetchall()

        best = None
        best_score = 0.0
        for row in rows:
            item = self._row_to_dict(row)
            memory_tokens = _tokenize(self._candidate_text(item))
            score = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
            if score > best_score:
                best = item
                best_score = score
        if best and best_score >= 0.45:
            best["similarity_score"] = best_score
            return best
        return None

    def add_or_update_experience(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record_id = record.get("id") or str(uuid.uuid4())
        now = _now()
        session_id = str(record.get("session_id", "default_user"))
        tags = record.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        reward = float(record.get("reward", 0.0))
        quality = str(record.get("quality") or _quality_from_reward(reward))

        similar = self._find_similar(record, session_id)
        if similar and similar.get("reward", 0.0) >= reward + 0.15:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE experiences SET usage_count = usage_count + 1, updated_at = ? WHERE id = ?",
                    (now, similar["id"]),
                )
            return {"id": similar["id"], "action": "keep"}

        if similar and reward > -0.2:
            merged_profile = _compact_text(f"{similar.get('profile', '')}\n{record.get('profile', '')}", 1000)
            merged_insight = _compact_text(f"{similar.get('insight', '')}\n{record.get('insight', '')}", 1600)
            merged_utility = _compact_text(f"{similar.get('utility', '')}\n{record.get('utility', '')}", 1200)
            merged_reward = max(float(similar.get("reward", 0.0)), reward)
            merged_tags = sorted(set((similar.get("tags") or []) + tags))
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE experiences
                    SET updated_at = ?, profile = ?, insight = ?, utility = ?, tags_json = ?,
                        quality = ?, reward = ?, reward_source = ?, usage_count = usage_count + 1,
                        success_count = success_count + ?, failure_count = failure_count + ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        merged_profile,
                        merged_insight,
                        merged_utility,
                        _json_dumps(merged_tags),
                        _quality_from_reward(merged_reward),
                        merged_reward,
                        str(record.get("reward_source", "heuristic")),
                        1 if reward >= 0.7 else 0,
                        1 if reward <= -0.2 else 0,
                        similar["id"],
                    ),
                )
            return {"id": similar["id"], "action": "merge"}

        if reward <= -0.6 and similar:
            return {"id": similar["id"], "action": "prune"}

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiences (
                    id, created_at, updated_at, session_id, question, profile, insight,
                    utility, tags_json, quality, plan_json, tool_trace_json, answer_summary,
                    reward, reward_source, usage_count, success_count, failure_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    record_id,
                    now,
                    now,
                    session_id,
                    str(record.get("question", "")),
                    str(record.get("profile", "")),
                    str(record.get("insight", "")),
                    str(record.get("utility", "")),
                    _json_dumps(tags),
                    quality,
                    _json_dumps(record.get("plan_json", {})),
                    _json_dumps(record.get("tool_trace_json", [])),
                    str(record.get("answer_summary", "")),
                    reward,
                    str(record.get("reward_source", "heuristic")),
                    1 if reward >= 0.7 else 0,
                    1 if reward <= -0.2 else 0,
                ),
            )
            if self.has_fts:
                conn.execute(
                    """
                    INSERT INTO experiences_fts(id, question, profile, insight, utility, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        str(record.get("question", "")),
                        str(record.get("profile", "")),
                        str(record.get("insight", "")),
                        str(record.get("utility", "")),
                        " ".join([str(tag) for tag in tags]),
                    ),
                )
            self._enforce_limit(conn)
        return {"id": record_id, "action": "add"}

    def retrieve(
        self,
        question: str,
        session_id: str = "default_user",
        top_k: int = 3,
        min_reward: float = 0.3,
    ) -> List[Dict[str, Any]]:
        query_tokens = _tokenize(question)
        rows = []
        with self._connect() as conn:
            if self.has_fts and query_tokens:
                fts_query = " OR ".join(sorted(query_tokens)[:16])
                try:
                    rows = conn.execute(
                        """
                        SELECT e.* FROM experiences_fts f
                        JOIN experiences e ON e.id = f.id
                        WHERE experiences_fts MATCH ?
                          AND e.reward >= ?
                          AND (e.session_id = ? OR e.session_id = 'global')
                        LIMIT 50
                        """,
                        (fts_query, min_reward, session_id),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                rows = conn.execute(
                    """
                    SELECT * FROM experiences
                    WHERE reward >= ?
                      AND (session_id = ? OR session_id = 'global')
                    ORDER BY updated_at DESC
                    LIMIT 100
                    """,
                    (min_reward, session_id),
                ).fetchall()

        scored = []
        for row in rows:
            item = self._row_to_dict(row)
            memory_tokens = _tokenize(self._candidate_text(item))
            lexical_score = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
            item["similarity_score"] = lexical_score + max(0.0, float(item.get("reward", 0.0))) * 0.1
            scored.append(item)
        scored.sort(key=lambda item: (item["similarity_score"], item.get("reward", 0.0)), reverse=True)
        return scored[:top_k]

    def apply_feedback(
        self,
        experience_id: str,
        session_id: str,
        score: float,
        comment: str = "",
    ) -> bool:
        with self._connect() as conn:
            exists = conn.execute("SELECT id FROM experiences WHERE id = ?", (experience_id,)).fetchone()
            if not exists:
                return False
            conn.execute(
                """
                INSERT INTO feedback(id, experience_id, session_id, score, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), experience_id, session_id, float(score), comment or "", _now()),
            )
            new_reward = 1.0 if score > 0 else -1.0 if score < 0 else 0.0
            conn.execute(
                """
                UPDATE experiences
                SET reward = ?, reward_source = 'explicit_feedback', updated_at = ?,
                    success_count = success_count + ?,
                    failure_count = failure_count + ?
                WHERE id = ?
                """,
                (new_reward, _now(), 1 if score > 0 else 0, 1 if score < 0 else 0, experience_id),
            )
        return True


def get_memory_store() -> AgentMemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = AgentMemoryStore(memory_config.get("storage_path", "./data/memory/agent_memory.sqlite3"))
    return _memory_store


def _memory_enabled() -> bool:
    return bool(memory_config.get("enabled", False))


def _format_memory_context(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return "无可用历史经验"
    chunks = []
    for idx, memory in enumerate(memories, start=1):
        chunks.append(
            f"[经验 {idx} | reward={float(memory.get('reward', 0.0)):.2f} | quality={memory.get('quality', 'mixed')}]\n"
            f"profile: {_compact_text(memory.get('profile'), 360)}\n"
            f"insight: {_compact_text(memory.get('insight'), 480)}\n"
            f"utility: {_compact_text(memory.get('utility'), 360)}"
        )
    return "\n\n".join(chunks)


def retrieve_memory_node(state: GraphState) -> Dict[str, Any]:
    question = state.get("original_question") or state.get("question", "")
    session_id = state.get("session_id", "default_user")
    if not _memory_enabled():
        return {
            "session_id": session_id,
            "retrieved_memories": [],
            "memory_context": "记忆系统未启用",
            "memory_trace": ["memory_disabled"],
        }

    memories = get_memory_store().retrieve(
        question,
        session_id=session_id,
        top_k=int(memory_config.get("retrieve_top_k", 3)),
        min_reward=float(memory_config.get("min_reward_to_reuse", 0.3)),
    )
    return {
        "session_id": session_id,
        "retrieved_memories": memories,
        "memory_context": _format_memory_context(memories),
        "memory_trace": [f"retrieved_memories={len(memories)}"],
    }


def _heuristic_reward(state: GraphState) -> float:
    reward = 0.0
    if str(state.get("generation", "")).strip():
        reward += 0.35
    if state.get("documents") or state.get("step_results"):
        reward += 0.25
    if state.get("tool_trace"):
        reward += 0.2
    if not state.get("retrieval_error"):
        reward += 0.1
    if state.get("retrieval_error"):
        reward -= 0.4
    if not state.get("documents") and not state.get("step_results"):
        reward -= 0.2
    return max(-1.0, min(1.0, reward))


def _fallback_reflection(state: GraphState, reward: float) -> Dict[str, Any]:
    question = state.get("original_question") or state.get("question", "")
    plan = state.get("retrieval_plan", [])
    tool_trace = state.get("tool_trace", [])
    tools = sorted({trace.get("tool_name") for trace in tool_trace if trace.get("tool_name")})
    styles = sorted({trace.get("search_style") for trace in tool_trace if trace.get("search_style")})
    return {
        "profile": f"问题类型：{_compact_text(question, 180)}",
        "insight": (
            f"本轮规划 {len(plan)} 步，工具={', '.join(tools) or 'none'}，"
            f"检索风格={', '.join(styles) or 'auto'}，reward={reward:.2f}。"
        ),
        "utility": "遇到相似问题时，可参考该轮的模态选择、检索风格和多步拆解方式；不要把该经验当作事实证据。",
        "tags": tools + styles,
        "quality": _quality_from_reward(reward),
    }


def _reflect_experience(state: GraphState, reward: float) -> Dict[str, Any]:
    if not bool(memory_config.get("use_llm_reflection", True)):
        return _fallback_reflection(state, reward)

    state_summary = {
        "question": state.get("original_question") or state.get("question", ""),
        "plan": state.get("retrieval_plan", []),
        "tool_trace": state.get("tool_trace", []),
        "answer_summary": _compact_text(state.get("generation", ""), 800),
        "retrieval_error": state.get("retrieval_error", ""),
        "reward": reward,
    }
    try:
        response = _get_llm_client().generate(
            memory_prompts.get("system_prompt", "你是 Agent 经验反思器。"),
            memory_prompts.get("user_template", "{state_json}").format(state_json=_json_dumps(state_summary)),
            temperature=0.0,
        )
        reflection = _extract_json_object(response)
        return {
            "profile": str(reflection.get("profile", ""))[:800],
            "insight": str(reflection.get("insight", ""))[:1200],
            "utility": str(reflection.get("utility", ""))[:800],
            "tags": reflection.get("tags", []) if isinstance(reflection.get("tags", []), list) else [],
            "quality": str(reflection.get("quality", _quality_from_reward(reward))),
        }
    except Exception:
        return _fallback_reflection(state, reward)


def update_memory_node(state: GraphState) -> Dict[str, Any]:
    session_id = state.get("session_id", "default_user")
    trace = list(state.get("memory_trace", []))
    if not _memory_enabled() or not bool(memory_config.get("write_after_chat", True)):
        return {
            "session_id": session_id,
            "memory_record_id": "",
            "memory_action": "disabled",
            "memory_trace": trace + ["memory_write_disabled"],
        }

    reward = _heuristic_reward(state)
    reflection = _reflect_experience(state, reward)
    record = {
        **reflection,
        "session_id": session_id,
        "question": state.get("original_question") or state.get("question", ""),
        "plan_json": {"steps": state.get("retrieval_plan", [])},
        "tool_trace_json": state.get("tool_trace", []),
        "answer_summary": _compact_text(state.get("generation", ""), 1200),
        "reward": reward,
        "reward_source": "heuristic",
    }
    result = get_memory_store().add_or_update_experience(record)
    return {
        "session_id": session_id,
        "memory_record_id": result["id"],
        "memory_action": result["action"],
        "memory_trace": trace + [f"memory_{result['action']}={result['id']}"],
    }


def apply_memory_feedback(
    memory_id: str,
    session_id: str = "default_user",
    score: float = 0.0,
    comment: str = "",
) -> bool:
    if not _memory_enabled():
        return False
    return get_memory_store().apply_feedback(memory_id, session_id, score, comment)
