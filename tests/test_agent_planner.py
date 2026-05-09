import os
import sys
from pathlib import Path

os.environ.setdefault("LLM_API_KEY", "test-key")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent.router import (
    _basic_plan_issues,
    _format_dependency_query_hint,
    _format_prior_context,
    _merge_dependency_hint_into_query,
    _normalize_plan,
)


def test_normalize_plan_keeps_valid_steps():
    raw_plan = {
        "steps": [
            {
                "id": "step_1",
                "sub_question": "找出题干中的原文锚点",
                "mode": "text",
                "depends_on": [],
                "purpose": "定位核心文本",
            },
            {
                "id": "step_2",
                "sub_question": "结合前一步证据分析表现手法",
                "mode": "hybrid",
                "depends_on": ["step_1"],
                "purpose": "补充同类题型范例",
            },
        ]
    }

    plan = _normalize_plan(raw_plan, "分析划线句作用")

    assert len(plan) == 2
    assert plan[0]["mode"] == "text"
    assert plan[1]["depends_on"] == ["step_1"]


def test_normalize_plan_downgrades_to_text_only():
    raw_plan = {
        "steps": [
            {
                "sub_question": "查看扫描页版面信息",
                "mode": "vision",
            }
        ]
    }

    plan = _normalize_plan(raw_plan, "题目", text_only=True)

    assert plan[0]["mode"] == "text"


def test_normalize_plan_falls_back_for_empty_steps():
    plan = _normalize_plan({"steps": []}, "原始问题")

    assert len(plan) == 1
    assert plan[0]["sub_question"] == "原始问题"
    assert plan[0]["mode"] == "hybrid"


def test_dependent_step_does_not_leak_prior_answer():
    raw_plan = {
        "steps": [
            {
                "id": "step_1",
                "sub_question": "《马说》的作者是谁？",
                "mode": "text",
                "depends_on": [],
                "purpose": "确定作品作者",
            },
            {
                "id": "step_2",
                "sub_question": "韩愈的祖籍是什么？",
                "mode": "text",
                "depends_on": ["step_1"],
                "purpose": "查询作者祖籍",
            },
        ]
    }

    plan = _normalize_plan(raw_plan, "《马说》的作者是谁？他的祖籍是什么？")

    assert plan[1]["depends_on"] == ["step_1"]
    assert "韩愈" not in plan[1]["sub_question"]
    assert "step_1" in plan[1]["sub_question"]


def test_compound_author_origin_question_is_flagged_for_repair():
    raw_plan = {
        "steps": [
            {
                "id": "step_1",
                "sub_question": "《马说》的作者及其祖籍是什么？",
                "mode": "text",
                "depends_on": [],
                "purpose": "回答作品作者和作者祖籍",
            }
        ]
    }

    plan = _normalize_plan(raw_plan, "《马说》的作者是谁？他的祖籍是什么？")

    assert len(plan) == 1
    assert "single_step_for_likely_multi_hop_question" in _basic_plan_issues(
        plan, "《马说》的作者是谁？他的祖籍是什么？"
    )


def test_pronoun_followup_question_is_flagged_for_repair():
    raw_plan = {
        "steps": [
            {
                "id": "step_1",
                "sub_question": "苏轼是谁？他的代表作是什么？",
                "mode": "text",
                "depends_on": [],
                "purpose": "回答人物及其代表作",
            }
        ]
    }

    plan = _normalize_plan(raw_plan, "苏轼是谁？他的代表作是什么？")

    assert len(plan) == 1
    assert "single_step_for_likely_multi_hop_question" in _basic_plan_issues(
        plan, "苏轼是谁？他的代表作是什么？"
    )


def test_chained_author_attribute_question_is_flagged_for_repair():
    raw_plan = {
        "steps": [
            {
                "id": "step_1",
                "sub_question": "《马说》的作者的祖籍是什么？",
                "mode": "hybrid",
                "depends_on": [],
                "purpose": "查询作品作者祖籍",
            }
        ]
    }

    plan = _normalize_plan(raw_plan, "《马说》的作者的祖籍是什么？")

    assert len(plan) == 1
    assert "single_step_for_likely_multi_hop_question" in _basic_plan_issues(
        plan, "《马说》的作者的祖籍是什么？"
    )


def test_author_article_tail_question_is_flagged_for_repair():
    raw_plan = {
        "steps": [
            {
                "id": "step_1",
                "sub_question": "找到韩愈写的文章中最后一句话的内容",
                "mode": "hybrid",
                "depends_on": [],
                "purpose": "查找文章最后一句",
            }
        ]
    }

    plan = _normalize_plan(raw_plan, "找到韩愈写的文章中最后一句话的内容")

    assert len(plan) == 1
    assert "single_step_for_likely_multi_hop_question" in _basic_plan_issues(
        plan, "找到韩愈写的文章中最后一句话的内容"
    )


def test_prior_context_prefers_structured_evidence_summary():
    context = _format_prior_context(
        [
            {
                "step": {"id": "step_1"},
                "query": "查看图表信息",
                "evidence_summary": "视觉代理描述：图表显示了三类数据。",
                "documents": [],
            }
        ],
        ["step_1"],
    )

    assert "视觉代理描述" in context
    assert "图表显示" in context


def test_dependency_query_hint_is_merged_into_dependent_query():
    step_results = [
        {
            "step": {"id": "step_1"},
            "resolved_facts": {
                "facts": [
                    {"name": "目标对象", "value": "对象甲", "evidence": "证据片段"}
                ],
                "summary": "已确定目标对象。",
            },
        }
    ]

    hint = _format_dependency_query_hint(step_results, ["step_1"])
    query = _merge_dependency_hint_into_query(
        "step_1 中确定的对象的目标属性是什么？",
        hint,
        "step_1 中确定的对象的目标属性是什么？",
    )

    assert "对象甲" in hint
    assert "证据片段" not in hint
    assert "对象甲" in query
    assert "step_1" not in query


if __name__ == "__main__":
    test_normalize_plan_keeps_valid_steps()
    test_normalize_plan_downgrades_to_text_only()
    test_normalize_plan_falls_back_for_empty_steps()
    test_dependent_step_does_not_leak_prior_answer()
    test_compound_author_origin_question_is_flagged_for_repair()
    test_pronoun_followup_question_is_flagged_for_repair()
    test_chained_author_attribute_question_is_flagged_for_repair()
    test_author_article_tail_question_is_flagged_for_repair()
    test_prior_context_prefers_structured_evidence_summary()
    test_dependency_query_hint_is_merged_into_dependent_query()
    print("test_agent_planner passed")
