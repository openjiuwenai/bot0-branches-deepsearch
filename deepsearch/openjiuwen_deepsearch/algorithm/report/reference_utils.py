# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Reference management: deduplication, renumbering, and citation replacement."""

import logging
import re

from openjiuwen_deepsearch.common.common_constants import CHINESE
from openjiuwen_deepsearch.utils.log_utils.log_manager import LogManager

logger = logging.getLogger(__name__)

_REF_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_CITATION_RE = re.compile(r"\[citation:(\d+)\]")


def _deduplicate_and_renumber_refs(
    ref_blocks: list[str],
) -> tuple[str, dict[str, int]]:
    """去重引用并重编号，按段落返回 ``{段落号-原编号: 新编号}``。

    ``ref_blocks`` 必须按正文段落顺序给出每个段落的引用块，段落号直接取
    1-based 下标。调用方本就是这个顺序，不需要（也不应该）再从文本里用
    "空行/``[1]``"反推段落边界：那种启发式在引用块内出现空行、或某段引用
    不是从 ``[1]`` 起编号时会整体漂移，导致 ``ref_map`` 的键与消费端按
    ``f"{i + 1}-{index}"`` 拼出的键对不上，整段引用都不会被重编号。
    """
    seen: dict[str, int] = {}
    renumbered: list[str] = []
    mapping: dict[str, int] = {}
    next_index = 1

    for paragraph_no, block in enumerate(ref_blocks, start=1):
        if not block:
            continue
        for line in block.splitlines():
            match = _REF_LINE_RE.match(line.strip())
            if match is None:
                continue
            content = match.group(2).strip()
            if not content:
                continue
            if content not in seen:
                seen[content] = next_index
                renumbered.append(f"[{next_index}] {content}")
                next_index += 1
            key = f"{paragraph_no}-{match.group(1)}"
            if key in mapping:
                # 同一个 key 只可能来自"同一段落里重复出现同一原编号"。此时后一行
                # 若直接覆盖，前一条文献的映射会被静默丢弃，消费端会把引用改指到
                # 后一条文献上（正是本模块要修的"引用张冠李戴"），因此保留首次
                # 映射并告警，由调用方决定是否修正输入。
                if mapping[key] != seen[content]:
                    logger.warning(
                        "Duplicate reference key '%s' in paragraph %d: keep the first "
                        "mapping (-> %d), ignore the later one (-> %d). Citation numbers "
                        "may point to the wrong reference; check the reference block.",
                        key, paragraph_no, mapping[key], seen[content],
                    )
                continue
            mapping[key] = seen[content]

    return "\n\n".join(renumbered), mapping


def _split_ref_blocks(raw_text: str) -> list[str]:
    """把单段文本切成引用块。

    兼容入口的历史调用形态是把各段落的引用块用单个 ``\\n`` 拼起来（每块首行
    是 ``[1]``），只按空行切分会让多个段落塌进同一段、并因 ``f"{n}-1"`` 键
    冲突互相覆盖映射。因此这里恢复旧实现的启发式：空行、或"行首再次出现
    ``[1]``"都视为新块的边界。
    """
    if not raw_text.strip():
        return []
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        match = _REF_LINE_RE.match(stripped)
        if match is not None and match.group(1) == "1" and current:
            blocks.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(block) for block in blocks]


def _deduplicate_and_renumber_ref(raw_text: str) -> tuple[str, dict[str, int]]:
    """单段文本入口，保留给旧调用方；新代码请用 :func:`_deduplicate_and_renumber_refs`。

    空行分隔的多个引用块按顺序当作多个段落；块内再次出现行首 ``[1]`` 也当作
    新段落（历史调用形态），段落号不再随空行数漂移。
    """
    return _deduplicate_and_renumber_refs(_split_ref_blocks(raw_text))


def _replace_citations_and_classified_index(
    paragraphs: list[str],
    classified_contents: list[list[dict]],
    ref_map: dict[str, int],
) -> tuple[list[str], list[list[dict]]]:
    if not ref_map or not classified_contents:
        return paragraphs, classified_contents

    updated_paragraphs: list[str] = []
    updated_classified_contents: list[list[dict]] = []

    for i, para in enumerate(paragraphs):
        if i >= len(classified_contents):
            updated_paragraphs.append(para)
            continue
        sub_classified_contents = classified_contents[i]
        if not sub_classified_contents:
            updated_paragraphs.append(para)
            updated_classified_contents.append([])
            continue

        # Build index mapping: original index -> new number
        index_map = {
            str(item["index"]): ref_map.get(f"{i + 1}-{item['index']}")
            for item in sub_classified_contents
        }

        # 单次替换：逐条 re.sub 会把上一轮刚换好的编号当成原文编号再换一次
        # （1->2、2->3 时 [citation:1] 会被连换两次）。
        updated_para = _CITATION_RE.sub(
            lambda m: (
                f"[citation:{index_map[m.group(1)]}]"
                if index_map.get(m.group(1)) is not None
                else m.group(0)
            ),
            para,
        )
        updated_paragraphs.append(updated_para)

        # Update index field in reference entries
        updated_sub_classified_content: list[dict] = []
        for item in sub_classified_contents:
            updated_item = item.copy()
            final_index = index_map.get(str(item["index"]))
            if final_index is not None:
                updated_item["index"] = final_index
            updated_sub_classified_content.append(updated_item)

        updated_classified_contents.append(updated_sub_classified_content)

    return updated_paragraphs, updated_classified_contents


class ReferenceMixin:
    """Mixin for reference management operations."""

    @staticmethod
    def add_references(sub_section_content: str, references: list, language: str):
        """Add references for subsection content"""
        logger.info(f"Adding references to sub_section_content")
        if not references:
            logger.info(f"No references found. can not add references.")
            return sub_section_content
        if sub_section_content:
            if language == CHINESE:
                append = "\n## 参考文章\n"
            else:
                append = "\n## References\n"
            temp_ref = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(references))
            sub_section_content = sub_section_content + append + temp_ref
        return sub_section_content

    @staticmethod
    def refresh_reference(sub_reports_content, sub_references, all_classified_contents):
        """Refresh references"""
        refreshed_references = ""
        raw_references = "\n".join(sub_references) if sub_references else ""
        if raw_references:
            refreshed_references, ref_map = _deduplicate_and_renumber_refs(
                sub_references
            )
            if not LogManager.is_sensitive():
                logger.info("refreshed_references: [%s]", refreshed_references)
            sub_reports_content, all_classified_contents = (
                _replace_citations_and_classified_index(
                    sub_reports_content, all_classified_contents, ref_map
                )
            )

        return dict(
            sub_reports_content="\n\n".join(sub_reports_content),
            sub_references=refreshed_references,
            refreshed_all_classified_contents=all_classified_contents,
        )
