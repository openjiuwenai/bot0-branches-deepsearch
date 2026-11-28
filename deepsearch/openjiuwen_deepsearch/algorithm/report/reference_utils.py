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


def _normalize_ref_no(value) -> str:
    """把引用编号归一成与 ``ref_map`` 键一致的十进制字符串。

    参考文献写成 ``[02]`` 时映射键是 ``2``；正文 ``[citation:02]`` 若直接用
    原始字符串 "02" 查找就会落空并改写成 ``[citation:?]``。正文替换、文献
    解析和资料条目的 index 查找必须共用同一套归一化规则。
    """
    text = str(value).strip()
    try:
        return str(int(text))
    except (TypeError, ValueError):
        return text



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
            # 原编号一律按整数归一化：参考文献写成 "[02]" 时，消费端按资料 index
            # （整数 2）拼出的键是 "段落-2"，保留原始字符串会拼出 "段落-02" 而查不到。
            orig_no = int(match.group(1))
            key = f"{paragraph_no}-{orig_no}"
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
    冲突互相覆盖映射。因此这里恢复旧实现的启发式：空行、或"块内已有引用后
    再次出现行首 ``[1]``"都视为新块的边界。

    首行是 ``## References`` 这类标题时，标题后面的 ``[1]`` 是同一块的第一条
    引用而不是新段落起点，否则整块会被标题行占走、映射键整体后移一段。
    """
    if not raw_text.strip():
        return []
    blocks: list[list[str]] = []
    current: list[str] = []
    current_has_ref = False
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
                current_has_ref = False
            continue
        match = _REF_LINE_RE.match(stripped)
        if match is not None and int(match.group(1)) == 1 and current_has_ref:
            blocks.append(current)
            current = [line]
            current_has_ref = True
            continue
        if match is not None:
            current_has_ref = True
        current.append(line)
    if current:
        blocks.append(current)
    # 只把含有效引用的块计入段落编号：``## References`` 与 ``[1]`` 之间有空行
    # 时，标题会被空行切成单独一块，段落号整体后移，后续映射键全部错位。
    return [
        "\n".join(block)
        for block in blocks
        if any(_REF_LINE_RE.match(line.strip()) for line in block)
    ]


def _deduplicate_and_renumber_ref(raw_text: str) -> tuple[str, dict[str, int]]:
    """单段文本入口，保留给旧调用方；新代码请用 :func:`_deduplicate_and_renumber_refs`。

    空行分隔的多个引用块按顺序当作多个段落；块内再次出现行首 ``[1]`` 也当作
    新段落（历史调用形态），段落号不再随空行数漂移。
    """
    return _deduplicate_and_renumber_refs(_split_ref_blocks(raw_text))


def _paragraph_ref_map(ref_map: dict[str, int], paragraph_no: int) -> dict[str, int]:
    """取出某个段落自己的 ``{原编号: 新编号}`` 映射。"""
    prefix = f"{paragraph_no}-"
    return {
        key[len(prefix):]: value
        for key, value in ref_map.items()
        if key.startswith(prefix)
    }


def _clear_skipped_sections(
    classified_contents: list[list[dict]],
    section_indexes: list[int] | None,
) -> list[list[dict]]:
    """让未进入报告的章节资料退出全局引用映射。

    报告组装会跳过正文为空的章节，但这些章节的资料仍留在列表里。下游按
    ``content["index"]`` 建全局引用映射时先读到它们：被跳过章节的资料 A 占住
    编号 1，进入报告的第二章里编号同样从 1 开始的资料 B 只能并到 A 上，
    最终引用链接指向 A。

    列表长度保持不变（图表数据收集/溯源推理按章节下标取资料），只把未进入
    报告的章节置空；为 ``None`` 时沿用旧的"下标一一对齐"语义。
    """
    if section_indexes is None or not classified_contents:
        return classified_contents
    kept = set(section_indexes)
    return [
        list(items) if index in kept else []
        for index, items in enumerate(classified_contents)
    ]


def _replace_citations_and_classified_index(
    paragraphs: list[str],
    classified_contents: list[list[dict]],
    ref_map: dict[str, int],
    section_indexes: list[int] | None = None,
) -> tuple[list[str], list[list[dict]]]:
    """按段落自己的引用块重写 ``[citation:N]``，并同步资料条目的 index。

    重编号直接用该段落引用块给出的映射，不再依赖 ``classified_contents`` 是否
    完整：资料列表偏短时，正文该改的编号照样要改，否则第二章的 ``[citation:1]``
    会原样保留、在下游碰巧链接到第一章的文献。

    ``section_indexes[i]`` 给出第 i 段正文对应的 ``classified_contents`` 下标。
    报告组装会跳过正文为空的章节，正文列表与资料列表因此不再按下标对齐
    （中间章节为空时，后面章节会读到上一章的资料并把引用改错）。

    段落引用了自己引用块里不存在的编号时，输入本身已经损坏，原编号在全局表里
    可能"碰巧"命中别的章节的文献，因此改写为 ``[citation:?]`` 并告警，
    而不是原样保留一个指向错误文献的编号。
    """
    if not ref_map:
        return paragraphs, classified_contents

    classified_contents = classified_contents or []
    updated_paragraphs: list[str] = []
    updated_classified_contents = [list(items) for items in classified_contents]
    unresolved: list[tuple[int, str]] = []

    for i, para in enumerate(paragraphs):
        paragraph_no = i + 1
        para_map = _paragraph_ref_map(ref_map, paragraph_no)

        def _replace(match, _map=para_map, _para_no=paragraph_no):
            new_index = _map.get(_normalize_ref_no(match.group(1)))
            if new_index is None:
                unresolved.append((_para_no, match.group(0)))
                return "[citation:?]"
            return f"[citation:{new_index}]"

        # 单次替换：逐条 re.sub 会把上一轮刚换好的编号当成原文编号再换一次
        # （1->2、2->3 时 [citation:1] 会被连换两次）。
        updated_paragraphs.append(_CITATION_RE.sub(_replace, para))

        target_index = (
            section_indexes[i]
            if section_indexes is not None and i < len(section_indexes)
            else i
        )
        if not 0 <= target_index < len(updated_classified_contents):
            continue
        sub_classified_contents = updated_classified_contents[target_index]
        if not sub_classified_contents:
            continue

        # Update index field in reference entries
        updated_sub_classified_content: list[dict] = []
        for item in sub_classified_contents:
            updated_item = item.copy()
            final_index = para_map.get(_normalize_ref_no(item["index"]))
            if final_index is not None:
                updated_item["index"] = final_index
            updated_sub_classified_content.append(updated_item)
        updated_classified_contents[target_index] = updated_sub_classified_content

    if unresolved:
        logger.warning(
            "Citations without a matching reference in their own paragraph were "
            "marked as unresolved: %s",
            unresolved[:10],
        )

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
    def refresh_reference(
        sub_reports_content,
        sub_references,
        all_classified_contents,
        section_indexes=None,
    ):
        """Refresh references

        ``section_indexes``：第 i 段正文对应的原始章节下标。报告组装会跳过正文
        为空的章节，正文/引用块列表与 ``all_classified_contents`` 因此可能不再
        按下标对齐，需要显式传入对应关系（不传则沿用按下标对齐的旧语义）。
        """
        refreshed_references = ""
        raw_references = "\n".join(sub_references) if sub_references else ""
        if raw_references:
            refreshed_references, ref_map = _deduplicate_and_renumber_refs(
                sub_references
            )
            if not LogManager.is_sensitive():
                logger.info("refreshed_references: [%s]", refreshed_references)
            all_classified_contents = _clear_skipped_sections(
                all_classified_contents or [], section_indexes
            )
            sub_reports_content, all_classified_contents = (
                _replace_citations_and_classified_index(
                    sub_reports_content,
                    all_classified_contents,
                    ref_map,
                    section_indexes,
                )
            )

        return dict(
            sub_reports_content="\n\n".join(sub_reports_content),
            sub_references=refreshed_references,
            refreshed_all_classified_contents=all_classified_contents,
        )
