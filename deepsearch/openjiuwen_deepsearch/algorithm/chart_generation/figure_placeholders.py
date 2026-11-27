# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
图表插入位识别模块

功能：
1. 对报告按一级标题进行划分
2. LLM识别可插入图表的内容
3. 生成占位符 @import:{图表名称}+figure_{id}
4. 生成信息收集任务
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Tuple

from openjiuwen_deepsearch.utils.constants_utils.node_constants import AgentLlmName
from openjiuwen_deepsearch.algorithm.chart_generation.utils import call_model, CallModelInput
from openjiuwen_deepsearch.common.exception import CustomValueException
from openjiuwen_deepsearch.common.status_code import StatusCode

logger = logging.getLogger(__name__)

_H1_LINE_RE = re.compile(r"^#\s+(.+?)\s*$")
_H2_LINE_RE = re.compile(r"^##\s+(.+?)\s*$")
# 围栏最多缩进 3 个空格；缩进 4 格（或一个 tab）在 Markdown 里是缩进代码块，
# 里面的 ``` 只是示例文本，不能当成围栏开关，否则会把后面的 H1 全吞进代码块
_FENCE_LINE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_CHAPTER_TITLE_RE = re.compile(
    r"^(第\s*[0-9一二三四五六七八九十百千]+\s*[章节篇]"
    r"|[0-9]+\s*[.、)）]"
    r"|chapter\s+\d+"
    r"|part\s+\d+)",
    re.IGNORECASE,
)
_EXCLUDED_TITLES = frozenset(
    {
        "目录",
        "table of contents",
        "contents",
        "摘要",
        "abstract",
        "结论",
        "conclusion",
        "参考文章",
        "参考文献",
        "reference articles",
        "references",
    }
)


# 目录行：标题 + 点线/省略号引导符 + 页码
_TOC_LINE_RE = re.compile(r"^.{1,80}?(?:\.{3,}|…{2,}|·{3,})\s*\d{1,4}\s*$")
# 前言小节标题：目录 / 摘要 / Table of Contents ...
_PREFACE_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:"
    + "|".join(re.escape(title) for title in sorted(_EXCLUDED_TITLES))
    + r")\s*$",
    re.IGNORECASE,
)
# 正文里的章节锚点链接（目录条目）：[第一章](#chapter-1)
_TOC_LINK_RE = re.compile(r"\]\(#chapter-\d+\)")


def _is_preface_body(body: str) -> bool:
    """判断首个 H1 下面的内容是不是"前言"（目录 / 摘要），而不是正文首章。

    报告总标题下面经常跟着目录或摘要，这些内容有固定形态：带点线和页码的目录行，
    或 "目录 / 摘要" 小节标题。据此可以把"总标题 + 前言"和"正文直接从第一章开始"
    区分开，而不必依赖标题有没有章节号。

    围栏代码块里的行必须跳过：首章正文里放一段 Markdown 示例，示例里的
    ``## Abstract`` 会被当成前言标题，导致整章被当成总标题丢掉。
    """
    lines = body.splitlines()
    fenced = _fenced_lines(lines)
    candidates = [
        line.strip()
        for idx, line in enumerate(lines)
        if idx not in fenced and line.strip()
    ]
    if not candidates:
        return True
    if any(_PREFACE_HEADING_RE.match(line) for line in candidates):
        return True
    toc_lines = sum(1 for line in candidates if _TOC_LINE_RE.match(line))
    return toc_lines >= 2 and toc_lines * 2 >= len(candidates)


def _has_chapter_links(body: str) -> bool:
    """正文里是否已经有成组的章节锚点链接（生成器写目录时用 ``#chapter-N``）。"""
    return len(_TOC_LINK_RE.findall(body)) >= 2


def _looks_like_report_title(
    block: Dict[str, str], following_titles: List[str]
) -> bool:
    """判断首个 H1 是不是整篇报告的总标题。

    只看标题形态会两头出错：``# 2025 年行业观察`` 没有章节号，但它可能是正文第一章；
    反过来，大纲标题也可能恰好长成 ``# 1. 2025 年行业观察``。所以按结构信号判断：

    * 正文为空、是目录/摘要这类前言、或正文里已经成组出现章节锚点链接 → 总标题；
    * 紧随其后的 H1 已经是带编号的章节（``1.`` / ``第一章``）→ 编号从那一章开始，
      当前位置只能是总标题，**与导语长短无关**。此前用"正文不超过 200 字"兜底，
      一段长导语就让总标题被当成正文首章，后面的章节整体后移一章，图表任务按
      章节索引取资料时错位到下一章、最后一章还会因索引越界直接失败；
    * 以上结构标记都没有（如 ``# Background`` 只有一句正文、后面接 ``# Outlook``）
      → 按正文首章保留，丢掉它等于整章内容蒸发。
    """
    if not following_titles:
        return False
    body = (block.get("content") or "").strip()
    if _is_preface_body(body) or _has_chapter_links(body):
        return True
    if _CHAPTER_TITLE_RE.match(block.get("title") or ""):
        return False
    return bool(_CHAPTER_TITLE_RE.match(following_titles[0]))


def _close_length_suffix_max(
    lines: List[str], ignored: "set[int]", char: str
) -> List[int]:
    """后缀最大闭合长度：``result[i]`` 是 i..末尾里可用作闭合围栏的最大长度。

    只需 O(1) 就能判断"从这里往后根本没有闭合围栏"，这是把未闭合围栏的回退
    从二次方扫描降成线性的关键。
    """
    result = [-1] * (len(lines) + 1)
    for idx in range(len(lines) - 1, -1, -1):
        result[idx] = result[idx + 1]
        if idx in ignored:
            continue
        match = _FENCE_LINE_RE.match(lines[idx].rstrip("\r\n"))
        if not match:
            continue
        fence = match.group("fence")
        if (
            fence[0] == char
            and not match.group("info").strip()
            and len(fence) > result[idx]
        ):
            result[idx] = len(fence)
    return result


def _scan_fences(lines: List[str], ignored: "set[int]") -> Tuple["set[int]", int]:
    """扫描围栏代码块，返回（完整代码块的行号集合, 未闭合围栏的行号）。

    闭合行必须与开启行同字符且不短于开启长度（``~~~`` 关不掉 ````` `````），
    也不允许带 info string；开启行的 info string 里出现反引号时不是围栏。
    """
    fenced: "set[int]" = set()
    # 后续不存在可用闭合围栏的"起始标记"直接按普通文本跳过：此前每遇到一个
    # 这样的标记就要从文件头重扫全文（连续 4,000 行 ~~~text 实测约 28 秒）。
    close_len = {
        "`": _close_length_suffix_max(lines, ignored, "`"),
        "~": _close_length_suffix_max(lines, ignored, "~"),
    }
    open_char = ""
    open_len = 0
    open_idx = -1
    for idx, raw in enumerate(lines):
        if idx in ignored:
            continue
        match = _FENCE_LINE_RE.match(raw.rstrip("\r\n"))
        if not match:
            continue
        fence = match.group("fence")
        char, length = fence[0], len(fence)
        if not open_char:
            if char == "`" and "`" in match.group("info"):
                continue
            suffix_max = close_len.get(char) or ()
            if idx + 1 >= len(suffix_max) or suffix_max[idx + 1] < length:
                continue
            open_char, open_len, open_idx = char, length, idx
        elif char == open_char and length >= open_len and not match.group("info").strip():
            fenced.update(range(open_idx, idx + 1))
            open_char, open_len, open_idx = "", 0, -1
    return fenced, open_idx


def _fenced_lines(lines: List[str]) -> "set[int]":
    """返回完整围栏代码块覆盖的行号。

    文件末尾没闭合的围栏按普通文本处理：否则半个代码块会把后面的标题全吞掉，
    报告直接少掉几章。
    """
    ignored: "set[int]" = set()
    while True:
        fenced, unclosed = _scan_fences(lines, ignored)
        if unclosed < 0:
            return fenced
        ignored.add(unclosed)


def _slice_block_content(content: str, start: int, end: int) -> str:
    """切出章节正文：只去掉标题行之后的空行，**保留首行缩进**。

    直接用 ``.strip()`` 会把首行的缩进一起吃掉。4 个空格在 Markdown 里是
    缩进代码块，缩进一消失，本应只是字面量的 ``~~~`` 就变成真正的开围栏，
    把后面的 ``## 1.1`` 及正文全吞进代码块，H2 定位整段失效。
    """
    return content[start:end].lstrip("\r\n").rstrip()


def _split_by_heading(content: str, pattern: "re.Pattern[str]") -> List[Dict[str, str]]:
    """按标题行切分 *content*，返回 ``[{"title", "content"}, ...]``。

    用正则 match 的偏移量切片，而不是 ``content.find(title)``：后者永远命中
    title 的第一次出现，遇到重复标题时后续章节的 content 会错位甚至为空。
    围栏代码块（``` / ~~~）内的行不参与切分，避免把 ``# 注释`` 当成标题。
    """
    blocks: List[Dict[str, str]] = []
    lines = content.splitlines(keepends=True)
    fenced = _fenced_lines(lines)
    offset = 0
    current = None

    for idx, line in enumerate(lines):
        if idx in fenced:
            offset += len(line)
            continue
        match = pattern.match(line.rstrip("\r\n"))
        if match:
            if current is not None:
                current["content"] = _slice_block_content(
                    content, current.pop("_start"), offset
                )
            current = {"title": match.group(1).strip(), "_start": offset + len(line)}
            blocks.append(current)
        offset += len(line)

    if current is not None:
        current["content"] = _slice_block_content(
            content, current.pop("_start"), len(content)
        )

    return blocks


class FigurePlaceholderGenerator:
    """图表占位符生成器"""

    def __init__(self, llm_model_name: str):
        """
        初始化

        Args:
            llm_model_name: LLM模型名称
        """
        self._llm_model = llm_model_name
        self.figure_id_counter = 0

    async def run(self, report_content: str
                  ) -> Dict[int, List[Dict[str, Any]]]:
        """运行图表占位符生成器"""
        """
        1. 移除报告中的溯源信息
        2. 移除报告中的表格
        3. 按一级标题划分报告
        4. 以二级标题拆分一级章节
        5. 先预定位图片插入点，让图片选择预定位范围内的坐标进行插入
        6. 选出插入锚点，输出图表生成信息序列
        
        Args:
            report_content: 报告内容
        Returns:
            Dict[int, List[Dict[str, Any]]]: 按章节划分的图表生成信息序列
                - 图表描述
                - 图表类型
                - 信息收集任务序列
                - 插入锚点id
                - 图表所在二级章节内容
                - 图表所在一级章节索引
                - 图表前文
        
        """
        try:
            report_content_rm = self._remove_table(report_content)
            sections = self._split_report_by_h1(report_content_rm)
            gen_chart_tasks = await self.generate_chart_tasks(sections)
        except Exception as e:
            error_msg = f"[CHART GENERATION] Chart Task Generator Error: {repr(e)}"
            logger.error(error_msg)
            raise CustomValueException(StatusCode.CHART_PLACEHOLDER_ERROR.code,
                                      StatusCode.CHART_PLACEHOLDER_ERROR.errmsg.format(e=error_msg)) from e
        return gen_chart_tasks

    @staticmethod
    def _remove_table(report_content: str) -> str:
        """表格不参与生成图，移除报告中的表格"""
        # 匹配完整的markdown表格：表头、分隔符行、数据行
        table_pattern = re.compile(
            r"""
            ^\|.*\|$              # 表头行：以|开头和结尾
            \n                    # 换行
            ^\|[-:\s|]+\|$       # 分隔符行：包含-、:、|和空格
            \n                    # 换行
            (?:^\|.*\|$\n?)+     # 数据行：一个或多个以|开头和结尾的行
            """,
            re.MULTILINE | re.VERBOSE,
        )
        return table_pattern.sub("", report_content)

    @staticmethod
    def _split_report_by_h1(report_content: str) -> List[Dict[str, str]]:
        """
        按一级标题划分报告

        Args:
            report_content: 报告内容

        Returns:
            List[Dict[str, str]]: 划分后的报告块列表，每个包含:
                - index: 章节序号 (从1开始)
                - title: 一级标题
                - content: 章节内容
        """
        blocks = _split_by_heading(report_content, _H1_LINE_RE)
        if not blocks:
            # 没有一级标题，将整个报告作为一个块
            return [{"index": 1, "title": "全文", "content": report_content}]

        report_sections = []
        for position, section in enumerate(blocks):
            normalized_title = re.sub(r"\s+", " ", section["title"]).strip().casefold()
            if normalized_title in _EXCLUDED_TITLES:
                continue
            # 首个 H1 要不要当总标题丢掉，按结构信号判断（见
            # _looks_like_report_title）：后面紧跟的是带编号的章节、或该块正文
            # 是目录/摘要前言时才算总标题；否则按正文首章保留，不能整章丢失。
            if position == 0 and _looks_like_report_title(
                section, [block["title"] for block in blocks[1:]]
            ):
                continue
            report_sections.append(section)

        for index, section in enumerate(report_sections, start=1):
            section["index"] = index

        logger.info(f"Split report into {len(report_sections)} sections by H1")
        return report_sections

    @staticmethod
    def _split_section_by_h2(section: Dict) -> List[Dict[str, str]]:
        """以二级标题拆分一级章节"""
        section_content = section.get("content", "")
        index = section.get("index", -1)
        
        blocks = _split_by_heading(section_content, _H2_LINE_RE)
        if not blocks:
            # 没有二级标题，将整个章节作为一个块
            return [
                {
                    "index": index,
                    "index_h2": 0,
                    "title": section.get("title", ""),
                    "content": section_content,
                }
            ]

        sub_sections = [
            {
                "index": index,
                "index_h2": position,
                "title": block["title"],
                "content": block["content"],
            }
            for position, block in enumerate(blocks)
        ]

        logger.info(f"Split section_{index} into {len(sub_sections)} sub_sections by H2")
        return sub_sections

    @staticmethod
    def _position_anchor(section: str) -> Tuple[str, Dict[str, str]]:
        """先预定位图片插入点，让图片选择预定位范围内的坐标进行插入"""
        section_paragraphs = section.split("\n")
        pos_char = "@import_chart_{index}"
        section_insert_anchor = []
        anchor_msg = {}
        for index, para in enumerate(section_paragraphs):
            if para:
                anchor_msg[index] = para # 锚点信息中的报告内容不包含占位符
                para += pos_char.format(index=index)
                section_insert_anchor.append(para)
        return ''.join(section_insert_anchor), anchor_msg

    async def generate_chart_tasks(
        self, sections: List[Dict[str, str]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        识别可插入图表的位置并生成占位符

        Args:
            sections: 划分后的报告章节列表

        Returns:
            Dict[int, List[Dict[str, Any]]]:
                - 全文包含的一级章节索引
                   - 一级章节包含的二级章节序列
                       - 二级章节包含的图表生成信息序列
        """
        # 将一级章节划分为二级章节
        sub_sections_h1 = []
        for section in sections:
            # 每个一级章节包含多个二级章节
            sub_sections_h1.append(self._split_section_by_h2(section))
            
        # 并行处理每个一级章节
        tasks = []
        for sec_h1 in sub_sections_h1:
            task = self._process_section_h1(sec_h1)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        if len(results) != len(sections):
            # 一级标题结果数量与章节数量不符，将导致图表信息错位，抛出异常
            error_msg = f"Error processing sections_report: {len(results)} != {len(sections)}"
            logger.error(error_msg)
            raise CustomValueException(StatusCode.CHART_PLACEHOLDER_ERROR.code,
                                      StatusCode.CHART_PLACEHOLDER_ERROR.errmsg.format(e=error_msg))

        # 将一级章节索引添加到结果中
        section_inder_chart_tasks = {}
        for i, result in enumerate(results):
            if result:
                section_inder_chart_tasks[i + 1] = result

        return section_inder_chart_tasks

    async def _process_section_h1(self, sections: List[Dict[str, str]]
                                  ) -> List[Dict[str, Any]]:
        """
        异步处理一级章节，识别图表插入位

        Args:
            sections: 一级章节中包含的所有二级章节列表

        Returns:
            List[Dict[str, Any]]: 图表生成信息序列
        """
        tasks = []
        for section in sections:
            task = self._process_section_h2(section)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        if len(results) != len(sections):
            # 生成的结果章节数量不符不会导致异常，但需要记录日志
            logger.error(f"Error processing sections_h1: {len(results)} != {len(sections)}")
        h1_tasks = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("[CHART GENERATION] section task failed: %s", result)
                continue
            if isinstance(result, list):
                h1_tasks.extend(result)
            else:
                logger.warning("[CHART GENERATION] skip non-list result: %r (type=%s)", result, type(result).__name__)
        # 为每个一级章节下的图表生成唯一的图表id
        chart_id_in_section = 0
        for task in h1_tasks:
            chart_id_in_section += 1
            task["chart_id_in_section"] = chart_id_in_section
        
        return h1_tasks

    async def _process_section_h2(self, section: Dict[str, str]
                                  ) -> List[Dict[str, Any]]:
        """
        处理单个二级章节，识别图表插入位

        Args:
            section: 章节信息

        Returns:
            List[Dict[str,str,List[str],str,int]]:
                - 图表生成信息序列
                    - 图表描述
                    - 图表类型
                    - 信息收集任务序列
                    - 图表前文内容
                    - 图表所在章节索引
        """
        section_title = section["title"]
        section_content = section["content"]

        """图表修改： 这一步首先将section按照换行符划分成段，将段序列输入llm识别需要添加图表的段, 
        输出与段个数和顺序对应的json sheama：[图表描述/图表类型/数据收集任务集]，如果不生成图表则返回空占位序列"""
        section_with_anchor, sub_anchor_msg = self._position_anchor(section_content)

        try:
            call_model_input = CallModelInput(
                model_name=self._llm_model, 
                prompt="vlm_find_insert_point_prompt", 
                user_input={"section_contents": section_with_anchor}, 
                agent_name=AgentLlmName.VLM_CHART_GENERATOR_FIND_INSERT_POINT.value
                )
            response = await call_model(call_model_input)

            # 解析LLM响应, 将定位锚点和图表生成信息记录并返回
            return self._parse_llm_response(response, sub_anchor_msg, section)
        except Exception as e:
            logger.error(
                f"[CHART GENERATION] Error calling LLM for section {section_title}: {e}"
            )

    @staticmethod
    def _parse_llm_response(
        response: List,
        sub_anchor_msg: Dict,
        section: Dict,
    ) -> List[Dict[str, Any]]:
        """
        解析LLM响应，生成占位符和信息收集任务

        Args:
            response: LLM响应内容
            sub_anchor_msg: 当前处理的二级章节内容的图表锚点信息
            section: 二级章节信息

        Returns:
            List[Dict[str,str,List[str],str,int]]:
                - 图表生成信息序列
                    - 图表描述
                    - 图表类型
                    - 信息收集任务序列
                    - 图表前文内容
                    - 图表所在章节索引
        """
        gen_chart_tasks = []
        try:
            for res in response:
                if not isinstance(res, dict):
                    continue
                res["context"] = section.get("content", "")
                res["section_index"] = section.get("index", -1)
                placeholder_index = res.get("placeholder_index", -1)
                if placeholder_index == -1 or placeholder_index not in sub_anchor_msg:
                    continue
                res["anchor_match_para"] = sub_anchor_msg[placeholder_index]
                gen_chart_tasks.append(res)
        except Exception as e:
            logger.error(f"[CHART GENERATION] Error parsing LLM response: {e}")
            return []
        return gen_chart_tasks
