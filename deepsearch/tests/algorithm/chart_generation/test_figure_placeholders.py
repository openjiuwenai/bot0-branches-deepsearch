import pytest

from openjiuwen_deepsearch.algorithm.chart_generation.figure_placeholders import (
    FigurePlaceholderGenerator,
)


pytestmark = pytest.mark.unit


def _report_with_sections(include_toc: bool) -> str:
    toc = "# 目录\n\n[第一章](#chapter-1)\n\n" if include_toc else ""
    return (
        "# 报告标题\n\n"
        f"{toc}"
        "# 摘要\n\n摘要内容\n\n"
        "# 1. 第一章\n\n## 1.1 小节\n\n第一章正文\n\n"
        "# 2. 第二章\n\n## 2.1 小节\n\n第二章正文\n\n"
        "# 结论\n\n结论内容\n\n"
        "# 参考文章\n\n参考内容\n"
    )


@pytest.mark.parametrize("include_toc", [False, True])
def test_split_report_by_h1_excludes_non_body_sections_and_reindexes(include_toc):
    sections = FigurePlaceholderGenerator._split_report_by_h1(
        _report_with_sections(include_toc)
    )

    assert [section["title"] for section in sections] == ["1. 第一章", "2. 第二章"]
    assert [section["index"] for section in sections] == [1, 2]


def test_split_report_by_h1_titles_clean_with_chapter_anchors():
    """锚点行在 H1 之后，切章 title 不应含锚点 HTML。"""
    report = (
        "# 报告标题\n\n"
        "# 目录\n\n[1. 第一章](#chapter-1)\n\n"
        "# 摘要\n\n摘要内容\n\n"
        '# 1. 第一章\n<a id="chapter-1"></a>\n\n## 1.1 小节\n\n第一章正文\n\n'
        '# 2. 第二章\n<a id="chapter-2"></a>\n\n## 2.1 小节\n\n第二章正文\n\n'
        "# 结论\n\n结论内容\n\n"
        "# 参考文章\n\n参考内容\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 第一章", "2. 第二章"]
    for section in sections:
        assert "<a id" not in section["title"]


def test_split_report_by_h1_handles_duplicate_titles():
    """同名 H1 出现两次时，第二个章节的 content 不能为空、也不能吞掉后续章节。"""
    report = (
        "# 报告标题\n\n"
        "# 背景\n\n第一段背景。\n\n"
        "# 背景\n\n第二段背景。\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["背景", "背景"]
    assert [section["content"] for section in sections] == ["第一段背景。", "第二段背景。"]


def test_split_report_by_h1_ignores_headings_inside_code_fence():
    """代码块里的 ``# 注释`` 不是 H1，不能切出新章节。"""
    report = (
        "# 报告标题\n\n"
        "# 1. 使用说明\n\n"
        "示例如下：\n\n"
        "```bash\n# install deps\npip install x\n```\n\n"
        "# 2. 结束\n\n收尾。\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 使用说明", "2. 结束"]
    assert "pip install x" in sections[0]["content"]
    assert "install deps" in sections[0]["content"]


def test_split_report_by_h1_keeps_first_chapter_when_report_has_no_title():
    """报告没有总标题时，第一个 H1 就是正文首章，不能被当成标题丢掉。"""
    report = "# 1. 第一章\n\n正文一\n\n# 2. 第二章\n\n正文二\n"
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 第一章", "2. 第二章"]
    assert [section["content"] for section in sections] == ["正文一", "正文二"]


def test_split_section_by_h2_handles_duplicate_titles():
    section = {
        "index": 2,
        "title": "2. 第二章",
        "content": "## 2.1 小节\n\n甲内容\n\n## 2.1 小节\n\n乙内容\n",
    }
    sub_sections = FigurePlaceholderGenerator._split_section_by_h2(section)

    assert [sub["content"] for sub in sub_sections] == ["甲内容", "乙内容"]
    assert [sub["index_h2"] for sub in sub_sections] == [0, 1]




def test_split_report_by_h1_fence_must_match_char_and_length():
    """~~~ 关不掉 ```，短围栏也关不掉长围栏，里面的 # 行不能当标题。"""
    report = (
        "# 报告标题\n\n"
        "# 1. 第一章\n\n"
        "````markdown\n"
        "~~~\n"
        "\n"
        "# 仍是代码块的注释\n"
        "````\n\n"
        "# 2. 第二章\n\n正文二\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 第一章", "2. 第二章"]
    assert "仍是代码块的注释" in sections[0]["content"]


def test_split_report_by_h1_short_fence_cannot_close_long_fence():
    report = (
        "# 报告标题\n\n"
        "# 1. 第一章\n\n"
        "````\n"
        "```\n"
        "\n"
        "# 代码块里的注释\n"
        "````\n\n"
        "# 2. 第二章\n\n正文二\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 第一章", "2. 第二章"]


def test_split_report_by_h1_unclosed_fence_falls_back_to_plain_text():
    """文件末尾半个代码块不能把后面的章节吞掉。"""
    report = (
        "# 报告标题\n\n"
        "# 1. 第一章\n\n"
        "```python\nprint(1)\n\n"
        "# 2. 第二章\n\n正文二\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 第一章", "2. 第二章"]
    assert "print(1)" in sections[0]["content"]
    assert sections[1]["content"] == "正文二"


def test_split_report_by_h1_drops_title_before_numbered_chapters_with_long_intro():
    """总标题 + 长导语 + 编号章节：按"后续章节已带编号"判为总标题。

    此前用"正文不超过 200 字"兜底，一段长导语就让总标题被当成正文首章，
    后面章节整体后移一章，图表任务按章节索引取资料时错位到下一章、
    最后一章还会因索引越界直接失败。
    """
    report = (
        "# Industry report\n\n"
        + "导语内容。" * 60
        + "\n\n# 1. Market\n\n市场正文\n\n# 2. Outlook\n\n展望正文\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. Market", "2. Outlook"]
    assert [section["index"] for section in sections] == [1, 2]


def test_split_report_by_h1_keeps_first_chapter_without_structural_marker():
    """既无编号章节也无前言的输入：首个 H1 按正文首章保留，不能整章丢掉。"""
    report = "# Background\n\nOrders grew by 20%.\n\n# Outlook\n\n展望正文\n"
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["Background", "Outlook"]
    assert sections[0]["content"] == "Orders grew by 20%."


def test_split_report_by_h1_drops_numbered_title_with_toc_body():
    """总标题恰好以章节号开头（``# 1. xxx``）且下面跟着目录/摘要时，仍要当标题丢掉。"""
    report = (
        "# 1. 2025年行业观察\n\n"
        "## 目录\n\n"
        "1. 行业概况 .............. 3\n"
        "2. 竞争格局 .............. 7\n"
        "3. 趋势判断 .............. 12\n\n"
        "## 摘要\n\n"
        "本报告从行业概况、竞争格局、趋势判断三个角度展开，"
        "基于公开数据与访谈记录对 2025 年的行业走向给出判断，"
        "并对关键变量做了敏感性分析，供决策参考。\n\n"
        "# 1. 行业概况\n\n正文一\n\n"
        "# 2. 竞争格局\n\n正文二\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 行业概况", "2. 竞争格局"]
    assert [section["index"] for section in sections] == [1, 2]


def test_split_report_by_h1_drops_title_with_long_preface_body():
    """总标题没有章节号、但正文是整篇目录时，不能被当成正文首章。"""
    report = (
        "# 2025年行业观察\n\n"
        "1. 行业概况 .............. 3\n"
        "2. 竞争格局 .............. 7\n"
        "3. 趋势判断 .............. 12\n\n"
        "# 1. 行业概况\n\n正文一\n\n"
        "# 2. 竞争格局\n\n正文二\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 行业概况", "2. 竞争格局"]


def test_split_report_by_h1_ignores_indented_fence_as_code_block():
    """缩进 4 格（或 tab）的围栏是缩进代码块，不能当围栏开关把后面的章节吞掉。"""
    report = (
        "# 1. 第一章\n\n"
        "    示例：\n\n"
        "    ```\n"
        "    echo before\n"
        "\n"
        "# 2. 第二章\n\n正文二\n"
        "    ```\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. 第一章", "2. 第二章"]
    assert sections[1]["content"] == "正文二\n    ```"


def test_split_section_by_h2_keeps_indented_first_line():
    """首章正文以 4 空格缩进的字面量 ~~~ 开头时，后面的真实 H2 仍要被切出来。

    直接对 H1 切片做 .strip() 会吃掉首行缩进，缩进代码块里的 ~~~ 就变成真正的
    开围栏，把 ``## 1.1 Demand`` 及正文一起吞进代码块。
    """
    report = (
        "# 1. Benchmark\n\n"
        "    ~~~\n\n"
        "## 1.1 Demand\n\n需求正文\n\n"
        "```text\n示例\n```\n\n"
        "## 1.2 Outlook\n\n展望正文\n\n"
        "# 2. Results\n\n结果正文\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)
    sub_sections = FigurePlaceholderGenerator._split_section_by_h2(sections[0])

    assert [sub["title"] for sub in sub_sections] == ["1.1 Demand", "1.2 Outlook"]
    assert "需求正文" in sub_sections[0]["content"]


def test_split_report_by_h1_ignores_preface_heading_inside_code_fence():
    """首章正文里的 Markdown 围栏示例含 ``## Abstract`` 时，整章不能被当前言丢掉。"""
    report = (
        "# 1. Benchmark\n\n"
        "首章正文。\n\n"
        "```markdown\n## Abstract\n示例内容\n```\n\n"
        "# 2. Results\n\n结果正文\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["1. Benchmark", "2. Results"]
    assert sections[0]["content"].startswith("首章正文。")


def test_fenced_lines_handles_many_unclosed_fences_linearly():
    """连续未闭合的围栏起始标记不能退化成二次方扫描（曾 4,000 行耗时约 28 秒）。"""
    import time

    from openjiuwen_deepsearch.algorithm.chart_generation.figure_placeholders import (
        _fenced_lines,
    )

    lines = ["~~~text\n"] * 4000
    started = time.perf_counter()
    fenced = _fenced_lines(lines)
    elapsed = time.perf_counter() - started

    assert fenced == set()
    assert elapsed < 2.0, f"_fenced_lines took {elapsed:.1f}s for 4000 lines"
