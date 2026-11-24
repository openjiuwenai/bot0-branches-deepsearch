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


def test_split_report_by_h1_keeps_titled_first_h1_with_long_body():
    """首个 H1 没有章节号，但下面已是成段正文，说明它是正文首章而非总标题。"""
    report = (
        "# 背景与现状\n\n"
        + "这是一段正文。" * 60
        + "\n\n# 1. 第一章\n\n正文一\n"
    )
    sections = FigurePlaceholderGenerator._split_report_by_h1(report)

    assert [section["title"] for section in sections] == ["背景与现状", "1. 第一章"]


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
