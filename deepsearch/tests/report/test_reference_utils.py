import logging

import pytest

from openjiuwen_deepsearch.algorithm.report.report import (
    Reporter,
    _deduplicate_and_renumber_ref,
    _deduplicate_and_renumber_refs,
    _replace_citations_and_classified_index,
)
from openjiuwen_deepsearch.common.common_constants import CHINESE, ENGLISH


@pytest.mark.parametrize("content, refs, lang, expected", [
    # 中文引用
    ("这是正文", ["参考A", "参考B"], CHINESE,
     "这是正文\n## 参考文章\n[1] 参考A\n[2] 参考B"),

    # 英文引用
    ("This is content", ["Ref A", "Ref B"], ENGLISH,
     "This is content\n## References\n[1] Ref A\n[2] Ref B"),

    # 没有引用
    ("正文内容", [], CHINESE, "正文内容"),

    # 没有正文但有引用（返回空字符串）
    ("", ["Ref A"], ENGLISH, ""),

    # 未知语言 → 默认走英文逻辑
    ("Contenu", ["Réf A"], "fr",
     "Contenu\n## References\n[1] Réf A"),
])
def test_add_references(content, refs, lang, expected):
    result = Reporter.add_references(content, refs, lang)
    assert result == expected


def test_deduplicate_and_renumber_with_ref_empty_input():
    text = ""
    result, mapping = _deduplicate_and_renumber_ref(text)
    assert result == ""
    assert mapping == {}


def test_deduplicate_and_renumber_with_ref_single_reference():
    text = "[1] First reference"
    result, mapping = _deduplicate_and_renumber_ref(text)
    assert result == "[1] First reference"
    assert mapping == {"1-1": 1}


def test_deduplicate_and_renumber_with_ref_duplicate_references_same_paragraph():
    text = "[1] First reference\n[2] First reference"
    result, mapping = _deduplicate_and_renumber_ref(text)
    # 去重后只保留一个
    assert result == "[1] First reference"
    # 两个 key 都映射到同一个编号
    assert mapping == {"1-1": 1, "1-2": 1}


def test_deduplicate_and_renumber_with_multiple_paragraphs_and_sections():
    text = "[1] First reference\n\n[1] Second reference\n[2] First reference"
    result, mapping = _deduplicate_and_renumber_ref(text)
    # 应该有两个不同的引用
    assert "[1] First reference" in result
    assert "[2] Second reference" in result
    # 映射按"空行分隔的引用块"给出段落号，不再随空行数量漂移
    assert mapping["1-1"] == 1  # 第一段第一条
    assert mapping["2-1"] == 2  # 第二段第一条
    assert mapping["2-2"] == 1  # 第二段第二条重复了第一段的内容


def test_deduplicate_and_renumber_refs_paragraph_ids_match_block_order():
    # 引用块内部有空行时，旧实现会把段落号推成 3，与消费端 f"{i + 1}-{index}" 对不上
    blocks = ["[1] A\n\n[2] B", "[1] C"]
    _, mapping = _deduplicate_and_renumber_refs(blocks)
    assert mapping == {"1-1": 1, "1-2": 2, "2-1": 3}


def test_deduplicate_and_renumber_refs_skips_empty_blocks():
    blocks = ["", "[1] A", ""]
    result, mapping = _deduplicate_and_renumber_refs(blocks)
    assert result == "[1] A"
    assert mapping == {"2-1": 1}


def test_deduplicate_and_renumber_with_ignore_lines_without_reference():
    text = "This is not a ref\n[1] Valid reference"
    result, mapping = _deduplicate_and_renumber_ref(text)
    assert result == "[1] Valid reference"
    # 只有"当前块已含有效引用"时行首 [1] 才算新段落起点；开头的非引用行
    # （如 ## References 标题）与后面的引用属于同一块，段落号不后移。
    assert mapping == {"1-1": 1}


def test_deduplicate_and_renumber_ref_splits_blocks_on_leading_index_one():
    """历史调用形态：各段落引用块用单个 \\n 拼接，每块首行是 [1]。"""
    text = "[1] A\n[2] B\n[1] C\n[2] D"
    result, mapping = _deduplicate_and_renumber_ref(text)
    assert result == "[1] A\n\n[2] B\n\n[3] C\n\n[4] D"
    # 两块各自独立编号，第二块不应覆盖第一块的映射
    assert mapping == {"1-1": 1, "1-2": 2, "2-1": 3, "2-2": 4}


def test_deduplicate_and_renumber_refs_keeps_first_mapping_on_duplicate_key(caplog):
    """同一段出现重复原编号时保留首次映射并告警，避免引用被静默改指。"""
    with caplog.at_level(logging.WARNING):
        result, mapping = _deduplicate_and_renumber_refs(["[1] A\n[1] B"])
    assert result == "[1] A\n\n[2] B"
    assert mapping == {"1-1": 1}
    assert any("Duplicate reference key '1-1'" in record.message for record in caplog.records)


@pytest.mark.parametrize("paragraphs, classified_contents, ref_map, expected", [
    # 测试用例1：正常情况
    (
            ["This is a paragraph [citation:1].", "Another paragraph [citation:2]."],
            [
                [{"index": 1, "content": "First citation"}],
                [{"index": 2, "content": "Second citation"}]
            ],
            {"1-1": 10, "2-2": 20},
            (["This is a paragraph [citation:10].", "Another paragraph [citation:20]."], [
                [{"index": 10, "content": "First citation"}],
                [{"index": 20, "content": "Second citation"}]
            ])
    ),

    # 测试用例2：没有引用映射
    (
            ["This is a paragraph [citation:1].", "Another paragraph [citation:2]."],
            [
                [{"index": 1, "content": "First citation"}],
                [{"index": 2, "content": "Second citation"}]
            ],
            {},
            (["This is a paragraph [citation:1].", "Another paragraph [citation:2]."], [
                [{"index": 1, "content": "First citation"}],
                [{"index": 2, "content": "Second citation"}]
            ])
    ),

    # 测试用例3：没有分类内容
    #   重编号只看该段落自己的引用块映射，不依赖分类资料是否齐全；
    #   资料缺失时正文编号仍按引用块改写，避免旧编号在下游碰巧命中别的章节。
    (
            ["This is a paragraph [citation:1].", "Another paragraph [citation:2]."],
            [],
            {"1-1": 10, "2-2": 20},
            (["This is a paragraph [citation:10].", "Another paragraph [citation:20]."], [])
    ),

    # 测试用例4：空段落及分类内容
    (
            [],
            [],
            {"1-1": 10, "2-2": 20},
            ([], [])
    )
])
def test_replace_citations_and_classified_index(paragraphs, classified_contents, ref_map, expected):
    result = _replace_citations_and_classified_index(paragraphs, classified_contents, ref_map)
    assert result == expected


def test_replace_citations_is_not_cascaded():
    # 第二段：1->2（重复引用）、2->3（新引用）。逐条 re.sub 会把 [citation:1]
    # 先换成 [citation:2]，再把刚生成的 [citation:2] 换成 [citation:3]。
    _, ref_map = _deduplicate_and_renumber_refs(["[1] x\n[2] y", "[1] y\n[2] z"])
    assert ref_map == {"1-1": 1, "1-2": 2, "2-1": 2, "2-2": 3}

    paragraphs = ["P0 [citation:1] [citation:2]", "P1 [citation:1] [citation:2]"]
    classified = [
        [{"index": 1, "content": "x"}, {"index": 2, "content": "y"}],
        [{"index": 1, "content": "y"}, {"index": 2, "content": "z"}],
    ]
    updated_paragraphs, updated_classified = _replace_citations_and_classified_index(
        paragraphs, classified, ref_map
    )
    assert updated_paragraphs == [
        "P0 [citation:1] [citation:2]",
        "P1 [citation:2] [citation:3]",
    ]
    assert [[item["index"] for item in c] for c in updated_classified] == [[1, 2], [2, 3]]


def test_replace_citations_uses_paragraph_id_from_block_order():
    # 生产者按段落顺序编号后，第二段的重编号不再因为空行而失效
    _, ref_map = _deduplicate_and_renumber_refs(["[1] a\n\n[2] b", "[1] c"])
    paragraphs = ["P0 [citation:1]", "P1 [citation:1]"]
    classified = [[{"index": 1, "content": "a"}], [{"index": 1, "content": "c"}]]
    updated_paragraphs, updated_classified = _replace_citations_and_classified_index(
        paragraphs, classified, ref_map
    )
    assert updated_paragraphs == ["P0 [citation:1]", "P1 [citation:3]"]
    assert updated_classified[1][0]["index"] == 3


def test_replace_citations_uses_own_block_when_classified_contents_is_short():
    """资料列表偏短时，第二章的 [citation:1] 仍要按它自己的引用块改成 [citation:2]。

    旧实现只有在 classified_contents 里有条目时才重写正文，第二章的 [citation:1]
    会原样保留，下游按全局编号溯源就链接到了第一章的文献 A。
    """
    ref_map = {"1-1": 1, "2-1": 2}
    paragraphs = ["Ch1 [citation:1]", "Ch2 [citation:1]"]
    classified = [[{"index": 1, "content": "A"}]]

    updated_paragraphs, updated_classified = _replace_citations_and_classified_index(
        paragraphs, classified, ref_map
    )

    assert updated_paragraphs == ["Ch1 [citation:1]", "Ch2 [citation:2]"]
    assert updated_classified == [[{"index": 1, "content": "A"}]]


def test_replace_citations_marks_unresolvable_citation(caplog):
    """段落引用了自己引用块里没有的编号时，明确标记为不可解析而不是原样保留。"""
    ref_map = {"1-1": 5}
    with caplog.at_level(logging.WARNING):
        updated_paragraphs, _ = _replace_citations_and_classified_index(
            ["P0 [citation:1] [citation:2]"], [], ref_map
        )
    assert updated_paragraphs == ["P0 [citation:5] [citation:?]"]
    assert any("unresolved" in record.message for record in caplog.records)


def test_replace_citations_pairs_sections_by_original_index():
    """中间章节正文为空时，第三个章节要用第三组资料，而不是第二组。"""
    paragraphs = ["Ch1 [citation:1]", "Ch3 [citation:1]"]
    classified = [
        [{"index": 1, "content": "A"}],
        [],
        [{"index": 1, "content": "C"}],
    ]
    ref_map = {"1-1": 1, "2-1": 2}

    updated_paragraphs, updated_classified = _replace_citations_and_classified_index(
        paragraphs, classified, ref_map, section_indexes=[0, 2]
    )

    assert updated_paragraphs == ["Ch1 [citation:1]", "Ch3 [citation:2]"]
    assert [item["index"] for item in updated_classified[0]] == [1]
    assert updated_classified[1] == []
    assert [item["index"] for item in updated_classified[2]] == [2]


def test_refresh_reference_keeps_section_alignment_with_empty_section():
    """报告组装跳过空章节后，资料仍按原始章节下标配对。"""
    result = Reporter.refresh_reference(
        ["Ch1 [citation:1]", "Ch3 [citation:1]"],
        ["[1] A", "[1] C"],
        [[{"index": 1, "content": "A"}], [], [{"index": 1, "content": "C"}]],
        [0, 2],
    )

    assert result["sub_reports_content"] == "Ch1 [citation:1]\n\nCh3 [citation:2]"
    assert result["sub_references"] == "[1] A\n\n[2] C"
    assert [item["index"] for item in result["refreshed_all_classified_contents"][2]] == [2]


def test_deduplicate_and_renumber_ref_normalizes_leading_zero_index():
    """参考文献写成 [02] 时按整数归一化，映射键要与资料 index 对得上。"""
    text = "[1] B\n[02] C"
    result, mapping = _deduplicate_and_renumber_ref(text)
    assert result == "[1] B\n\n[2] C"
    assert mapping == {"1-1": 1, "1-2": 2}


def test_deduplicate_and_renumber_ref_keeps_reference_title_block():
    """单字符串入口带 ## References 标题时，标题不能自成一个空段落。"""
    text = "## References\n[1] A\n[2] B"
    result, mapping = _deduplicate_and_renumber_ref(text)
    assert result == "[1] A\n\n[2] B"
    assert mapping == {"1-1": 1, "1-2": 2}
