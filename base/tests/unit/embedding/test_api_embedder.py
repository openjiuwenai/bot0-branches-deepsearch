"""Tests for :class:`ApiEmbedder` cache lifecycle and vector ordering."""

import asyncio
import os
import sqlite3
from contextlib import closing

import pytest

from openjiuwen_search_base.embedding.api_embedder import (
    ApiEmbedder,
    EmbedderSettings,
    EmbeddingRetryError,
)


def _settings(cache_dir, **overrides) -> EmbedderSettings:
    return EmbedderSettings(
        url="https://example.com/v1/embeddings",
        model="test-model",
        api_key=bytearray(),
        cache_dir=str(cache_dir),
        **overrides,
    )


def _cache_rows(cache_dir) -> int:
    """统计缓存表里的向量条数，用于确认失败批次没有落库。"""
    db = os.path.join(str(cache_dir), "_emb_test-model.db")
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


def _fast_fail_settings(cache_dir) -> EmbedderSettings:
    """关掉退避等待，让契约不一致的响应快速失败。"""
    return _settings(cache_dir, max_retries=1, retry_backoff_seconds=0.0)



class _ConnectionProxy:
    """Records ``close()`` calls made through the embedder's connections."""

    def __init__(self, conn: sqlite3.Connection, closed: list) -> None:
        self._conn = conn
        self._closed = closed

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._closed.append(True)
        self._conn.close()


def test_cache_connections_are_closed(tmp_path, monkeypatch):
    """``with conn`` 只提交不关闭；每次缓存读写都必须真正 close。"""
    closed: list = []
    real_connect = sqlite3.connect

    def _tracked_connect(*args, **kwargs):
        return _ConnectionProxy(real_connect(*args, **kwargs), closed)

    monkeypatch.setattr(sqlite3, "connect", _tracked_connect)

    async def transport(payload):
        return 200, {"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}]}

    embedder = ApiEmbedder(_settings(tmp_path), transport=transport)
    assert asyncio.run(embedder.async_encode(["a", "b"])) == [[1.0], [2.0]]

    # __init__ 建表 1 次 + 缓存未命中查 1 次 + 写回 1 次
    assert len(closed) == 3


def test_out_of_order_response_is_reordered_by_index(tmp_path):
    """响应乱序返回时，向量必须按 index 对齐到请求文本。"""
    sent: list = []

    async def transport(payload):
        sent.append(list(payload["input"]))
        return 200, {
            "data": [
                {"index": 2, "embedding": [3.0]},
                {"index": 0, "embedding": [1.0]},
                {"index": 1, "embedding": [2.0]},
            ]
        }

    embedder = ApiEmbedder(_settings(tmp_path), transport=transport)
    vectors = asyncio.run(embedder.async_encode(["a", "b", "c"]))

    assert vectors == [[1.0], [2.0], [3.0]]
    # 缓存里也必须按原文本存对：二次调用应全部命中，不再请求服务
    again = asyncio.run(embedder.async_encode(["a", "b", "c"]))
    assert again == [[1.0], [2.0], [3.0]]
    assert sent == [["a", "b", "c"]]


def test_missing_index_keeps_response_order(tmp_path):
    """没有 index 字段时保持原有数组顺序，行为与旧实现一致。"""

    async def transport(payload):
        return 200, {"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]}

    embedder = ApiEmbedder(_settings(tmp_path), transport=transport)

    assert asyncio.run(embedder.async_encode(["a", "b"])) == [[1.0], [2.0]]


def test_non_int_index_keeps_response_order(tmp_path):
    """index 全部不是整数时退回数组顺序（视为服务端不发 index）。"""

    async def transport(payload):
        return 200, {
            "data": [
                {"index": "x", "embedding": [1.0]},
                {"index": "y", "embedding": [2.0]},
            ]
        }

    embedder = ApiEmbedder(_settings(tmp_path), transport=transport)

    assert asyncio.run(embedder.async_encode(["a", "b"])) == [[1.0], [2.0]]


def test_partial_index_is_rejected(tmp_path):
    """部分条目带 index、部分不带：契约不一致，不能按数组顺序猜，且不写缓存。"""

    async def transport(payload):
        return 200, {
            "data": [
                {"index": 0, "embedding": [1.0]},
                {"embedding": [2.0]},
            ]
        }

    embedder = ApiEmbedder(_fast_fail_settings(tmp_path), transport=transport)

    with pytest.raises(EmbeddingRetryError):
        asyncio.run(embedder.async_encode(["a", "b"]))
    assert _cache_rows(tmp_path) == 0


def test_duplicate_index_is_rejected(tmp_path):
    """index 重复会让一条向量覆盖另一条，必须判失败而不是静默错配。"""

    async def transport(payload):
        return 200, {
            "data": [
                {"index": 0, "embedding": [1.0]},
                {"index": 0, "embedding": [2.0]},
            ]
        }

    embedder = ApiEmbedder(_fast_fail_settings(tmp_path), transport=transport)

    with pytest.raises(EmbeddingRetryError):
        asyncio.run(embedder.async_encode(["a", "b"]))
    assert _cache_rows(tmp_path) == 0


def test_out_of_range_index_is_rejected(tmp_path):
    """index 越界说明服务端批次与请求不一致，不能按位置硬塞进结果。"""

    async def transport(payload):
        return 200, {
            "data": [
                {"index": 0, "embedding": [1.0]},
                {"index": 2, "embedding": [2.0]},
            ]
        }

    embedder = ApiEmbedder(_fast_fail_settings(tmp_path), transport=transport)

    with pytest.raises(EmbeddingRetryError):
        asyncio.run(embedder.async_encode(["a", "b"]))
    assert _cache_rows(tmp_path) == 0


def test_missing_embedding_field_is_rejected(tmp_path):
    """条目缺 embedding 字段时不抛 KeyError、也不产生错位结果。"""

    async def transport(payload):
        return 200, {"data": [{"index": 0, "embedding": [1.0]}, {"index": 1}]}

    embedder = ApiEmbedder(_fast_fail_settings(tmp_path), transport=transport)

    with pytest.raises(EmbeddingRetryError):
        asyncio.run(embedder.async_encode(["a", "b"]))
    assert _cache_rows(tmp_path) == 0


def test_non_list_data_is_rejected(tmp_path):
    """data 不是数组时直接判失败，不进入缓存写入路径。"""

    async def transport(payload):
        return 200, {"data": "oops"}

    embedder = ApiEmbedder(_fast_fail_settings(tmp_path), transport=transport)

    with pytest.raises(EmbeddingRetryError):
        asyncio.run(embedder.async_encode(["a", "b"]))
    assert _cache_rows(tmp_path) == 0


def test_bool_index_falls_back_to_response_order(tmp_path):
    """bool 是 int 的子类，但不能被当成 0/1；全 bool 视作没有合法 index。"""

    async def transport(payload):
        return 200, {
            "data": [
                {"index": True, "embedding": [1.0]},
                {"index": False, "embedding": [2.0]},
            ]
        }

    embedder = ApiEmbedder(_settings(tmp_path), transport=transport)

    assert asyncio.run(embedder.async_encode(["a", "b"])) == [[1.0], [2.0]]
