# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""OpenAI 兼容 embedding 客户端：SQLite 缓存 + 有限重试指数退避 + 可注入传输层。"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Awaitable, Callable, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openjiuwen_search_base.security import reveal_secret

logger = logging.getLogger(__name__)

# transport(payload) -> (status_code, parsed_json_or_text)
Transport = Callable[[dict], Awaitable[tuple[int, object]]]


class EmbeddingRetryError(Exception):
    """重试耗尽。携带 retries 与最后错误信息，消费方可包装为自己的异常体系。"""

    def __init__(self, retries: int, last_error: object) -> None:
        self.retries = retries
        self.last_error = last_error
        super().__init__(f"Embedding failed after {retries} retries: {last_error}")


class EmbedderSettings(BaseModel):
    """密钥以 bytearray 存储，构造时可传字符串（自动规范化）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str = "https://openrouter.ai/api/v1/embeddings"
    model: str = ""
    api_key: bytearray = Field(default_factory=bytearray)
    query_prefix: str = ""
    cache_dir: str = "."
    max_retries: int = 8
    retry_backoff_seconds: float = 2.0
    max_chars: int = 65535
    # 单次请求总超时（防服务端 accept 后挂住导致重试永不触发）
    timeout_seconds: float = 60.0

    @field_validator("url")
    @classmethod
    def _http_scheme_only(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"embedding url must be http(s), got: {v[:32]}")
        return v


class ApiEmbedder:
    def __init__(self, settings: EmbedderSettings, transport: Optional[Transport] = None) -> None:
        self._settings = settings
        self._transport = transport or self._aiohttp_transport
        self._session = None  # 惰性创建、跨请求复用的 aiohttp session（连接池/免重复握手）
        os.makedirs(settings.cache_dir, exist_ok=True)
        safe_model_name = settings.model.replace("/", "_")
        self._cache_db_path = os.path.join(settings.cache_dir, f"_emb_{safe_model_name}.db")
        with self._connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (text_hash TEXT PRIMARY KEY, embedding TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._cache_db_path)
        # WAL + busy_timeout：多任务并发索引时避免 "database is locked"
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """提交并**关闭**连接。

        ``with sqlite3.connect(...) as conn`` 只是事务上下文：正常退出时 commit，
        但不会 close，长驻索引进程会随每次调用泄漏文件句柄。
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    async def close(self) -> None:
        """释放持久连接（服务化/长驻进程的生命周期收口）。"""
        if self._session is not None:
            await self._session.close()
            self._session = None

    # --- cache ---
    # 缓存键 = **实际发送文本**（含 query 前缀与截断）的 sha256：
    # 同一文本作为文档与作为查询（instruct 前缀不同）产出不同向量，必须分键缓存。
    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _fetch_cached(self, chunks: list[str]) -> tuple[list, list[int], list[str]]:
        embeddings: list = [None] * len(chunks)
        missing_indices: list[int] = []
        missing_chunks: list[str] = []
        with self._connection() as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                cursor.execute(
                    "SELECT embedding FROM cache WHERE text_hash = ?", (self._text_hash(chunk),)
                )
                row = cursor.fetchone()
                if row:
                    embeddings[i] = json.loads(row[0])
                else:
                    missing_indices.append(i)
                    missing_chunks.append(chunk)
        return embeddings, missing_indices, missing_chunks

    def _save_cached(self, chunks: list[str], indices: list[int], vectors: list) -> None:
        with self._connection() as conn:
            for idx, emb in zip(indices, vectors):
                conn.execute(
                    "INSERT OR IGNORE INTO cache (text_hash, embedding) VALUES (?, ?)",
                    (self._text_hash(chunks[idx]), json.dumps(emb)),
                )

    # --- request ---
    def _prepare_input(self, chunks: list[str], is_query: bool) -> list[str]:
        if is_query and self._settings.query_prefix:
            chunks = [self._settings.query_prefix + c for c in chunks]
        return [c[: self._settings.max_chars] for c in chunks]

    async def _aiohttp_transport(self, payload: dict) -> tuple[int, object]:
        import aiohttp  # guarded import

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._settings.timeout_seconds)
            )
        headers = {
            "Authorization": f"Bearer {reveal_secret(self._settings.api_key)}",
            "Content-Type": "application/json",
        }
        async with self._session.post(self._settings.url, headers=headers, json=payload) as resp:
            status = resp.status
            try:
                body = await resp.json()
            except Exception:
                body = await resp.text()
            return status, body

    @staticmethod
    def _ordered_embeddings(items: list) -> list:
        """按响应里的 ``index`` 还原向量顺序。

        OpenAI 兼容服务（vLLM、部分自建代理）允许 ``data`` 乱序返回，只保证每条
        带 ``index``；直接按数组顺序 zip 会把向量错配到别的文本上，且不会报错。

        三种情形分别处理：

        * ``index`` 恰好构成 ``0..n-1`` 的全排列：按 ``index`` 定点回填，顺序确定；
        * 完全没有合法整型 ``index``：退回数组顺序，兼容不发 ``index`` 的服务；
        * 部分缺失 / 重复 / 越界 / 条目缺 ``embedding``：属于服务端契约不一致，
          返回空列表让上层长度校验判失败，宁可重试也不把错配向量写进缓存。
        """
        if not isinstance(items, list) or not all(
            isinstance(item, dict) and "embedding" in item for item in items
        ):
            return []

        raw_indices = [item.get("index") for item in items]
        # bool 是 int 的子类，必须先排除，否则 True/False 会被当成 0/1
        valid_ints = [
            idx for idx in raw_indices if isinstance(idx, int) and not isinstance(idx, bool)
        ]

        if len(valid_ints) == len(items) and set(valid_ints) == set(range(len(items))):
            ordered: list = [None] * len(items)
            for idx, item in zip(valid_ints, items):
                ordered[idx] = item["embedding"]
            return ordered

        if not valid_ints:
            return [item["embedding"] for item in items]

        return []

    async def async_encode(self, chunks: list[str], is_query: bool = False) -> list[list[float]]:
        # 先构造实际发送文本（前缀+截断），缓存查/存一律以它为键；
        # sqlite 读写走线程，不占事件循环
        prepared = self._prepare_input(chunks, is_query)
        embeddings, missing_indices, missing_prepared = await asyncio.to_thread(
            self._fetch_cached, prepared
        )
        if not missing_prepared:
            return embeddings

        payload = {
            "model": self._settings.model,
            "input": missing_prepared,
            "encoding_format": "float",
        }

        last_error: object = None
        for attempt in range(self._settings.max_retries):
            try:
                status, body = await self._transport(payload)
            except Exception as e:  # 网络异常与非 200 同路径重试
                status, body = -1, str(e)
            if status == 200 and isinstance(body, dict) and "data" in body:
                vectors = self._ordered_embeddings(body["data"])
                if len(vectors) == len(missing_prepared):
                    await asyncio.to_thread(self._save_cached, prepared, missing_indices, vectors)
                    for idx, emb in zip(missing_indices, vectors):
                        embeddings[idx] = emb
                    return embeddings
                # 空向量列表说明响应条目缺 embedding，或 index 部分缺失/重复/越界
                last_error = (
                    f"vector count mismatch or inconsistent index: got {len(vectors)}, "
                    f"expected {len(missing_prepared)}"
                )
            else:
                last_error = f"status={status} body={str(body)[:500]}"
            if attempt + 1 >= self._settings.max_retries:
                break  # 最后一轮失败直接抛，不再空等退避
            backoff = self._settings.retry_backoff_seconds * (2**attempt)
            logger.warning(
                "Embedding attempt %d failed (%s); retrying in %.1fs",
                attempt + 1, last_error, backoff,
            )
            await asyncio.sleep(backoff)

        raise EmbeddingRetryError(self._settings.max_retries, last_error)
