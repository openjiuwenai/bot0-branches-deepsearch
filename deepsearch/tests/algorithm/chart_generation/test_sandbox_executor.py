# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""沙箱逃逸回归用例：读写白名单、子模块黑名单、进程启动入口。"""

import asyncio
import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from openjiuwen_deepsearch.algorithm.chart_generation.sandbox.sandbox_executor import (
    AsyncCodeExecutor,
)

# 用测试文件所在目录作为"仓库外"路径：它既不在工作目录，也不在系统临时目录
OUTSIDE_DIR = Path(__file__).resolve().parent

_HAS_MPL = importlib.util.find_spec("matplotlib") is not None
requires_matplotlib = pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装")


@pytest.fixture(scope="module")
def executor(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("sandbox_workdir")
    return AsyncCodeExecutor(working_dir=str(workdir), exec_timeout=120)


def _run(executor, code):
    return asyncio.run(executor.execute(code))


def _lit(value) -> str:
    return json.dumps(str(value))


def test_os_open_cannot_write_outside_workdir(executor):
    target = OUTSIDE_DIR / "_sandbox_denied_write.txt"
    code = (
        "import os\n"
        f"fd = os.open({_lit(target)}, os.O_WRONLY | os.O_CREAT)\n"
        "os.write(fd, b'x')\n"
        "os.close(fd)\n"
        "print('wrote')\n"
    )
    result = _run(executor, code)

    assert result["error"] is True
    assert "PermissionError" in result["stderr"]
    assert not target.exists()


def test_os_open_cannot_read_outside_allowlist(executor):
    target = Path(__file__).resolve()
    code = (
        "import os\n"
        f"fd = os.open({_lit(target)}, os.O_RDONLY)\n"
        "print(len(os.read(fd, 16)))\n"
        "os.close(fd)\n"
    )
    result = _run(executor, code)

    assert result["error"] is True
    assert "PermissionError" in result["stderr"]


def test_pathlib_write_cannot_escape_workdir(executor):
    target = OUTSIDE_DIR / "_sandbox_denied_pathlib.txt"
    code = f"from pathlib import Path\nPath({_lit(target)}).write_text('x')\nprint('wrote')\n"
    result = _run(executor, code)

    assert result["error"] is True
    assert "PermissionError" in result["stderr"]
    assert not target.exists()


def test_os_remove_cannot_touch_files_outside_workdir(executor):
    target = OUTSIDE_DIR / "_sandbox_denied_remove.txt"
    target.write_text("keep me\n", encoding="utf-8")
    code = f"import os\nos.remove({_lit(target)})\nprint('removed')\n"
    try:
        result = _run(executor, code)
        assert result["error"] is True
        assert "PermissionError" in result["stderr"]
        assert target.exists()
    finally:
        target.unlink(missing_ok=True)


def test_dotted_blacklist_blocks_asyncio_subprocess(executor):
    """限制名单里的子模块条目必须生效（此前只比较点号前的包名）。"""
    code = "import asyncio.subprocess\nprint('imported')\n"
    result = _run(executor, code)

    assert result["error"] is True
    assert "ImportError" in result["stderr"]


@pytest.mark.skipif(sys.platform != "win32", reason="os.startfile 仅 Windows 存在")
def test_os_startfile_is_blocked(executor):
    code = "import os\nos.startfile('calc.exe')\nprint('started')\n"
    result = _run(executor, code)

    assert result["error"] is True
    assert "PermissionError" in result["stderr"]


def test_legit_write_and_read_inside_workdir_still_work(executor):
    code = (
        "from pathlib import Path\n"
        "import os\n"
        "Path('legit.txt').write_text('ok', encoding='utf-8')\n"
        "fd = os.open('legit2.txt', os.O_WRONLY | os.O_CREAT)\n"
        "os.write(fd, b'ok')\n"
        "os.close(fd)\n"
        "print(Path('legit.txt').read_text(encoding='utf-8'))\n"
    )
    result = _run(executor, code)

    assert result["error"] is False, result["stderr"]
    assert "ok" in result["stdout"]


def test_os_symlink_to_outside_allowlist_is_denied(executor):
    """symlink 的 src 也要校验：否则可先挂链、再顺着链接读取白名单外的文件。

    守卫在真正调用 os.symlink 之前就拒绝，因此该用例不依赖平台是否允许创建
    符号链接。
    """
    link = Path(executor.working_dir) / "_sandbox_denied_link"
    code = (
        "import os\n"
        f"os.symlink({_lit(OUTSIDE_DIR)}, {_lit(link)})\n"
        "print('linked')\n"
    )
    result = _run(executor, code)

    assert result["error"] is True
    assert "PermissionError" in result["stderr"]
    assert not link.exists()


def test_read_through_symlink_cannot_escape_workdir(executor):
    """已存在的符号链接指向白名单外时，读取必须按真实目标（realpath）判定。"""
    secret = OUTSIDE_DIR / "_sandbox_symlink_secret.txt"
    link = Path(executor.working_dir) / "_sandbox_outside_link"
    secret.write_text("secret\n", encoding="utf-8")
    try:
        os.symlink(str(OUTSIDE_DIR), str(link), target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - 环境不支持符号链接
        secret.unlink(missing_ok=True)
        pytest.skip(f"cannot create symlink: {exc}")
    try:
        code = (
            "from pathlib import Path\n"
            "print(Path('_sandbox_outside_link/_sandbox_symlink_secret.txt').read_text())\n"
        )
        result = _run(executor, code)

        assert result["error"] is True
        assert "PermissionError" in result["stderr"]
        assert "secret" not in result["stdout"]
    finally:
        link.unlink(missing_ok=True)
        secret.unlink(missing_ok=True)


def test_symlink_inside_workdir_is_still_allowed(executor):
    """白名单内的符号链接不能被误伤。"""
    code = (
        "from pathlib import Path\n"
        "import os\n"
        "Path('_sandbox_link_target.txt').write_text('ok', encoding='utf-8')\n"
        "os.symlink('_sandbox_link_target.txt', '_sandbox_link_alias.txt')\n"
        "print(Path('_sandbox_link_alias.txt').read_text(encoding='utf-8'))\n"
    )
    result = _run(executor, code)

    if result["error"] is True:
        # Windows 默认不允许普通用户创建符号链接（WinError 1314），此时跳过而非判失败
        if "WinError 1314" in result["stderr"] or "privilege" in result["stderr"].lower():
            pytest.skip("environment cannot create symlinks")
    assert result["error"] is False, result["stderr"]
    assert "ok" in result["stdout"]


def test_keyword_call_forms_are_still_guarded(executor):
    """关键字传参不能绕过守卫：`open(file=...)`、`os.remove(path=...)` 等。

    旧实现只按位置下标取路径，参数写在 kwargs 里时下标取不到，守卫形同虚设。
    """
    cases = {
        "open": "open(file={p}, mode='w')\n",
        "remove": "import os\nos.remove(path={p})\n",
        "rename": "import os\nos.rename(src={p}, dst='moved.txt')\n",
        "mkdir": "import os\nos.mkdir(path={p})\n",
    }
    target = OUTSIDE_DIR / "_sandbox_denied_keyword.txt"
    target.write_text("keep me\n", encoding="utf-8")
    try:
        for name, snippet in cases.items():
            code = snippet.format(p=_lit(target))
            result = _run(executor, code)
            assert result["error"] is True, name
            assert "PermissionError" in result["stderr"], (name, result["stderr"])
        assert target.exists()
    finally:
        target.unlink(missing_ok=True)


def test_from_import_of_restricted_submodule_is_blocked(executor):
    """`from asyncio import subprocess` 与 importlib 相对导入都要按解析后的模块名判定。"""
    for code in (
        "from asyncio import subprocess\nprint('imported')\n",
        "import importlib\nimportlib.import_module('.subprocess', 'asyncio')\nprint('imported')\n",
    ):
        result = _run(executor, code)
        assert result["error"] is True, code
        assert "ImportError" in result["stderr"], result["stderr"]
        assert "imported" not in result["stdout"]


@requires_matplotlib
def test_plain_line_chart_still_returns_png(executor):
    """端到端：带中文轴标签的普通折线图必须回传有效 PNG。

    字体是在渲染阶段才按需读取的，读写白名单若把字体目录一起拦掉，
    就会退化成 chart_base64=None 且 error=False。
    """
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3], [3, 1, 2])\n"
        "ax.set_title('销售趋势')\n"
        "ax.set_xlabel('月份')\n"
        "ax.set_ylabel('销售额（万元）')\n"
        "plt.tight_layout()\n"
    )
    result = _run(executor, code)

    assert result["error"] is False, result["stderr"]
    assert result["chart_base64"], result["stderr"]
    assert base64.b64decode(result["chart_base64"])[:8] == b"\x89PNG\r\n\x1a\n"


@requires_matplotlib
def test_render_failure_is_reported_as_error(executor):
    """渲染失败不能静默成“没有图”，必须作为执行错误返回。"""
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2], [2, 1])\n"
        "def _boom(*a, **kw):\n"
        "    raise RuntimeError('render boom')\n"
        "fig.savefig = _boom\n"
    )
    result = _run(executor, code)

    assert result["error"] is True
    assert result["chart_base64"] is None
    assert "__CHART_RENDER_ERROR__" in result["stderr"]
    assert "render boom" in result["stderr"]

