import os
import shlex
import subprocess

from .registry import ToolSpec, ToolOutcome


def _split_windows(command: str) -> list[str]:
    """按 Windows 命令行规则（CommandLineToArgvW）拆参数。

    POSIX 的 shlex 把反斜杠当转义符，会吃掉 ``C:\\tmp`` 里的分隔符；
    ``posix=False`` 又只在 token 以引号开头时才当引用，``a"b c"d`` 会被拆成
    ``a"b`` 和 ``c"d``。这里按 Windows 自己的规则分词：反斜杠只在双引号前成对
    转义（n 个反斜杠加一个引号，输出 n // 2 个反斜杠，奇数个再补一个字面量
    引号），双引号内的空格不分割，其余位置的反斜杠原样保留。
    """
    args: list[str] = []
    current: list[str] = []
    in_quotes = False
    pending_slashes = 0
    has_token = False
    for char in command:
        if char == "\\":
            pending_slashes += 1
            continue
        if char == '"':
            current.extend("\\" * (pending_slashes // 2))
            if pending_slashes % 2:
                current.append('"')
            else:
                in_quotes = not in_quotes
                has_token = True
            pending_slashes = 0
            continue
        current.extend("\\" * pending_slashes)
        pending_slashes = 0
        if char in " \t" and not in_quotes:
            if current or has_token:
                args.append("".join(current))
            current = []
            has_token = False
            continue
        current.append(char)
        has_token = True
    current.extend("\\" * pending_slashes)
    if current or has_token:
        args.append("".join(current))
    return args


def split_command(command: str) -> list[str]:
    """把命令行拆成参数列表，按平台选择分词规则。"""
    if os.name != "nt":
        return shlex.split(command)
    return _split_windows(command)


async def execute(ctx, args: dict) -> ToolOutcome:
    command = args.get("command")
    if not command:
        return ToolOutcome(error="Error: command is required.")

    try:
        cmd_args = split_command(command)
        result = subprocess.run(
            cmd_args, shell=False, cwd=ctx.repo_dir, capture_output=True, text=True, timeout=300
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout[:2000]}\n"  # Truncate to avoid context explosion
        if result.stderr:
            output += f"STDERR:\n{result.stderr[:2000]}\n"
        return ToolOutcome(message=output)
    except subprocess.TimeoutExpired:
        return ToolOutcome(error="Error: Command timed out after 300 seconds.")
    except Exception as e:
        return ToolOutcome(error=f"Error executing command: {e}")


SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_bash_command",
        "description": (
            "Executes a bash command in the repository root directory. "
            "Use this to run tests or verify code changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g. 'pytest tests/').",
                }
            },
            "required": ["command"],
        },
    },
}

SPEC = ToolSpec(name="run_bash_command", schema=SCHEMA, executor=execute)
