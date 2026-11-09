"""普通报告文字中的本机路径脱敏，不改变网页地址。"""

import re


_PATH = re.compile(r"(?i)(?:[a-z]:(?:\\|/(?!/))|\\\\)[^\r\n；;，,。]+")
_REMAINDER = re.compile(r"[^\s；;，。<>\"']*\\[^\s；;，。<>\"']*")


def redact_local_paths(value: str) -> str:
    """第一轮移除绝对路径，第二轮清除被标点截断后遗留的目录。"""
    def basename(match):
        text = match.group(0).replace("\\", "/").rstrip("/")
        return text.rsplit("/", 1)[-1]

    return _REMAINDER.sub(basename, _PATH.sub(basename, value))
