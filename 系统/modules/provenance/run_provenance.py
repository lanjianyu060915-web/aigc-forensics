"""文件标识解析命令行；成功发布 JSON 即返回 0。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

PRODUCT_ROOT = Path(__file__).resolve().parents[2]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

from modules.provenance import analyze_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读解析图片、视频的生成标识与溯源元数据")
    parser.add_argument("--input", required=True, type=Path, help="待解析媒体路径")
    parser.add_argument("--output", required=True, type=Path, help="结果 JSON 路径")
    parser.add_argument("--ffprobe", type=Path, help="ffprobe 可执行文件路径")
    args = parser.parse_args(argv)
    temporary = None
    try:
        if args.input.resolve() == args.output.resolve() or (
            args.input.exists() and args.output.exists() and os.path.samefile(args.input, args.output)
        ):
            raise ValueError("输出不得覆盖输入媒体")
        result = analyze_file(args.input, ffprobe=args.ffprobe)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=".provenance-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, args.output)
        return 0
    except Exception as exc:
        print(f"结果写入失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
