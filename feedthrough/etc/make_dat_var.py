"""将 FLUTE 的 dat 数据编码为 C++ 字符串变量。

这是原 ``MakeDatVar.tcl`` 的跨平台 Python 等价实现，避免 Windows 运行时
额外依赖 Tcl、base64 和 tr 命令。
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path


def main() -> None:
    """读取二进制 dat 文件，并生成定义 ``std::string`` 的 C++ 源文件。"""
    if len(sys.argv) != 4:
        raise SystemExit("Usage: make_dat_var.py <variable> <output.cpp> <input.dat>")
    variable, output_path, data_path = sys.argv[1:]
    encoded = base64.b64encode(Path(data_path).read_bytes()).decode("ascii")
    source = (
        "#include <string>\n"
        "namespace Flute {\n"
        f'std::string {variable} = "{encoded}";\n'
        "}\n"
    )
    Path(output_path).write_text(source, encoding="ascii")


if __name__ == "__main__":
    main()
