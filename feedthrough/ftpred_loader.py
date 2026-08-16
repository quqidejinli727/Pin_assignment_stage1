"""
作用为从block.json和pingroup.json(函数未被引用时，被引用时通过上层文件传输数据，参考text.py)生成ftpred所需的modules/nets数据，并发送到ftpred。

传输数据方式：
  - modules 与 nets 都通过 stdin 输入给 ftpred(argv[1]=="-" 且 argv[2]=="-")，中间用一行 '---NETS---' 分隔。
  - ftpred 的输出支持写到 stdout：当 argv[3]=="-" 时，结果逐行输出到 stdout。
"""

import argparse
import os
import subprocess
import sys
import threading
import struct
from typing import Optional, Union, Any

from PlaceDB import PlaceDB


class _OneNetDBView:
    """一个只包含单个 net 的轻量 DB 视图。

    用途：复用 build_nets_text / assign_feedthrough 的逻辑，但只针对一个 net。
    只复制必要字段（modules 与 nets_list）。
    """

    def __init__(self, *, all_modules_list, all_module_dict, nets_list):
        self.all_modules_list = all_modules_list
        self.all_module_dict = all_module_dict
        self.nets_list = nets_list


def _one_net_view(db: PlaceDB, net_idx: int) -> _OneNetDBView:
    nets = getattr(db, "nets_list", [])
    if net_idx < 0 or net_idx >= len(nets):
        raise IndexError(f"net index {net_idx} out of range (0..{len(nets) - 1})")
    return _OneNetDBView(
        all_modules_list=getattr(db, "all_modules_list", []),
        all_module_dict=getattr(db, "all_module_dict", {}),
        nets_list=[nets[net_idx]],
    )


def _one_net_view_from_net(db: PlaceDB, net) -> _OneNetDBView:
    """根据 net 对象构造 one-net view（不依赖 net_idx）。"""
    if net is None:
        raise ValueError("net is None")
    return _OneNetDBView(
        all_modules_list=getattr(db, "all_modules_list", []),
        all_module_dict=getattr(db, "all_module_dict", {}),
        nets_list=[net],
    )


def _is_nonleaf_module(db: PlaceDB, parent_inst: str) -> bool:
    mod = db.all_module_dict.get(parent_inst)
    return bool(mod and bool(getattr(mod, "children", None)))


def _pin_coord(pin):
    return (float(getattr(pin, "x")), float(getattr(pin, "y")))


def _pin_full_name(pin) -> Optional[str]:
    """Return the canonical pingroup full name: parent_inst + '.' + pingroup_name.

    The uploaded pingroup.json uses exactly this convention in `successors`:
      successor string == target.parent_inst + "." + target.pingroup_name

    We still accept a few compatibility attributes, but the strict key is always
    preferred when parent_inst and pingroup_name are present.
    """
    parent = (
        getattr(pin, "parent_inst", None)
        or getattr(pin, "parent", None)
        or getattr(pin, "inst", None)
    )
    pgname = (
        getattr(pin, "pingroup_name", None)
        or getattr(pin, "pin_group_name", None)
        or getattr(pin, "pin_name", None)
        or getattr(pin, "name", None)
    )
    if parent and pgname:
        return f"{parent}.{pgname}"

    for attr in ("full_name", "fullname", "global_name", "full_pin_name"):
        val = getattr(pin, attr, None)
        if val:
            return str(val)
    return None


def _pin_lookup_keys(pin) -> list[Any]:
    """Keys used to resolve successor references.

    Primary key is the pingroup full name. Extra keys are compatibility only.
    """
    keys: list[Any] = [pin, id(pin)]

    full = _pin_full_name(pin)
    if full:
        keys.append(full)

    parent = getattr(pin, "parent_inst", None)
    pgname = getattr(pin, "pingroup_name", None)
    if parent and pgname:
        keys.append((parent, pgname))

    for attr in (
        "id", "uid", "idx", "index",
        "pingroup_name", "pin_group_name", "pin_name", "name",
        "full_name", "fullname", "global_name", "full_pin_name",
    ):
        val = getattr(pin, attr, None)
        if val is not None:
            keys.append(val)
            keys.append(str(val))

    # De-duplicate while preserving order and ignoring unhashable values.
    out: list[Any] = []
    seen = set()
    for k in keys:
        try:
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
        except TypeError:
            out.append(k)
    return out


def _build_pin_lookup(pins: list) -> dict[Any, Any]:
    lookup: dict[Any, Any] = {}
    for i, pin in enumerate(pins):
        lookup[i] = pin
        lookup[str(i)] = pin
        for key in _pin_lookup_keys(pin):
            try:
                lookup[key] = pin
            except TypeError:
                pass
    return lookup


def _normalize_successor_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value.decode() if isinstance(value, bytes) else value]
    if isinstance(value, dict):
        # e.g. {successor_name: weight}. Successor references are the keys.
        return list(value.keys())
    try:
        return list(value)
    except TypeError:
        return [value]


def _successor_refs(pin) -> list:
    refs: list = []
    for attr in ("successors", "successor", "succs", "succ", "next_pins", "next"):
        if hasattr(pin, attr):
            refs.extend(_normalize_successor_list(getattr(pin, attr)))
    return refs


def _resolve_pin_ref(ref, lookup: dict[Any, Any]):
    """Resolve a successor reference to the target pin object.

    The expected reference form is a string such as
    'TOP.U_X...<parent_inst>.<pingroup_name>'. Direct object references and
    compatibility keys are also accepted.
    """
    if hasattr(ref, "x") and hasattr(ref, "y"):
        return ref
    candidates = [ref]
    try:
        candidates.append(str(ref))
    except Exception:
        pass
    for key in candidates:
        try:
            if key in lookup:
                return lookup[key]
        except TypeError:
            pass
    return None


def _extract_successor_pairs_strict(net):
    """Extract explicit pin->successor pairs from a PlaceDB net.

    This implements the pingroup.json semantics:
      src_full_name = src.parent_inst + "." + src.pingroup_name
      each src.successors item is dst_full_name

    No p0->p1->... chain fallback is used here. If a successor cannot be
    resolved inside this net's pin set, it is reported in the returned stats.
    """
    pins = list(getattr(net, "pins", []) or [])
    lookup = _build_pin_lookup(pins)
    pairs: list[tuple[Any, Any]] = []
    seen = set()
    missing: list[tuple[str, str]] = []

    def add_pair(src, dst_ref):
        dst = _resolve_pin_ref(dst_ref, lookup)
        if dst is None:
            src_name = _pin_full_name(src) or str(getattr(src, "pingroup_name", "<unknown-src>"))
            missing.append((src_name, str(dst_ref)))
            return
        if src is dst:
            return
        key = (id(src), id(dst))
        if key in seen:
            return
        seen.add(key)
        pairs.append((src, dst))

    # 1) Preferred: pin-level successors, matching pingroup.json.
    for pin in pins:
        for dst_ref in _successor_refs(pin):
            add_pair(pin, dst_ref)

    # 2) Compatibility: net-level edge tables if PlaceDB materialized them.
    if not pairs:
        for attr in ("successor_edges", "edges", "pairs", "pin_pairs"):
            raw_edges = getattr(net, attr, None)
            if not raw_edges:
                continue
            for e in _normalize_successor_list(raw_edges):
                if isinstance(e, dict):
                    src = e.get("src", e.get("source", e.get("from", e.get("u"))))
                    dst = e.get("dst", e.get("target", e.get("to", e.get("v"))))
                    if src is not None and dst is not None:
                        src_pin = _resolve_pin_ref(src, lookup)
                        if src_pin is not None:
                            add_pair(src_pin, dst)
                elif isinstance(e, (list, tuple)) and len(e) >= 2:
                    src_pin = _resolve_pin_ref(e[0], lookup)
                    if src_pin is not None:
                        add_pair(src_pin, e[1])
            if pairs:
                break

    return pairs, {"missing_successors": missing, "pin_count": len(pins), "edge_count": len(pairs)}


def build_successor_edge_coords(
    db: PlaceDB,
    net,
    *,
    split_nonleaf_nets: bool = True,
    return_stats: bool = False,
):
    """Expand one PlaceDB net into two-pin successor edges.

    Strict pingroup semantics are used. The function does NOT interpret
    net.pins order as p0->p1->...; unordered pins without explicit successors
    produce zero child edges.
    """
    pairs, stats = _extract_successor_pairs_strict(net)
    edges = [(_pin_coord(a), _pin_coord(b)) for a, b in pairs]
    if return_stats:
        return edges, stats
    return edges


def _find_parent_nonleaf_inst(db: PlaceDB, leaf_inst: str) -> Optional[str]:
    """
    给出一个叶模块，向上找出其最近的非叶模块。
    找不到则返回 None。
    """
    if not leaf_inst:
        return None
    parts = leaf_inst.split(".")
    """
    逐级弹出
    """
    while len(parts) > 1:
        parts.pop()
        cand = ".".join(parts)
        if _is_nonleaf_module(db, cand):
            return cand
    return None


def filter_nonleaf_pins_inplace(db: PlaceDB, *, remove_from_net: bool = False) -> int:
    """
    识别 net 中哪些 pin 来自非叶模块(children 非空)，并将其剔除。
    Returns:被判定为“非叶模块 pin”并被剔除的 pin 数量。
    作为备用手段
    """
    removed_pins = 0
    for net in getattr(db, "nets_list", []):
        pins = getattr(net, "pins", [])
        if not pins:
            continue

        keep = []
        for pin in pins:
            parent_inst = getattr(pin, "parent_inst", None)
            mod = db.all_module_dict.get(parent_inst) if parent_inst else None
            is_nonleaf = bool(getattr(mod, "children", None)) if mod else False
            if is_nonleaf:
                removed_pins += 1
            else:
                keep.append(pin)

        if remove_from_net and len(keep) != len(pins):
            net.pins = keep

    return removed_pins


def filter_nets_inplace(db: PlaceDB, min_pins: int = 2) -> int:
    """
    过滤只有单个引脚的nets(存在于同构模块中)。
    Returns:被过滤掉的 net 数量
    """
    original = len(getattr(db, "nets_list", []))
    db.nets_list = [net for net in db.nets_list if len(getattr(net, "pins", [])) >= min_pins]
    return original - len(db.nets_list)


def build_modules_text(db: PlaceDB) -> str:
    """
    将模块转换为 ftpred 需要的文本格式，并分配 id。
    """
    modules = []  
    next_id = 1
    for m in db.all_modules_list:
        if (not m.children) and hasattr(m, "vertex") and m.vertex and len(m.vertex) >= 3:
            modules.append((next_id, m.module_name if m.module_name else m.name, m.vertex))
            next_id += 1

    lines = [str(len(modules))]
    for mid, name, verts in modules:
        line = f"{mid} {name} {len(verts)}"
        for v in verts:
            line += f" {v[0]} {v[1]}"
        lines.append(line)
    return "\n".join(lines)


def build_nets_text_all(
    db: PlaceDB,
    *,
    return_stats: bool = True,
    split_nonleaf_nets: bool = True,
):
    """将所有 nets 转换为 ftpred 文本格式。

    当前语义：一个 PlaceDB net 会被严格展开为 pingroup.json 中定义的
    pin->successor 二端 child nets。

    full pin name = parent_inst + "." + pingroup_name；successors 中的
    字符串必须能在同一个 net 的 pin 集合中解析到目标 pin。
    不再把 net.pins 的相邻顺序当作 p0->p1->... chain fallback。
    """
    child_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    orig_to_children: dict[int, list[int]] = {}
    missing_by_orig: dict[int, list[tuple[str, str]]] = {}
    edge_count_by_orig: dict[int, int] = {}

    for orig_idx, net in enumerate(getattr(db, "nets_list", []) or []):
        edge_coords, edge_stats = build_successor_edge_coords(
            db,
            net,
            split_nonleaf_nets=split_nonleaf_nets,
            return_stats=True,
        )
        child_ids: list[int] = []
        for edge in edge_coords:
            child_ids.append(len(child_edges))
            child_edges.append(edge)
        orig_to_children[orig_idx] = child_ids
        if edge_stats.get("missing_successors"):
            missing_by_orig[orig_idx] = edge_stats["missing_successors"]
        edge_count_by_orig[orig_idx] = int(edge_stats.get("edge_count", len(edge_coords)))

    lines = [str(len(child_edges))]
    for (x1, y1), (x2, y2) in child_edges:
        lines.append(f"2 {x1} {y1} {x2} {y2}")
    nets_text = "\n".join(lines)

    if return_stats:
        info = {
            "orig_to_children": orig_to_children,
            "child_net_count": len(child_edges),
            "mode": "strict_successor_edges",
            "missing_successors": missing_by_orig,
            "edge_count_by_orig": edge_count_by_orig,
        }
        return nets_text, info
    return nets_text

def build_nets_text(
    db: PlaceDB,
    net,
    *,
    return_stats: bool = True,
    split_nonleaf_nets: bool = True,
):
    """只针对一个 net 生成 Nets 文本（仍可能拆成多个 child nets）。

    输入：
      - db: 用于判断模块层级、取 modules 字典等
      - net: 单个 net 对象（通常来自 db.nets_list[i]）

    返回：
      - (nets_text, build_info) 或 nets_text

    注意：build_info['orig_to_children'] 的 key 会是 0（因为 view 内只有一个 net）。
    """
    view = _one_net_view_from_net(db, net)
    return build_nets_text_all(view, return_stats=return_stats, split_nonleaf_nets=split_nonleaf_nets)


def build_nets_text_by_index(
    db: PlaceDB,
    net_idx: int,
    *,
    return_stats: bool = True,
    split_nonleaf_nets: bool = True,
):
    """兼容接口：通过 net_idx 调 build_nets_text。后续建议改用 build_nets_text(db, net)。"""
    view = _one_net_view(db, net_idx)
    return build_nets_text(db, view.nets_list[0], return_stats=return_stats, split_nonleaf_nets=split_nonleaf_nets)


def run_ftpred_for_one_net(
    db: PlaceDB,
    net,
    ftpred_path: str,
    *,
    Modules: Optional[str] = None,
    split_nonleaf_nets: bool = True,
) -> float:
    """只跑一个 net，并把聚合后的 feedthrough 回写到该 net 对象的 feedthrough 属性。"""
    if Modules is None:
        Modules = build_modules_text(db)

    Nets, build_info = build_nets_text(
        db,
        net,
        return_stats=True,
        split_nonleaf_nets=split_nonleaf_nets,
    )

    out = run_ftpred(ftpred_path, Modules, Nets)

    view = _one_net_view_from_net(db, net)
    assign_feedthrough(view, out, build_info)
    ft = float(getattr(view.nets_list[0], "feedthrough", 0.0))
    setattr(net, "feedthrough", ft)
    return ft


def run_ftpred(ftpred_path: str, Modules: str, Nets: str) -> str:
    """运行 ftpred 并返回其输出文本(stdout)。"""
    ftpred_path = os.path.normpath(ftpred_path)
    cmd = [ftpred_path, "-", "-", "-"]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        print("Failed to start ftpred:", e, file=sys.stderr)
        sys.exit(1)

    payload = Modules + "\n" + "---NETS---" + "\n" + Nets + "\n"
    out, err = p.communicate(payload)

    if err:
        print(f"ftpred stderr: {err}", file=sys.stderr)
    if p.returncode != 0:
        print(f"ftpred failed with return code: {p.returncode}", file=sys.stderr)
        sys.exit(p.returncode)

    return out or ""


class FtpredSession:
    """常驻 ftpred 进程会话（方案A）。

    协议：
      - 首次：发送 Modules 文本 + '\n---NETS---\n'（不紧跟 nets block 也可以）
      - 每次计算：发送 '---NETS---\n' + <nets block>，其中 nets block 格式与旧版相同
      - 每次计算后：ftpred 会输出若干行 'Net i feedthrough = x' 并 flush
      - 结束：发送 '---QUIT---\n' 并关闭 stdin
    """

    def __init__(self, ftpred_path: str, Modules: str):
        self.ftpred_path = os.path.normpath(ftpred_path)
        cmd = [self.ftpred_path, "-", "-", "-"]
        self._p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        if not self._p.stdin or not self._p.stdout:
            raise RuntimeError("Failed to open pipes for ftpred")

        # 后台收集 stderr，避免缓冲区塞满卡死
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # 发送 modules 段
        self._p.stdin.write(Modules)
        if not Modules.endswith("\n"):
            self._p.stdin.write("\n")
        self._p.stdin.write("---NETS---\n")
        self._p.stdin.flush()

        # 做一次“空 block 握手”，用于：
        # 1) 立刻验证当前运行的 ftpred.exe 是否支持 streaming + ---END---
        # 2) 避免后续首次真实 block 才暴露“协议不一致/旧版本/解析卡死”的问题
        # 空 block 的输入为：nNets=0，不会产生 Net 行，只会输出 ---END---
        try:
            self._warmup_handshake(timeout_s=10.0)
        except Exception as e:
            tail = "".join(self._stderr_lines[-50:])
            raise RuntimeError(
                "FtpredSession handshake failed. "
                "This usually means you're running an older ftpred.exe that doesn't emit ---END---, "
                "or the stdin streaming protocol is not matched. "
                f"Inner error: {e}\nLast stderr lines:\n{tail}"
            )


def _pack_bin_modules(modules_text: str) -> bytes:
    """把 build_modules_text 的文本解析回几何数据，并按二进制协议打包。

    说明：
      - 为了最小侵入先复用 modules_text（Python 端已经有）
      - 这里解析文本生成 (id, verts) 后打包为二进制 MODULES 帧 payload
    """
    lines = [ln.strip() for ln in (modules_text or "").splitlines() if ln.strip()]
    if not lines:
        n = 0
        payload = struct.pack("<B I", 1, n)
        return payload

    n = int(lines[0])
    payload_parts: list[bytes] = [struct.pack("<B I", 1, n)]
    for i in range(1, 1 + n):
        # line: id name vcount x y ...
        parts = lines[i].split()
        mid = int(parts[0])
        vcount = int(parts[2])
        # verts start at index 3
        coords = list(map(float, parts[3:3 + 2 * vcount]))
        payload_parts.append(struct.pack("<i I", mid, vcount))
        for vi in range(vcount):
            x = coords[2 * vi]
            y = coords[2 * vi + 1]
            payload_parts.append(struct.pack("<dd", x, y))

    return b"".join(payload_parts)


def _pack_bin_nets_from_text(nets_text: str) -> bytes:
    """把 build_nets_text 的文本解析并打包为 NETS payload。保留兼容。"""
    lines = [ln.strip() for ln in (nets_text or "").splitlines() if ln.strip()]
    if not lines:
        return struct.pack("<B I", 2, 0)
    n = int(lines[0])
    payload_parts: list[bytes] = [struct.pack("<B I", 2, n)]
    for i in range(1, 1 + n):
        parts = lines[i].split()
        pcount = int(parts[0])
        payload_parts.append(struct.pack("<I", pcount))
        coords = list(map(float, parts[1:1 + 2 * pcount]))
        for pi in range(pcount):
            x = coords[2 * pi]
            y = coords[2 * pi + 1]
            payload_parts.append(struct.pack("<dd", x, y))
    return b"".join(payload_parts)


def _pack_bin_successor_edges(
    edge_coords: list[tuple[tuple[float, float], tuple[float, float]]]
) -> bytes:
    """直接把二端 successor edges 打包为 NETS payload。

    相比旧路径：net -> 文本 Nets -> 解析文本 -> 二进制，
    这里直接 net -> 二进制，避免重复字符串构造和解析。
    """
    payload_parts: list[bytes] = [struct.pack("<B I", 2, len(edge_coords))]
    for (x1, y1), (x2, y2) in edge_coords:
        payload_parts.append(struct.pack("<I", 2))
        payload_parts.append(struct.pack("<dd", float(x1), float(y1)))
        payload_parts.append(struct.pack("<dd", float(x2), float(y2)))
    return b"".join(payload_parts)

def _write_frame(w, payload: bytes) -> None:
    w.write(struct.pack("<I", len(payload)))
    if payload:
        w.write(payload)
    w.flush()


def _read_exact(r, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = r.read(n - len(buf))
        if not chunk:
            raise EOFError(f"EOF while reading {n} bytes (got {len(buf)})")
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(r) -> bytes:
    hdr = _read_exact(r, 4)
    (ln,) = struct.unpack("<I", hdr)
    if ln == 0:
        return b""
    return _read_exact(r, ln)


class FtpredBinSession:
    """跨平台二进制帧协议的常驻会话实现（推荐路径）。

    生命周期应由上层持有，例如 text.py 中的：
        with FtpredBinSession(ftpred_path, Modules) as sess:
            sess.run_one_net(...)

    不做 feedthrough 结果缓存；net 坐标变化时每次都会重新发送当前坐标。
    """

    def __init__(self, ftpred_path: str, Modules: str):
        self.ftpred_path = os.path.normpath(ftpred_path)
        cmd = [self.ftpred_path, "-", "-", "-", "--bin"]
        self._p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        if not self._p.stdin or not self._p.stdout:
            raise RuntimeError("Failed to open pipes for ftpred (bin)")

        self._stderr_chunks: list[bytes] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr_bin, daemon=True)
        self._stderr_thread.start()

        # MODULES 只发送一次；C++ 端会缓存 module grid。
        _write_frame(self._p.stdin, _pack_bin_modules(Modules))

        # 空 NETS 握手：验证当前 ftpred 支持二进制协议。
        _write_frame(self._p.stdin, struct.pack("<B I", 2, 0))
        resp = _read_frame(self._p.stdout)
        if len(resp) < 4:
            tail = self._stderr_tail()
            raise RuntimeError(f"Handshake failed (resp too short). stderr tail:\n{tail}")
        (n,) = struct.unpack("<I", resp[:4])
        if n != 0:
            raise RuntimeError(f"Handshake failed: expected n=0, got n={n}")

    def _drain_stderr_bin(self):
        try:
            if not self._p.stderr:
                return
            for chunk in iter(lambda: self._p.stderr.read(4096), b""):
                if not chunk:
                    break
                self._stderr_chunks.append(chunk)
        except Exception:
            return

    def _stderr_tail(self, max_bytes: int = 8192) -> str:
        data = b"".join(self._stderr_chunks[-32:])[-max_bytes:]
        return data.decode("utf-8", errors="replace")

    def run_one_net(self, db: PlaceDB, net, *, split_nonleaf_nets: bool = True) -> float:
        """计算单个 net 的 feedthrough，并写回 net.feedthrough。

        内部会把 net 严格展开成若干二端 successor child nets。
        不使用 net.pins 相邻顺序 fallback。
        """
        edge_coords, edge_stats = build_successor_edge_coords(
            db,
            net,
            split_nonleaf_nets=split_nonleaf_nets,
            return_stats=True,
        )
        if edge_stats.get("missing_successors"):
            setattr(net, "ftpred_missing_successors", edge_stats["missing_successors"])
        setattr(net, "ftpred_child_edge_count", int(edge_stats.get("edge_count", len(edge_coords))))
        nets_payload = _pack_bin_successor_edges(edge_coords)
        _write_frame(self._p.stdin, nets_payload)

        resp = _read_frame(self._p.stdout)
        if len(resp) < 4:
            raise RuntimeError(f"Malformed response frame. stderr tail:\n{self._stderr_tail()}")

        (n,) = struct.unpack("<I", resp[:4])
        off = 4
        total = 0.0
        for _ in range(n):
            if off + 4 > len(resp):
                raise RuntimeError("Malformed response frame: truncated feedthrough array")
            (ft_i,) = struct.unpack("<i", resp[off:off + 4])
            total += float(ft_i)
            off += 4

        # 不做缓存：上层 net 坐标可能反复变化。
        setattr(net, "feedthrough", total)
        return total

    def close(self):
        if getattr(self, "_p", None) and self._p.stdin:
            try:
                # 二进制协议 QUIT 帧：msgType=3。
                _write_frame(self._p.stdin, struct.pack("<B", 3))
                self._p.stdin.close()
            except Exception:
                pass
        if getattr(self, "_p", None):
            try:
                self._p.wait(timeout=2)
            except Exception:
                try:
                    self._p.kill()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

def _can_open_ftpred(ftpred_path: str, *, timeout_s: float = 2.0) -> tuple[bool, str]:
    """快速检查 ftpred.exe 是否能启动且一启动就退出（打印 usage 也算）。

    目的：帮助定位“卡在启动 ftpred / CreateProcess / 权限 / 依赖缺失”等问题。
    """
    ftpred_path = os.path.normpath(ftpred_path)
    try:
        p = subprocess.Popen(
            [ftpred_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        return False, f"Failed to spawn: {e}"
    try:
        out, err = p.communicate(timeout=float(timeout_s))
    except Exception as e:
        try:
            p.kill()
        except Exception:
            pass
        return False, f"Spawned but did not exit within {timeout_s}s: {e}"
    msg = (out or "") + (err or "")
    return True, msg.strip()


def assign_feedthrough(db: PlaceDB, ftpred_output: str, build_info=None):
    """解析 ftpred 输出，把 feedthrough 写回到 db.nets_list。

    当 build_info 里包含 orig_to_children 时，说明 nets 输入经过了拆分：
    - ftpred 输出的 Net i 对应“子 net i”
    - 原始 net 的 feedthrough = sum(所有子 net 的 feedthrough)
    """
    lines = ftpred_output.strip().split("\n") if ftpred_output else []
    child_vals = {}
    parsed = 0
    for line in lines:
        line = line.strip()
        if not line.startswith("Net"):
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[2] == "feedthrough" and parts[3] == "=":
            try:
                net_id = int(parts[1])
                val = float(parts[4])
            except ValueError:
                continue
            child_vals[net_id] = val
            parsed += 1

    orig_to_children = None
    if isinstance(build_info, dict):
        orig_to_children = build_info.get("orig_to_children")

    if orig_to_children:
        count = 0
        for orig_idx, child_list in orig_to_children.items():
            total = 0.0
            for cid in child_list:
                total += float(child_vals.get(cid, 0.0))
            if 0 <= orig_idx < len(db.nets_list):
                db.nets_list[orig_idx].feedthrough = total
                count += 1
        print(f"Parsed {parsed} child nets; assigned aggregated feedthrough for {count} original nets.")
    else:
        count = 0
        for cid, val in child_vals.items():
            if 0 <= cid < len(db.nets_list):
                db.nets_list[cid].feedthrough = float(val)
                count += 1
        print(f"Parsed and assigned feedthrough for {count} nets.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ftpred", required=True, help="ftpred 可执行文件路径, e.g. ./build/ftpred")
    parser.add_argument("--block", default="block.json", help="block.json 路径")
    parser.add_argument("--pingroup", default="pingroup.json", help="pingroup.json 路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印生成的内容，不调用 ftpred")
    parser.add_argument("--net", type=int, required=True, help="只处理一个 net 的 index (db.nets_list 的下标)")
    args = parser.parse_args()

    db = PlaceDB(args.block, args.pingroup)
    removed = filter_nets_inplace(db, min_pins=2)
    if removed:
        print(f"Filtered out {removed} single-pin nets. Remaining nets: {len(db.nets_list)}")

    modules_text = build_modules_text(db)

    nets_text, build_info = build_nets_text_by_index(db, args.net)

    if args.dry_run:
        print(modules_text)
        print("\n---NETS---\n")
        print(nets_text)
        return

    out = run_ftpred(args.ftpred, modules_text, nets_text)

    # 将结果写回到指定的原始 net：用 one-net view 聚合后拷贝回真实 db
    view = _one_net_view(db, args.net)
    assign_feedthrough(view, out, build_info)
    db.nets_list[args.net].feedthrough = float(getattr(view.nets_list[0], "feedthrough", 0.0))
    print(f"Single-net mode: net {args.net} feedthrough = {db.nets_list[args.net].feedthrough}")


if __name__ == "__main__":
    main()
