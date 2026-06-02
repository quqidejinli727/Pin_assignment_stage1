"""text.py

一个“参考模板”脚本：演示如何用 `ftpred_loader` 跑指定的 net，并把 feedthrough 回写到 PlaceDB。

这份脚本的目标：
    - 尽量短、好抄；下一位用户想用 `ftpred_loader` 时可直接照着写
    - 走常驻会话（`FtpredBinSession`，二进制帧协议），避免反复启动 ftpred.exe

典型流程：
    1) 读取 PlaceDB(block.json + pingroup.json)
    2) 给每个 pin 设置坐标（这里示例用“模块质心”）
    3) build Modules（一次）
    4) session 内循环跑 nets：sess.run_one_net(db, db.nets_list[i])
"""

import os
import sys
import time
from typing import Tuple

from PlaceDB import PlaceDB
import ftpred_loader


# ===== 用户在这里改配置即可（无需传命令行参数） =====

# 要跑哪些 net（示例：net0 + net1）
NET_INDICES = [0, 1, 3]

# 是否过滤掉单引脚网（建议 True；会改变 net 索引的含义，需与你的目标一致）
FILTER_SINGLE_PIN_NETS = True

# ftpred 可执行文件路径（默认指向 build/ 下生成物）
def _default_ftpred_path(base_dir: str) -> str:
    return os.path.join(base_dir, "build", "ftpred.exe" if os.name == "nt" else "ftpred")


def compute_module_centroid(mod) -> Tuple[float, float]:
    """将所有模块的质心计算出来，作为 pin 的坐标使用"""
    vs = getattr(mod, 'vertex', None)
    xs = [float(v[0]) for v in vs]
    ys = [float(v[1]) for v in vs]
    if len(xs) > 0:
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))
    return 0.0, 0.0


def main():
    t0 = time.perf_counter()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ftpred_path = os.path.normpath(_default_ftpred_path(base_dir))
    block_path = os.path.normpath(os.path.join(base_dir, "block.json"))
    pingroup_path = os.path.normpath(os.path.join(base_dir, "pingroup.json"))

    # 1) PlaceDB
    if not os.path.exists(block_path) or not os.path.exists(pingroup_path):
        print("Error: block.json or pingroup.json not found.", file=sys.stderr)
        sys.exit(1)
    db = PlaceDB(block_path, pingroup_path)

    # 2) 可选：过滤单引脚网
    if FILTER_SINGLE_PIN_NETS:
        removed = ftpred_loader.filter_nets_inplace(db, min_pins=2)
        if removed:
            print(f"Filtered out {removed} single-pin nets. Remaining nets: {len(db.nets_list)}")

    # 3) 给 pin 设置坐标（示例：模块质心）
    for net in db.nets_list:
        for pin in getattr(net, "pins", []):
            parent_name = getattr(pin, "parent_inst", None)
            mod = db.all_module_dict.get(parent_name) if parent_name else None
            if mod and hasattr(mod, "vertex"):
                cx, cy = compute_module_centroid(mod)
                pin.x = cx
                pin.y = cy

    # 4) 用户指定要跑哪些 net（脚本内配置）
    net_indices = list(NET_INDICES)
    for ni in net_indices:
        if ni < 0 or ni >= len(db.nets_list):
            print(f"Error: NET_INDICES has {ni} out of range (0..{len(db.nets_list)-1})", file=sys.stderr)
            sys.exit(1)

    # 5) Modules（一次）+ 常驻会话（多次 net）
    Modules = ftpred_loader.build_modules_text(db)
    with ftpred_loader.FtpredBinSession(ftpred_path, Modules) as sess:
        for ni in net_indices:
            net_obj = db.nets_list[ni]
            t1 = time.perf_counter()
            ft = sess.run_one_net(db, net_obj)
            t2 = time.perf_counter()
            print(f"Net {ni} feedthrough={ft} dt={t2 - t1:.4f}s")

    print(f"Total elapsed: {time.perf_counter() - t0:.4f}s")


if __name__ == "__main__":
    main()
