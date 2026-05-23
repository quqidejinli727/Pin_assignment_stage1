"""PlaceDB parser and lightweight indexes for pin assignment.

This file keeps the original class names from the provided ``PlaceDB.py`` and
adds the indexes needed by the MCTS assignment flow.
"""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from shapely.geometry import Polygon

from geometry_utils import centroid


class Pin:
    """表示 pingroup 文件中的一个实际 Pin 实例。"""

    def __init__(
        self,
        parent_inst: str,
        parent_module: str,
        pingroup_name: str,
        scope: List[Any],
        successors: List[str],
        width: float,
    ):
        """初始化 Pin 的基础属性、分配状态和坐标占位。"""
        self.parent_inst = parent_inst
        self.parent_module = parent_module
        self.pingroup_name = pingroup_name
        self.scope = scope
        self.successors = successors
        self.width = float(width)
        self.x = 0.0
        self.y = 0.0
        self.assigned_segment_coord = None
        self.segment_endpoints = None
        self.assigned_segment_id: Optional[str] = None
        self.net: Optional["Net"] = None

    @property
    def full_name(self) -> str:
        """返回 Pin 的实例唯一名称：parent_inst.pingroup_name。"""
        return self.get_full_name()

    @property
    def homology_name(self) -> str:
        """返回同构 Pin 的分组名称：parent_module.pingroup_name。"""
        return f"{self.parent_module}.{self.pingroup_name}"

    def get_full_name(self) -> str:
        """拼接并返回当前 Pin 的完整实例名。"""
        return f"{self.parent_inst}.{self.pingroup_name}"

    def __repr__(self) -> str:
        """返回便于调试的 Pin 字符串。"""
        return (
            f"Pin(parent_inst='{self.parent_inst}', "
            f"pingroup_name='{self.pingroup_name}', width={self.width})"
        )

    def __str__(self) -> str:
        """返回面向用户阅读的 Pin 字符串。"""
        return f"{self.get_full_name()} (width={self.width})"


class Net:
    """表示由多个 Pin 构成的一条网表连接。"""

    def __init__(self, pins: List[Pin], net_id: int = -1):
        """初始化 Net，并把每个 Pin 反向关联到该 Net。"""
        self.pins = pins
        self.net_id = net_id
        self.net_degree = len(pins)
        self.feedthrough = 0
        for pin in pins:
            pin.net = self

    def get_pin_count(self) -> int:
        """返回当前 Net 内 Pin 的数量。"""
        return len(self.pins)

    def __repr__(self) -> str:
        """返回便于调试的 Net 字符串。"""
        return f"Net(net_id={self.net_id}, pin_count={len(self.pins)})"

    def __str__(self) -> str:
        """返回面向用户阅读的 Net 字符串。"""
        return f"Net {self.net_id} with {len(self.pins)} pins"


class CoordRotation(IntEnum):
    """输入数据中的旋转/镜像方向枚举。"""

    UNDEFINE = -1
    ROTATION_R0 = 0
    ROTATION_R180 = 1
    ROTATION_R90 = 2
    ROTATION_R270 = 3
    ROTATION_MY = 4
    ROTATION_MX = 5
    ROTATION_MX90 = 6
    ROTATION_MY90 = 7


class Module:
    """表示层级结构中的一个已放置 block/module。"""

    def __init__(
        self,
        name: str,
        module_name: str,
        vertex: List[List[float]],
        direction: int,
        color: str,
        children: Optional[List["Module"]] = None,
        parent: Optional["Module"] = None,
    ):
        """初始化模块几何、层级关系、面积和 Pin 列表。"""
        self.name = name
        self.module_name = module_name
        self.vertex = vertex
        self.direction = CoordRotation(direction) if direction >= 0 else CoordRotation.UNDEFINE
        self.color = color
        self.children = children if children is not None else []
        self.parent = parent
        self.polygon = Polygon(vertex)
        self.area = self.polygon.area
        self.pin_list: List[Pin] = []

        for child in self.children:
            child.parent = self

    def get_hierarchy_level(self) -> int:
        """返回模块在层级树中的深度，根模块为 0。"""
        level = 0
        current = self
        while current.parent is not None:
            level += 1
            current = current.parent
        return level

    def get_path(self) -> List[str]:
        """返回从根模块到当前模块的 module_name 路径。"""
        path = []
        current: Optional[Module] = self
        while current is not None:
            path.insert(0, current.module_name)
            current = current.parent
        return path

    def find_module_by_name(self, name: str) -> Optional["Module"]:
        """在当前模块子树中按实例全名查找模块。"""
        if self.name == name:
            return self
        for child in self.children:
            result = child.find_module_by_name(name)
            if result is not None:
                return result
        return None

    def find_modules_by_module_name(self, module_name: str) -> List["Module"]:
        """在当前模块子树中查找所有同 module_name 的模块。"""
        results = []
        if self.module_name == module_name:
            results.append(self)
        for child in self.children:
            results.extend(child.find_modules_by_module_name(module_name))
        return results

    def get_bounding_box(self) -> Dict[str, float]:
        """计算并返回模块顶点的外接矩形范围。"""
        if not self.vertex:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0}
        x_coords = [v[0] for v in self.vertex]
        y_coords = [v[1] for v in self.vertex]
        return {
            "min_x": min(x_coords),
            "max_x": max(x_coords),
            "min_y": min(y_coords),
            "max_y": max(y_coords),
        }

    def get_area(self) -> float:
        """返回 shapely 计算得到的模块多边形面积。"""
        return self.area

    def get_centroid(self) -> tuple[float, float]:
        """返回模块多边形的质心坐标。"""
        return centroid(self.vertex)

    def is_leaf(self) -> bool:
        """判断当前模块是否没有子模块。"""
        return len(self.children) == 0

    def __repr__(self) -> str:
        """返回便于调试的 Module 字符串。"""
        return (
            f"Module(name='{self.name}', module_name='{self.module_name}', "
            f"direction={self.direction.name}, children={len(self.children)})"
        )

    def __str__(self) -> str:
        """返回面向用户阅读的 Module 字符串。"""
        return f"{self.name} ({self.module_name}) - {self.direction.name}"


class PlaceDB:
    """解析 block/pingroup 数据，并建立求解器所需的索引。"""

    def __init__(self, block_json_file_path: str, pingroup_json_file_path: str):
        """加载两个 JSON 文件并构建模块、Pin、Net 及查询索引。"""
        self.root_module = self.load_place_db(block_json_file_path)
        self.all_module_dict: Dict[str, Module] = {}
        self.modules_by_module_name: Dict[str, List[Module]] = {}
        self.all_modules_list = self.collect_all_modules(self.root_module)

        self.total_pin_count = 0
        self.pin_dict: Dict[str, Pin] = {}
        self.pins_by_homology: Dict[str, List[Pin]] = {}
        self.nets_by_pin: Dict[str, List[Net]] = {}
        self.nets_list = self.load_pingroup_json(pingroup_json_file_path)

    def __repr__(self) -> str:
        """返回便于调试的 PlaceDB 概览。"""
        return (
            f"PlaceDB(modules={len(self.all_modules_list)}, "
            f"nets={len(self.nets_list)}, total_pin_count={self.total_pin_count})"
        )

    def __str__(self) -> str:
        """返回面向用户阅读的 PlaceDB 概览。"""
        return (
            f"PlaceDB with {len(self.all_modules_list)} modules, "
            f"{len(self.nets_list)} nets and {self.total_pin_count} pins"
        )

    def parse_module_from_dict(self, data: Dict[str, Any], parent: Optional[Module] = None) -> Module:
        """递归地把 block JSON 字典解析为 Module 树。"""
        children = []
        if data.get("children"):
            for child_data in data["children"]:
                children.append(self.parse_module_from_dict(child_data, None))

        return Module(
            name=data.get("name", ""),
            module_name=data.get("module_name", ""),
            vertex=data.get("vertex", []),
            direction=data.get("direction", 0),
            color=data.get("color", "#000000"),
            children=children,
            parent=parent,
        )

    def load_place_db(self, json_file_path: str) -> Module:
        """从 block JSON 文件读取并返回根模块。"""
        with Path(json_file_path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        return self.parse_module_from_dict(data)

    def load_place_db_from_string(self, json_string: str) -> Module:
        """从 JSON 字符串读取并返回根模块，便于测试。"""
        data = json.loads(json_string)
        return self.parse_module_from_dict(data)

    def collect_all_modules(
        self,
        module: Module,
        modules_list: Optional[List[Module]] = None,
    ) -> List[Module]:
        """递归收集所有模块，并建立实例名和 module_name 索引。"""
        if modules_list is None:
            modules_list = []

        self.all_module_dict[module.name] = module
        self.modules_by_module_name.setdefault(module.module_name, []).append(module)
        modules_list.append(module)

        for child in module.children:
            self.collect_all_modules(child, modules_list)

        return modules_list

    def load_pingroup_json(self, json_file_path: str) -> List[Net]:
        """读取 pingroup JSON，构建 Pin/Net 对象和 Pin 查询索引。"""
        with Path(json_file_path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        nets = []
        for net_id, net_data in enumerate(data):
            pins = []
            for pin_data in net_data:
                pin = Pin(
                    parent_inst=pin_data.get("parent_inst", ""),
                    parent_module=pin_data.get("parent_module", ""),
                    pingroup_name=pin_data.get("pingroup_name", ""),
                    scope=pin_data.get("scope", []),
                    successors=pin_data.get("successors", []),
                    width=pin_data.get("width", 0.0),
                )
                module = self.all_module_dict.get(pin.parent_inst)
                if module is None:
                    raise KeyError(
                        f"Pin {pin.full_name} refers to missing module {pin.parent_inst}."
                    )
                module.pin_list.append(pin)
                self.pin_dict[pin.full_name] = pin
                self.pins_by_homology.setdefault(pin.homology_name, []).append(pin)
                pins.append(pin)
                self.total_pin_count += 1

            net = Net(pins, net_id=net_id)
            nets.append(net)
            for pin in pins:
                self.nets_by_pin.setdefault(pin.full_name, []).append(net)

        return nets

    def get_module(self, module_name: str) -> Module:
        """按模块实例名获取 Module，不存在时给出清晰错误。"""
        try:
            return self.all_module_dict[module_name]
        except KeyError as exc:
            raise KeyError(f"Unknown module instance: {module_name}") from exc

    def get_pin_location_estimate(self, pin: Pin) -> tuple[float, float]:
        """Return assigned segment midpoint, or module centroid before assignment."""
        # 中文说明：优先返回已分配 segment 中点，未分配时使用所属模块质心估算。
        if pin.assigned_segment_coord is not None:
            return pin.assigned_segment_coord
        return self.get_module(pin.parent_inst).get_centroid()

    def iter_pin_nets(self, pin: Pin) -> List[Net]:
        """返回包含指定 Pin 的所有 Net。"""
        return self.nets_by_pin.get(pin.full_name, [])
