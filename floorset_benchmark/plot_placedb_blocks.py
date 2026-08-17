"""Render hierarchical FloorSet cases loaded through :class:`PlaceDB`.

The renderer intentionally consumes the parsed ``Module`` objects exposed by
``PlaceDB`` instead of drawing directly from the JSON dictionaries.  This
keeps the visual check on the same block-loading path used by assignment and
also makes malformed module references fail before an image is written.

The project runtime already provides Pillow, so this module does not require
matplotlib or any additional dependency.  Run from the repository root with::

    python floorset_benchmark/plot_placedb_blocks.py

By default the four calibrated cases are rendered to a ``plots`` directory
under each case.  A small JSON report is written beside each image with the
PlaceDB counts and geometry checks used while rendering.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# When invoked as ``python floorset_benchmark/...py``, Python puts the script
# directory first on sys.path.  Add the repository root so the real PlaceDB
# module is imported, not a test stub or an unrelated installed package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageColor, ImageDraw, ImageFont  # noqa: E402

from PlaceDB import Module, PlaceDB  # noqa: E402


DEFAULT_CASES = ("20k_r60", "20k_r95", "30k_r60", "30k_r95")
DEFAULT_BULK_DIR = REPO_ROOT / "outputs" / "floorset_mcts_tree_depth_calibrated" / "bulk"


@dataclass(frozen=True)
class PlotConfig:
    """Pixel/layout settings for a deterministic render."""

    width: int = 2200
    height: int = 1400
    left_margin: int = 80
    top_margin: int = 100
    bottom_margin: int = 80
    legend_width: int = 410

    @property
    def plot_right(self) -> int:
        return self.width - self.legend_width

    @property
    def plot_bottom(self) -> int:
        return self.height - self.bottom_margin


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load a common Windows font, with a Pillow fallback for CI."""

    candidates = (
        Path("C:/Windows/Fonts/segoeui.ttf") if not bold else Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arial.ttf") if not bold else Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _type_color(module_name: str) -> tuple[int, int, int]:
    """Return a deterministic pastel RGB color for one module type."""

    digest = hashlib.sha256(module_name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65536.0
    saturation = 0.58 + digest[2] / 255.0 * 0.18
    lightness = 0.66 + digest[3] / 255.0 * 0.10
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return tuple(round(channel * 255) for channel in rgb)


def _darken(color: tuple[int, int, int], factor: float = 0.62) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)


def _module_level(module: Module) -> int:
    return module.get_hierarchy_level()


def _descendants(module: Module) -> Iterable[Module]:
    for child in module.children:
        yield child
        yield from _descendants(child)


def _polygon_pixels(
    module: Module,
    *,
    min_x: float,
    min_y: float,
    scale: float,
    left: int,
    bottom: int,
) -> list[tuple[int, int]]:
    """Convert a PlaceDB module polygon to equal-aspect image coordinates."""

    return [
        (
            round(left + (float(vertex[0]) - min_x) * scale),
            round(bottom - (float(vertex[1]) - min_y) * scale),
        )
        for vertex in module.vertex
    ]


def _bbox_from_module(module: Module) -> tuple[float, float, float, float]:
    bbox = module.get_bounding_box()
    return bbox["min_x"], bbox["max_x"], bbox["min_y"], bbox["max_y"]


def _reuse_label(case_name: str) -> str:
    token = case_name.lower()
    if "r95" in token:
        return "R95"
    if "r60" in token:
        return "R60"
    return "reuse unknown"


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    *,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (30, 30, 30),
    line_gap: int = 7,
) -> int:
    """Draw lines and return the y coordinate after the block."""

    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def render_case(
    case_dir: Path,
    *,
    output_path: Path | None = None,
    config: PlotConfig = PlotConfig(),
) -> dict[str, object]:
    """Load one case through PlaceDB and render its hierarchy to PNG.

    Returns the report that is also written beside the image.  The function
    checks the expected 1 + 12 + 228 hierarchy before drawing; a mismatch is
    an error instead of a silently misleading plot.
    """

    block_path = case_dir / "block.json"
    pingroup_path = case_dir / "pingroup.json"
    if not block_path.is_file():
        raise FileNotFoundError(block_path)
    if not pingroup_path.is_file():
        raise FileNotFoundError(pingroup_path)

    placedb = PlaceDB(str(block_path), str(pingroup_path))
    modules = list(placedb.all_modules_list)
    root = placedb.root_module
    parents = [module for module in modules if _module_level(module) == 1]
    leaves = [module for module in modules if module.is_leaf() and _module_level(module) == 2]
    nonroot_count = len(modules) - 1
    level_counts: dict[str, int] = {}
    for module in modules:
        key = str(_module_level(module))
        level_counts[key] = level_counts.get(key, 0) + 1

    expected = {"root": 1, "parents": 12, "leaves": 228, "nonroot": 240}
    actual = {
        "root": 1 if _module_level(root) == 0 else 0,
        "parents": len(parents),
        "leaves": len(leaves),
        "nonroot": nonroot_count,
    }
    if actual != expected:
        raise ValueError(f"{case_dir.name}: expected hierarchy {expected}, got {actual}")

    min_x, max_x, min_y, max_y = _bbox_from_module(root)
    geometry_width = max(max_x - min_x, 1.0)
    geometry_height = max(max_y - min_y, 1.0)
    plot_width = config.plot_right - config.left_margin
    plot_height = config.plot_bottom - config.top_margin
    scale = min(plot_width / geometry_width, plot_height / geometry_height)
    draw_left = round(config.left_margin + (plot_width - geometry_width * scale) / 2)
    draw_bottom = round(config.plot_bottom - (plot_height - geometry_height * scale) / 2)

    image = Image.new("RGB", (config.width, config.height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    subtitle_font = _font(22, bold=True)
    body_font = _font(19)
    small_font = _font(16)

    # Root is a light canvas.  The same scale is used for x/y so rectangles
    # and rotations retain their PlaceDB geometry proportions.
    root_polygon = _polygon_pixels(
        root,
        min_x=min_x,
        min_y=min_y,
        scale=scale,
        left=draw_left,
        bottom=draw_bottom,
    )
    if len(root_polygon) >= 3:
        draw.polygon(root_polygon, fill=(247, 248, 250), outline=(25, 25, 30))

    parent_palette = (
        (76, 114, 176),
        (221, 132, 74),
        (92, 153, 108),
        (166, 92, 154),
        (198, 147, 54),
        (67, 156, 171),
    )
    # Parent fills are intentionally light; leaf fills drawn afterwards make
    # the 12 parent regions and the 228 children visible simultaneously.
    for index, parent in enumerate(sorted(parents, key=lambda item: item.name)):
        polygon = _polygon_pixels(
            parent,
            min_x=min_x,
            min_y=min_y,
            scale=scale,
            left=draw_left,
            bottom=draw_bottom,
        )
        if len(polygon) < 3:
            continue
        color = parent_palette[index % len(parent_palette)]
        draw.polygon(polygon, fill=tuple(round((channel + 255) / 2) for channel in color))
        draw.line(polygon + [polygon[0]], fill=color, width=4, joint="curve")

    type_names = sorted({str(leaf.module_name) for leaf in leaves})
    for leaf in sorted(leaves, key=lambda item: item.name):
        polygon = _polygon_pixels(
            leaf,
            min_x=min_x,
            min_y=min_y,
            scale=scale,
            left=draw_left,
            bottom=draw_bottom,
        )
        if len(polygon) < 3:
            continue
        color = _type_color(str(leaf.module_name))
        outline = _darken(color)
        draw.polygon(polygon, fill=color)
        draw.line(polygon + [polygon[0]], fill=outline, width=2, joint="curve")

    # Redraw parent boundaries and labels over child fills.
    for index, parent in enumerate(sorted(parents, key=lambda item: item.name)):
        polygon = _polygon_pixels(
            parent,
            min_x=min_x,
            min_y=min_y,
            scale=scale,
            left=draw_left,
            bottom=draw_bottom,
        )
        if len(polygon) < 3:
            continue
        color = parent_palette[index % len(parent_palette)]
        draw.line(polygon + [polygon[0]], fill=color, width=5, joint="curve")
        cx, cy = parent.get_centroid()
        px = round(draw_left + (cx - min_x) * scale)
        py = round(draw_bottom - (cy - min_y) * scale)
        label = parent.name.rsplit(".", 1)[-1]
        label_box = draw.textbbox((0, 0), label, font=subtitle_font)
        draw.rounded_rectangle(
            (px - 7, py - 4, px + label_box[2] + 9, py + label_box[3] - label_box[1] + 4),
            radius=4,
            fill=(255, 255, 255),
            outline=color,
            width=2,
        )
        draw.text((px + 1, py), label, font=subtitle_font, fill=(35, 35, 35))

    # Root outline and label are last so the top-level boundary is clear.
    if len(root_polygon) >= 3:
        draw.line(root_polygon + [root_polygon[0]], fill=(20, 20, 25), width=6, joint="curve")
        draw.text((draw_left + 8, draw_top := max(12, config.top_margin - 56)), "TOP", font=subtitle_font, fill=(20, 20, 25))

    # Right-hand annotation panel: counts come from PlaceDB objects above.
    panel_x = config.plot_right + 30
    draw.line((config.plot_right, 40, config.plot_right, config.height - 40), fill=(210, 210, 215), width=2)
    case_name = case_dir.name
    draw.text((panel_x, 42), f"FloorSet {case_name}", font=title_font, fill=(20, 20, 25))
    y = 92
    y = _draw_text_block(
        draw,
        [
            f"reuse target: {_reuse_label(case_name)}",
            f"PlaceDB modules: {len(modules)}",
            f"non-root instances: {nonroot_count}",
            f"parents / leaves: {len(parents)} / {len(leaves)}",
            f"module types: {len(placedb.modules_by_module_name)}",
            f"nets / pins: {len(placedb.nets_list)} / {placedb.total_pin_count}",
            f"equal-aspect scale: {scale:.3f} px/unit",
        ],
        x=panel_x,
        y=y,
        font=body_font,
        line_gap=9,
    )
    y += 16
    draw.text((panel_x, y), "Legend", font=subtitle_font, fill=(20, 20, 25))
    y += 38
    legend_items = [
        ((247, 248, 250), (20, 20, 25), "TOP/root boundary"),
        ((210, 220, 235), parent_palette[0], "parent regions P00–P11"),
        (_type_color(type_names[0]) if type_names else (180, 180, 180), (60, 60, 60), "leaf fill = module type"),
    ]
    for fill, outline, label in legend_items:
        draw.rectangle((panel_x, y + 2, panel_x + 27, y + 28), fill=fill, outline=outline, width=2)
        draw.text((panel_x + 40, y), label, font=small_font, fill=(45, 45, 45))
        y += 39
    y += 12
    draw.text((panel_x, y), "Sample leaf module types", font=subtitle_font, fill=(20, 20, 25))
    y += 35
    for type_name in type_names[:8]:
        color = _type_color(type_name)
        draw.rectangle((panel_x, y + 2, panel_x + 23, y + 24), fill=color, outline=_darken(color), width=1)
        display_name = type_name if len(type_name) <= 24 else f"{type_name[:21]}..."
        draw.text((panel_x + 34, y), display_name, font=small_font, fill=(45, 45, 45))
        y += 30
    if len(type_names) > 8:
        draw.text((panel_x, y + 4), f"... {len(type_names) - 8} more types", font=small_font, fill=(90, 90, 90))

    if output_path is None:
        output_path = case_dir / "plots" / "block_layout.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)

    report: dict[str, object] = {
        "case": case_name,
        "source": {
            "block_json": str(block_path),
            "pingroup_json": str(pingroup_path),
            "loader": "PlaceDB(block_json_file_path, pingroup_json_file_path)",
        },
        "placedb": {
            "module_count": len(modules),
            "nonroot_instance_count": nonroot_count,
            "root_count": actual["root"],
            "parent_count": len(parents),
            "leaf_count": len(leaves),
            "module_type_count": len(placedb.modules_by_module_name),
            "net_count": len(placedb.nets_list),
            "pin_count": placedb.total_pin_count,
            "levels": level_counts,
        },
        "geometry": {
            "root_bounds": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
            "equal_aspect_scale_px_per_unit": scale,
        },
        "output": {
            "path": str(output_path),
            "width": config.width,
            "height": config.height,
            "bytes": output_path.stat().st_size,
        },
    }
    report_path = output_path.with_name("plot_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-dir", type=Path, default=DEFAULT_BULK_DIR)
    parser.add_argument("--case", dest="cases", action="append", help="case name; repeat for multiple cases")
    parser.add_argument("--width", type=int, default=PlotConfig.width)
    parser.add_argument("--height", type=int, default=PlotConfig.height)
    args = parser.parse_args(argv)

    case_names = tuple(args.cases) if args.cases else DEFAULT_CASES
    config = PlotConfig(width=args.width, height=args.height)
    reports = []
    for case_name in case_names:
        report = render_case(args.bulk_dir / case_name, config=config)
        reports.append(report)
        output = report["output"]
        assert isinstance(output, dict)
        print(
            f"{case_name}: PlaceDB modules={report['placedb']['module_count']} "
            f"nonroot={report['placedb']['nonroot_instance_count']} "
            f"parents={report['placedb']['parent_count']} leaves={report['placedb']['leaf_count']} "
            f"-> {output['path']} ({output['bytes']} bytes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
