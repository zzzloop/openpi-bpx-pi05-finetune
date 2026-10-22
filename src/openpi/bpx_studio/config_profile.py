"""Preview literal config bindings without importing the training runtime."""

from __future__ import annotations

import ast
import dataclasses
import pathlib


@dataclasses.dataclass(frozen=True)
class AssetBinding:
    assets_dir: str | None
    asset_id: str | None


@dataclasses.dataclass(frozen=True)
class DataBinding:
    repo_id: str
    assets: AssetBinding


@dataclasses.dataclass(frozen=True)
class ConfigBinding:
    name: str
    assets_base_dir: str
    data: DataBinding
    validation_manifest: str | None = None


def _keywords(node: ast.AST) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Call) or any(item.arg is None for item in node.keywords):
        raise ValueError("配置预览要求显式的 data/assets 字段，不执行动态 Python 表达式")
    return {item.arg: item.value for item in node.keywords}


def _literal(node: ast.AST, label: str) -> str | None:
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
        return node.value
    raise ValueError(f"{label} 不是字符串字面量，无法安全预览；请将数据路径显式写入 Config")


def read_config_binding(project_root: pathlib.Path, config_name: str) -> ConfigBinding:
    """Read actual source, not a hardcoded config-name/path table. Never eval/import it."""
    source = project_root / "src/openpi/training/config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("TrainConfig", "AssetsConfig"):
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and field.value is not None:
                    defaults[node.name, field.target.id] = field.value
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "TrainConfig":
            continue
        fields = {item.arg: item.value for item in node.keywords if item.arg is not None}
        name = fields.get("name")
        if not isinstance(name, ast.Constant) or name.value != config_name:
            continue
        if "data" not in fields:
            raise ValueError(f"Config '{config_name}' 没有可预览的 LeRobot data 定义")
        data = _keywords(fields["data"])
        repo = _literal(data.get("repo_id", ast.Constant(None)), "data.repo_id")
        if not repo or repo == "fake":
            raise ValueError(f"Config '{config_name}' 没有可用的 LeRobot repo_id")
        assets = _keywords(data["assets"]) if "assets" in data else {}
        base = _literal(fields.get("assets_base_dir", defaults.get(
            ("TrainConfig", "assets_base_dir"), ast.Constant("./assets"))), "assets_base_dir")
        directory = _literal(assets.get("assets_dir", defaults.get(
            ("AssetsConfig", "assets_dir"), ast.Constant(None))), "assets.assets_dir")
        asset_id = _literal(assets.get("asset_id", defaults.get(
            ("AssetsConfig", "asset_id"), ast.Constant(None))), "assets.asset_id")
        return ConfigBinding(config_name, base or "./assets", DataBinding(repo, AssetBinding(directory, asset_id)))
    raise ValueError(f"找不到可静态预览的 Config '{config_name}'；请输入已注册的配置名，不是数据目录")
