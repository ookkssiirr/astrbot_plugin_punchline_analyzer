#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstrBot 笑点解析插件 独立备份与回滚工具 (Reviewable Standalone Tool)

核心设计准则：
1. 备份必须放置于独立目录（如 /opt/astrbot/data/plugin_backups/），严禁置于插件扫描目录。
2. 每次备份生成带时间戳与唯一 UUID 的独立目录与 manifest.json。
3. 严格记录并区分“首次安装（原目录/配置不存在）”与“升级覆盖（原目录/配置已存在）”两种状态。
4. 回滚时若原状态为不存在，则彻底清除新写入的文件，恢复不存在状态；若原状态为存在，则100%校验哈希并恢复旧版本。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict


def compute_file_sha256(filepath: str | Path) -> str:
    """计算单个文件的 SHA-256 哈希值"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_manifest(dirpath: str | Path) -> Dict[str, str]:
    """遍历目录计算所有相对路径文件的哈希表"""
    base = Path(dirpath)
    manifest = {}
    if not base.exists() or not base.is_dir():
        return manifest
    for root, _, files in os.walk(base):
        for f in files:
            full = Path(root) / f
            rel = full.relative_to(base).as_posix()
            manifest[rel] = compute_file_sha256(full)
    return manifest


def create_backup(
    plugin_dir: str | Path,
    config_file: str | Path,
    backup_base_dir: str | Path,
) -> Path:
    """
    执行独立备份操作并生成规范的 manifest.json
    """
    p_dir = Path(plugin_dir).resolve()
    c_file = Path(config_file).resolve()
    b_base = Path(backup_base_dir).resolve()

    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_id = f"backup_{now_str}_{uuid.uuid4().hex[:8]}"
    dest_dir = b_base / backup_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "version": "1.0",
        "backup_id": backup_id,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plugin_dir_target": str(p_dir),
        "plugin_dir_existed": p_dir.exists(),
        "plugin_dir_backed_up": False,
        "plugin_dir_hashes": {},
        "config_file_target": str(c_file),
        "config_file_existed": c_file.exists(),
        "config_file_backed_up": False,
        "config_file_hash": None,
    }

    # 1. 备份插件目录
    if p_dir.exists() and p_dir.is_dir():
        saved_p_dir = dest_dir / "plugin_content"
        shutil.copytree(p_dir, saved_p_dir)
        manifest["plugin_dir_backed_up"] = True
        manifest["plugin_dir_hashes"] = compute_directory_manifest(saved_p_dir)

    # 2. 备份配置文件
    if c_file.exists() and c_file.is_file():
        saved_c_dir = dest_dir / "config_content"
        saved_c_dir.mkdir(parents=True, exist_ok=True)
        saved_c_file = saved_c_dir / c_file.name
        shutil.copy2(c_file, saved_c_file)
        manifest["config_file_backed_up"] = True
        manifest["config_file_hash"] = compute_file_sha256(saved_c_file)

    # 3. 写入清单
    manifest_path = dest_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return dest_dir


def execute_rollback(backup_dir: str | Path) -> bool:
    """
    根据 manifest.json 严格执行回滚恢复
    """
    b_dir = Path(backup_dir).resolve()
    manifest_path = b_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"备份清单不存在: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    p_target = Path(manifest["plugin_dir_target"])
    c_target = Path(manifest["config_file_target"])

    # 1. 恢复插件目录状态
    if manifest["plugin_dir_existed"]:
        saved_p = b_dir / "plugin_content"
        if not saved_p.exists():
            raise FileNotFoundError(f"备份中缺少插件目录内容: {saved_p}")
        if p_target.exists():
            shutil.rmtree(p_target)
        shutil.copytree(saved_p, p_target)
        # 校验哈希
        current_hashes = compute_directory_manifest(p_target)
        if current_hashes != manifest.get("plugin_dir_hashes", {}):
            raise ValueError("插件目录回滚后文件哈希校验失败！")
    else:
        # 原始状态为不存在：彻底清除新装文件
        if p_target.exists():
            shutil.rmtree(p_target)

    # 2. 恢复配置文件状态
    if manifest["config_file_existed"]:
        saved_c = b_dir / "config_content" / c_target.name
        if not saved_c.exists():
            raise FileNotFoundError(f"备份中缺少配置文件内容: {saved_c}")
        c_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_c, c_target)
        if compute_file_sha256(c_target) != manifest.get("config_file_hash"):
            raise ValueError("配置文件回滚后文件哈希校验失败！")
    else:
        # 原始状态为不存在：删除生成的新配置
        if c_target.exists():
            c_target.unlink()

    return True


def run_scenario_verifications(base_work_dir: str | Path | None = None) -> bool:
    """
    自动化验证两种典型运维场景：
    场景 1：全新安装（部署前原目录与配置文件均不存在）-> 回滚后100%还原为不存在
    场景 2：版本升级（部署前原目录与配置文件均存在旧版）-> 回滚后100%还原为旧版内容并校验哈希
    """
    work_dir = Path(base_work_dir) if base_work_dir else Path(tempfile.mkdtemp(prefix="astrbot_verify_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[验证启动] 工作隔离根目录: {work_dir}")

    try:
        # ==========================================
        # 场景 1 验证：全新安装
        # ==========================================
        print("\n--- 开始验证场景 1：全新安装（原目录与配置均不存在） ---")
        sc1_plugins = work_dir / "sc1_plugins"
        sc1_configs = work_dir / "sc1_configs"
        sc1_backups = work_dir / "sc1_backups"
        sc1_plugins.mkdir(parents=True, exist_ok=True)
        sc1_configs.mkdir(parents=True, exist_ok=True)
        sc1_backups.mkdir(parents=True, exist_ok=True)

        target_plugin = sc1_plugins / "astrbot_plugin_punchline_analyzer"
        target_config = sc1_configs / "astrbot_plugin_punchline_analyzer_config.json"

        # 确保初始不存在
        assert not target_plugin.exists(), "场景1初始插件目录必须不存在"
        assert not target_config.exists(), "场景1初始配置文件必须不存在"

        # 1. 部署前备份
        b1_dir = create_backup(target_plugin, target_config, sc1_backups)
        with open(b1_dir / "manifest.json", "r", encoding="utf-8") as f:
            m1 = json.load(f)
        assert m1["plugin_dir_existed"] is False, "场景1清单必须记录 plugin_dir_existed=False"
        assert m1["config_file_existed"] is False, "场景1清单必须记录 config_file_existed=False"
        print(f"[场景 1 步骤 1] 备份成功，清单正确识别目标不存在: {b1_dir.name}")

        # 2. 模拟部署（写入新版本文件）
        target_plugin.mkdir(parents=True, exist_ok=True)
        (target_plugin / "main.py").write_text("# new deployed code v1.0", encoding="utf-8")
        target_config.write_text('{"deployed": true, "version": "1.0"}', encoding="utf-8")
        assert target_plugin.exists() and (target_plugin / "main.py").exists()
        assert target_config.exists()
        print("[场景 1 步骤 2] 模拟新插件与新配置部署成功")

        # 3. 执行回滚
        ok1 = execute_rollback(b1_dir)
        assert ok1 is True
        assert not target_plugin.exists(), "场景1回滚后插件目录必须彻底被清除！"
        assert not target_config.exists(), "场景1回滚后配置文件必须彻底被清除！"
        print("[场景 1 步骤 3] 回滚成功，目标目录与配置已 100% 还原为不存在状态！")

        # ==========================================
        # 场景 2 验证：已有版本升级覆盖
        # ==========================================
        print("\n--- 开始验证场景 2：已有版本升级覆盖（原目录与配置均存在） ---")
        sc2_plugins = work_dir / "sc2_plugins"
        sc2_configs = work_dir / "sc2_configs"
        sc2_backups = work_dir / "sc2_backups"
        sc2_plugins.mkdir(parents=True, exist_ok=True)
        sc2_configs.mkdir(parents=True, exist_ok=True)
        sc2_backups.mkdir(parents=True, exist_ok=True)

        target_plugin2 = sc2_plugins / "astrbot_plugin_punchline_analyzer"
        target_config2 = sc2_configs / "astrbot_plugin_punchline_analyzer_config.json"

        # 初始旧版本
        target_plugin2.mkdir(parents=True, exist_ok=True)
        old_file_content = "# legacy stable code v0.9\nprint('old version')"
        (target_plugin2 / "legacy.py").write_text(old_file_content, encoding="utf-8")
        old_hash = compute_file_sha256(target_plugin2 / "legacy.py")
        old_cfg_content = '{"mode": "legacy", "threshold": 10}'
        target_config2.write_text(old_cfg_content, encoding="utf-8")
        old_cfg_hash = compute_file_sha256(target_config2)

        # 1. 部署前备份
        b2_dir = create_backup(target_plugin2, target_config2, sc2_backups)
        with open(b2_dir / "manifest.json", "r", encoding="utf-8") as f:
            m2 = json.load(f)
        assert m2["plugin_dir_existed"] is True, "场景2清单必须记录 plugin_dir_existed=True"
        assert m2["config_file_existed"] is True, "场景2清单必须记录 config_file_existed=True"
        assert m2["plugin_dir_hashes"]["legacy.py"] == old_hash
        assert m2["config_file_hash"] == old_cfg_hash
        print(f"[场景 2 步骤 1] 备份成功，清单完整记录旧版存在状态与文件哈希: {b2_dir.name}")

        # 2. 模拟升级（写入新代码与修改配置，删除旧文件）
        (target_plugin2 / "legacy.py").unlink()
        (target_plugin2 / "main.py").write_text("# newly upgraded code v2.0", encoding="utf-8")
        target_config2.write_text('{"mode": "v2.0", "threshold": 50}', encoding="utf-8")
        assert not (target_plugin2 / "legacy.py").exists()
        assert (target_plugin2 / "main.py").exists()
        print("[场景 2 步骤 2] 模拟新版覆盖升级成功（旧文件被替换，新配置已生效）")

        # 3. 执行回滚
        ok2 = execute_rollback(b2_dir)
        assert ok2 is True
        assert target_plugin2.exists()
        assert (target_plugin2 / "legacy.py").exists(), "场景2回滚后旧文件必须被恢复！"
        assert not (target_plugin2 / "main.py").exists(), "场景2回滚后新文件必须被清除！"
        assert compute_file_sha256(target_plugin2 / "legacy.py") == old_hash, "场景2回滚后文件哈希必须与旧版完全一致！"
        assert compute_file_sha256(target_config2) == old_cfg_hash, "场景2回滚后配置哈希必须与旧版完全一致！"
        print("[场景 2 步骤 3] 回滚成功，旧版目录结构、文件哈希与配置文件 100% 完整复原！")

        print("\n[全部通过] 两种场景的备份、识别、回滚与哈希校验均已成功通过验证！")
        return True

    finally:
        # 清理测试隔离工作目录
        if base_work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="AstrBot Plugin Backup & Rollback Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # backup
    p_backup = subparsers.add_parser("backup", help="Create an isolated backup with manifest")
    p_backup.add_argument("--plugin-dir", required=True, help="Target plugin directory")
    p_backup.add_argument("--config-file", required=True, help="Target config file")
    p_backup.add_argument("--backup-base-dir", required=True, help="Base directory to store backups")

    # rollback
    p_rollback = subparsers.add_parser("rollback", help="Execute rollback using a specific backup directory")
    p_rollback.add_argument("--backup-dir", required=True, help="Directory containing the backup and manifest.json")

    # verify-scenarios
    p_verify = subparsers.add_parser("verify-scenarios", help="Verify both fresh-install and upgrade scenarios")
    p_verify.add_argument("--temp-dir", help="Optional specific directory to run tests in")

    args = parser.parse_args()

    if args.command == "backup":
        b_dir = create_backup(args.plugin_dir, args.config_file, args.backup_base_dir)
        print(f"BACKUP_SUCCESS: {b_dir}")
        sys.exit(0)
    elif args.command == "rollback":
        ok = execute_rollback(args.backup_dir)
        if ok:
            print(f"ROLLBACK_SUCCESS: {args.backup_dir}")
            sys.exit(0)
        else:
            print(f"ROLLBACK_FAILED: {args.backup_dir}")
            sys.exit(1)
    elif args.command == "verify-scenarios":
        ok = run_scenario_verifications(args.temp_dir)
        if ok:
            print("VERIFY_SCENARIOS_SUCCESS")
            sys.exit(0)
        else:
            print("VERIFY_SCENARIOS_FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
