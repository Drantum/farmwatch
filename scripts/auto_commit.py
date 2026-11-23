#!/usr/bin/env python3
"""
Automatisches Git-Commit Skript.
Funktioniert sowohl wenn 'git' im System installiert ist, als auch rein über Python,
sodass nach jeder Änderung garantiert ein valider Git-Commit im Repository angelegt wird.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path


def commit_with_git_cli(message: str) -> bool:
    """Versucht einen Commit über die reguläre Git-CLI zu erstellen."""
    git_bin = shutil.which("git")
    if not git_bin:
        return False
    try:
        subprocess.run([git_bin, "add", "-A"], check=True, capture_output=True)
        res = subprocess.run([git_bin, "commit", "-m", message], capture_output=True, text=True)
        if res.returncode == 0 or "nothing to commit" in res.stdout:
            print(f"[git] Commit erfolgreich: {message.strip()}")
            return True
    except Exception:
        pass
    return False


def commit_with_python(message: str) -> str:
    """Erstellt einen standardkonformen Git-Commit rein über Python."""
    repo_dir = Path(".").resolve()
    git_dir = repo_dir / ".git"

    if not git_dir.exists():
        raise RuntimeError("Kein .git-Verzeichnis gefunden.")

    def write_object(obj_type: str, data: bytes) -> str:
        header = f"{obj_type} {len(data)}\x00".encode()
        full = header + data
        sha = hashlib.sha1(full).hexdigest()
        obj_dir = git_dir / "objects" / sha[:2]
        obj_dir.mkdir(parents=True, exist_ok=True)
        obj_path = obj_dir / sha[2:]
        if not obj_path.exists():
            with open(obj_path, "wb") as f:
                f.write(zlib.compress(full))
        return sha

    ignored_names = {".git", ".env", "data", "__pycache__", ".pytest_cache", "venv", ".venv"}

    def build_tree(current_dir: Path) -> str:
        entries = []
        items = sorted(os.listdir(current_dir))
        for item in items:
            if item in ignored_names or item.endswith((".pyc", ".db")):
                continue
            p = current_dir / item
            if p.is_dir():
                sub_sha = build_tree(p)
                if sub_sha:
                    entries.append((item + "/", b"40000 " + item.encode("utf-8") + b"\x00" + bytes.fromhex(sub_sha)))
            elif p.is_file():
                with open(p, "rb") as f:
                    content = f.read()
                blob_sha = write_object("blob", content)
                entries.append((item, b"100644 " + item.encode("utf-8") + b"\x00" + bytes.fromhex(blob_sha)))
        entries.sort(key=lambda x: x[0])
        tree_data = b"".join(e[1] for e in entries)
        return write_object("tree", tree_data)

    current_tree_sha = build_tree(repo_dir)

    # Aktuellen Parent-Commit aus refs/heads/main holen
    main_ref = git_dir / "refs" / "heads" / "main"
    parent_sha = main_ref.read_text(encoding="utf-8").strip() if main_ref.exists() else None

    # Prüfen, ob sich der Tree geändert hat
    if parent_sha:
        obj_file = git_dir / "objects" / parent_sha[:2] / parent_sha[2:]
        if obj_file.exists():
            raw = zlib.decompress(obj_file.read_bytes())
            _, body = raw.split(b"\x00", 1)
            lines = body.split(b"\n")
            tree_lines = [l for l in lines if l.startswith(b"tree ")]
            if tree_lines:
                parent_tree_sha = tree_lines[0].split()[1].decode()
                if parent_tree_sha == current_tree_sha:
                    print("[python-git] Keine Änderungen vorhanden. Kein neuer Commit nötig.")
                    return parent_sha

    now = int(time.time())
    tz_offset = "+0100"
    author_str = f"Frederik <frederik@kergers.de> {now} {tz_offset}"
    parent_line = f"parent {parent_sha}\n" if parent_sha else ""
    commit_content = f"tree {current_tree_sha}\n{parent_line}author {author_str}\ncommitter {author_str}\n\n{message.strip()}\n".encode("utf-8")

    commit_sha = write_object("commit", commit_content)
    with open(main_ref, "w", encoding="utf-8") as f:
        f.write(f"{commit_sha}\n")

    print(f"[python-git] Neuer Commit erstellt: {commit_sha[:7]} - {message.strip()}")
    return commit_sha


def auto_commit(message: str = "Update code") -> None:
    if not commit_with_git_cli(message):
        commit_with_python(message)


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "Auto-commit nach Änderung"
    auto_commit(msg)
