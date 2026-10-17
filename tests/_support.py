"""Shared test helpers: MASY src on sys.path and synthetic temp Git repos.

Synthetic repositories are created only inside temporary directories; no
test initializes Git in MASY or an upstream tree, imports real upstream
code, invokes transport or reads datasets.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

IMPLEMENTATION = Path(__file__).resolve().parents[1]
SRC = IMPLEMENTATION / "src"
LAUNCHER = IMPLEMENTATION / "tools" / "masy_preflight.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GIT = shutil.which("git")
requires_git = unittest.skipUnless(GIT, "git executable not available")


def git(repo: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    completed = subprocess.run(
        ["git", "-c", "user.name=MASY test", "-c", "user.email=masy-test@invalid", "-c", "commit.gpgsign=false",
         "-c", "init.defaultBranch=main", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
        check=True, capture_output=True, env=env, timeout=60,
    )
    return completed.stdout.decode()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def make_repo(base: Path, name: str, package: str, project_name: str,
              files: dict[str, str] | None = None) -> Path:
    """Create and commit a synthetic project: pyproject + src/<package>/."""
    root = base / name
    root.mkdir(parents=True)
    write(root / "pyproject.toml", f'[project]\nname = "{project_name}"\nversion = "0"\n')
    write(root / "src" / package / "__init__.py", f'"""synthetic {package}"""\n')
    for relative, text in (files or {}).items():
        write(root / relative, text)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "synthetic")
    return root


def commit_all(root: Path, message: str = "change") -> None:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="masy01-test-")
        self.tmp = Path(os.path.realpath(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()


# Fake SYRUP modules for end-to-end launcher tests (no real upstream code).
FAKE_DEPENDENCY = '''
class MapleDependencyError(RuntimeError):
    pass


class MapleApiIncompatibleError(MapleDependencyError):
    pass


REQUIRED_MAPLE_API = ()


def check_required_api(requirements=REQUIRED_MAPLE_API):
    return None


def resolve_maple_dependency(expected_source_root=None, *, requirements=REQUIRED_MAPLE_API):
    {mutation}
    raise MapleDependencyError(
        f"expected MAPLE source root {{expected_source_root}}, but the imported maple resolves to "
        "source root None"
    )
'''

MUTATE_MAPLE = '''import pathlib
    with open(pathlib.Path(expected_source_root) / "src" / "maple" / "__init__.py", "a") as handle:
        handle.write("# concurrent edit\\n")'''

FAKE_PROVENANCE = '''
import hashlib
import os
import subprocess
from dataclasses import dataclass

SOURCE_DIGEST_ALGORITHM = "sha256(sorted relpath NUL file_sha256 LF), excl __pycache__/*.pyc/*.pyo; v1"


@dataclass(frozen=True)
class SourceTreeDigest:
    root: str
    algorithm: str
    digest_sha256: str
    file_count: int
    total_bytes: int

    def as_record(self, *, include_files=False):
        return {"root": self.root, "algorithm": self.algorithm, "digest_sha256": self.digest_sha256,
                "file_count": self.file_count, "total_bytes": self.total_bytes}


def source_tree_digest(root, *, max_files=5000, max_bytes=1 << 26):
    entries, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "rb") as handle:
                data = handle.read()
            total += len(data)
            entries.append((os.path.relpath(path, root).replace(os.sep, "/"), hashlib.sha256(data).hexdigest()))
    hasher = hashlib.sha256()
    for relative, file_hash in sorted(entries):
        hasher.update(relative.encode() + b"\\0" + file_hash.encode() + b"\\n")
    return SourceTreeDigest(str(root), SOURCE_DIGEST_ALGORITHM, hasher.hexdigest(), len(entries), total)


def scoped_git_status(repo_root, pathspecs, *, timeout_s=30.0):
    out = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo_root), "status", "--porcelain=v1", "-z",
         "--untracked-files=all", "--", *pathspecs], capture_output=True, timeout=timeout_s, check=True,
    ).stdout
    return {"status": "dirty" if out else "clean", "pathspecs": list(pathspecs)}
'''


def make_fake_pair(base: Path, *, mutate: bool = False) -> tuple[Path, Path]:
    maple = make_repo(base, "MAPLE", "maple", "maple")
    dependency = FAKE_DEPENDENCY.format(mutation=MUTATE_MAPLE if mutate else "pass")
    syrup = make_repo(base, "SYRUP", "maple_syrup", "maple-syrup", files={
        "src/maple_syrup/dependency.py": dependency,
        "src/maple_syrup/provenance.py": FAKE_PROVENANCE,
    })
    return maple, syrup


def write_config(path: Path, maple: Path, syrup: Path, extra: str = "") -> Path:
    path.write_text(
        'schema = "masy.preflight.v1"\n'
        f'[sources.maple]\nroot = "{maple}"\n'
        f'[sources.syrup]\nroot = "{syrup}"\n' + extra
    )
    return path
