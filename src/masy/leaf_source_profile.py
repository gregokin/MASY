"""Literal source profiles + sealed receipt for the scheduled LEAF-water participant (task phase1_leaf_calendar_impl_20261112; Design.md section 2). Derived from the accepted f257/Leaf1e468 reference; the OLD profile ids are not part of this module.

Two DISTINCT explicit profiles pin the SAME actual sources by literals: the isolated LEAF reference = git base 1e468e14 PLUS the qualified scoped delta (NOT an unchanged 1e468, NOT live 5480, NOT a new commit; 23 Python modules + pyproject),
the SYRUP water109 package (109 production sources, digest afcad536...) and the MAPLE native9f package. A tree of the old 21-module reference fails the literal census. `CPU_PROFILE_ID` refuses any CuPy import; `CUDA_PROFILE_ID` is a separate sealed sibling (the receipt carries its own
literal, kind and identity digest: a CPU receipt is never accepted for a CUDA lifecycle and vice versa). Cold verification is host-only for both (no device work, no CuPy import here). Nothing widens the
historical 95ef/308 MASY profiles; no `production_committed` recipe is claimed; a mutable live LEAF is never bound.

The MASY side is verified against a CALLER-DECLARED frozen manifest (`masy_manifest`, never auto-built from the current tree and never self-hashed circularly) with the bounded `source_census` primitive of the
accepted adapter, plus module association: every loaded `masy.*` module must be a manifest member under the fenced root, and the adapter/profile modules and classes are bound by identity.
`verify_profile` is COLD (never per step) and ordered, with a fresh strict free-memory reading and an explicit staging admission BEFORE each stage reads anything: (1) MASY manifest census, (2) LEAF file census
BEFORE any LEAF import, (3) lazy import of the LEAF modules [the fixed interpreter/import cost is an EXPLICIT exclusion], (4) module/file/class association, (5) the genuine `native_case_profile.verify_source_profile` for the materialized MAPLE bundle (runtime check + exact bundle identity), then bounded SYRUP / LEAF tree digests with the literal
file-count and byte bounds below (the qualified `source_tree_digest` with explicit limits, never its defaults). `check_hot` is the per-call O(#bindings) identity/metadata check (no hash, no census).
LEAF is reached only through the process `PYTHONPATH` that the caller's fresh child received (no runtime sys.path / sys.modules substitution here).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = ["verify_profile_delivery", "VARIANTS", "LEAF_BASE_COMMIT_8DA", "LEAF_PACKAGE_DIGEST_8DA", "LEAF_MANIFEST_8DA", "CPU_PROFILE_8DA_ID", "CUDA_PROFILE_8DA_ID", "LEAF_BASE_COMMIT", "LEAF_DELTA", "NATIVE_BUNDLE_MAX_FILES", "CPU_PROFILE_ID", "CUDA_PROFILE_ID", "PROFILE_ID", "LEAF_COMMIT", "LEAF_PACKAGE_DIGEST", "LEAF_MANIFEST", "SYRUP_PACKAGE_DIGEST", "MAPLE_PACKAGE_DIGEST", "TREE_BOUNDS",
           "SourceProfileError", "SourceReceipt", "verify_profile", "check_hot", "read_free"]

CPU_PROFILE_ID = "leaf23_d4fe_water109_native9f_cpu_v1"
CUDA_PROFILE_ID = "leaf23_d4fe_water109_native9f_cuda_v1"
PROFILE_ID = CPU_PROFILE_ID  # historical alias of the first stage
_PROFILES = {"cpu": CPU_PROFILE_ID, "cuda": CUDA_PROFILE_ID}
LEAF_BASE_COMMIT = "1e468e1410992430e0aa5045ca44637b382ac0e9"
LEAF_COMMIT = LEAF_BASE_COMMIT  # historical alias: the BASE only; the bound source is base + the qualified scoped delta below (no commit exists for it)
LEAF_DELTA = "qualified scoped LEAF biology-API delta (biocalendar, plant_pet, growth/interevent additions) over the 1e468 base; package digest below"
LEAF_PACKAGE_DIGEST = "d4fe6cd23a728e97d027827df3ed5d544833c234d07be84293c14d651ab65c50"  # provenance.source_tree_digest of the `leaf/` package directory
SYRUP_PACKAGE_DIGEST = "afcad53691b9d295ed560116d313419c08fc5ffe9b227a4bdd9c922f93c52671"
MAPLE_PACKAGE_DIGEST = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
MAPLE_BUNDLE_ID = "55d1485a03c385240bab57edf7fba6cad5230628729b30b8a19d96d094ddaa5c"
#: Literal bounds of the tree digests (exact file counts and total bytes measured on the fixed sources; the digest helper refuses more than either).
TREE_BOUNDS = MappingProxyType({"syrup": (109, 4_776_197), "leaf": (23, 912_739), "leaf_8da": (23, 945_984)})
#: MAPLE is verified by the ACTUAL `native_case_profile.verify_source_profile` (check_runtime + verified materialized bundle + pinned digests); its declared staging is the bundle `Limits.max_files`.
NATIVE_BUNDLE_MAX_FILES = 20_000
LEAF_MANIFEST = MappingProxyType({
    "leaf/__init__.py": "5d0f3e40077546da9287d043c9f53058efcc0e495f077478426cd18f3f2d4f1d",
    "leaf/__main__.py": "83b4589b1aeab57d43ad04c69d32b80daf6eb976004b38ae4b57410d861b8ee1",
    "leaf/biocalendar.py": "c2a911ad66e84b420fe948f40a1f80b4d286bd569bec065d10b468ad0570e8e5",
    "leaf/canopy.py": "6f16b97e7fbd248685495959b4e0322f44ef9b2ba950bb35ded0d70826adde0e",
    "leaf/checkpoint.py": "02eddb43da85ecb8637791967d8e17f8cb5b21d1acaf3be1d424da8b29788357",
    "leaf/coupling.py": "2a9de99773f976d260617d1506be15f9b5950233880625dc40389e6d99e9b247",
    "leaf/demography.py": "7648b8f4a23016dc07196b9c790498bccb3cf18db0a20bbbda408974834dce1a",
    "leaf/disturbance.py": "38a9dfa4521cdcb2512b904a83d5f2af4fbad97fa1a0c972bb70b0cb6906f9c0",
    "leaf/events.py": "2ec912dac4a5bb4ae1f52a64ef99e74e973d7f7dc68993f0f083fceb3cbed98e",
    "leaf/forcing.py": "fa79d12f2c96a15aac8012d713976ece8b4c055ca567968251225635fa9d7474",
    "leaf/growth.py": "6664c7194ca7df3f731fe82f9527cf6f60835c5517aca60dbc591cf719d2aa58",
    "leaf/initialization.py": "25e58f759761b525c2021eb18d64cd2b6bede8543b22ba983a8eb2d65967aeab",
    "leaf/interevent.py": "51c00435d827b66dcad082165421eed8ba615d3104e4b4b8cbd73728743b45b9",
    "leaf/plant_pet.py": "ec831f57d62655bbf9766c27e36dc17c7a89de96cf2b3e18a2bc0448af0d1b17",
    "leaf/recording.py": "12ea3fee1dd660756b2e800adb20265f25b867a42604f456f01ea2728fa2c584",
    "leaf/recruitment.py": "a4918c363e84d10465fc9bea6d1c22664c6236eb05ad139d7daf4f925f99e632",
    "leaf/roots.py": "6cf7ab0fd2ee5c508d48ef6d8071022dd58faa67fcbc8de9974ad9d752601b50",
    "leaf/runtime.py": "abbea7b0f873d87139f92c9eb4a6989f657def504c2080379d728f7ceb121826",
    "leaf/state.py": "187ecb5f9e042cc47beb8ae5ba118f4111ee7cdc2d725ef820e730cec349e6a0",
    "leaf/stepwat.py": "981f9b04cd15421c889c923453957f3a7f77034d535aca2f8e4aa4f70858f76a",
    "leaf/syrup_water.py": "baefec2a100b751cc35bda8739ea5d04339681f0d3f9d6b900972986161db215",
    "leaf/traits.py": "4aecd751ae6a46fb996dfa295129d3f5c01e57abf98943d150103782f9ce8527",
    "leaf/uptake.py": "675e4bf4b9a9c202ae166f50444032c105d4d691f366f94479034814952bfe2d",
    "pyproject.toml": "76fae76c6026ab990d979349ffbc641059622d5bc44ac20aa9911c39df6e2216",
})

#: The SECOND literal LEAF family: git base 9fd4bbbe PLUS the independently qualified four-file API delta (plain plant CSV + companion metadata), package digest 8da24af5... Exact closed table; no wildcard,
#: prefix or hash fallback. The d4fe literals above are unchanged and a d4fe receipt/profile refuses this tree (and vice versa) by the literal census.
LEAF_BASE_COMMIT_8DA = "9fd4bbbe2faae907922ed7194f633caa1da3624f"
LEAF_DELTA_8DA = "independently qualified four-file LEAF API delta (state/traits/initialization/interevent plain plant CSV + companion metadata) over the 9fd4bbbe base; package digest below"
LEAF_PACKAGE_DIGEST_8DA = "8da24af5a672138bab335d8fd1d4518c5a17ae90db9b400c5d211cc0c768fa4b"
CPU_PROFILE_8DA_ID = "leaf23_8da_water109_native9f_cpu_v1"
CUDA_PROFILE_8DA_ID = "leaf23_8da_water109_native9f_cuda_v1"
LEAF_MANIFEST_8DA = MappingProxyType({
    "leaf/__init__.py": "e8d289c0c597dbaf987a94e1f53b036ebd5f1dc08f2900be7cc6b3aacb5e78b5",
    "leaf/__main__.py": "e1e40edddf48c656e0b55cf2b5d8bd8387bf8f581439d23609ffbc3d99b27294",
    "leaf/biocalendar.py": "c2a911ad66e84b420fe948f40a1f80b4d286bd569bec065d10b468ad0570e8e5",
    "leaf/canopy.py": "6f16b97e7fbd248685495959b4e0322f44ef9b2ba950bb35ded0d70826adde0e",
    "leaf/checkpoint.py": "b59b555b475277f428d14f125c04e22ef13ac328df6f21436703e413867f935d",
    "leaf/coupling.py": "2a9de99773f976d260617d1506be15f9b5950233880625dc40389e6d99e9b247",
    "leaf/demography.py": "7648b8f4a23016dc07196b9c790498bccb3cf18db0a20bbbda408974834dce1a",
    "leaf/disturbance.py": "38a9dfa4521cdcb2512b904a83d5f2af4fbad97fa1a0c972bb70b0cb6906f9c0",
    "leaf/events.py": "2ec912dac4a5bb4ae1f52a64ef99e74e973d7f7dc68993f0f083fceb3cbed98e",
    "leaf/forcing.py": "fa79d12f2c96a15aac8012d713976ece8b4c055ca567968251225635fa9d7474",
    "leaf/growth.py": "6664c7194ca7df3f731fe82f9527cf6f60835c5517aca60dbc591cf719d2aa58",
    "leaf/initialization.py": "df3e2184e1c9e8c5483884b8e371472dac059cda50e549e27251f8f8caa52a19",
    "leaf/interevent.py": "50bbc6ea064862156c3391d0cc4a74e76c287de243bd366b2869aa2e4a88c4ac",
    "leaf/plant_pet.py": "ec831f57d62655bbf9766c27e36dc17c7a89de96cf2b3e18a2bc0448af0d1b17",
    "leaf/recording.py": "49564345dd53505148de90de00b5bcb02d7d0e23b8b483e75ebac96af2ff9ef0",
    "leaf/recruitment.py": "a4918c363e84d10465fc9bea6d1c22664c6236eb05ad139d7daf4f925f99e632",
    "leaf/roots.py": "6cf7ab0fd2ee5c508d48ef6d8071022dd58faa67fcbc8de9974ad9d752601b50",
    "leaf/runtime.py": "abbea7b0f873d87139f92c9eb4a6989f657def504c2080379d728f7ceb121826",
    "leaf/state.py": "a9eefbc5520b094d0a5ec63b5ca5e8f1a1f2d0595595bb1b80a6ca33774a62cd",
    "leaf/stepwat.py": "981f9b04cd15421c889c923453957f3a7f77034d535aca2f8e4aa4f70858f76a",
    "leaf/syrup_water.py": "baefec2a100b751cc35bda8739ea5d04339681f0d3f9d6b900972986161db215",
    "leaf/traits.py": "693b0a0b3093480e198a072240ea4d93177af66823929c6b8747464388db7b89",
    "leaf/uptake.py": "675e4bf4b9a9c202ae166f50444032c105d4d691f366f94479034814952bfe2d",
    "pyproject.toml": "76fae76c6026ab990d979349ffbc641059622d5bc44ac20aa9911c39df6e2216",
})

#: CLOSED variant table: leaf_variant -> (device -> profile id, base commit, delta text, package digest, literal manifest, tree-bound key).
VARIANTS = MappingProxyType({
    "d4fe": MappingProxyType({"profiles": MappingProxyType({"cpu": CPU_PROFILE_ID, "cuda": CUDA_PROFILE_ID}), "base": LEAF_BASE_COMMIT, "delta": LEAF_DELTA, "digest": LEAF_PACKAGE_DIGEST,
                              "manifest": LEAF_MANIFEST, "bounds": "leaf"}),
    "8da": MappingProxyType({"profiles": MappingProxyType({"cpu": CPU_PROFILE_8DA_ID, "cuda": CUDA_PROFILE_8DA_ID}), "base": LEAF_BASE_COMMIT_8DA, "delta": LEAF_DELTA_8DA, "digest": LEAF_PACKAGE_DIGEST_8DA,
                             "manifest": LEAF_MANIFEST_8DA, "bounds": "leaf_8da"}),
})
_ALL_PROFILES = {pid: (v, dev) for v, spec in VARIANTS.items() for dev, pid in spec["profiles"].items()}

#: LEAF modules the adapter binds (module -> classes bound by identity).
_BOUND = (("leaf.uptake", ("RootMap", "UptakePlan", "UptakeFeedback")), ("leaf.growth", ("CompiledSpecies",)), ("leaf.forcing", ("TimestepRecord",)), ("leaf.biocalendar", ("Schedule",)),
          ("leaf.plant_pet", ("CanopyMap",)), ("leaf.state", ("PlantState", "Domain")), ("leaf.traits", ("SpeciesCatalog",)))
_ADAPTER = (("masy.leaf_water_interval", ("IntervalOperator", "LeafBoundary", "IntervalHead", "IntervalCandidate")), ("masy.leaf_source_profile", ("SourceReceipt",)),
           ("masy.leaf_calendar", ("LeafCalendar",)))
_SEAL = object()
_CENSUSED: list[str] = []  # LEAF roots whose FILES were censused earlier in THIS process (a later verification may then find LEAF already imported)
_STAGE_CHUNK = 1 << 20


class SourceProfileError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def read_free(fn: Any, what: str = "available_bytes_fn") -> int:
    """Strict reading: the callable is invoked fresh; the result must be a non-Boolean int >= 0 (a False-valued int subclass is honoured as its integer value)."""
    if not callable(fn):
        raise SourceProfileError("bad_reader", f"{what} must be callable")
    value = fn()
    if isinstance(value, (bool,)) or not isinstance(value, int) or int(value) < 0:
        raise SourceProfileError("bad_reading", f"{what} must return a non-Boolean integer >= 0, got {type(value).__name__}")
    return int(value)


class SourceReceipt:
    """Sealed immutable result of the cold verification: scalars + the module/class identities the hot check compares."""

    __slots__ = ("profile", "device_kind", "variant", "leaf_root", "syrup_root", "native_root", "masy_root", "census", "masy_census", "digests", "stages", "native", "delivery", "_modules", "_classes", "_adapter", "_seal")

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise SourceProfileError("forged_receipt", "SourceReceipt is minted only by verify_profile()")
        fields.setdefault("delivery", None)  # None for every receipt minted by the historical verify_profile()
        for name in self.__slots__:
            if name != "_seal":
                object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise SourceProfileError("forged_receipt", f"unexpected fields {sorted(fields)}")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SourceReceipt is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SourceReceipt is immutable")

    def record(self) -> dict[str, Any]:
        return {"profile": self.profile, "device_kind": self.device_kind, "leaf_variant": self.variant, "leaf_base_commit": VARIANTS[self.variant]["base"], "leaf_delta": VARIANTS[self.variant]["delta"], "leaf_root": self.leaf_root, "syrup_root": self.syrup_root, "native_root": self.native_root,
                "masy_root": self.masy_root, "census": dict(self.census), "masy_census": dict(self.masy_census), "digests": dict(self.digests), "stages": dict(self.stages), "native": dict(self.native),
                "exclusions": "fixed interpreter/import cost of the LEAF modules, allocator pools, whole-process memory",
                **({} if self.delivery is None else {"delivery": dict(self.delivery)})}  # old receipts: record byte-identical (no `delivery` key)

    def module(self, name: str) -> Any:
        return self._modules[name]


def _under(path: Any, root: str) -> bool:
    try:
        Path(str(path)).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _stage(free_fn: Any, need: int, name: str, stages: dict[str, int]) -> None:
    """A fresh strict reading BEFORE every cold stage reads anything; refuse when the stage's staging does not fit."""
    free = read_free(free_fn)
    if int(need) > free:
        raise SourceProfileError("cold_admission", f"cold stage {name!r} needs {int(need)} B of staging but {free} B are reported; nothing was read")
    stages[name] = int(need)


def verify_profile(*, device: str = "cpu", leaf_variant: str = "d4fe", leaf_root: Any, syrup_root: Any, native_target: Any, masy_root: Any, masy_manifest: Any, available_bytes_fn: Any, reserve_bytes: int = 0,
                   limits: Any = None) -> SourceReceipt:
    """COLD verification (see module docstring). `device` selects the explicit sibling profile ('cpu' | 'cuda'); both verify host-only. `masy_manifest` is the caller-declared frozen
    `{relative path: sha256}` of the MASY source directory `masy_root` (e.g. its `src/masy`), never derived from the tree being verified."""
    return _verify(device=device, leaf_variant=leaf_variant, leaf_root=leaf_root, syrup_root=syrup_root, native_target=native_target, masy_root=masy_root, masy_manifest=masy_manifest,
                   available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, limits=limits, delivery=None)


def verify_profile_delivery(*, delivery: Any, device: str = "cpu", leaf_variant: str = "8da", leaf_root: Any, syrup_root: Any, native_target: Any, masy_root: Any, masy_manifest: Any,
                            available_bytes_fn: Any, reserve_bytes: int = 0, limits: Any = None) -> SourceReceipt:
    """PRODUCTION-ONLY sibling of `verify_profile` for a verified NEW delivery of the SAME pinned 9f source (the bundle id also hashes export path/tool, so a fresh export of identical bytes has a different id).
    The cold verification is the same engine; only the bundle-id gate differs and is strictly stronger than a label: `delivery` is the closed `masy.native_delivery.v1` record built by
    `masy.production_runtime.resolve` from the verified V1 lock + Root-selected recipe, and the ACTUAL native verification must match it (bundle id, the actual `bundle-manifest.json` bytes, the literal archive
    sha256, target/leaf/SYRUP/MASY roots, profile/device/variant). The reference-id gate of `verify_profile` is not reachable from here and there is no flag that disables it."""
    if delivery is None:  # the sibling can never fall back to the reference-id mode
        raise SourceProfileError("delivery_required", "verify_profile_delivery requires the verified-delivery record; use verify_profile for the reference bundle")
    return _verify(device=device, leaf_variant=leaf_variant, leaf_root=leaf_root, syrup_root=syrup_root, native_target=native_target, masy_root=masy_root, masy_manifest=masy_manifest,
                   available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes, limits=limits, delivery=delivery)


def _check_delivery_binding(delivery: Any, native: Any, ncp: Any, stage: Any, free_fn: Any, reserve: int, stages: dict) -> None:
    """The ACTUAL native verification must be the delivery the production binding named: bundle id, the bytes of the materialized `bundle-manifest.json` and the literal archive sha256."""
    if native.bundle_id != delivery["bundle_id"] or Path(native.target).name != delivery["bundle_id"]:
        raise SourceProfileError("delivery_mismatch", f"the checked bundle {native.bundle_id} is not the verified delivery {delivery['bundle_id']}")
    if delivery["archive_sha256"] != ncp.ARCHIVE_SHA256:
        raise SourceProfileError("delivery_mismatch", "delivery.archive_sha256 is not the pinned 9f archive sha256 literal")
    path = Path(native.target) / "bundle-manifest.json"
    try:
        size = os.lstat(path).st_size
    except OSError as exc:
        raise SourceProfileError("delivery_mismatch", f"the materialized bundle manifest cannot be examined: {exc}") from None
    if size > (16 << 20):
        raise SourceProfileError("delivery_mismatch", f"the materialized bundle manifest is {size} B, over the 16 MiB bound (checked before the body)")
    stage(free_fn, 2 * int(size) + _STAGE_CHUNK + reserve, "delivery_manifest", stages)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        data = b""
        while len(data) <= size:
            chunk = os.read(fd, min(1 << 20, size - len(data) + 1))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) != size or hashlib.sha256(data).hexdigest() != delivery["manifest_sha256"]:
        raise SourceProfileError("delivery_mismatch", "the bytes of the materialized bundle manifest differ from delivery.manifest_sha256")


def _check_delivery_record(delivery: Any) -> Any:
    from masy import production_runtime as prod  # stdlib-only module; lazy so the historical path never imports it

    try:
        return prod.validate_delivery(delivery)
    except Exception as exc:  # noqa: BLE001 - GateRefusal from the closed validator
        raise SourceProfileError("delivery_invalid", f"the verified-delivery record is not the closed masy.native_delivery.v1 record: {getattr(exc, 'detail', exc)}") from None


def _verify(*, delivery: Any, device: str = "cpu", leaf_variant: str = "d4fe", leaf_root: Any, syrup_root: Any, native_target: Any, masy_root: Any, masy_manifest: Any, available_bytes_fn: Any,
            reserve_bytes: int = 0, limits: Any = None) -> SourceReceipt:
    from masy import native_water_calendar as nwc  # lazy: the bounded census of the accepted adapter (no-follow, descriptor based)

    if type(device) is not str or device not in _PROFILES:
        raise SourceProfileError("bad_device", f"device must be one of the strings {sorted(_PROFILES)}")
    if type(leaf_variant) is not str or leaf_variant not in VARIANTS:
        raise SourceProfileError("bad_variant", f"leaf_variant must be one of the strings {sorted(VARIANTS)} (an exact closed table, no wildcard)")
    var = VARIANTS[leaf_variant]
    manifest_lit = var["manifest"]
    if delivery is not None:  # production delivery: the closed record and its static joins are checked BEFORE any census/import
        delivery = _check_delivery_record(delivery)
        for key, got in (("device", device), ("leaf_variant", leaf_variant), ("leaf_profile", var["profiles"][device]), ("leaf_root", str(leaf_root)), ("syrup_root", str(syrup_root)),
                         ("masy_root", str(masy_root)), ("target", str(native_target))):
            if delivery[key] != got:
                raise SourceProfileError("delivery_mismatch", f"delivery.{key} {delivery[key]!r} differs from the requested {got!r}")
    if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int) or reserve_bytes < 0:
        raise SourceProfileError("bad_reserve", "reserve_bytes must be a non-Boolean integer >= 0")
    limits = nwc.CensusLimits() if limits is None else limits
    if type(masy_manifest) is not dict or not 0 < len(masy_manifest) <= limits.max_files or not all(type(k) is str and type(v) is str for k, v in masy_manifest.items()):
        raise SourceProfileError("bad_manifest", f"masy_manifest must be a plain dict of 1..{limits.max_files} str -> str entries (validated before any copy or census)")
    reserve = int(reserve_bytes)
    stages: dict[str, int] = {}
    if device == "cpu" and "cupy" in sys.modules:
        raise SourceProfileError("gpu_import", "CuPy is imported; the CPU profile refuses GPU modules")
    leaf_root, masy_root = str(leaf_root), str(masy_root)
    # (1) MASY: the caller-declared manifest is validated (bounded) and the tree censused BEFORE anything else is read
    _stage(available_bytes_fn, limits.staging_bytes(len(masy_manifest)) + reserve, "masy_census", stages)
    masy_census = nwc.source_census(masy_root, dict(masy_manifest), limits)
    if not masy_census["clean"]:
        raise SourceProfileError("masy_census", f"MASY source differs from the caller-declared manifest: changed={masy_census['changed']} missing={masy_census['missing']} unexpected={masy_census['unexpected']} unsafe={masy_census['unsafe']}")
    # (2) LEAF files BEFORE any LEAF import
    pre = [n for n in sys.modules if n == "leaf" or n.startswith("leaf.")]
    if pre and leaf_root not in _CENSUSED:
        raise SourceProfileError("leaf_preloaded", f"LEAF modules {sorted(pre)[:3]} were imported BEFORE the file census; the cold order is census first")
    _stage(available_bytes_fn, limits.staging_bytes(len(manifest_lit)) + reserve, "leaf_census", stages)
    census = nwc.source_census(leaf_root, dict(manifest_lit), limits)
    if not census["clean"]:
        raise SourceProfileError("leaf_census", f"LEAF reference differs from the literal manifest: changed={census['changed']} missing={census['missing']} unexpected={census['unexpected']} unsafe={census['unsafe']}")
    if leaf_root not in _CENSUSED:
        _CENSUSED.append(leaf_root)
    # (3)+(4) imports, then module/file/class association
    _stage(available_bytes_fn, reserve, "leaf_import", stages)  # fixed interpreter/import cost is an EXPLICIT exclusion (not modelled)
    modules, classes = {}, {}
    for name, names in _BOUND:
        mod = importlib.import_module(name)
        file = getattr(mod, "__file__", None)
        if not file or not _under(file, leaf_root):
            raise SourceProfileError("leaf_origin", f"{name} was imported from {file!r}, not from the fenced LEAF root {leaf_root!r}")
        modules[name] = mod
        for cls in names:
            classes[f"{name}.{cls}"] = getattr(mod, cls)
    for n in sorted(x for x in sys.modules if x == "leaf" or x.startswith("leaf.")):
        f = getattr(sys.modules[n], "__file__", None)
        if f is not None and (not _under(f, leaf_root) or os.path.relpath(Path(f).resolve(), Path(leaf_root).resolve()) not in manifest_lit):
            raise SourceProfileError("leaf_origin", f"loaded LEAF module {n} ({f}) is not a member of the literal manifest under the fenced root")
    import maple
    import maple_syrup
    from maple_syrup import provenance as prov

    syrup_pkg = Path(maple_syrup.__file__).resolve().parent
    if str(syrup_pkg) != str(Path(str(syrup_root)).resolve()):
        raise SourceProfileError("syrup_origin", f"maple_syrup is imported from {syrup_pkg}, not from the fenced SYRUP root {syrup_root}")
    native_src = Path(str(native_target)) / "source" / "src"
    if not _under(maple.__file__, str(native_src)):
        raise SourceProfileError("native_origin", f"maple is imported from {maple.__file__}, not from the native bundle {native_target}")
    # the genuine native source profile (closed CPU/CUDA name): bundle manifest + check_runtime + pinned digests + imported-module census, admitted BEFORE it reads anything
    from maple_syrup import native_case_profile as ncp

    _stage(available_bytes_fn, _STAGE_CHUNK + 512 * NATIVE_BUNDLE_MAX_FILES + reserve, "native_bundle_profile", stages)
    try:
        native = ncp.verify_source_profile(native_target, expected_syrup_digest=SYRUP_PACKAGE_DIGEST, profile=ncp.PROFILE_NAME if device == "cpu" else ncp.CUDA_SOURCE_PROFILE_NAME)
    except ncp.NativeCaseProfileError as exc:
        raise SourceProfileError("native_profile", f"the materialized native bundle/runtime check failed: {exc}") from None
    if delivery is None:
        if native.bundle_id != MAPLE_BUNDLE_ID or native.package_digest != MAPLE_PACKAGE_DIGEST or native.syrup_package_digest != SYRUP_PACKAGE_DIGEST:
            raise SourceProfileError("native_profile", f"the checked bundle {native.bundle_id} / package {native.package_digest} is not the declared MAPLE_BUNDLE_ID / MAPLE_PACKAGE_DIGEST")
    else:
        if native.package_digest != MAPLE_PACKAGE_DIGEST or native.syrup_package_digest != SYRUP_PACKAGE_DIGEST:
            raise SourceProfileError("native_profile", f"the checked native package {native.package_digest} / SYRUP {native.syrup_package_digest} is not the declared MAPLE_PACKAGE_DIGEST / SYRUP_PACKAGE_DIGEST")
        _check_delivery_binding(delivery, native, ncp, _stage, available_bytes_fn, reserve, stages)
    native_rec = {"profile": native.profile, "bundle_id": native.bundle_id, "target": native.target, "package_digest": native.package_digest, "project_digest": native.project_digest,
                  "maple_commit": native.maple_commit, "n_maple_modules": len(native.maple_modules), "python": native.python, "dependency": native.dependency}
    for n in sorted(x for x in sys.modules if x.split(".")[0] in ("maple_syrup", "masy", "maple")):
        f = getattr(sys.modules[n], "__file__", None)
        top = n.split(".")[0]
        if f is None:
            continue
        if top == "maple_syrup" and not _under(f, str(syrup_pkg)):
            raise SourceProfileError("syrup_origin", f"loaded module {n} lies outside the fenced SYRUP root")
        if top == "maple" and not _under(f, str(native_src)):
            raise SourceProfileError("native_origin", f"loaded module {n} lies outside the native bundle source")
        if top == "masy":
            rel = os.path.relpath(Path(f).resolve(), Path(masy_root).resolve())
            if not _under(f, masy_root) or rel not in masy_manifest:
                raise SourceProfileError("masy_origin", f"loaded masy module {n} is not a member of the caller-declared manifest under the fenced root (foreign MASY namespace)")
    adapter, adapter_cls = {}, {}
    for name, names in _ADAPTER:
        mod = sys.modules.get(name)
        if mod is None:
            raise SourceProfileError("adapter_not_loaded", f"{name} must be imported before the verification so its identity is bound")
        adapter[name] = mod
        for cls in names:
            adapter_cls[f"{name}.{cls}"] = getattr(mod, cls)
    # (5) bounded tree digests: the qualified helper with the EXPLICIT literal file-count/byte bounds, each stage admitted first
    digests = {}
    trees = (("syrup", syrup_pkg, SYRUP_PACKAGE_DIGEST), ("leaf", Path(modules["leaf.uptake"].__file__).resolve().parent, var["digest"]))
    for key, root, want in trees:
        n_files, n_bytes = TREE_BOUNDS[var["bounds"] if key == "leaf" else key]
        _stage(available_bytes_fn, _STAGE_CHUNK + 512 * n_files + reserve, f"{key}_tree_digest", stages)
        try:
            digests[key] = prov.source_tree_digest(root, max_files=n_files, max_bytes=n_bytes).digest_sha256
        except prov.ProvenanceScopeError as exc:
            raise SourceProfileError(f"{key}_digest", f"{key} tree exceeds its literal bounds ({n_files} files, {n_bytes} B) or is unsafe: {exc}") from None
        if digests[key] != want:
            raise SourceProfileError(f"{key}_digest", f"{key} package digest {digests[key]} != literal {want}")
    digests["maple"] = native.package_digest  # established by the genuine native profile above
    census_rec = {k: census[k] for k in ("root", "files", "bytes_read", "manifest_sha256", "clean")}
    masy_rec = {k: masy_census[k] for k in ("root", "files", "bytes_read", "manifest_sha256", "clean")}
    modules = dict(modules, **adapter)
    classes = dict(classes, **adapter_cls)
    return SourceReceipt(_SEAL, profile=var["profiles"][device], device_kind=device, variant=leaf_variant, leaf_root=leaf_root, syrup_root=str(syrup_pkg), native_root=str(native_target), masy_root=masy_root,
                         census=MappingProxyType(census_rec), masy_census=MappingProxyType(masy_rec), digests=MappingProxyType(digests), stages=MappingProxyType(stages), native=MappingProxyType(native_rec),
                         _modules=MappingProxyType(modules), _classes=MappingProxyType(classes), _adapter=MappingProxyType(adapter_cls), delivery=None if delivery is None else MappingProxyType(dict(delivery)))


def check_hot(receipt: Any) -> None:
    """Per-call metadata/identity check (no hash, census or file read): genuine sealed receipt; the recorded modules (LEAF AND the adapter/profile) are still the loaded ones; the bound classes are the
    same objects. The CPU profile also refuses a CuPy import; the CUDA sibling does not (its device gate is the genuine lifecycle/column preflight)."""
    if type(receipt) is not SourceReceipt or getattr(receipt, "_seal", None) is not _SEAL:
        raise SourceProfileError("forged_receipt", "a genuine SourceReceipt from verify_profile() is required")
    if receipt.profile not in _ALL_PROFILES or _ALL_PROFILES[receipt.profile] != (receipt.variant, receipt.device_kind):
        raise SourceProfileError("profile", "unknown or inconsistent source profile")
    for name, mod in receipt._modules.items():
        if sys.modules.get(name) is not mod:
            raise SourceProfileError("module_replaced", f"{name} is no longer the verified module object")
    for key, cls in receipt._classes.items():
        modname, _, clsname = key.rpartition(".")
        if getattr(sys.modules.get(modname), clsname, None) is not cls:
            raise SourceProfileError("class_replaced", f"{key} is no longer the verified class")
    if receipt.device_kind == "cpu" and "cupy" in sys.modules:
        raise SourceProfileError("gpu_import", "CuPy is imported; the CPU profile refuses GPU modules")
