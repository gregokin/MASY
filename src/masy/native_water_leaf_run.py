"""LEAF half of the v2 file-configured run (imported LAZILY by `native_water_worker` only for a `masy.native_water_run.v2` configuration; a v1 run never imports it, LEAF or the leaf bridge).

Composes only accepted pieces: the stdlib plant-file I/O control (this module), the genuine source receipt (`leaf_source_profile.verify_profile`, 8da family; census BEFORE any LEAF import), the PUBLIC
LEAF 8da loaders (`traits.SpeciesCatalog`, `state.parse_plant_csv(metadata=...)`, `PlantState.resolve_establishment`, `biocalendar.rebuild_geometry`, `growth.compile_species`) and the session API
(`create_leaf(_cuda)` / `resume_leaf(_cuda)` / `checkpoint_leaf`). It owns no plant, water or calendar law. The stdlib control (byte caps, streamed row count against the companion metadata and
`leaf.io.max_plant_records`) is I/O admission only: the bounded public LEAF parser stays the authority on every plant value.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io as _io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

__all__ = ["LeafRunRefused", "plant_file_control", "verify_receipt", "reference_operator", "build_participant", "leaf_limits", "static_pins", "biology_summary"]


class LeafRunRefused(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def count_csv_records(csv_bytes: bytes, max_records: int) -> int:
    """Data-record count of a plant CSV with the SAME record semantics as the public LEAF parser (`csv.reader(strict=True)` over the UTF-8 text, newline untranslated; the header is the first record; an empty record
    is skipped): quoted fields (even multi-line ones) are ONE record. Streamed through a bounded text wrapper: no line list, no row history, no whole-file decode; stops as soon as `max_records` is exceeded.
    Counting only: the LEAF parser stays the authority on header, values, SURFACE tokens and everything else."""
    text = _io.TextIOWrapper(_io.BytesIO(csv_bytes), encoding="utf-8", newline="")
    try:
        reader = csv.reader(text, strict=True)
        try:
            next(reader)  # header
        except StopIteration:
            return 0
        n = 0
        for row in reader:
            if len(row) == 0:
                continue
            n += 1
            if n > max_records:
                raise LeafRunRefused("plant_records_limit", f"more than leaf.io.max_plant_records={max_records} plant records")
        return n
    except UnicodeDecodeError as exc:
        raise LeafRunRefused("plants_invalid", f"the plant CSV is not valid UTF-8: {exc}") from None
    except csv.Error as exc:
        raise LeafRunRefused("plants_invalid", f"malformed plant CSV: {exc}") from None
    finally:
        text.detach()


def plant_file_control(csv_bytes: bytes, meta_bytes: bytes, io: Any, pins: Any, csv_pin: str, meta_pin: str) -> dict[str, Any]:
    """STDLIB I/O control BEFORE any plant array: the bytes equal their pins, the sizes respect the declared caps, the CSV row count (streamed, header excluded) equals the companion's `n_plants`
    and respects `io.max_plant_records`, and the companion's recorded CSV byte count equals the file. Not replacement plant validation: LEAF's parser verifies sha256, catalog, domain, time and values."""
    if hashlib.sha256(csv_bytes).hexdigest() != csv_pin or hashlib.sha256(meta_bytes).hexdigest() != meta_pin:
        raise LeafRunRefused("input_changed_after_pin", "a leaf input file changed after the configuration pinned it")
    if len(csv_bytes) > io["max_plants_bytes"] or len(meta_bytes) > io["max_catalog_bytes"]:
        raise LeafRunRefused("reference_too_large", "a leaf plant file exceeds its declared leaf.io cap")
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LeafRunRefused("plant_metadata_invalid", f"the companion metadata is not valid JSON: {exc}") from None
    if not isinstance(meta, dict) or type(meta.get("n_plants")) is not int or not isinstance(meta.get("csv"), dict) or type(meta["csv"].get("n_bytes")) is not int:
        raise LeafRunRefused("plant_metadata_invalid", "the companion metadata lacks integer n_plants / csv.n_bytes")
    rows = count_csv_records(csv_bytes, int(io["max_plant_records"]))
    if meta["n_plants"] != rows or meta["csv"]["n_bytes"] != len(csv_bytes):
        raise LeafRunRefused("plant_files_inconsistent", f"the companion declares n_plants={meta['n_plants']}, csv.n_bytes={meta['csv']['n_bytes']} but the CSV has {rows} records and {len(csv_bytes)} B")
    if rows > io["max_plant_records"]:
        raise LeafRunRefused("plant_records_limit", f"{rows} plant records exceed leaf.io.max_plant_records={io['max_plant_records']}")
    return {"n_records": rows, "csv_bytes": len(csv_bytes), "metadata_bytes": len(meta_bytes)}


def verify_receipt(cfg: Any, masy_manifest: Any, avail: Any, device: str) -> Any:
    """The genuine source receipt: the leaf bridge modules are imported FIRST (their identities are bound), then the cold verification (MASY manifest census, LEAF literal census BEFORE any LEAF import).
    Historical four-argument public API (reference-bundle gate): a production_committed configuration is refused here (production uses `verify_receipt_production`)."""
    if cfg.source["mode"] == "production_committed":
        raise LeafRunRefused("leaf_delivery_required", "a production_committed run uses verify_receipt_production with the verified-delivery record; the reference-bundle API is not used in production")
    return _verify_receipt_engine(cfg, masy_manifest, avail, device, None)


def verify_receipt_production(cfg: Any, masy_manifest: Any, avail: Any, device: str, delivery: Any) -> Any:
    """Explicit production sibling: REQUIRES the verified-delivery record (`masy.native_delivery.v1`) and a production_committed configuration; a development configuration or a missing record is refused
    BEFORE any import or source work. Uses `leaf_source_profile.verify_profile_delivery` (never the reference-id path)."""
    if cfg.source["mode"] != "production_committed":
        raise LeafRunRefused("leaf_delivery_unexpected", "a verified-delivery record is accepted only by a production_committed run (development keeps the reference bundle gate)")
    if delivery is None:
        raise LeafRunRefused("leaf_delivery_required", "a production_committed run requires the verified-delivery record")
    return _verify_receipt_engine(cfg, masy_manifest, avail, device, delivery)


def _verify_receipt_engine(cfg: Any, masy_manifest: Any, avail: Any, device: str, delivery: Any) -> Any:
    from masy import leaf_calendar  # noqa: F401
    from masy import leaf_water_interval  # noqa: F401
    from masy import leaf_source_profile as lsp

    root, prefix = Path(cfg.source["masy_root"]), "src/masy/"
    manifest = {k[len(prefix):]: v for k, v in masy_manifest.items() if k.startswith(prefix)}
    common = dict(device=device, leaf_variant=cfg.leaf["leaf_variant"], leaf_root=cfg.leaf["leaf_root"], syrup_root=Path(cfg.source["syrup_root"]) / "src" / "maple_syrup",
                  native_target=cfg.source["native_target"], masy_root=root / "src" / "masy", masy_manifest=manifest, available_bytes_fn=avail, reserve_bytes=int(cfg.resources["caller_reserve_bytes"]))
    try:
        return lsp.verify_profile(**common) if delivery is None else lsp.verify_profile_delivery(delivery=delivery, **common)
    except lsp.SourceProfileError as exc:
        raise LeafRunRefused("leaf_" + exc.code, exc.detail) from None


# ------------------------------------------------------------------ cold HOST/DEVICE admission (resource engineering only; the LEAF components keep their own caps)
#: Conservative declared models of the stdlib/interpreter side (LEAF's own estimates are reused wherever it publishes one). A reading is not a reservation; interpreter, JIT, CUDA context, pool rounding and RSS are excluded.
CATALOG_BYTES_PER_INPUT_BYTE = 24  # decoded text (<= 4 B/char) + parsed JSON objects of the catalog
PARSE_BYTES_PER_CSV_BYTE = 4  # the parser's decoded text
PARSE_BYTES_PER_RECORD = 1024  # the parser's per-record Python row objects (strings, lists) before the arrays exist
METADATA_BYTES_PER_INPUT_BYTE = 24
POPULATION_BYTES_PER_LIVE_PLANT = 21  # B, E (f8 each), species index (i4), live (bool): the copies handed to the boundary (host; also uploaded on CUDA)
STAGE_SLACK_BYTES = 4096


def _readers(lop: Any) -> tuple[Any, Any]:
    """The REAL caller-provided readers of the lifecycle operator: host reader (CUDA: the lifecycle's own host reader, never the device one) and the device reader (CUDA only)."""
    if lop._cuda is not None:
        return lop._cuda.host_available_bytes_fn, lop.column.available_bytes_fn
    return lop.column.available_bytes_fn, None


def _admit_cold(cfg: Any, lop: Any, cold: list, stage: str, host: int, device: int = 0) -> None:
    """Fresh strict readings of the host (and, on CUDA, device) reader against the stage's own estimated NEW bytes plus the explicit reserve (host: `resources.caller_reserve_bytes`; device: `cuda.device_reserve_bytes`).
    Already-resident buffers (the held input files, earlier stages' results) are excluded: a fresh reading already accounts for them. Refuses BEFORE the stage's native call; records the figures."""
    from masy import leaf_source_profile as lsp

    host_fn, dev_fn = _readers(lop)
    h_res = int(cfg.resources["caller_reserve_bytes"])
    d_res = int(cfg.cuda["device_reserve_bytes"]) if (cfg.cuda is not None and dev_fn is not None) else 0
    try:
        h_free = lsp.read_free(host_fn, "host available_bytes_fn")
        d_free = lsp.read_free(dev_fn, "device available_bytes_fn") if dev_fn is not None and device > 0 else None
    except lsp.SourceProfileError as exc:
        raise LeafRunRefused("leaf_cold_reader", f"{exc.code}: {exc.detail}") from None
    rec = {"stage": stage, "host_need_bytes": int(host), "host_reserve_bytes": h_res, "host_free_bytes": h_free, "device_need_bytes": int(device), "device_reserve_bytes": d_res, "device_free_bytes": d_free}
    if int(host) + h_res > h_free:
        raise LeafRunRefused("leaf_cold_memory", f"cold stage {stage!r} needs an estimated {int(host)} B of new host bytes + reserve {h_res} B but only {h_free} B are reported; no native LEAF object was built for it")
    if d_free is not None and int(device) + d_res > d_free:
        raise LeafRunRefused("leaf_cold_memory", f"cold stage {stage!r} needs an estimated {int(device)} B of new device bytes + reserve {d_res} B but only {d_free} B are reported; no native LEAF object was built for it")
    cold.append(rec)


def _meta_counts(meta_bytes: bytes) -> tuple[int, int]:
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
        q, n = meta["n_plants"], meta["csv"]["n_bytes"]
    except Exception:
        raise LeafRunRefused("plant_metadata_invalid", "the companion metadata lacks integer n_plants / csv.n_bytes") from None
    if type(q) is not int or type(n) is not int or q < 0 or n < 0:
        raise LeafRunRefused("plant_metadata_invalid", "the companion metadata lacks non-negative integer n_plants / csv.n_bytes")
    return q, n


def _geometry_stage_bytes(cfg: Any, backend: str, n_all: int, n_live: int) -> tuple[int, int]:
    """(host, device) NEW bytes of the initial `rebuild_geometry`: LEAF's own materialize/geometry per-plant constants over ALL base records Q and the live rows, plus the DECLARED component caps (maps are
    bounded by `geometry.max_bytes + canopy_max_bytes`; those are caps, used here as the conservative size of the maps, never as an availability reading)."""
    from leaf import biocalendar as bc

    g = cfg.leaf["geometry"]
    cuda = backend == "cuda"
    mat = bc.MATERIALIZE_BYTES_PER_PLANT * n_all + 16 * n_live + (16 * n_live if cuda else 0)
    geo = (bc.ROW_BYTES_PER_PLANT * n_all + (16 * n_live if cuda else 0) + bc.GEOMETRY_BYTES_PER_PLANT * n_all + bc.VALIDATE_SCRATCH_BYTES_PER_PLANT * n_all + 16 * n_live)
    caps = int(g["max_bytes"]) + int(g["canopy_max_bytes"])
    host = max(mat, geo) + caps
    return host, (caps + 16 * n_live if cuda else 0)


def _table_bytes(growth: Any, n_species: int, backend: str) -> tuple[int, int]:
    items = growth.table_memory(max(int(n_species), 1), backend)["items"]
    return sum(i["bytes"] for i in items if i["where"] == "host"), sum(i["bytes"] for i in items if i["where"] == "device")


def _mask_stage_bytes(lop: Any) -> int:
    return 3 * int(lop.n_cells) + STAGE_SLACK_BYTES  # the flat contiguous host copy made here + the operator's own admitted owned copy and hash staging


def _plan_total(cfg: Any, lop: Any, n_species: int, q: int, csv_len: int, meta_len: int, resume: bool) -> tuple[int, int]:
    """Whole remaining cold plan (a conservative SUM of the stages' new bytes) used for the up-front admission."""
    from leaf import growth

    th, td = _table_bytes(growth, n_species, lop.backend)
    host, dev = th + _mask_stage_bytes(lop), td
    if not resume:
        gh, gd = _geometry_stage_bytes(cfg, lop.backend, q, q)
        host += (PARSE_BYTES_PER_CSV_BYTE * csv_len + PARSE_BYTES_PER_RECORD * q + METADATA_BYTES_PER_INPUT_BYTE * meta_len + 128 * q + gh + POPULATION_BYTES_PER_LIVE_PLANT * q + STAGE_SLACK_BYTES)
        dev += gd + (POPULATION_BYTES_PER_LIVE_PLANT * q if lop.backend == "cuda" else 0)
    return host, dev


def _host_mask(lop: Any) -> Any:
    """The KNOWN shared valid-cell mask: the retained HOST `PondTopology.active` of the water setup as a flat C-contiguous bool array (one copy), or None when every cell is active (None = whole domain valid is then
    exactly equivalent). The SAME mask feeds the initial canopy build and the interval operator (so every later growth rebuild and the resume operator use it too)."""
    import numpy as np

    act = np.asarray(lop.topology.active)
    if act.dtype != np.bool_ or act.shape != tuple(int(v) for v in lop.topology.shape):
        raise LeafRunRefused("mask_invalid", "the water topology's active mask must be a host bool array of the case shape")
    if bool(act.all()):
        return None
    return np.ascontiguousarray(act.reshape(-1))


def _catalog_table(cfg: Any, receipt: Any, catalog_bytes: bytes, lop: Any, dev: Any, cold: list, *, q: int, csv_len: int, meta_len: int, resume: bool) -> tuple[Any, Any]:
    # stage 0: BEFORE any native LEAF catalog / table / state / geometry object: the whole cold plan with the smallest possible S (S is unknown until the catalog is parsed)
    from leaf import growth, traits  # modules only (already census-verified and imported by the receipt); nothing is built

    h0, d0 = _plan_total(cfg, lop, 1, q, csv_len, meta_len, resume)
    _admit_cold(cfg, lop, cold, "cold_plan_lower_bound", h0 + CATALOG_BYTES_PER_INPUT_BYTE * len(catalog_bytes) + STAGE_SLACK_BYTES, d0)
    try:
        cat = traits.SpeciesCatalog(traits.parse_strict_json(catalog_bytes.decode("utf-8"), what="leaf.catalog"))
    except Exception as exc:  # the public LEAF catalog validation is the authority
        raise LeafRunRefused("catalog_invalid", f"{type(exc).__name__}: {exc}") from None
    if int(cat.root_layers) != int(lop.n_layers):
        raise LeafRunRefused("layer_mismatch", f"the catalog has {cat.root_layers} root layers but the water column has {int(lop.n_layers)}")
    # stage 1: the actual S is known: the remaining plan (table + mask [+ parse + geometry + population]) is admitted in full BEFORE the table is compiled
    h1, d1 = _plan_total(cfg, lop, cat.n_species, q, csv_len, meta_len, resume)
    _admit_cold(cfg, lop, cold, "cold_plan_remaining", h1, d1)
    backend = lop.backend
    try:
        table = growth.compile_species(cat, backend, int(cfg.leaf["budgets"]["leaf_budget_bytes"]), **({"device": int(dev.device_id)} if backend == "cuda" else {}))
    except Exception as exc:
        raise LeafRunRefused("species_table_refused", f"{type(exc).__name__}: {exc}") from None
    return cat, table


def _operator(cfg: Any, receipt: Any, lop: Any, cat: Any, table: Any, mask: Any, cold: list) -> Any:
    from masy import leaf_water_interval as lwi

    g = cfg.leaf["geometry"]
    opts = dict(max_edges=g["max_edges"], max_bytes=g["max_bytes"], canopy_max_edges=g["canopy_max_edges"], canopy_max_bytes=g["canopy_max_bytes"], builder=g["builder"])
    if mask is not None:
        opts["valid_cell_mask"] = mask
        _admit_cold(cfg, lop, cold, "valid_cell_mask", _mask_stage_bytes(lop))
    try:
        return lwi.prepare_interval_operator(receipt, lop, table=table, leaf_budget_bytes=int(cfg.leaf["budgets"]["leaf_budget_bytes"]), catalog=cat, geometry=opts)
    except lwi.IntervalError as exc:
        raise LeafRunRefused("leaf_operator_refused", str(exc)) from None


def reference_operator(cfg: Any, receipt: Any, catalog_bytes: bytes, lop: Any, dev: Any, *, cold: Any = None, plants_meta_bytes: bytes = b"{}") -> Any:
    """The resume path: catalog/table/operator over the REFERENCE lifecycle operator with the SAME shared valid-cell mask as the create path. The initial plants are NOT read (the accepted biology comes
    from the joint artifact). `cold` (optional caller-owned list) receives the cold admission records."""
    cold = [] if cold is None else cold
    cat, table = _catalog_table(cfg, receipt, catalog_bytes, lop, dev, cold, q=0, csv_len=0, meta_len=0, resume=True)
    return _operator(cfg, receipt, lop, cat, table, _host_mask(lop), cold)


def _initial_surface(lop: Any, st: Any) -> Any:
    """The ACTUAL initial native bed on the HOST for both backends: the retained `PondTopology.source_bed_m` (HELD routing-source ground elevation of the initial build, never a water level; no D2H, no
    second bed). Validated: generation 0, shape, float64, finite everywhere, and every unresolved anchor lies in a known ACTIVE cell. Returns a read-only (ny, nx) view; nothing is invented or filled."""
    import numpy as np

    topo = lop.topology
    z = topo.source_bed_m
    if int(topo.generation) != 0:
        raise LeafRunRefused("surface_source_invalid", "SURFACE anchors resolve only from the INITIAL native bed (topology generation 0)")
    if type(z) is not np.ndarray or z.dtype != np.float64 or z.shape != tuple(int(v) for v in topo.shape):
        raise LeafRunRefused("surface_source_invalid", "the retained initial bed must be a host float64 (ny, nx) array of the case shape")
    if not bool(np.isfinite(z).all()):
        raise LeafRunRefused("surface_source_invalid", "the retained initial bed holds non-finite cells; refusing (nothing is filled or invented)")
    m = np.asarray(st.establishment_unresolved)
    if bool(m.any()):
        row, col = st.domain.cell_indices(st.x_m[m], st.y_m[m])
        if not bool(np.asarray(topo.active)[row, col].all()):
            raise LeafRunRefused("surface_anchor_inactive", "an unresolved SURFACE anchor lies in an inactive (unknown-bed) cell")
    return z


def build_participant(cfg: Any, receipt: Any, catalog_bytes: bytes, csv_bytes: bytes, meta_bytes: bytes, scenario: Any, prep: Any, lop: Any, state: Any, dev: Any, *, cold: Any = None) -> SimpleNamespace:
    """The create path: catalog/table, the initial population through the PUBLIC loaders, explicit grid/layer/species/cell-area compatibility joins, SURFACE anchors resolved only from the ACTUAL native
    initial bed (host, both backends), geometry through `rebuild_geometry` (with the shared known valid-cell mask), the operator and the initial `LeafBoundary` at the initial boundary state with the calendar
    at the scenario start. Every native stage is preceded by an explicit cold HOST/DEVICE admission (records in `cold` / the result's `.cold`)."""
    import numpy as np
    from leaf import biocalendar as bc
    from leaf import state as ls

    from masy import leaf_calendar as lcal
    from masy import leaf_water_interval as lwi

    cold = [] if cold is None else cold
    q, _ = _meta_counts(meta_bytes)
    cat, table = _catalog_table(cfg, receipt, catalog_bytes, lop, dev, cold, q=q, csv_len=len(csv_bytes), meta_len=len(meta_bytes), resume=False)
    _admit_cold(cfg, lop, cold, "plant_parse", PARSE_BYTES_PER_CSV_BYTE * len(csv_bytes) + PARSE_BYTES_PER_RECORD * q + METADATA_BYTES_PER_INPUT_BYTE * len(meta_bytes) + 128 * q + STAGE_SLACK_BYTES)
    try:
        st = ls.parse_plant_csv(csv_bytes, cat, metadata=meta_bytes, source="leaf.plants")
    except Exception as exc:
        raise LeafRunRefused("plants_invalid", f"{type(exc).__name__}: {exc}") from None
    topo = lop.topology
    ny, nx = topo.shape
    dx = float(topo.dx_m)
    d = st.domain
    if (d.nx, d.ny) != (int(nx), int(ny)) or d.dx_m != dx or d.dy_m != dx or d.origin_x_m != 0.0 or d.origin_y_m != 0.0 or not d.square_cells:
        raise LeafRunRefused("grid_mismatch", f"the plant domain (nx={d.nx}, ny={d.ny}, dx={d.dx_m}, dy={d.dy_m}, origin=({d.origin_x_m}, {d.origin_y_m})) differs from the MAPLE case grid (nx={nx}, ny={ny}, dx={dx}, origin 0)")
    if abs(float(lop.column.cell_area_m2) - dx * dx) > 1e-12 * dx * dx:
        raise LeafRunRefused("cell_area_mismatch", "the water column's cell area differs from dx*dx of the case grid")
    if not st.establishment_resolved:
        surface = _initial_surface(lop, st)
        try:
            st.resolve_establishment(surface)  # the PUBLIC LEAF resolver over the ACTUAL retained initial host bed
        except Exception as exc:
            raise LeafRunRefused("surface_resolution_refused", f"{type(exc).__name__}: {exc}") from None
    rows = np.flatnonzero(np.asarray(st.status) == ls.LIVE).astype(np.int64)
    mask = _host_mask(lop)
    g = cfg.leaf["geometry"]
    gh, gd = _geometry_stage_bytes(cfg, lop.backend, int(st.n), int(len(rows)))
    _admit_cold(cfg, lop, cold, "geometry", gh, gd)
    canopy = {"max_edges": g["canopy_max_edges"], "max_bytes": g["canopy_max_bytes"], "builder": g["builder"]}
    if mask is not None:
        canopy["valid_cell_mask"] = mask
    try:
        geo = bc.rebuild_geometry(st, cat, st.agb_dry_kg[rows], st.seasonal_agb_dry_kg[rows], elapsed_days=0.0, max_edges=g["max_edges"], max_bytes=g["max_bytes"], backend=lop.backend if lop.backend != "numba" else "numba",
                                  builder=g["builder"], canopy=canopy, live_rows=rows)
    except Exception as exc:
        raise LeafRunRefused("geometry_refused", f"{type(exc).__name__}: {exc}") from None
    op = _operator(cfg, receipt, lop, cat, table, mask, cold)
    cal = lcal.LeafCalendar.new(receipt.module("leaf.biocalendar"), scenario.calendar.start, dt.timedelta(days=int(cfg.leaf["cadence_days"])))
    xp = lop.xp
    _admit_cold(cfg, lop, cold, "population", POPULATION_BYTES_PER_LIVE_PLANT * int(len(rows)) + STAGE_SLACK_BYTES, POPULATION_BYTES_PER_LIVE_PLANT * int(len(rows)) if lop.backend == "cuda" else 0)
    try:
        boundary = lwi.new_boundary(op, state, B_kg=xp.asarray(np.array(st.agb_dry_kg[rows])), E_kg=xp.asarray(np.array(st.seasonal_agb_dry_kg[rows])),
                                    species_index=xp.asarray(np.array(st.species_index[rows], dtype=np.int32)), live=xp.ones(len(rows), dtype=bool), canopy_area_m2=geo.canopy_area_m2, root_map=geo.root_map,
                                    canopy_map=geo.canopy_map, base_state=st, live_rows=rows, calendar=cal)
    except lwi.IntervalError as exc:
        raise LeafRunRefused("leaf_boundary_refused", str(exc)) from None
    cm = geo.canopy_map
    valid_area = None if (cm is None or lop.backend == "cuda") else float(np.sum(cm.view("area_total_m2")))  # zero-copy host view (CPU); the CUDA map is not exported for a summary scalar
    return SimpleNamespace(op=op, boundary=boundary, canopy_valid_area_m2=valid_area, n_base=int(st.n), n_live=int(len(rows)), n_species=len(cat.species_ids), catalog_hash=str(cat.hash), resolved=bool(st.establishment_resolved),
                           cold=cold, masked=mask is not None)


def leaf_limits(cfg: Any, cuda: bool) -> dict[str, int]:
    ck = cfg.leaf["budgets"]["checkpoint"]
    return {"max_host_bytes": int(ck["max_host_bytes"]), "max_device_bytes": int(ck["max_device_bytes"]) if cuda else 0, "max_disk_bytes": int(ck["max_disk_bytes"]), "max_biology_bytes": int(ck["max_biology_bytes"])}


def static_pins(cfg: Any) -> dict[str, str]:
    """Input pins the joint checkpoint/resume join adds for a v2 run: the three file hashes and ONE digest of the closed static LEAF settings (cadence, geometry caps, budgets, I/O caps, family)."""
    leaf = cfg.leaf
    settings = {"cadence_days": leaf["cadence_days"], "variant": leaf["leaf_variant"], "geometry": dict(leaf["geometry"]), "leaf_budget_bytes": leaf["budgets"]["leaf_budget_bytes"], "io": dict(leaf["io"])}
    return {"leaf/catalog": leaf["pins"]["catalog_sha256"], "leaf/plants": leaf["pins"]["plants_sha256"], "leaf/plants_metadata": leaf["pins"]["metadata_sha256"],
            "leaf/settings": hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()}


def biology_summary(session: Any) -> dict[str, Any]:
    """Bounded current biology/water summary (scalars only; no arrays, no history)."""
    lf = session.leaf
    st = session.state
    return {"participant": session.participant, "n_live": int(lf.B_kg.shape[0]), "n_base": int(lf.base_state.n), "generation": int(lf.generation), "cursor": lf.calendar.cursor.isoformat(),
            "pending_days": str(lf.calendar.pending_days), "season_due": bool(lf.calendar.schedule.season_due), "request_sequence": int(st.request_sequence), "n_t_requests": int(st.lifetime.n_t_requests),
            "transpiration_m3": float(st.lifetime.transpiration_m3), "requested_transpiration_m3": float(st.lifetime.requested_transpiration_m3)}
