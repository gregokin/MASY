"""Biology-ONLY checkpoint codec of the scheduled LEAF-water participant (Design.md section 6; CPU authored and run, CUDA chunked path authored only).

The SYRUP physical V2 codec owns ALL water (soil, bed, ledgers, request lifetime): nothing of it is stored here. This codec stores ONLY the LEAF side of an accepted `LeafBoundary`:

  biology.json   canonical scalars: version, source profile + LEAF base/delta/package identity, catalog identity, species, P/S, live rows, generation, the exact `biocalendar.Schedule`
                 (cursor, grid, deferred season, pending ENABLED credit as an exact Fraction), the water continuity MARK and ANCHOR (request-sequence anchor + delta: the biology request count since the
                 LEAF start, NOT the lifetime request total), the exact elapsed days of the last geometry rebuild, the digests of the root-map CSR / canopy CSR / canopy areas, and the sha256 of the
                 two payload members;
  base_plants.csv  the canonical LEAF plant CSV (`leaf.state.plant_csv_text`) of the BASE state the schedule anchor stands at (a rebuild needs it);
  arrays.bin     little-endian float64: B, E, D (period sum), U (period sum) of the P live plants, then the S-sized A and T integrals.

Restoration NEVER uses `leaf.checkpoint.save_session/load_session` (they duplicate native water), never `make_head` and never replays: it parses the canonical CSV with LEAF's own parser, validates
every digest, REBUILDS the root map / canopy map / canopy areas with LEAF's pure `rebuild_geometry` at the recorded elapsed days and requires their digests to equal the stored ones, rebuilds the
calendar through `leaf.biocalendar.Schedule` (which re-validates it) and joins the water continuity mark of the restored lifecycle state; any inequality is a typed `BiologyCodecError` before a boundary
exists. Bounds (member count, bytes, P, S) are derived and checked BEFORE any payload is parsed.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from fractions import Fraction
from typing import Any

import numpy as np

from masy import leaf_water_interval as lwi

__all__ = ["VERSION", "MEMBERS", "STAGING_NOTE", "BiologyCodecError", "encode", "decode", "member_limits", "root_digest", "canopy_digest", "required_encode_bytes", "required_decode_bytes"]

VERSION = "masy.leaf_biology.v2"  # v2 (unreleased; v1 was never a published artifact): adds the base-record count `n_records` (Q) so the CSV / parsed-state bounds use ALL base records, not the live count P
MEMBERS = ("biology.json", "base_plants.csv", "arrays.bin")  # the CLOSED member set of `biology/` (3 regular files)
_LE = np.dtype("<f8")
MAX_JSON_BYTES = 1 << 20
STAGING_NOTE = ("v1 stages the P/S payload and the root/canopy CSR host export in full host memory (declared and admitted by `required_encode_bytes` / `required_decode_bytes` BEFORE the allocation; "
                "no chunked or streamed D2H is claimed). A device population must be copied to host by the caller within this admission; a chunked CUDA sibling is future work.")
_PAYLOAD_NAMES = ("arrays.bin", "base_plants.csv")


class BiologyCodecError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def _refuse(code: str, detail: str) -> BiologyCodecError:
    return BiologyCodecError(code, detail)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canon(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _hash_arrays(arrs: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(k.encode("ascii") + b"|" + str(a.dtype).encode("ascii") + b"|" + str(a.shape).encode("ascii") + b"|")
        h.update(memoryview(a).cast("B") if a.size else b"")  # streamed from the array buffer: no `.tobytes()` copy
    return h.hexdigest()


def root_digest(root_map: Any) -> str:
    """Digest of the actual root-map CSR arrays (`RootMap.host_arrays`)."""
    return _hash_arrays(root_map.host_arrays())


def canopy_digest(canopy_map: Any) -> str:
    if canopy_map is None:
        return "none"
    return _hash_arrays(canopy_map.host_arrays())


def member_limits(n_plants: int, n_species: int, n_records: int | None = None) -> dict[str, int]:
    """Derived (never default) byte caps of the closed member set. P = LIVE plants (arrays member exactly (4P + 2S) * 8 B, sparse root/canopy payload), S species, Q = ALL base-state records, live AND non-live
    (legal initial LEAF data; the CSV is bounded per RECORD: 4096 + 512(Q + 8) B). Q defaults to P only for a population without non-live records; Q >= P is required."""
    q = n_plants if n_records is None else n_records
    for name, v in (("n_plants", n_plants), ("n_species", n_species), ("n_records", q)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise _refuse("bad_bounds", f"{name} must be a non-Boolean integer >= 0")
    if q < n_plants:
        raise _refuse("bad_bounds", "n_records (all base records) cannot be smaller than the live count")
    return {"biology.json": MAX_JSON_BYTES, "arrays.bin": (4 * n_plants + 2 * n_species) * 8, "base_plants.csv": 4096 + 512 * (q + 8), "members": len(MEMBERS)}


def _host(x: Any) -> np.ndarray:
    get = getattr(x, "get", None)
    return np.asarray(get() if get is not None else x, dtype=np.float64)


def _iso(t: _dt.datetime) -> str:
    return t.isoformat()


def required_encode_bytes(op: Any, leaf: Any) -> dict[str, int]:
    """Peak FRESH bytes of `encode`, HOST and DEVICE kept separate (pure arithmetic; see STAGING_NOTE). The device side allocates nothing (a device population is copied to host, one array at a time).
    Host peak = the `arrays.bin` build buffer (8(4P+2S)) + its single `.tobytes()` member copy (same size, both alive at the copy) + on CUDA one transient device->host copy (8P) + the CSV (bounded per plant)
    + the JSON cap + the full host export of the root-map CSR (`resident_bytes`) AND of the canopy CSR (`export_bytes`) that the digests read (hashing streams from the buffers: no `.tobytes()`)."""
    P, S, Q = int(leaf.B_kg.shape[0]), op.n_species, int(leaf.base_state.n)
    lim = member_limits(P, S, Q)
    arrays = 8 * (4 * P + 2 * S)
    export = int(leaf.root_map.resident_bytes) + (0 if leaf.canopy_map is None else int(leaf.canopy_map.export_bytes))
    host = 2 * arrays + (8 * P if op.cuda else 0) + 2 * lim["base_plants.csv"] + lim["biology.json"] + export  # CSV text + its encoded bytes are alive together
    return {"host": host, "device": 0}


def required_decode_bytes(op: Any, n_plants: int, n_species: int, member_bytes: int, n_records: int | None = None) -> dict[str, int]:
    """Peak FRESH bytes of `decode` before any rebuild result exists, HOST and DEVICE separate. Host: the retained members, the six float arrays (8(4P+2S)) and their `frombuffer`-converted copies, and the
    parsed CSV state (<= its derived member bound). The declared geometry-rebuild cap `geometry.max_bytes` is charged to the compute side (device on CUDA; host on CPU), where `rebuild_geometry` itself
    admits against it."""
    q = n_plants if n_records is None else n_records
    lim = member_limits(n_plants, n_species, q)
    arrays = 8 * (4 * n_plants + 2 * n_species)
    parsed = op.leaf["biocalendar"].MATERIALIZE_BYTES_PER_PLANT * q  # the parsed PlantState (all Q records) + the decoded CSV text
    host = int(member_bytes) + 2 * arrays + lim["base_plants.csv"] + parsed
    cap = int(op.geometry["max_bytes"]) + int(op.geometry["canopy_max_bytes"])
    return {"host": host + (0 if op.cuda else cap), "device": cap if op.cuda else 0}


def _admit_codec(op: Any, need: dict[str, int], stage: str) -> None:
    """Strict fresh admissions BEFORE the allocation: the HOST figure against the host reader and (CUDA) the DEVICE figure against the device reader; on CPU everything is host."""
    if op.cuda:
        lwi._admit_host(op, need["host"], stage + "_host")
        if need["device"]:
            lwi._admit(op, need["device"], stage + "_device")
    else:
        lwi._admit(op, need["host"] + need["device"], stage)


def _check_variant_d4fe(op: Any) -> None:
    if getattr(op.receipt, "variant", "d4fe") != "d4fe":
        raise _refuse("variant", "this codec (v2) serves only the d4fe LEAF family; the 8da family uses leaf_biology_codec_v3")


def encode(op: Any, leaf: Any) -> dict[str, bytes]:
    """Serialise an accepted `LeafBoundary` (see the module docstring). Pure: nothing is mutated; returns {member name: bytes} for the closed member set."""
    lwi._check_op(op)
    _check_variant_d4fe(op)
    if type(leaf) is not lwi.LeafBoundary or leaf not in lwi._MINTED:
        raise _refuse("bad_boundary", "leaf must be a genuine LeafBoundary")
    st_mod = op.leaf["state"]
    P, S, Q = int(leaf.B_kg.shape[0]), op.n_species, int(leaf.base_state.n)
    _admit_codec(op, required_encode_bytes(op, leaf), "biology_encode")  # fresh strict readings BEFORE the first allocation
    buf = np.empty(4 * P + 2 * S, dtype=_LE)
    for k, src in enumerate((leaf.B_kg, leaf.E_kg, leaf.D_sum, leaf.U_sum)):
        buf[k * P:(k + 1) * P] = _host(src)  # one transient host copy at a time on CUDA
    buf[4 * P:4 * P + S] = np.asarray(leaf.A_int, dtype=np.float64)
    buf[4 * P + S:] = np.asarray(leaf.T_int, dtype=np.float64)
    raw = buf.tobytes()
    del buf
    if len(raw) != member_limits(P, S, Q)["arrays.bin"]:
        raise _refuse("internal", "arrays member size differs from its derived size")
    csv = st_mod.plant_csv_text(leaf.base_state, initial=False).encode("utf-8")
    if len(csv) > member_limits(P, S, Q)["base_plants.csv"]:  # the encoded size must fit the derived bound (a too-large result is refused, never written)
        raise _refuse("member_limit", f"base_plants.csv would be {len(csv)} B > its derived {member_limits(P, S, Q)['base_plants.csv']} B for Q={Q} records")
    sch = leaf.calendar.schedule
    rm, cm = leaf.root_map, leaf.canopy_map
    doc = {
        "version": VERSION, "profile": op.receipt.profile, "leaf_package_digest": op.receipt.digests["leaf"], "syrup_package_digest": op.receipt.digests["syrup"],
        "catalog_hash": str(leaf.base_state.catalog_hash), "species": list(leaf.base_state.species_names), "n_plants": P, "n_records": Q, "n_species": S, "n_layers": int(op.n_layers), "n_cells": int(op.n_cells),
        "live_rows": [int(x) for x in np.asarray(leaf.live_rows)], "generation": int(leaf.generation),
        "schedule": sch.to_dict(), "geo_elapsed_days": str(Fraction(leaf.geo_elapsed)),
        "water_digest": leaf.water_digest, "mark": list(leaf.mark), "anchor": list(leaf.anchor), "biology_requests_since_anchor": int(leaf.mark[1]) - int(leaf.anchor[1]),
        "root_digest": root_digest(rm), "root_edges": int(rm.n_edges), "canopy_digest": canopy_digest(cm), "area_digest": _hash_arrays({"area": _host(leaf.canopy_area_m2)}),
        "members": {"base_plants.csv": {"sha256": _sha(csv), "bytes": len(csv)}, "arrays.bin": {"sha256": _sha(raw), "bytes": len(raw)}},
    }
    js = _canon(doc)
    if len(js) > MAX_JSON_BYTES:
        raise _refuse("json_limit", f"biology.json would be {len(js)} B > {MAX_JSON_BYTES}")
    return {"biology.json": js, "base_plants.csv": csv, "arrays.bin": raw}


def _check_restored_values(op: Any, doc: dict, B: Any, E: Any, D: Any, U: Any, A: Any, T: Any) -> None:
    """Malformed-state checks under the ACTUAL LEAF limits (no new biology law): finite >= 0 B/E/D/U/A/T, E <= B, U <= D(1 + rtol), T <= A EXACT, A <= the weekly cap, a non-negative non-Boolean
    integer generation and counts, and the anchor/mark request counts (anchor <= mark, both non-negative ints)."""
    gr = op.leaf["growth"]
    for name, a in (("B", B), ("E", E), ("D", D), ("U", U), ("A", A), ("T", T)):
        if a.size and (not bool(np.all(np.isfinite(a))) or float(a.min()) < 0.0):
            raise _refuse("arrays", f"{name} must be finite and >= 0")
    if B.size and bool(np.any(E > B)):
        raise _refuse("arrays", "seasonal biomass E exceeds B")
    rtol = float(getattr(gr, "OVERDELIVERY_RTOL", 0.0))
    if D.size and bool(np.any(U > D * (1.0 + rtol))):
        raise _refuse("arrays", "delivered U exceeds requested D beyond LEAF's over-delivery tolerance")
    if A.size and (bool(np.any(T > A)) or float(A.max()) > gr.MAX_ACTIVITY_INTEGRAL * (1.0 + gr.ACTIVITY_INTEGRAL_SLACK)):
        raise _refuse("arrays", "T > A or A above the weekly activity cap (LEAF's actual integral limits)")
    g = doc["generation"]
    if isinstance(g, bool) or not isinstance(g, int) or g < 0:
        raise _refuse("generation", "generation must be a non-negative integer")
    for name in ("mark", "anchor"):
        v = doc[name]
        if (type(v) is not list or len(v) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in v[:3]) or type(v[3]) is not str):
            raise _refuse("anchor", f"{name} must be [request_sequence, n_requests, reset_count, lop_sha] with non-negative integers")
    if doc["anchor"][1] > doc["mark"][1] or isinstance(doc["biology_requests_since_anchor"], bool) or not isinstance(doc["biology_requests_since_anchor"], int) or doc["biology_requests_since_anchor"] < 0:
        raise _refuse("anchor", "request anchor must not exceed the recorded count and delta must be a non-negative integer")


def _parse_schedule(bc: Any, d: Any) -> Any:
    keys = {"version", "anchor", "cursor", "next_boundary", "cadence_seconds_exact", "season_due", "pending_days"}
    if type(d) is not dict or set(d) != keys:
        raise _refuse("bad_schedule", f"schedule requires exactly {sorted(keys)}")
    try:
        parse = _dt.datetime.fromisoformat
        sch = bc.Schedule(parse(d["anchor"]), _dt.timedelta(seconds=int(d["cadence_seconds_exact"])), parse(d["cursor"]), parse(d["next_boundary"]), d["season_due"], Fraction(d["pending_days"]))
    except Exception as exc:  # LEAF re-validates every invariant in Schedule.__post_init__
        raise _refuse("bad_schedule", f"{type(exc).__name__}: {exc}") from None
    if sch.to_dict() != d:
        raise _refuse("bad_schedule", "the schedule does not round-trip exactly")
    return sch


def decode(op: Any, members: dict[str, bytes], water: Any, *, calendar_cls: Any, expected_profile: str | None = None, expected_cursor: Any = None) -> Any:
    """Rebuild an accepted `LeafBoundary` for the restored lifecycle state `water` (role boundary or completed storm) from the closed members; see the module docstring for every check. Nothing is adopted
    or mutated; the result is returned only when ALL identities/digests/joins hold."""
    lwi._check_op(op)
    _check_variant_d4fe(op)
    if type(members) is not dict or set(members) != set(MEMBERS) or any(type(v) is not bytes for v in members.values()):
        raise _refuse("bad_members", f"members must be exactly {MEMBERS} as bytes")
    if len(members["biology.json"]) > MAX_JSON_BYTES:
        raise _refuse("json_limit", "biology.json exceeds its cap")
    try:
        doc = json.loads(members["biology.json"].decode("ascii"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except Exception as exc:
        raise _refuse("bad_json", f"{type(exc).__name__}: {exc}") from None
    need = {"version", "profile", "leaf_package_digest", "syrup_package_digest", "catalog_hash", "species", "n_plants", "n_records", "n_species", "n_layers", "n_cells", "live_rows", "generation", "schedule",
            "geo_elapsed_days", "water_digest", "mark", "anchor", "biology_requests_since_anchor", "root_digest", "root_edges", "canopy_digest", "area_digest", "members"}
    if type(doc) is not dict or set(doc) != need or _canon(doc) != members["biology.json"]:
        raise _refuse("bad_json", "biology.json must be the canonical closed record")
    if doc["version"] != VERSION:
        raise _refuse("version", f"unsupported biology version {doc['version']!r}")
    if doc["profile"] != op.receipt.profile or (expected_profile is not None and doc["profile"] != expected_profile):
        raise _refuse("profile", f"the artifact was written under profile {doc['profile']!r}, the live source profile is {op.receipt.profile!r}")
    if doc["leaf_package_digest"] != op.receipt.digests["leaf"] or doc["syrup_package_digest"] != op.receipt.digests["syrup"]:
        raise _refuse("source", "LEAF/SYRUP package digests differ from the live verified sources")
    P, S, Q = doc["n_plants"], doc["n_species"], doc["n_records"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (P, S, Q)) or Q < P or S != op.n_species or doc["n_layers"] != op.n_layers or doc["n_cells"] != op.n_cells:
        raise _refuse("shape", "species/layer/cell counts differ from the live operator")
    lim = member_limits(P, S, Q)
    rec = doc["members"]
    if type(rec) is not dict or set(rec) != set(_PAYLOAD_NAMES):
        raise _refuse("members_record", f"members must record exactly {_PAYLOAD_NAMES}")
    for name in _PAYLOAD_NAMES:
        meta = rec[name]
        if (type(meta) is not dict or set(meta) != {"sha256", "bytes"} or type(meta["bytes"]) is not int or type(meta["sha256"]) is not str or len(members[name]) != meta["bytes"]
                or _sha(members[name]) != meta["sha256"]):
            raise _refuse("member_digest", f"{name} differs from its recorded size/sha256")
    if len(members["arrays.bin"]) != lim["arrays.bin"] or len(members["base_plants.csv"]) > lim["base_plants.csv"]:  # pure length checks BEFORE the admission, text or parse
        raise _refuse("member_limit", "a payload member exceeds its derived size")
    _admit_codec(op, required_decode_bytes(op, P, S, sum(len(v) for v in members.values()), Q), "biology_decode")  # BEFORE the float arrays / parsed state / rebuild exist
    bc, st_mod = op.leaf["biocalendar"], op.leaf["state"]
    cat = op.catalog
    if str(doc["catalog_hash"]) != str(cat.hash) or list(doc["species"]) != list(op.table.species_ids):
        raise _refuse("catalog", "catalog identity or species order differs from the live catalog/table")
    try:
        base = st_mod.parse_plant_csv(members["base_plants.csv"].decode("utf-8"), cat)
    except Exception as exc:  # the LEAF parser validates the canonical format and the catalog
        raise _refuse("base_plants", f"{type(exc).__name__}: {exc}") from None
    if int(base.n) != Q:
        raise _refuse("n_records", f"the base CSV holds {int(base.n)} records but the closed metadata declares Q={Q}")
    if str(base.catalog_hash) != doc["catalog_hash"]:
        raise _refuse("catalog", "the base plant CSV was written for another catalog")
    rows = np.asarray(doc["live_rows"], dtype=np.int64)
    if rows.shape != (P,) or (P and (int(rows.min()) < 0 or int(rows.max()) >= base.n)) or not np.array_equal(np.flatnonzero(np.asarray(base.status) == st_mod.LIVE), rows):
        raise _refuse("live_rows", "live_rows do not match the live plants of the base CSV")
    raw = np.frombuffer(members["arrays.bin"], dtype=_LE)
    B, E, D, U = (raw[i * P:(i + 1) * P].astype(np.float64) for i in range(4))
    A, T = raw[4 * P:4 * P + S].astype(np.float64), raw[4 * P + S:4 * P + 2 * S].astype(np.float64)
    if not bool(np.all(np.isfinite(raw))):
        raise _refuse("arrays", "non-finite value in arrays.bin")
    _check_restored_values(op, doc, B, E, D, U, A, T)
    sch = _parse_schedule(bc, doc["schedule"])
    cal = calendar_cls.from_schedule(bc, sch)
    if expected_cursor is not None and sch.cursor != expected_cursor:
        raise _refuse("clock_join", f"the stored LEAF cursor {sch.cursor.isoformat()} != the scheduler's accepted instant {expected_cursor.isoformat()}")
    try:
        geo_el = Fraction(doc["geo_elapsed_days"])
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise _refuse("geo_elapsed", f"malformed geometry elapsed days: {exc}") from None
    if geo_el < 0 or geo_el > bc.elapsed_days(sch.anchor, sch.cursor):
        raise _refuse("geo_elapsed", "the geometry elapsed days must lie in [0, elapsed(anchor, cursor)] (no future geometry)")
    try:
        rb = bc.rebuild_geometry(base, cat, B, E, elapsed_days=float(geo_el), max_edges=op.geometry["max_edges"], max_bytes=op.geometry["max_bytes"], backend=op.leaf_backend,
                                 builder=op.geometry["builder"], live_rows=rows,
                                 canopy={k: v for k, v in (("max_edges", op.geometry["canopy_max_edges"]), ("max_bytes", op.geometry["canopy_max_bytes"]), ("builder", op.geometry["builder"]),
                                                          ("valid_cell_mask", op.geometry["valid_cell_mask"])) if v is not None})
    except Exception as exc:
        raise _refuse("rebuild", f"{type(exc).__name__}: {exc}") from None
    if root_digest(rb.root_map) != doc["root_digest"] or int(rb.root_map.n_edges) != doc["root_edges"]:
        raise _refuse("root_map", "the rebuilt root-map CSR differs from the stored digest (catalog/geometry/biomass identity mismatch)")
    if canopy_digest(rb.canopy_map) != doc["canopy_digest"] or _hash_arrays({"area": _host(rb.canopy_area_m2)}) != doc["area_digest"]:
        raise _refuse("canopy_map", "the rebuilt canopy CSR/areas differ from the stored digests")
    mark = tuple(doc["mark"])
    if str(doc["water_digest"]) != str(water.digest):
        raise _refuse("water_join", "the restored lifecycle state's digest differs from the state the biology was released at (a different water lifetime/branch)")
    if mark != lwi._mark_of(op, water):
        raise _refuse("water_join", f"the restored lifecycle state's continuity mark {lwi._mark_of(op, water)} != the stored biology mark {mark}")
    anchor = tuple(doc["anchor"])
    if len(mark) != 4 or len(anchor) != 4 or int(mark[1]) - int(anchor[1]) != doc["biology_requests_since_anchor"] or anchor[3] != mark[3]:
        raise _refuse("anchor", "request anchor + delta do not reproduce the recorded mark")
    xp = op.lop.xp
    live = np.ones(P, dtype=bool)
    sp = np.array(base.species_index[rows], dtype=np.int32)
    area = np.array(_host(rb.canopy_area_m2))
    leaf = lwi._boundary_from_restore(op, B_kg=B, E_kg=E, species_index=sp, live=live, canopy_area_m2=area, root_map=rb.root_map, canopy_map=rb.canopy_map, base_state=base, live_rows=rows,
                                      D_sum=xp.asarray(D), U_sum=xp.asarray(U), A_int=A, T_int=T, calendar=cal, generation=int(doc["generation"]), mark=mark, anchor=anchor, geo_elapsed=geo_el, water_digest=str(doc["water_digest"]))
    return leaf
