"""Biology-ONLY checkpoint codec v3 (`masy.leaf_biology.v3`) for the 8da LEAF family: the PUBLIC plain plant CSV plus its companion metadata (`leaf.state.plant_files_text` /
`parse_plant_csv(metadata=...)`), a DISTINCT schema from the unreleased d4fe codec `leaf_biology_codec` (v2), which refuses an 8da operator (and this module refuses a d4fe one).

The SYRUP physical V2 codec owns ALL water; this codec stores nothing of the lifecycle state except two join tokens (the state digest and the request continuity mark). Never
`leaf.checkpoint.save_session` (it duplicates water). Closed members of `biology/` (4 regular files):

  base_plants.csv            plain CSV of ALL Q base records (live AND non-live, legal initial data) from `plant_files_text`
  base_plants.metadata.json  the companion metadata of the pair (LEAF verifies size, sha256, row count, catalog, domain, time, initial/SURFACE flags on parse)
  arrays.bin                 little-endian float64: B,E,D,U of the P live plants (compute side) then the HOST species integrals A,T (S each)
  biology.json               canonical closed record (schedule with the exact pending Fraction, generation, geometry elapsed, water join tokens, request anchor+delta, root/canopy/area digests,
                             live-row digest, member size+sha256 records). Live rows are NOT listed (a P-scaled JSON list could outgrow any fixed cap): they are derived from the CSV status
                             column on load and joined by digest.

Bounds are derived, never defaulted: CSV from Q and the longest species name (worst-case CSV quoting), metadata by LEAF's own `plant_metadata_size` (a streaming conservative bound; at decode time of a
template state over the live catalog, because the companion size does not depend on Q), arrays exactly 8(4P+2S), JSON from the species names. Staging is honest FULL HOST memory (declared and admitted
before the allocation; no chunked/streamed claim); the arrays member is ONE preallocated `bytearray` (no `tobytes()` copy). No unresolved SURFACE anchor is ever given an invented elevation: the
`initial` flag round-trips through LEAF's own metadata or the pair is refused.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from fractions import Fraction
from typing import Any

import numpy as np

from masy import leaf_biology_codec as v2
from masy import leaf_water_interval as lwi

__all__ = ["VERSION", "MEMBERS", "BiologyCodecError", "BiologyMembers", "encode", "decode", "member_limits", "metadata_cap", "required_encode_bytes", "required_decode_bytes"]

VERSION = "masy.leaf_biology.v3"
MEMBERS = ("base_plants.csv", "base_plants.metadata.json", "arrays.bin", "biology.json")  # the CLOSED member set of `biology/` (4 regular files)
_PAYLOAD = ("base_plants.csv", "base_plants.metadata.json", "arrays.bin")
_LE = np.dtype("<f8")
BiologyCodecError = v2.BiologyCodecError
_refuse = v2._refuse
_sha = v2._sha
_canon = v2._canon
PROVENANCE_KEY = "writer"


class BiologyMembers(dict):
    """{member name: bytes-like} of the closed set (`arrays.bin` is a bytearray)."""


def _check_variant(op: Any) -> None:
    if getattr(op.receipt, "variant", None) != "8da":
        raise _refuse("variant", "codec v3 serves only the 8da LEAF family (the d4fe codec is leaf_biology_codec)")


def _provenance(op: Any) -> dict[str, str]:
    return {PROVENANCE_KEY: VERSION, "profile": op.receipt.profile, "leaf_package_digest": op.receipt.digests["leaf"]}


def _max_name(names: Any) -> int:
    return max([len(str(n)) for n in names] or [0])


def member_limits(n_base: int, n_live: int, n_species: int, max_name: int, meta_cap: int, json_cap: int | None = None) -> dict[str, int]:
    """Derived byte caps. CSV: header + Q rows, each row = id (<=21) + species name (worst case every char quoted/doubled: 2n+2) + six floats (<=25 each: x, y, anchor, B, E, age) + status (<=13) + 9
    separators. Arrays: exactly 8(4P+2S). Metadata: `meta_cap` (LEAF's size API). JSON: `json_cap` (default derived from the species table)."""
    for name, v in (("n_base", n_base), ("n_live", n_live), ("n_species", n_species), ("max_name", max_name), ("meta_cap", meta_cap)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise _refuse("bad_bounds", f"{name} must be a non-Boolean integer >= 0")
    if n_live > n_base:
        raise _refuse("bad_bounds", "n_live cannot exceed n_base (all base records)")
    row = 21 + (2 * max_name + 2) + 6 * 25 + 13 + 9
    return {"base_plants.csv": 256 + n_base * row, "base_plants.metadata.json": int(meta_cap), "arrays.bin": 8 * (4 * n_live + 2 * n_species),
            "biology.json": int(json_cap) if json_cap is not None else 4096 + n_species * (2 * max_name + 8) + 1024, "members": len(MEMBERS)}


_META_CAP: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()


def metadata_cap(op: Any) -> int:
    """THE ONE companion-metadata cap (used by the plan, encode, decode, the member reads and the retained-member bound; there is no extra accepted slack anywhere). It is LEAF's own conservative size
    API (`plant_metadata_size`, a streaming count) over an ACTUAL empty `PlantState` of the live catalog (the companion text does not depend on Q) + 1024 B for the domain numbers. Computing it performs one
    small actual-LEAF pass (zero-length arrays and S-sized containers): that cost is EXPLICITLY admitted (4096 + 1024*S host bytes, strict fresh reading) the first time, and the scalar is cached per
    operator, so later planning calls are pure arithmetic. No fake template class: the public `PlantState.empty` / `plant_metadata_size`."""
    _check_variant(op)  # an old-family (d4fe) operator is refused BEFORE any probe of the 8da-only metadata API (`plant_metadata_size`) or any allocation
    cached = _META_CAP.get(op)
    if cached is not None:
        return cached
    v2._admit_codec(op, {"host": 4096 + 1024 * int(op.n_species), "device": 0}, "biology_metadata_planning")
    st = op.leaf["state"]
    tmpl = st.PlantState.empty(op.catalog, st.Domain(1, 1, 1.0), 0.0)
    cap = int(st.plant_metadata_size(tmpl, initial=True, provenance=_provenance(op))) + 1024
    _META_CAP[op] = cap
    return cap


def required_encode_bytes(op: Any, leaf: Any) -> dict[str, int]:
    """Peak FRESH bytes of `encode`, HOST and DEVICE separate (pure arithmetic once `metadata_cap` is cached; the FIRST call per operator performs the explicitly admitted actual empty-`PlantState`/S-sized LEAF pass, it is not literally array-free): the ONE arrays buffer (8(4P+2S); no second copy), the CSV text AND its encoded bytes alive together, the metadata text and
    bytes, the JSON cap, on CUDA one transient device->host copy (8P), and the full host export of the root-map CSR (`resident_bytes`) AND the canopy CSR (`export_bytes`). Device: nothing."""
    _check_variant(op)
    P, S, Q = int(leaf.B_kg.shape[0]), op.n_species, int(leaf.base_state.n)
    lim = member_limits(Q, P, S, _max_name(op.catalog.species_ids), metadata_cap(op))
    export = int(leaf.root_map.resident_bytes) + (0 if leaf.canopy_map is None else int(leaf.canopy_map.export_bytes))
    host = lim["arrays.bin"] + (8 * P if op.cuda else 0) + 2 * lim["base_plants.csv"] + 2 * lim["base_plants.metadata.json"] + lim["biology.json"] + export
    return {"host": host, "device": 0}


def required_decode_bytes(op: Any, n_base: int, n_live: int, n_species: int, member_bytes: int, *, members_resident: bool = False) -> dict[str, int]:
    """CONSERVATIVE SUM (an upper bound, not the exact allocation peak) of the fresh bytes of `decode`, HOST and DEVICE separate; on CPU "device" is empty and everything is host.
    HOST: the retained members (`member_bytes`; charged ONLY by the whole-call PLAN, i.e. `members_resident=False`; the decode itself runs with the members already allocated, so its own admission
    passes `members_resident=True`: a fresh free reading already excludes them, no double charge), the host float arrays B/E/D/U/A/T (8(4P+2S)), the decoded CSV text, the metadata text + parsed
    dict (2 x its cap), the parsed all-Q `PlantState` (`MATERIALIZE_BYTES_PER_PLANT * Q`), the transient host exports of the rebuilt root/canopy CSR for the digests (<= their declared caps) and, on CUDA, the
    transient host copy of the area (8P) and the int32 species vector (4P) before upload. COMPUTE side (device on CUDA; host on CPU): the rebuilt maps (<= the declared caps) and the population
    B/E/D/U (8P each) + species int32 + live bool + the rebuilt area (8P). EXPLICIT CONSERVATISM: on CPU `xp.asarray` ALIASES the host float arrays B/E/D/U, so the population term double-reserves
    them beside the host staging; that is a safe over-reservation by design (never an underflow), not a statement of the actual allocation peak. This is distinct from the forbidden double charge of the
    retained members, which `members_resident` removes."""
    lim = member_limits(n_base, n_live, n_species, _max_name(op.catalog.species_ids), metadata_cap(op))
    parsed = op.leaf["biocalendar"].MATERIALIZE_BYTES_PER_PLANT * n_base
    cap = int(op.geometry["max_bytes"]) + int(op.geometry["canopy_max_bytes"])
    pop = 8 * 4 * n_live + 4 * n_live + n_live + 8 * n_live  # B/E/D/U (8P each) + species int32 + live bool + the rebuilt area
    host = (0 if members_resident else int(member_bytes)) + 8 * (4 * n_live + 2 * n_species) + lim["base_plants.csv"] + 2 * lim["base_plants.metadata.json"] + parsed + cap  # + CSR export staging
    if op.cuda:
        return {"host": host + 8 * n_live + 4 * n_live, "device": cap + pop}
    return {"host": host + cap + pop, "device": 0}  # CPU: maps + population live on the host beside the staging


def encode(op: Any, leaf: Any) -> BiologyMembers:
    """Serialise an accepted `LeafBoundary`. Pure; every cap is checked before the text/array exists and again on the real lengths before the members are returned."""
    lwi._check_op(op)
    _check_variant(op)
    if type(leaf) is not lwi.LeafBoundary or leaf not in lwi._MINTED:
        raise _refuse("bad_boundary", "leaf must be a genuine LeafBoundary")
    st_mod = op.leaf["state"]
    base = leaf.base_state
    P, S, Q = int(leaf.B_kg.shape[0]), op.n_species, int(base.n)
    names_max = _max_name(op.catalog.species_ids)
    mcap = metadata_cap(op)
    lim = member_limits(Q, P, S, names_max, mcap)
    v2._admit_codec(op, required_encode_bytes(op, leaf), "biology_encode")  # fresh strict host (and device) readings BEFORE the first allocation
    initial = not bool(base.establishment_resolved)  # an unresolved SURFACE anchor is carried by LEAF's own initial flag, never given an elevation
    prov = _provenance(op)
    declared_meta = int(st_mod.plant_metadata_size(base, initial=initial, provenance=prov))  # LEAF's streaming bound, BEFORE serialising
    if declared_meta > lim["base_plants.metadata.json"]:
        raise _refuse("member_limit", f"the companion metadata would need up to {declared_meta} B > the derived cap {lim['base_plants.metadata.json']} B")
    buf = bytearray(8 * (4 * P + 2 * S))
    view = np.frombuffer(buf, dtype=_LE)
    for k, src in enumerate((leaf.B_kg, leaf.E_kg, leaf.D_sum, leaf.U_sum)):
        view[k * P:(k + 1) * P] = v2._host(src)  # one transient host copy at a time on CUDA
    view[4 * P:4 * P + S] = np.asarray(leaf.A_int, dtype=np.float64)
    view[4 * P + S:] = np.asarray(leaf.T_int, dtype=np.float64)
    del view
    csv_text, meta_text = st_mod.plant_files_text(base, initial=initial, provenance=prov)
    csv, meta = csv_text.encode("utf-8"), meta_text.encode("ascii")
    del csv_text, meta_text
    if len(csv) > lim["base_plants.csv"] or len(meta) > lim["base_plants.metadata.json"] or len(buf) != lim["arrays.bin"]:
        raise _refuse("member_limit", f"encoded sizes (csv {len(csv)}, metadata {len(meta)}, arrays {len(buf)}) exceed the derived bounds {lim}")
    rows = np.flatnonzero(np.asarray(base.status) == st_mod.LIVE).astype(np.int64)
    if not np.array_equal(rows, np.asarray(leaf.live_rows)) or len(rows) != P:
        raise _refuse("live_rows", "the boundary's live rows are not exactly the live records of its base state")
    sch, rm, cm = leaf.calendar.schedule, leaf.root_map, leaf.canopy_map
    doc = {
        "version": VERSION, "profile": op.receipt.profile, "leaf_package_digest": op.receipt.digests["leaf"], "syrup_package_digest": op.receipt.digests["syrup"], "catalog_hash": str(base.catalog_hash),
        "species": list(base.species_names), "n_base": Q, "n_live": P, "n_species": S, "n_layers": int(op.n_layers), "n_cells": int(op.n_cells), "initial": initial,
        "live_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "generation": int(leaf.generation), "schedule": sch.to_dict(), "geo_elapsed_days": str(Fraction(leaf.geo_elapsed)),
        "water_digest": leaf.water_digest, "mark": list(leaf.mark), "anchor": list(leaf.anchor), "biology_requests_since_anchor": int(leaf.mark[1]) - int(leaf.anchor[1]),
        "root_digest": v2.root_digest(rm), "root_edges": int(rm.n_edges), "canopy_digest": v2.canopy_digest(cm), "area_digest": v2._hash_arrays({"area": v2._host(leaf.canopy_area_m2)}),
        "members": {"csv": {"sha256": _sha(csv), "bytes": len(csv)}, "metadata": {"sha256": _sha(meta), "bytes": len(meta)}, "arrays": {"sha256": _sha(buf), "bytes": len(buf)}},
    }
    js = _canon(doc)
    if len(js) > lim["biology.json"]:
        raise _refuse("json_limit", f"biology.json would be {len(js)} B > its derived cap {lim['biology.json']} B")
    return BiologyMembers({"base_plants.csv": csv, "base_plants.metadata.json": meta, "arrays.bin": buf, "biology.json": js})


_NEED = {"version", "profile", "leaf_package_digest", "syrup_package_digest", "catalog_hash", "species", "n_base", "n_live", "n_species", "n_layers", "n_cells", "initial", "live_rows_sha256", "generation",
         "schedule", "geo_elapsed_days", "water_digest", "mark", "anchor", "biology_requests_since_anchor", "root_digest", "root_edges", "canopy_digest", "area_digest", "members"}
_MEMBER_KEY = {"base_plants.csv": "csv", "base_plants.metadata.json": "metadata", "arrays.bin": "arrays"}


def decode(op: Any, members: Any, water: Any, *, calendar_cls: Any, expected_profile: str | None = None, expected_cursor: Any = None, expected_anchor: Any = None,
           expected_cadence: Any = None, expected_counts: Any = None) -> Any:
    """Rebuild an accepted `LeafBoundary` for the restored lifecycle state `water`. Nothing is adopted or mutated; the boundary is minted only when ALL joins hold: closed members and recorded
    sizes/sha256, LEAF's own pair verification (`parse_plant_csv(..., metadata=...)`), Q/P/S, catalog, live-row digest, malformed-value limits, the rebuilt root/canopy/area digests, the exact water
    join tokens and (when given) the scheduler's cursor, scenario anchor and cadence."""
    lwi._check_op(op)
    _check_variant(op)
    if not isinstance(members, dict) or set(members) != set(MEMBERS) or any(not isinstance(v, (bytes, bytearray)) for v in members.values()):
        raise _refuse("bad_members", f"members must be exactly {MEMBERS} as bytes-like")
    names_max = _max_name(op.catalog.species_ids)
    mcap = metadata_cap(op)
    jcap = member_limits(0, 0, op.n_species, names_max, mcap)["biology.json"]
    if len(members["biology.json"]) > jcap:
        raise _refuse("json_limit", "biology.json exceeds its derived cap")
    try:
        doc = json.loads(bytes(members["biology.json"]).decode("ascii"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    except Exception as exc:
        raise _refuse("bad_json", f"{type(exc).__name__}: {exc}") from None
    if type(doc) is not dict or set(doc) != _NEED or _canon(doc) != bytes(members["biology.json"]):
        raise _refuse("bad_json", "biology.json must be the canonical closed record")
    if doc["version"] != VERSION:
        raise _refuse("version", f"unsupported biology version {doc['version']!r}")
    if doc["profile"] != op.receipt.profile or (expected_profile is not None and doc["profile"] != expected_profile):
        raise _refuse("profile", f"the artifact was written under profile {doc['profile']!r}, the live source profile is {op.receipt.profile!r}")
    if doc["leaf_package_digest"] != op.receipt.digests["leaf"] or doc["syrup_package_digest"] != op.receipt.digests["syrup"]:
        raise _refuse("source", "LEAF/SYRUP package digests differ from the live verified sources")
    Q, P, S = doc["n_base"], doc["n_live"], doc["n_species"]
    if expected_counts is not None and tuple(expected_counts) != (Q, P, S):  # the scalar counts the caller PLANNED/read with must equal the ones that drive parsing and rebuild: joined BEFORE any allocation
        raise _refuse("count_join", f"the counts the caller planned with (n_base, n_live, n_species) = {tuple(expected_counts)} differ from biology.json's {(Q, P, S)}; nothing was parsed or rebuilt")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (Q, P, S)) or P > Q:
        raise _refuse("shape", "n_base/n_live/n_species must be non-negative integers with n_live <= n_base")
    if S != op.n_species or doc["n_layers"] != op.n_layers or doc["n_cells"] != op.n_cells or type(doc["initial"]) is not bool:
        raise _refuse("shape", "species/layer/cell counts differ from the live operator")
    lim = member_limits(Q, P, S, names_max, mcap)
    rec = doc["members"]
    if type(rec) is not dict or set(rec) != {"csv", "metadata", "arrays"}:
        raise _refuse("members_record", "members must record exactly csv, metadata, arrays")
    for name, key in _MEMBER_KEY.items():
        meta = rec[key]
        if (type(meta) is not dict or set(meta) != {"sha256", "bytes"} or type(meta["bytes"]) is not int or type(meta["sha256"]) is not str or len(members[name]) != meta["bytes"]
                or _sha(members[name]) != meta["sha256"]):
            raise _refuse("member_digest", f"{name} differs from its recorded size/sha256")
    if len(members["arrays.bin"]) != lim["arrays.bin"] or len(members["base_plants.csv"]) > lim["base_plants.csv"] or len(members["base_plants.metadata.json"]) > lim["base_plants.metadata.json"]:
        raise _refuse("member_limit", "a payload member exceeds its derived size")
    v2._admit_codec(op, required_decode_bytes(op, Q, P, S, sum(len(v) for v in members.values()), members_resident=True), "biology_decode")  # BEFORE the float arrays / parsed state / rebuild exist
    bc, st_mod = op.leaf["biocalendar"], op.leaf["state"]
    cat = op.catalog
    if str(doc["catalog_hash"]) != str(cat.hash) or list(doc["species"]) != list(op.table.species_ids):
        raise _refuse("catalog", "catalog identity or species order differs from the live catalog/table")
    try:
        base = st_mod.parse_plant_csv(bytes(members["base_plants.csv"]), cat, metadata=bytes(members["base_plants.metadata.json"]), source="biology/base_plants.csv")
    except Exception as exc:  # LEAF verifies the pair (size, sha256, rows, catalog, domain, time, initial/SURFACE flags)
        raise _refuse("base_plants", f"{type(exc).__name__}: {exc}") from None
    if int(base.n) != Q:
        raise _refuse("n_records", f"the base CSV holds {int(base.n)} records but the closed metadata declares Q={Q}")
    if str(base.catalog_hash) != doc["catalog_hash"] or (not doc["initial"] and not bool(base.establishment_resolved)):
        raise _refuse("catalog", "the base plant pair disagrees with the recorded catalog / initial flag (an unresolved SURFACE state is only legal as initial)")
    rows = np.flatnonzero(np.asarray(base.status) == st_mod.LIVE).astype(np.int64)
    if len(rows) != P or hashlib.sha256(rows.tobytes()).hexdigest() != doc["live_rows_sha256"]:
        raise _refuse("live_rows", "the live rows derived from the CSV statuses do not match the recorded count/digest")
    raw = np.frombuffer(members["arrays.bin"], dtype=_LE)
    B, E, D, U = (raw[i * P:(i + 1) * P].astype(np.float64) for i in range(4))
    A, T = raw[4 * P:4 * P + S].astype(np.float64), raw[4 * P + S:4 * P + 2 * S].astype(np.float64)
    if not bool(np.all(np.isfinite(raw))):
        raise _refuse("arrays", "non-finite value in arrays.bin")
    v2._check_restored_values(op, doc, B, E, D, U, A, T)
    sch = v2._parse_schedule(bc, doc["schedule"])
    if expected_anchor is not None and sch.anchor != expected_anchor:
        raise _refuse("clock_join", f"the schedule anchor {sch.anchor.isoformat()} != the scenario calendar start {expected_anchor.isoformat()}")
    if expected_cadence is not None and sch.cadence != expected_cadence:
        raise _refuse("clock_join", "the schedule cadence differs from the live cadence")
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
    if v2.root_digest(rb.root_map) != doc["root_digest"] or int(rb.root_map.n_edges) != doc["root_edges"]:
        raise _refuse("root_map", "the rebuilt root-map CSR differs from the stored digest (catalog/geometry/biomass identity mismatch)")
    if v2.canopy_digest(rb.canopy_map) != doc["canopy_digest"] or v2._hash_arrays({"area": v2._host(rb.canopy_area_m2)}) != doc["area_digest"]:
        raise _refuse("canopy_map", "the rebuilt canopy CSR/areas differ from the stored digests")
    mark, anchor = tuple(doc["mark"]), tuple(doc["anchor"])
    if str(doc["water_digest"]) != str(water.digest):
        raise _refuse("water_join", "the restored lifecycle state's digest differs from the state the biology was released at (a different water lifetime/branch)")
    if mark != lwi._mark_of(op, water):
        raise _refuse("water_join", f"the restored lifecycle state's continuity mark {lwi._mark_of(op, water)} != the stored biology mark {mark}")
    if len(mark) != 4 or len(anchor) != 4 or int(mark[1]) - int(anchor[1]) != doc["biology_requests_since_anchor"] or anchor[3] != mark[3]:
        raise _refuse("anchor", "request anchor + delta do not reproduce the recorded mark")
    # COMPUTE-side allocations (device arrays on CUDA, the same host arrays on CPU): B/E/D/U, the species index and the live mask are uploaded explicitly (the exact `_typed` contract is never relaxed: it
    # requires `type(a) is op.lop.xp.ndarray`); the rebuilt area is passed DIRECTLY: the qualified `biocalendar.rebuild_geometry` already returns a CuPy area for backend cuda and a NumPy area otherwise
    # (no D2H/H2D, no conversion fallback); the species integrals A/T stay HOST NumPy.
    xp = op.lop.xp
    area = rb.canopy_area_m2
    return lwi._boundary_from_restore(op, B_kg=xp.asarray(B), E_kg=xp.asarray(E), species_index=xp.asarray(np.array(base.species_index[rows], dtype=np.int32)), live=xp.ones(P, dtype=bool),
                                      canopy_area_m2=area, root_map=rb.root_map, canopy_map=rb.canopy_map, base_state=base, live_rows=rows, D_sum=xp.asarray(D), U_sum=xp.asarray(U), A_int=A, T_int=T,
                                      calendar=cal, generation=int(doc["generation"]), mark=mark, anchor=anchor, geo_elapsed=geo_el, water_digest=str(doc["water_digest"]))
