"""Opt-in static Dense CPU active-water store; HEAD remains the only authority.

This module owns bounded control metadata and calls the existing C/B/StageA
owners. It implements no Native clock, sediment, water, checkpoint or reset law.
Importing it does not import NumPy or either Native model.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import re
import stat
import sys
import threading
import uuid
from pathlib import Path

from masy import dense_boundary_store as c

SCHEMA = "masy.dense_boundary_store.activewater.v1"
PROFILE = "default_dense_numpy_activewater_v1"
_MAX_REVISION = 24
_MAX_CHAIN = 25
_MAX_PAUSES = 2
_ACTIVE_RAW_BYTES = 22 << 20
_ACTIVE_ENTRIES = 8
_PATH_CHARS = 4096
_KINDS = ("genesis", "water_paused", "event_accepted")
_STORE = (*c._STORE[:-1], "managed_water", "native_work_bytes", "schedule_sha256", "store_sha256")
_HEAD = ("schema", "store_id", "store_file_sha256", "revision", "accepted_sequence",
         "kind", "generation_name", "generation_file_sha256", "generation_sha256",
         "anchor_generation", "event_operation_id", "request_id", "pause_ordinal",
         "pause_count", "head_sha256")
_INTENT = ("schema", "store_id", "attempt_id", "revision", "accepted_sequence",
           "expected_parent_head_file_sha256", "expected_parent_generation",
           "entry_anchor_head_file_sha256", "event_operation_id", "request_id",
           "request_intent", "source_identity", "environment_sha256", "intent_sha256")
_GENERATION = ("schema", "store_id", "store_file_sha256", "attempt_id", "revision",
               "accepted_sequence", "kind", "parent_head_record",
               "parent_head_file_sha256", "parent_generation", "entry_anchor_head_record",
               "event_operation_id", "request_id", "request_intent", "intent_file_sha256",
               "image", "accepted_summary", "accepted_receipt_sha256", "water_ref",
               "ack_core", "ack_core_sha256", "generation_sha256")
_ACK = ("schema", "store_id", "event_operation_id", "request_id", "request_intent",
        "returned_kind", "revision", "accepted_sequence", "selected_parent_token",
        "pause_ordinal", "pause_count", "completion_receipt_sha256")
_MANAGED = ("path", "dev", "ino", "parent_path", "parent_dev", "parent_ino", "fs_id")
_WATER = ("attempt_name", "export_sha256", "ordinal", "pauses", "origin_s",
          "native_time_s", "plan_sha256")
_SELECTION_KEY = object()

# Closed opt-in sibling format. The old constants and old intent arithmetic are
# unchanged; selectors below have only these two Source-owned specializations.
SCHEMA_PROCESS = "masy.dense_boundary_store.activeprocess.v1"
PROFILE_PROCESS = "default_dense_numpy_activeprocess_v1"
_PROCESS_SCHEMAS = tuple(SCHEMA_PROCESS + suffix for suffix in
    ("", ".head", ".intent", ".generation", ".ack", ".request", ".storage", ".report"))
_PROCESS_KINDS = (*_KINDS, "wind_paused")
_PROCESS_STORE = (*tuple("managed_active" if x == "managed_water" else x for x in _STORE[:-1]),
                  "event_kinds", _STORE[-1])
_PROCESS_GENERATION = tuple("active_ref" if x == "water_ref" else x for x in _GENERATION)
_PROCESS_REF = ("process_kind", *_WATER)
# Cold wind re-export changes the encoded native/local/global clocks and step
# index as well as ordinal.  A signed finite binary64 difference has at most
# 632 numerator digits and 324 denominator digits: two quoted rational values
# (960 each), one encoded native time (1006), step index (4), and ordinal growth
# (3).  This bounds constructor bytes BEFORE the unchanged 1 MiB wire refusal.
_COMBINED_CAPSULE_GROWTH_BYTES = 2933


def _combined(value):
    return type(value) is dict and type(value.get("schema")) is str and value["schema"] in _PROCESS_SCHEMAS


def _schema(value):
    return SCHEMA_PROCESS if _combined(value) else SCHEMA


def _profile_name(value):
    return PROFILE_PROCESS if _combined(value) else PROFILE


def _kinds(value):
    return _PROCESS_KINDS if _combined(value) else _KINDS


def _store_fields(value):
    return _PROCESS_STORE if _combined(value) else _STORE


def _generation_fields_list(value):
    return _PROCESS_GENERATION if _combined(value) else _GENERATION


def _water_fields(value):
    return _PROCESS_REF if _combined(value) else _WATER


def _ref_key(value):
    return "active_ref" if _combined(value) else "water_ref"


def _managed_key(value):
    return "managed_active" if _combined(value) else "managed_water"


def _progress_key(value):
    return "active_process" if _combined(value) else "active_water"


def _sibling_prefix(value):
    return ".masy-active-" if _combined(value) else ".masy-water-"


def _paused(value):
    return value["kind"] == "water_paused" or (_combined(value) and value["kind"] == "wind_paused")


def _same_format(header, *values):
    if any(_combined(value) != _combined(header) for value in values):
        c._fail("active_format", "every selected control belongs to its actual STORE format")


def _pause_fields_for(value):
    kind = "water_paused" if _combined(value) and value["kind"] == "wind_paused" else value["kind"]
    return _pause_fields(kind, value["pause_ordinal"], value["pause_count"])


def _intent_for(kind, ordinal, header):
    if not _combined(header) or kind != "checkpoint_wind":
        return _intent(kind, ordinal)
    # Same bounded original-plan ordinal law, a distinct closed process tag.
    answer = _intent("checkpoint_water", ordinal)
    return {**answer, "kind": "checkpoint_wind"}


def _check_intent_for(value, header):
    if not _combined(header):
        return _check_intent(value)
    c._keys(value, ("kind", "boundary_ordinal"), "active process request intent")
    expected = _intent_for(value["kind"], value["boundary_ordinal"], header)
    if c.oi._json(value) != c.oi._json(expected):
        c._fail("active_request", "canonical closed process request required")
    return expected


def _active_reference(root, header, limits, head, generation):
    if _combined(header) and head["kind"] == "wind_paused":
        from masy import dense_active_wind_checkpoint as wind
        identity = (header["storage_contract"]["root_dev"], header["storage_contract"]["root_ino"])
        return wind.validate_retained_reference(root, header, limits, head, generation,
                                                _managed_at(root, header, identity))
    return _water_reference(root, header, limits, head, generation)


def _declare_for(header):
    _declare()
    if _combined(header):
        from masy import dense_active_wind_checkpoint as wind
        wind._declare()


def _next_kind(owner):
    return None if owner._next == len(owner._schedule.events) else owner._schedule.events[owner._next].kind


def _wind_profile(runner, reference):
    from masy import dense_wind_pause as wind
    api = wind._declare()
    wind._profile(runner, api)
    wind._runtime(api, reference)


@dataclasses.dataclass(frozen=True)
class ActiveWaterAcknowledgement:
    """Prepared immutable wire bytes; contains no runner, Native holder or token."""
    schema: str
    store_id: str
    event_operation_id: str
    request_id: str
    revision: int
    accepted_sequence: int
    head_token: str
    ack_bytes: bytes
    receipt_bytes: bytes | None


def _intent(kind, ordinal=None):
    if type(kind) is not str:
        c._fail("active_request", "plain request kind required")
    if kind == "finish":
        if ordinal is not None:
            c._fail("active_request", "finish has no ordinal")
        return {"kind": "finish", "boundary_ordinal": None}
    if kind == "checkpoint_water":
        if type(ordinal) is not int or not 1 <= ordinal <= 4096:
            c._fail("active_request", "plain positive original plan ordinal required")
        return {"kind": kind, "boundary_ordinal": ordinal}
    c._fail("active_request", "closed checkpoint_water/finish intent only")


def _check_intent(value):
    c._keys(value, ("kind", "boundary_ordinal"), "active request intent")
    expected = _intent(value["kind"], value["boundary_ordinal"])
    if c.oi._json(value) != c.oi._json(expected):
        c._fail("active_request", "canonical closed request intent required")
    return expected


def _request_id(header, anchor, selected, parent_raw, intent, schedule):
    """Validated accepted A fixes event operation; canonical selected R fixes request."""
    limits = c._limits_load(header["limits"])
    _head_fields(anchor, limits, header=header)
    _head_fields(selected, limits, header=header)
    _check_intent_for(intent, header)
    _same_format(header, anchor, selected)
    c._digest(schedule, "schedule_sha256")
    if (anchor["kind"] not in ("genesis", "event_accepted")
            or anchor["store_id"] != header["store_id"]
            or selected["store_id"] != header["store_id"]):
        c._fail("active_anchor", "actual accepted same-store anchor required")
    if (type(parent_raw) is not bytes or not 1 <= len(parent_raw) <= limits.max_head_bytes
            or parent_raw != c.oi._json(selected)):
        c._fail("active_request", "stat-admitted canonical selected raw HEAD required")
    operation = c._operation(header["store_id"], anchor,
                             anchor["accepted_sequence"], schedule)
    if _paused(selected):
        if (selected["anchor_generation"] != anchor["generation_name"]
                or selected["accepted_sequence"] != anchor["accepted_sequence"]
                or selected["event_operation_id"] != operation):
            c._fail("active_anchor", "selected pause differs from actual event anchor")
    elif parent_raw != c.oi._json(anchor):
        c._fail("active_anchor", "selected accepted HEAD must itself be the anchor")
    request = c._sha(c.oi._json({"schema": _schema(header) + ".request",
        "store_id": header["store_id"], "event_operation_id": operation,
        "selected_parent_head_file_sha256": c._sha(parent_raw),
        "request_intent": intent}))
    return operation, request


def _name(value, revision):
    match = c._GEN.fullmatch(value) if type(value) is str else None
    if match is None or int(match[1]) != revision:
        c._fail("active_generation", "exact bounded revision/nonce path required")
    return match[2]


def _pause_fields(kind, ordinal, count):
    if kind == "water_paused":
        if (type(ordinal) is not int or not 1 <= ordinal <= 4096
                or type(count) is not int or not 1 <= count <= _MAX_PAUSES):
            c._fail("active_pause", "original ordinal and pause count have distinct bounds")
    elif ordinal is not None or count is not None:
        c._fail("active_pause", "only water_paused carries original ordinal/count")


def _head_fields(value, limits, *, header=None):
    descriptor = value if header is None else header
    c._check_seal(value, _HEAD, "head_sha256", _schema(descriptor) + ".head")
    if type(value["store_id"]) is not str or not c._NONCE.fullmatch(value["store_id"]):
        c._fail("metadata_nonce", "active store id")
    revision = c._integer(value["revision"], _MAX_REVISION, "active revision")
    sequence = c._integer(value["accepted_sequence"], min(8, limits.max_accepted - 1),
                          "active accepted sequence")
    kind = value["kind"]
    if type(kind) is not str or kind not in _kinds(descriptor) or sequence > revision:
        c._fail("active_head", "kind/cursor/revision")
    _name(value["generation_name"], revision)
    for field in ("store_file_sha256", "generation_file_sha256", "generation_sha256",
                  "event_operation_id", "request_id"):
        c._digest(value[field], field)
    if type(value["anchor_generation"]) is not str or not c._GEN.fullmatch(value["anchor_generation"]):
        c._fail("active_anchor", "bounded generation path required")
    _pause_fields_for(value)
    if kind == "genesis":
        if revision != 0 or sequence != 0:
            c._fail("active_head", "genesis is revision/cursor zero")
    elif revision == 0 or (kind == "event_accepted" and sequence == 0):
        c._fail("active_head", "non-genesis revision/cursor")
    if _paused(value):
        anchor_match = c._GEN.fullmatch(value["anchor_generation"])
        if int(anchor_match[1]) >= revision:
            c._fail("active_anchor", "paused anchor must be a strict prior generation")
    if not _paused(value) and value["anchor_generation"] != value["generation_name"]:
        c._fail("active_anchor", "accepted HEAD is its next event anchor")
    return value


def _intent_fields(value, limits, *, header=None):
    descriptor = value if header is None else header
    c._check_seal(value, _INTENT, "intent_sha256", _schema(descriptor) + ".intent")
    if (type(value["store_id"]) is not str or not c._NONCE.fullmatch(value["store_id"])
            or type(value["attempt_id"]) is not str or not c._NONCE.fullmatch(value["attempt_id"])):
        c._fail("metadata_nonce", "actual active store/attempt nonce")
    revision = c._integer(value["revision"], _MAX_REVISION, "intent revision")
    sequence = c._integer(value["accepted_sequence"], min(8, limits.max_accepted - 1), "intent cursor")
    if sequence > revision or (revision == 0 and sequence != 0):
        c._fail("active_intent", "typed accepted cursor/revision")
    for field in ("event_operation_id", "request_id", "environment_sha256"):
        c._digest(value[field], field)
    parent = value["expected_parent_generation"]
    if revision == 0:
        if any(value[key] is not None for key in ("expected_parent_head_file_sha256",
                "expected_parent_generation", "entry_anchor_head_file_sha256", "request_intent")):
            c._fail("active_intent", "genesis has no input parent/request")
    else:
        if type(parent) is not str or not c._GEN.fullmatch(parent) or int(c._GEN.fullmatch(parent)[1]) != revision - 1:
            c._fail("active_intent", "original selected parent generation")
        for field in ("expected_parent_head_file_sha256", "entry_anchor_head_file_sha256"):
            c._digest(value[field], field)
        _check_intent_for(value["request_intent"], descriptor)
    return value


def _head_record(header, store_sha, name, generation, file_sha):
    water = generation[_ref_key(header)]
    anchor_name = (generation["entry_anchor_head_record"]["generation_name"]
                   if _paused(generation) else name)
    result = _seal_control({"schema": _schema(header) + ".head", "store_id": header["store_id"],
        "store_file_sha256": store_sha, "revision": generation["revision"],
        "accepted_sequence": generation["accepted_sequence"], "kind": generation["kind"],
        "generation_name": name, "generation_file_sha256": file_sha,
        "generation_sha256": generation["generation_sha256"], "anchor_generation": anchor_name,
        "event_operation_id": generation["event_operation_id"], "request_id": generation["request_id"],
        "pause_ordinal": None if water is None else water["ordinal"],
        "pause_count": None if water is None else water["pauses"]}, "head_sha256",
        c._limits_load(header["limits"]).max_head_bytes, "active HEAD")
    return _head_fields(result, c._limits_load(header["limits"]), header=header)


def _ack_fields(value, head, parent, *, header=None):
    descriptor = head if header is None else header
    c._keys(value, _ACK, "active ACK core")
    if (value["schema"] != _schema(descriptor) + ".ack" or value["store_id"] != head["store_id"]
            or value["event_operation_id"] != head["event_operation_id"]
            or value["request_id"] != head["request_id"]
            or value["returned_kind"] != head["kind"]):
        c._fail("active_ack", "ACK/head identity")
    for field, expected in (("revision", head["revision"]),
            ("accepted_sequence", head["accepted_sequence"]),
            ("pause_ordinal", head["pause_ordinal"]), ("pause_count", head["pause_count"])):
        if type(value[field]) is not type(expected) or value[field] != expected:
            c._fail("active_ack", "typed ACK cursor/progress")
    if (parent is None) != (head["kind"] == "genesis"):
        c._fail("active_ack", "only genesis lacks a direct parent")
    if parent is not None:
        expected_sequence = parent["accepted_sequence"] + (head["kind"] == "event_accepted")
        if head["revision"] != parent["revision"] + 1 or head["accepted_sequence"] != expected_sequence:
            c._fail("active_ack", "direct child revision/accepted-cursor step")
        c._digest(value["selected_parent_token"], "actual selected parent token")
    if value["selected_parent_token"] != (None if parent is None else c._token(parent)):
        c._fail("active_ack", "direct selected parent token")
    if parent is None:
        if value["request_intent"] is not None or value["completion_receipt_sha256"] is not None:
            c._fail("active_ack", "genesis has no request/completion")
    else:
        intent = _check_intent_for(value["request_intent"], descriptor)
        if _paused(head) and (intent["kind"] != ("checkpoint_wind" if head["kind"] == "wind_paused" else "checkpoint_water")
                or intent["boundary_ordinal"] != head["pause_ordinal"]):
            c._fail("active_ack", "wet ACK must match requested original boundary")
        receipt = value["completion_receipt_sha256"]
        if head["kind"] == "event_accepted":
            c._digest(receipt, "exact completion receipt")
        elif receipt is not None:
            c._fail("active_ack", "wet request has no completion receipt")
    return value


def _metadata_bytes(value, maximum, where):
    """Use existing bounded plain-metadata and encoder; no copied JSON codec."""
    c._plain_metadata(value, maximum, where)
    raw = c.oi._json(value)
    if len(raw) > maximum:
        c._fail("metadata_cap", where)
    return raw


def _seal_control(value, field, maximum, where):
    # The largest closed seal key plus its 64 hex characters fits 256 bytes.
    # Admit the unsealed body before the unchanged native C encoder is invoked.
    if field not in ("head_sha256", "intent_sha256", "generation_sha256", "store_sha256"):
        c._fail("metadata_seal", "closed owning control seal only")
    if type(maximum) is not int or maximum < 256:
        c._fail("metadata_cap", "finite control envelope required")
    c._plain_metadata(value, maximum - 256, where)
    sealed = c._seal(value, field)
    _metadata_bytes(sealed, maximum, where)
    return sealed


@dataclasses.dataclass(frozen=True, slots=True)
class _Selection:
    """Issued by this C owner, additionally checked against actual live files/state."""
    key: object
    state: object
    selected_raw: bytes
    anchor_raw: bytes
    image_raw: bytes
    intent_raw: bytes | None
    diagnostics_raw: bytes | None = None


def _anchor_row(chain):
    for row in chain:
        if row[0]["kind"] in ("genesis", "event_accepted"):
            return row
    c._fail("active_anchor", "selected chain has no genuine accepted ancestor")


def _selection(parent):
    if type(parent) is not c._WaterParent or type(parent._selection) is not _Selection:
        c._fail("active_parent", "actual C-issued active-water parent required")
    selection = parent._selection
    if selection.key is not _SELECTION_KEY or type(selection.state) is not c._State:
        c._fail("active_parent", "foreign private selection")
    return selection


def _parent_light(parent, handle):
    parent._own()
    selection = _selection(parent)
    if type(handle) is not c.DenseBoundaryStore or not c._active_store(handle._header):
        c._fail("active_parent", "exact active C handle required")
    handle._binding()
    if (handle._entered is not True or handle._state is not selection.state
            or handle._state.phase not in ("busy_before_head", "initializing")
            or c.oi._json(handle._header["source_identity"]) != parent._source
            or handle._header["environment_sha256"] != parent._environment):
        c._fail("active_parent", "selected handle/state/source binding changed")
    handle._stable()
    selected, selected_raw = handle._head()
    if selected_raw != selection.selected_raw:
        c._fail("active_parent", "selected R changed during private operation")
    chain = _generation_chain(handle._root, handle._header, c._sha(handle._store_raw),
                              selected, handle._limits)
    anchor, generation, _manifest = _anchor_row(chain)
    anchor_raw = c.oi._json(anchor)
    if (anchor_raw != selection.anchor_raw or parent._head != anchor_raw
            or parent._token != c._token(anchor)
            or parent._image != selection.image_raw
            or c.oi._json(generation["image"]) != selection.image_raw
            or anchor["accepted_sequence"] != selected["accepted_sequence"]):
        c._fail("active_parent", "accepted A/image no longer matches genuine selected ancestry")
    if selection.intent_raw is not None:
        if type(selection.intent_raw) is not bytes or not 1 <= len(selection.intent_raw) <= handle._limits.max_intent_bytes:
            c._fail("active_parent", "bounded retained intent bytes required before parser")
        intent = c._plain_metadata(c.oi._strict_json(selection.intent_raw, c._JSON),
                                    handle._limits.max_intent_bytes, "reserved active intent")
        _intent_fields(intent, handle._limits, header=handle._header)
        path = handle._root / "generations" / f"g-{intent['revision']:04d}-{intent['attempt_id']}" / "intent.json"
        if (intent["attempt_id"] != handle._state.attempt_id
                or intent["expected_parent_head_file_sha256"] != c._sha(selected_raw)
                or intent["entry_anchor_head_file_sha256"] != c._sha(anchor_raw)
                or c._read(path, handle._limits.max_intent_bytes) != selection.intent_raw):
            c._fail("active_parent", "real reserved intent/current selection changed")
    elif handle._state.phase != "initializing":
        c._fail("active_parent", "only explicit recovery has no new intent")
    return selection


def _checked_selection(parent, runner):
    parent._own()
    if (parent._phase != "entered" or parent._issued is None
            or parent._issued() is not runner):
        c._fail("active_parent", "foreign or unissued actual runner")
    return _parent_light(parent, parent._handle)


def _parent_record(parent, runner):
    selection = _checked_selection(parent, runner)
    handle = parent._handle
    # selection.anchor_raw comes from the fully checked accepted chain record.
    # handle._head() is selected R and must not be its hash source.
    return {"store_id": handle._header["store_id"], "token": parent._token,
            "accepted_sequence": c.oi._strict_json(selection.anchor_raw, c._JSON)["accepted_sequence"],
            "store_file_sha256": c._sha(handle._store_raw),
            "head_file_sha256": c._sha(selection.anchor_raw),
            "image": c.oi._strict_json(selection.image_raw, c._JSON),
            "originals": handle._header["originals"]}


def _borrow_parent(handle, intent):
    if type(handle) is not c.DenseBoundaryStore or not c._active_store(handle._header):
        c._fail("active_parent", "exact active C handle required")
    handle._binding()
    if (handle._entered is not True or type(handle._state) is not c._State
            or handle._state.phase not in ("busy_before_head", "initializing")):
        c._fail("active_parent", "only the real reserved operation or explicit recovery borrows")
    handle._stable()
    selected, raw = handle._head()
    chain = _generation_chain(handle._root, handle._header, c._sha(handle._store_raw),
                              selected, handle._limits)
    anchor, generation, _manifest = _anchor_row(chain)
    anchor_raw = c.oi._json(anchor)
    image_raw = c.oi._json(generation["image"])
    context = c._WaterParent(c._WATER_PARENT_KEY, handle, c._token(anchor), anchor_raw,
                            c.oi._json(handle._header["source_identity"]),
                            handle._header["environment_sha256"])
    context._image = image_raw
    context._selection = _Selection(_SELECTION_KEY, handle._state, raw, anchor_raw, image_raw,
                                    None if intent is None else c.oi._json(intent))
    del chain, anchor, generation, _manifest
    _parent_light(context, handle)
    return context


def _enter_parent(parent):
    parent._own()
    if parent._phase != "new":
        c._fail("active_parent", "one-shot active parent already entered/closed")
    handle = parent._handle
    _parent_light(parent, handle)
    parent._phase = "entered"
    prepared = owner = None
    try:
        selection = _selection(parent)
        anchor = c.oi._strict_json(selection.anchor_raw, c._JSON)
        prepared = _prepare_anchor(handle, anchor)
        diagnostics_raw = _metadata_bytes(prepared.diagnostics(), handle._limits.max_generation_bytes,
                                           "actual accepted-anchor diagnostics")
        owner = c._owner_for(handle._header, prepared)
        prepared = None
        if (owner._next != anchor["accepted_sequence"]
                or owner._limits.max_work_bytes != handle._header["native_work_bytes"]):
            c._fail("active_parent", "actual accepted owner cursor/work budget differs")
        parent._image = selection.image_raw
        parent._selection = dataclasses.replace(selection, diagnostics_raw=diagnostics_raw)
        parent._owner, owner = owner, None
    except BaseException:
        prepared = owner = None
        parent._image = None
        parent._outcome = "entry_failed"
        _release_parent(parent)
        raise
    return parent


def _release_parent(parent):
    # Outer C operation owns handle._entered until all temporary contexts close
    # and publication completes. This context must not clear that outer guard.
    parent._handle = None
    parent._owner = parent._issued = None
    parent._selection = None
    parent._phase = "closed"


def _close_parent(parent):
    """Normal publication requires the actual context to complete its close."""
    if type(parent) is not c._WaterParent:
        c._fail("active_parent", "exact C context close required")
    parent.close()
    if parent._phase != "closed":
        c._fail("active_parent", "non-closed context cannot precede publication")


def _managed(handle):
    return _managed_at(handle._root, handle._header, handle._root_identity)


def _managed_at(root, header, root_identity):
    value = header[_managed_key(header)]
    c._keys(value, _MANAGED, "managed sibling")
    expected = root.parent / ((_sibling_prefix(header)) + header["store_id"])
    if (type(value["path"]) is not str or len(value["path"]) > _PATH_CHARS
            or value["path"] != str(expected) or os.path.realpath(expected) != str(expected)
            or value["parent_path"] != str(root.parent)):
        c._fail("active_sibling", "canonical code-owned final sibling/parent required")
    for field in ("dev", "ino", "parent_dev", "parent_ino", "fs_id"):
        if type(value[field]) is not int or value[field] < 0:
            c._fail("active_sibling", "plain actual filesystem identity fields")
    outer = c.oi._open_dir(root.parent)
    sibling = None
    try:
        info = os.fstat(outer)
        if c._identity(info) != (value["parent_dev"], value["parent_ino"]):
            c._fail("active_sibling", "recorded sibling parent inode differs")
        sibling = c.oi._open_dir(expected)
        info = os.fstat(sibling)
        if (c._identity(info) != (value["dev"], value["ino"])
                or info.st_dev != root_identity[0]
                or os.fstatvfs(sibling).f_fsid != value["fs_id"]
                or value["fs_id"] != header["storage_contract"]["observed_fs_id"]):
            c._fail("active_sibling", "same verified filesystem/inode required")
        c._same_directory(expected, sibling)
        c._same_directory(root.parent, outer)
    finally:
        if sibling is not None:
            os.close(sibling)
        os.close(outer)
    return expected


def _inventory(handle, *, sibling=True):
    return _inventory_at(handle._root, handle._header, handle._limits,
                         handle._root_identity, sibling=sibling)


def _inventory_at(root, header, limits, root_identity, *, sibling=True):
    """Closed operational stat inventory before any new metadata hash/parse.

    It uses the existing C directory/identity primitives. Only C plus its named
    sibling is visited, never neighboring stores. No archive/model is opened.
    """
    count = used = 0
    generations, temps, water = set(), set(), set()

    def role(relative, name, is_water):
        if is_water:
            if not relative:
                _name(name, int(c._GEN.fullmatch(name)[1]) if type(name) is str and c._GEN.fullmatch(name) else -1)
                if int(c._GEN.fullmatch(name)[1]) > _MAX_REVISION:
                    c._fail("active_inventory", "water attempt revision")
                water.add(name)
                return True, 0
            pieces = relative.split("/")
            if len(pieces) == 1:
                if name == "entry" or name == "native" or re.fullmatch(r"\.native\.partial-[A-Za-z0-9_-]{1,64}", name):
                    return True, 0
                if name == "export.json":
                    return False, 1 << 20
            if len(pieces) == 2:
                if pieces[1] == "entry" and name == "capsule.json":
                    return False, 1 << 20
                if (pieces[1] == "native" or re.fullmatch(r"\.native\.partial-[A-Za-z0-9_-]{1,64}", pieces[1])):
                    if _combined(header) and pieces[1] == "native" and name == "continuation.bin":
                        return False, 8 << 20
                    if name == "checkpoint.json":
                        return False, 2 << 20
                    if name in ("maple_state.npz", "continuation.npz"):
                        return False, 9 << 20
            c._fail("active_inventory", "unknown caller-linked Native staging/child role")
        if not relative:
            match = c._TEMP.fullmatch(name)
            if match:
                temps.add(match[1]);return False, limits.max_head_bytes
            if name in ("originals", "generations", ".storage_probe"):
                return True, 0
            if name in ("STORE.json", "STORE.tmp"):
                return False, 128 << 10
            if name == "HEAD":
                return False, limits.max_head_bytes
            if name == ".writer.lock":
                return False, 0
        elif relative == "generations":
            match = c._GEN.fullmatch(name)
            if match and int(match[1]) <= _MAX_REVISION:
                generations.add(name);return True, 0
        elif re.fullmatch(r"generations/g-[0-9]{4}-[0-9a-f]{32}", relative):
            if name in ("physical", "image"):
                return True, 0
            if name in ("generation.json", "generation.tmp"):
                return False, limits.max_generation_bytes
            if name in ("intent.json", "failure.json"):
                return False, limits.max_intent_bytes
        elif relative == "originals":
            if re.fullmatch(r"o-[0-9a-f]{32}", name):
                return True, 0
        elif relative == ".storage_probe":
            if name in ("original", "linked", "result.json"):
                return False, 32 << 10
        elif re.fullmatch(r"originals/o-[0-9a-f]{32}", relative):
            if name in ("manifest.json", "manifest.tmp"):
                return False, 4 << 20
            if re.fullmatch(r"a_[0-9]{4}\.npy", name):
                return False, (64 << 20) + 4096
        elif re.fullmatch(r"generations/g-[0-9]{4}-[0-9a-f]{32}/image", relative):
            if name in ("manifest.json", "manifest.tmp"):
                return False, limits.image.max_metadata_bytes
            if re.fullmatch(r"a_000[0-6]\.npy", name):
                return False, limits.image.max_numeric_bytes + limits.image.max_header_bytes
        elif re.fullmatch(r"generations/g-[0-9]{4}-[0-9a-f]{32}/physical", relative):
            if name in ("checkpoint.json", "native_state.npz", "pending_state.npz") or re.fullmatch(r"\.manifest-[0-9a-f]{32}", name):
                return False, (1 << 20) if not name.endswith(".npz") else c._native_reservation(limits)
        c._fail("active_inventory", "unknown closed C directory/file role")

    def walk(path, relative, is_water, expected):
        nonlocal count, used
        if len(relative.split("/")) > 3:
            c._fail("active_inventory", "finite admitted directory depth")
        fd = c.oi._open_dir(path)
        try:
            if c._identity(os.fstat(fd)) != expected:
                c._fail("identity_drift", "admitted inventory directory changed")
            local = {}
            with os.scandir(fd) as iterator:
                for entry in iterator:
                    count += 1
                    if count > limits.max_entries:
                        c._fail("entry_cap", "combined C+sibling count before append/hash")
                    name = entry.name
                    if len(str(path / name)) > _PATH_CHARS:
                        c._fail("active_inventory", "bounded operational path before retention")
                    info = entry.stat(follow_symlinks=False)
                    directory, cap = role(relative, name, is_water)
                    local[name] = info
                    if directory:
                        if not stat.S_ISDIR(info.st_mode):
                            c._fail("active_inventory", "known directory type required")
                        walk(path / name, name if not relative else relative + "/" + name,
                             is_water, c._identity(info))
                    else:
                        if (not stat.S_ISREG(info.st_mode) or info.st_size > cap
                                or info.st_nlink not in ((1,) if is_water else (1, 2))):
                            c._fail("active_inventory", "exact regular/link/raw-size role required")
                        if is_water and (info.st_uid != os.geteuid()):
                            c._fail("active_inventory", "own Native attempt required")
                        used += info.st_size
                        if used > limits.max_store_bytes:
                            c._fail("store_raw_cap", "combined bytes before hash/parse")
            # Preserve original C's only allowed own manifest hardlink pairs.
            if not is_water:
                for name, info in local.items():
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    partner = {"STORE.json":"STORE.tmp", "STORE.tmp":"STORE.json",
                               "generation.json":"generation.tmp", "generation.tmp":"generation.json",
                               "manifest.json":"manifest.tmp", "manifest.tmp":"manifest.json"}.get(name)
                    if name == "checkpoint.json":
                        matches = [n for n in local if re.fullmatch(r"\.manifest-[0-9a-f]{32}", n)]
                        partner = matches[0] if len(matches) == 1 else None
                    elif re.fullmatch(r"\.manifest-[0-9a-f]{32}", name):
                        partner = "checkpoint.json"
                    other = local.get(partner)
                    if (info.st_nlink == 2 or other is not None) and (other is None
                            or not stat.S_ISREG(other.st_mode) or info.st_nlink != 2
                            or other.st_nlink != 2 or c._identity(info) != c._identity(other)):
                        c._fail("manifest_leftover", "only exact original own hardlink pair")
            c._same_directory(path, fd)
        finally:
            os.close(fd)

    walk(root, "", False, root_identity)
    if sibling:
        managed = _managed_at(root, header, root_identity)
        walk(managed, "", True, (header[_managed_key(header)]["dev"], header[_managed_key(header)]["ino"]))
        if not water <= generations:
            c._fail("active_inventory", "Native attempt has no retained owning C intent generation")
    if not temps <= {n.rsplit("-", 1)[1] for n in generations}:
        c._fail("active_inventory", "HEAD temporary lacks its own retained attempt")
    return {"entries": count, "bytes": used, "generations": generations,
            "temps": temps, "water_attempts": water}


def _parse_charge(raw_bytes):
    if type(raw_bytes) is not int or not 0 <= raw_bytes <= 8 << 20:
        c._fail("active_memory", "bounded exact raw metadata occurrence required")
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    coefficient = (sys.getsizeof({}) + sys.getsizeof([]) + sys.getsizeof(0.0)
                   + 8 * pointer + 5)
    return 2 * raw_bytes * coefficient + 6 * raw_bytes


def _context_charge(limits):
    """Fixed Source projection over finite metadata constructors, never arrays."""
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    # Three heads, intent, generation and separately retained ACK canonical view.
    records = (3 * _parse_charge(limits.max_head_bytes)
               + _parse_charge(limits.max_intent_bytes)
               + 2 * _parse_charge(limits.max_generation_bytes))
    # Name/stat/index/path records have <=64 scalar/ref slots, no alias credit.
    entry = (4 * sys.getsizeof({}) + 4 * sys.getsizeof(()) + sys.getsizeof([])
             + sys.getsizeof(os.stat_result((0,) * 10))
             + 64 * (pointer + sys.getsizeof(0))
             + 4 * sys.getsizeof("") + 4 * 4 * _PATH_CHARS)
    # Named non-array receivers, counted independently even when inherited C
    # already charges its own copy. No alias/phase-overlap subtraction.
    holders = ("old_idle", "old_identity_halt", "busy", "attempt_halt",
        "proposed_base", "proposed_halt", "proposed_idle", "publisher_possible",
        "publisher_uncertain", "work_parent", "work_selection", "reader_parent",
        "reader_selection", "ack_record", "chain_index", "report_mapping")
    fixed = len(holders) * (sys.getsizeof({}) + sys.getsizeof(())
                 + 64 * (pointer + sys.getsizeof(0)) + 4 * 4 * _PATH_CHARS)
    # One selected raw capsule is borrowed while actual three-file hashes run.
    # Parsed cold facts/readback are separately under each real active W.
    return records + limits.max_entries * entry + fixed + (1 << 20)


def _metadata_stat_charge(handle, inventory):
    """Exact stat sizes before reading/parsing old chain control metadata.

    Every potential row is charged independently; sharing an accepted image
    across several paused revisions gives no subtraction.
    """
    limits = handle._limits
    metadata_sizes = []
    image_max = 0
    generation_count = 0
    combined_future_peak = 0
    root = handle._root
    directory = c.oi._open_dir(root / "generations")
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                generation_count += 1
                if generation_count > _MAX_CHAIN + limits.max_orphans:
                    c._fail("active_chain", "finite selected+orphan generation count")
                name = entry.name
                match = c._GEN.fullmatch(name)
                if not match or int(match[1]) > _MAX_REVISION or not entry.is_dir(follow_symlinks=False):
                    c._fail("active_chain", "exact bounded generation name before metadata")
                path = root / "generations" / name
                for child, cap in (("intent.json", limits.max_intent_bytes),
                                   ("generation.json", limits.max_generation_bytes)):
                    file = path / child
                    if file.exists():
                        info = file.lstat()
                        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= cap:
                            c._fail("active_memory", "raw chain cap before metadata parse")
                        if len(metadata_sizes) >= 2 * (_MAX_CHAIN + limits.max_orphans):
                            c._fail("active_memory", "finite metadata occurrences before append")
                        metadata_sizes.append(info.st_size)
                file = path / "image/manifest.json"
                if file.exists():
                    info = file.lstat()
                    if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= limits.image.max_metadata_bytes:
                        c._fail("active_memory", "raw B metadata cap before parse")
                    image_max = max(image_max, info.st_size)
    finally:
        os.close(directory)
    # Old per-revision B metadata, head/intent/generation and a proposed B image.
    # C's original decode/native reservation remains fully additive.
    existing = sum(_parse_charge(x) for x in metadata_sizes)
    existing += min(generation_count, _MAX_CHAIN) * _parse_charge(image_max)
    managed = _managed(handle)
    # Each retained export parsed by chain validation is a separate occurrence;
    # capsule bytes are hashed without decoding here, and a.load separately
    # admits its own two parsed capsule/native reader graphs under real W.
    for name in inventory["water_attempts"]:
        path = managed / name / "export.json"
        if path.exists():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= 1 << 20:
                c._fail("active_memory", "actual raw export cap before chain parser")
            existing += _parse_charge(info.st_size)
        capsule = managed / name / "entry/capsule.json"
        if capsule.exists():
            info = capsule.lstat()
            if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= 1 << 20:
                c._fail("active_memory", "actual raw capsule cap before future capture metadata")
            if _combined(handle._header):
                # Keep EVERY current capsule occurrence.  Only one selected
                # cold child can be exported by an operation; its future two
                # constructor/comparison families are bounded by the maximum
                # over all candidates, including orphans.  Do not clamp a
                # constructor to the wire cap: it may exist before refusal.
                existing += _parse_charge(info.st_size)
                combined_future_peak = max(combined_future_peak, _parse_charge(
                    info.st_size + _COMBINED_CAPSULE_GROWTH_BYTES))
            else:
                # Preserve the original water-only arithmetic and wire domain.
                future = min(1 << 20, info.st_size + 3)
                existing += 2 * _parse_charge(future)
    # _load -> _prepare_anchor can retain a caller chain while validating a
    # second. _operate -> _borrow_parent -> _parent_light, including the
    # temporary-HEAD validation within _stable, has at most THREE chain
    # receiver families. Charge every row/parse/hash occurrence at all three
    # roles without alias credit; current early local drops do not lower U.
    # Preserve all three current history/descriptor/capsule receiver families.
    # The two prospective families are still paid at EACH of those three
    # roles, including primary/export-close OR primary/reader/work-close errors.
    return 3 * existing + 6 * combined_future_peak


def _reservation(limits, work_bytes, chain_charge=0, *, combined=False):
    original = c._reservation(limits, work_bytes)
    if type(chain_charge) is not int or chain_charge < 0:
        c._fail("active_memory", "actual bounded chain charge required")
    memory = (original["memory_bytes"] + 2 * work_bytes + chain_charge
              + _context_charge(limits) + _parse_charge(limits.image.max_metadata_bytes)
              + (2 << 20))
    extra = 0
    if combined:
        from masy import dense_active_wind_checkpoint as wind
        extra = wind.reservation_increment()["required_bytes"]
        memory += extra
    if memory > limits.max_memory_bytes:
        c._fail("memory_cap", "complete C+two active W+chain/context/future-B reservation")
    answer = {**original, "memory_bytes": memory,
            "disk_bytes": original["disk_bytes"] + _ACTIVE_RAW_BYTES,
            "entries": original["entries"] + _ACTIVE_ENTRIES,
            "active_native_work_roles": 2,
            "active_native_work_bytes_each": work_bytes,
            "active_chain_metadata_bytes": chain_charge,
            "active_context_metadata_bytes": _context_charge(limits),
            "active_future_image_metadata_bytes": _parse_charge(limits.image.max_metadata_bytes),
            "logical_Source_reservation_not_heap_fit": True}
    if combined:
        answer["active_wind_codec_increment_bytes"] = extra
    return answer


def _admit(handle, retained, max_work_bytes, *, initial=False):
    inventory = _inventory(handle)
    if type(max_work_bytes) is not int or max_work_bytes != handle._header["native_work_bytes"]:
        c._fail("active_memory", "same genuine original Native work limit required")
    if not initial and len(retained) > _MAX_CHAIN:
        c._fail("active_chain", "bounded selected revision chain")
    nonces = {name.rsplit("-", 1)[1] for name in inventory["generations"] - set(retained)} | inventory["temps"]
    if len(nonces) + (0 if initial else 1) > handle._limits.max_orphans:
        c._fail("orphan_cap", "retained failed attempts exhaust original allowance")
    if (handle._limits.max_head_bytes, handle._limits.max_generation_bytes,
            handle._limits.max_intent_bytes) != (32 << 10, 128 << 10, 32 << 10):
        c._fail("metadata_cap", "unchanged complete finite control envelope required")
    reservation = _reservation_for_handle(handle, max_work_bytes, _metadata_stat_charge(handle, inventory))
    if initial:
        reservation = {**reservation, "disk_bytes": reservation["disk_bytes"] + (72 << 20)}
    if (inventory["bytes"] + reservation["disk_bytes"] > handle._limits.max_store_bytes
            or inventory["entries"] + reservation["entries"] > handle._limits.max_entries):
        c._fail("store_cap", "complete C+sibling attempt before Native/decode work")
    free = os.fstatvfs(handle._rootfd)
    if (free.f_bavail * free.f_frsize < handle._limits.free_reserve_bytes + reservation["disk_bytes"]
            or free.f_favail < handle._limits.free_reserve_inodes + reservation["entries"]):
        c._fail("storage_cap", "same filesystem free disk/inodes plus original reserves")
    handle._reservation_record = reservation
    return inventory


def _water_reference(root, header, limits, head, generation):
    water = generation[_ref_key(header)]
    if head["kind"] != "water_paused":
        if water is not None:
            c._fail("active_water_reference", "only a wet generation has a Native child")
        return
    c._keys(water, _water_fields(header), "native water reference")
    if _combined(header) and water["process_kind"] != "water":
        c._fail("active_water_reference", "actual water tag required")
    if (water["attempt_name"] != head["generation_name"]
            or type(water["ordinal"]) is not int or water["ordinal"] != head["pause_ordinal"]
            or type(water["pauses"]) is not int or water["pauses"] != head["pause_count"]):
        c._fail("active_water_reference", "actual owned attempt/progress identity")
    c._digest(water["export_sha256"], "actual original-writer export SHA")
    c._digest(water["plan_sha256"], "original plan identity")
    from masy import dense_active_checkpoint as a
    from maple_syrup import checkpoint
    a._float(water["origin_s"])
    a._float(water["native_time_s"])
    root_identity = (header["storage_contract"]["root_dev"], header["storage_contract"]["root_ino"])
    path = _managed_at(root, header, root_identity) / water["attempt_name"]
    if (path / "export.json").lstat().st_mode & 0o222:
        c._fail("active_water_reference", "original descriptor must remain readonly")
    raw = c._read(path / "export.json", 1 << 20)
    if c._sha(raw) != water["export_sha256"]:
        c._fail("active_water_reference", "retained original-writer descriptor bytes differ")
    export = a._decode(raw, 1 << 20)
    c._keys(export, ("schema", "capsule_sha256", "capsule_bytes", "identity", "raw_files", "layout"), "retained export")
    if export["schema"] != a.EXPORT_SCHEMA:
        c._fail("active_water_reference", "exact StageA export schema")
    c._digest(export["capsule_sha256"], "original entry capsule")
    c._integer(export["capsule_bytes"], 1 << 20, "actual capsule raw cap")
    if (path / "entry/capsule.json").lstat().st_mode & 0o222:
        c._fail("active_water_reference", "original capsule must remain readonly")
    capsule_raw = c._read(path / "entry/capsule.json", 1 << 20)
    if len(capsule_raw) != export["capsule_bytes"] or c._sha(capsule_raw) != export["capsule_sha256"]:
        c._fail("active_water_reference", "retained capsule bytes differ")
    a._raw_files(path / "native", checkpoint, c, expected=export["raw_files"])


def _generation_fields(value, head, header, limits):
    c._check_seal(value, _generation_fields_list(header), "generation_sha256", _schema(header) + ".generation")
    _head_fields(head, limits, header=header)
    _same_format(header, head, value)
    if any(type(value[k]) is not type(head[k]) or value[k] != head[k] for k in
           ("store_id", "store_file_sha256", "revision", "accepted_sequence", "kind",
            "event_operation_id", "request_id")):
        c._fail("active_generation", "typed HEAD/generation identity")
    nonce = _name(head["generation_name"], head["revision"])
    if value["attempt_id"] != nonce or type(value["attempt_id"]) is not str:
        c._fail("active_generation", "real attempt nonce")
    for k in ("intent_file_sha256", "ack_core_sha256"):
        c._digest(value[k], k)
    parent = value["parent_head_record"]
    if head["kind"] == "genesis":
        if any(value[k] is not None for k in ("parent_head_record", "parent_head_file_sha256",
                "parent_generation", "entry_anchor_head_record", "request_intent")):
            c._fail("active_generation", "genesis has no previous revision or request")
    else:
        _head_fields(parent, limits, header=header)
        if (parent["revision"] + 1 != head["revision"]
                or value["parent_generation"] != parent["generation_name"]
                or c._sha(c.oi._json(parent)) != value["parent_head_file_sha256"]):
            c._fail("active_generation", "exact direct selected-parent bytes")
        _check_intent_for(value["request_intent"], head)
        anchor = value["entry_anchor_head_record"]
        _head_fields(anchor, limits, header=header)
        if anchor["kind"] not in ("genesis", "event_accepted"):
            c._fail("active_anchor", "input anchor must be genuinely accepted")
        expected_anchor = (anchor["generation_name"] if _paused(head)
                           else head["generation_name"])
        if (head["anchor_generation"] != expected_anchor
                or anchor["store_id"] != head["store_id"]
                or (_paused(head)
                    and anchor["accepted_sequence"] != head["accepted_sequence"])):
            c._fail("active_anchor", "HEAD anchor differs from recorded entry anchor")
    if head["store_id"] != header["store_id"]:
        c._fail("active_generation", "foreign store")
    core = _ack_fields(value["ack_core"], head, parent, header=header)
    if (c.oi._json(core["request_intent"]) != c.oi._json(value["request_intent"])
            or c._sha(c.oi._json(core)) != value["ack_core_sha256"]):
        c._fail("active_ack", "exact canonical request/ACK bytes")
    return value


def _generation_chain(root, header, store_sha, head, limits):
    """Actual new-format ancestry validation; no Native LOAD or transport."""
    _header_fields(header, limits)
    root_identity = (header["storage_contract"]["root_dev"], header["storage_contract"]["root_ino"])
    _inventory_at(root, header, limits, root_identity)
    c._digest(header["schedule_sha256"], "actual original resolved schedule")
    # The whole metadata/owner reservation additionally precedes this parser
    # from its genuine facade entry and every new owner decode/copy.

    chain, seen = [], set()
    current = head
    child = None
    while True:
        if len(chain) >= _MAX_CHAIN:
            c._fail("active_chain", "derived25-row chain before append")
        _head_fields(current, limits, header=header)
        name = current["generation_name"]
        if name in seen or current["store_id"] != header["store_id"] or current["store_file_sha256"] != store_sha:
            c._fail("active_chain", "cycle or foreign store")
        seen.add(name)
        path = root / "generations" / name
        generation, raw = c._record(path / "generation.json", limits.max_generation_bytes,
                                     links=c._published_links(path, "generation.json"))
        _generation_fields(generation, current, header, limits)
        if c._sha(raw) != current["generation_file_sha256"] or generation["generation_sha256"] != current["generation_sha256"]:
            c._fail("active_chain", "actual generation file/semantic seal differs")
        intent, intent_raw = c._record(path / "intent.json", limits.max_intent_bytes)
        _intent_fields(intent, limits, header=header)
        if c._sha(intent_raw) != generation["intent_file_sha256"]:
            c._fail("active_chain", "actual reserved intent differs")
        for field in ("store_id", "attempt_id", "revision", "event_operation_id", "request_id", "request_intent"):
            if c.oi._json(intent[field]) != c.oi._json(generation[field]):
                c._fail("active_chain", "intent/generation binding")
        if (c.oi._json(intent["source_identity"]) != c.oi._json(header["source_identity"])
                or intent["environment_sha256"] != header["environment_sha256"]
                or intent["expected_parent_generation"] != generation["parent_generation"]
                or intent["expected_parent_head_file_sha256"] != generation["parent_head_file_sha256"]):
            c._fail("active_chain", "input parent/runtime binding")
        expected_input_sequence = (0 if generation["parent_head_record"] is None
                                   else generation["parent_head_record"]["accepted_sequence"])
        if intent["accepted_sequence"] != expected_input_sequence:
            c._fail("active_chain", "intent binds input cursor before unknown Native completion/pause outcome")
        entry_anchor = generation["entry_anchor_head_record"]
        if intent["entry_anchor_head_file_sha256"] != (None if entry_anchor is None else c._sha(c.oi._json(entry_anchor))):
            c._fail("active_chain", "input accepted-anchor bytes differ")
        image = generation["image"]
        owner_name = current["anchor_generation"] if _paused(current) else name
        if type(image) is not dict or image.get("relative_path") != f"generations/{owner_name}/image":
            c._fail("active_chain", "image must belong to genuine accepted anchor")
        manifest, wrapper_raw = c._image_for(header, root, image, limits)
        c._preflight_for(header, manifest, header["source_identity"], header["environment_sha256"], limits.image)
        if (c.oi._json(manifest["originals"]) != c.oi._json(header["originals"])
                or c.oi._json(manifest["genesis"]) != c.oi._json(header["genesis_physical"])
                or manifest["physical"]["relative_path"] != f"generations/{owner_name}/physical"):
            c._fail("active_chain", "original/genesis/own physical reference")
        artifacts = c._artifacts_for(header, root, manifest["originals"], manifest["genesis"],
                                      manifest["physical"], limits.image, wrapper_raw)
        try:
            for ref in artifacts.refs:
                artifacts.hashed(ref)
            artifacts.check()
        finally:
            artifacts.close()
        sequence = current["accepted_sequence"]
        if _combined(header) and _paused(current):
            expected_kind = "wind" if current["kind"] == "wind_paused" else "rain"
            if sequence >= len(header["event_kinds"]) or header["event_kinds"][sequence] != expected_kind:
                c._fail("active_process_kind", "selected pause differs from the STORE-bound original event kind")
        if (manifest["state"]["next_index"] != sequence or manifest["state"]["n_accepted"] != sequence
                or len(manifest["receipts"]) != sequence
                or c.oi._json(generation["accepted_summary"]) != c.oi._json(c._summary(manifest))):
            c._fail("active_chain", "accepted image/cursor/summary")
        receipt = c._receipt(manifest)
        if generation["accepted_receipt_sha256"] != (None if receipt is None else c._sha(receipt)):
            c._fail("active_chain", "last accepted receipt identity")
        if generation["ack_core"]["completion_receipt_sha256"] != (c._sha(receipt) if current["kind"] == "event_accepted" else None):
            c._fail("active_ack", "request completion receipt differs from actual B receipt")
        if child is not None:
            child_head, child_gen, child_manifest = child
            advance = child_head["kind"] == "event_accepted"
            if (child_head["revision"] != current["revision"] + 1
                    or child_head["accepted_sequence"] != sequence + advance
                    or c.oi._json(child_manifest["receipts"][:-1] if advance else child_manifest["receipts"])
                        != c.oi._json(manifest["receipts"])):
                c._fail("active_chain", "exact accepted prefix across paused/accepted revisions")
            anchor = current if not _paused(current) else generation["entry_anchor_head_record"]
            if c.oi._json(child_gen["entry_anchor_head_record"]) != c.oi._json(anchor):
                c._fail("active_anchor", "nearest genuine accepted input anchor differs")
            operation, request = _request_id(header, anchor, current, c.oi._json(current),
                                             child_gen["request_intent"], header["schedule_sha256"])
            if (child_gen["event_operation_id"] != operation or child_gen["request_id"] != request):
                c._fail("active_request", "actual schedule/parent/request identity differs")
            if _paused(child_head):
                if _combined(header) and _paused(current) and child_head["kind"] != current["kind"]:
                    c._fail("active_process_kind", "one unaccepted event cannot switch Native process")
                previous_count = current["pause_count"] if _paused(current) else 0
                if child_head["pause_count"] != previous_count + 1:
                    c._fail("active_pause", "pause count must advance independently of original ordinal")
                if _paused(current) and child_head["pause_ordinal"] <= current["pause_ordinal"]:
                    c._fail("active_pause", "original pause ordinal must increase")
        _active_reference(root, header, limits, current, generation)
        chain.append((current, generation, manifest))
        child = chain[-1]
        if current["kind"] == "genesis":
            if c.oi._json(manifest["physical"]) != c.oi._json(header["genesis_physical"]):
                c._fail("active_chain", "actual genesis physical differs")
            break
        current = generation["parent_head_record"]
    return chain


def _header_fields(header, limits):
    c._check_seal(header, _store_fields(header), "store_sha256", _schema(header))
    if (type(header["profile"]) is not str or header["profile"] != _profile_name(header)
            or type(header["store_id"]) is not str or not c._NONCE.fullmatch(header["store_id"])):
        c._fail("store_identity", "actual explicit active-water store/profile")
    if c._limits_record(c._limits_load(header["limits"])) != c._limits_record(limits):
        c._fail("limit_identity", "actual original persisted limits required")
    c._integer(header["native_work_bytes"], 6 << 30, "genuine original work cap")
    if header["native_work_bytes"] == 0:
        c._fail("active_memory", "actual positive original Native work limit required")
    c._digest(header["schedule_sha256"], "actual resolved original schedule")
    c._digest(header["environment_sha256"], "actual controlling environment")
    c.b._reference(header["originals"], c.oi.SCHEMA)
    c.b._reference(header["genesis_physical"], "maple-exact-physical-checkpoint/1")
    if (not re.fullmatch(r"originals/o-[0-9a-f]{32}", header["originals"]["relative_path"])
            or not re.fullmatch(r"generations/g-0000-[0-9a-f]{32}/physical",
                                 header["genesis_physical"]["relative_path"])):
        c._fail("store_identity", "actual original/genesis paths")
    c._keys(header[_managed_key(header)], _MANAGED, "managed sibling")
    if _combined(header):
        kinds = header["event_kinds"]
        if (type(kinds) is not list or not 1 <= len(kinds) <= min(8, limits.image.max_events)
                or any(type(kind) is not str or kind not in ("wind", "rain") for kind in kinds)):
            c._fail("active_process_kind", "bounded actual original event-kind list required")
    return header


def _head(handle):
    value, raw = c._record(handle._root / "HEAD", handle._limits.max_head_bytes)
    _head_fields(value, handle._limits, header=handle._header)
    _same_format(handle._header, value)
    if value["store_id"] != handle._header["store_id"] or value["store_file_sha256"] != c._sha(handle._store_raw):
        c._fail("active_head", "actual selected store identity")
    return value, raw


def _temporary_heads(handle, inventory):
    for nonce in inventory["temps"]:
        head, _raw = c._record(handle._root / f"HEAD.{nonce}.tmp", handle._limits.max_head_bytes)
        _head_fields(head, handle._limits, header=handle._header)
        if _name(head["generation_name"], head["revision"]) != nonce or head["generation_name"] not in inventory["generations"]:
            c._fail("head_temporary", "exact own retained generation/nonce")
        _generation_chain(handle._root, handle._header, c._sha(handle._store_raw), head, handle._limits)


def _stable(handle):
    handle._binding()
    _header_fields(handle._header, handle._limits)
    inventory = _inventory(handle)
    # Existing and maximum proposed metadata are charged before read/hash/parse.
    _reservation_for_handle(handle, handle._header["native_work_bytes"],
                            _metadata_stat_charge(handle, inventory))
    raw = c._read(handle._root / "STORE.json", 128 << 10,
                  links=c._published_links(handle._root, "STORE.json"))
    if raw != handle._store_raw:
        c._fail("identity_drift", "immutable actual STORE changed")
    handle._storage()
    _managed(handle)
    source, environment = c._operation_runtime_binding(handle)
    if c.oi._json(source) != c.oi._json(handle._header["source_identity"]) or environment != handle._header["environment_sha256"]:
        c._fail("identity_drift", "actual current Source/environment differs")
    _temporary_heads(handle, inventory)


def _prepare_anchor(handle, anchor):
    _head_fields(anchor, handle._limits, header=handle._header)
    if anchor["kind"] not in ("genesis", "event_accepted"):
        c._fail("active_anchor", "accepted anchor preparation only")
    inventory = _inventory(handle)
    _reservation_for_handle(handle, handle._header["native_work_bytes"], _metadata_stat_charge(handle, inventory))
    chain = _generation_chain(handle._root, handle._header, c._sha(handle._store_raw), anchor, handle._limits)
    image = chain[0][1]["image"]
    prepared = c._prepare_for(handle._header, handle._root / image["relative_path"],
        artifact_root=handle._root, expected_manifest_sha256=image["manifest_sha256"],
        expected_source_identity=handle._header["source_identity"],
        expected_environment_sha256=handle._header["environment_sha256"], limits=handle._limits.image)
    owner = c._owner_for(handle._header, prepared)
    if (owner._limits.max_work_bytes != handle._header["native_work_bytes"]
            or owner._schedule.digest != handle._header["schedule_sha256"]
            or c.oi._json(prepared.diagnostics()["accepted"]) != c.oi._json(chain[0][1]["accepted_summary"])):
        c._fail("readback_identity", "actual Native/original owner limits/schedule/accepted state differ")
    if _combined(handle._header) and [event.kind for event in owner._schedule.events] != handle._header["event_kinds"]:
        c._fail("active_process_kind", "genuine original schedule differs from STORE-bound event kinds")
    return prepared


def _load(handle, head):
    inventory = _inventory(handle)
    _reservation_for_handle(handle, handle._header["native_work_bytes"], _metadata_stat_charge(handle, inventory))
    chain = _generation_chain(handle._root, handle._header, c._sha(handle._store_raw), head, handle._limits)
    prepared = _prepare_anchor(handle, _anchor_row(chain)[0])
    return prepared, chain


def _acknowledgement(header, head, generation, manifest):
    core_raw = _metadata_bytes(generation["ack_core"], 128 << 10, "bounded actual ACK")
    receipt = c._receipt(manifest) if head["kind"] == "event_accepted" else None
    return ActiveWaterAcknowledgement(_schema(header) + ".ack", header["store_id"],
        head["event_operation_id"], head["request_id"], head["revision"],
        head["accepted_sequence"], c._token(head), core_raw, receipt)


def _idle_state(handle, head, prepared, generation):
    owner = c._owner_for(handle._header, prepared)
    diagnostics = prepared.diagnostics()
    diagnostics[_progress_key(handle._header)] = copy.deepcopy(generation[_ref_key(handle._header)])
    if _combined(handle._header):
        diagnostics["next_event_kind"] = _next_kind(owner)
    # Keep only immutable bounded metadata on the facade, no live Native owner.
    ack = _acknowledgement(handle._header, head, generation,
                           c._image_for(handle._header, handle._root, generation["image"], handle._limits)[0])
    base = c._State("idle", copy.deepcopy(head), c._token(head), owner.report(), diagnostics, ack,
        sequence=head["accepted_sequence"], intended=c._token(head),
        max_work_bytes=owner._limits.max_work_bytes, schedule_sha256=owner._schedule.digest)
    halt = dataclasses.replace(base, phase="halted_before_head", intended=None)
    return dataclasses.replace(base, identity_halt=halt)


def report(handle):
    handle._binding()
    head = handle._state.head
    progress = None if head is None or not _paused(head) else {
        "revision": head["revision"], "original_boundary_ordinal": head["pause_ordinal"],
        "pause_count": head["pause_count"], "accepted_sequence": head["accepted_sequence"],
        "native_time_s": handle._state.diagnostics[_progress_key(handle._header)]["native_time_s"],
        "native_origin_s": handle._state.diagnostics[_progress_key(handle._header)]["origin_s"],
        "scope": "unaccepted Native progress; no accepted elapsed/cursor increment"}
    return {"schema": _schema(handle._header) + ".report", "phase": handle._state.phase,
        "head_token": handle._state.token, "intended_token": handle._state.intended,
        "store_file_sha256": None if handle._store_raw is None else c._sha(handle._store_raw),
        "accepted": copy.deepcopy(handle._state.report), "diagnostics": copy.deepcopy(handle._state.diagnostics),
        _progress_key(handle._header): progress, "reservation": copy.deepcopy(handle._reservation_record),
        "failure_evidence_error": handle._evidence_write_error,
        "scope": ("opt-in static Dense CPU active-process; production/dynamic/GPU/windowed separate"
                  if _combined(handle._header) else
                  "opt-in static Dense CPU active-water; default/dynamic/wind-active/GPU/windowed separate")}


def _declare():
    """Existing actual lazy declarations precede genuine C environment binding."""
    from masy import dense_active_checkpoint as a
    a._declare()
    c.oi._native()
    c.oi._registry()


def _make_managed(handle, store_id, probe, *, combined=False):
    path = handle._root.parent / ((".masy-active-" if combined else ".masy-water-") + store_id)
    if len(str(path)) > _PATH_CHARS:
        c._fail("active_sibling", "bounded canonical final sibling path")
    parent = c.oi._open_dir(handle._root.parent)
    sibling = None
    try:
        outer = os.fstat(parent)
        os.mkdir(path.name, 0o700, dir_fd=parent)
        os.fsync(parent)
        sibling = c.oi._open_dir(path)
        info = os.fstat(sibling)
        if info.st_dev != handle._root_identity[0] or os.fstatvfs(sibling).f_fsid != probe["f_fsid"]:
            c._fail("active_sibling", "same verified filesystem required before any child")
        c._same_directory(path, sibling)
        c._same_directory(path.parent, parent)
        return {"path": str(path), "dev": info.st_dev, "ino": info.st_ino,
                "parent_path": str(path.parent), "parent_dev": outer.st_dev,
                "parent_ino": outer.st_ino, "fs_id": os.fstatvfs(sibling).f_fsid}
    finally:
        if sibling is not None:
            os.close(sibling)
        os.close(parent)


def _create_active_store(path, *, runner, originals, runtime_reference,
                         limits=c._DEFAULT, storage_contract, combined=False):
    """Create the new opt-in format from an untouched genuine static runner only."""
    _protocol = {"schema": SCHEMA_PROCESS if combined else SCHEMA,
                 "profile": PROFILE_PROCESS if combined else PROFILE}
    c.BoundaryStoreLimits.__post_init__(limits)
    c.StorageContract.__post_init__(storage_contract)
    if type(runner) is not c.dr.DenseRunner:
        c._fail("runner_profile", "exact static Dense CPU genesis runner required")
    c.dr.DenseRunnerLimits.__post_init__(runner._limits)
    _reservation(limits, runner._limits.max_work_bytes, combined=combined)
    _declare_for(_protocol)
    c.b._runner_admission(runner, limits.image)
    if combined:
        _wind_profile(runner, runtime_reference)
    if (runner._next != 0 or runner._n_accepted != 0 or runner._receipts
            or runner._stop_receipt is not None or runner._active_evidence is not None):
        c._fail("fresh_runner_required", "no store migration or progressed runner import")
    if type(originals) is not c.oi.OriginalInputs:
        c._fail("original_identity", "actual newly captured static OriginalInputs required")
    source, environment = c._runtime_binding(runtime_reference, limits)
    if originals.source_identity != source:
        c._fail("original_identity", "actual complete Source differs from originals")
    handle = c.DenseBoundaryStore._open(path, limits, create=True)
    try:
        probe, probe_sha = c._probe(handle._root, handle._rootfd)
        os.mkdir(handle._root / "originals", 0o700)
        os.mkdir(handle._root / "generations", 0o700)
        store_id = uuid.uuid4().hex
        managed = _make_managed(handle, store_id, probe, combined=combined)
        handle._header = {"schema": _schema(_protocol), "store_id": store_id, "profile": _profile_name(_protocol),
            "source_identity": source, "runtime_reference": copy.deepcopy(runtime_reference),
            "environment_sha256": environment,
            "storage_contract": {"schema": _schema(_protocol) + ".storage", "operator_declares_local": True,
                "operator_declares_fsync_durable": True,
                "observed_fs_type": probe["observation"]["fs_type"], "observed_fs_id": probe["f_fsid"],
                "root_path": str(handle._root), "root_dev": probe["root_dev"], "root_ino": probe["root_ino"],
                "probe_file_sha256": probe_sha}, "limits": c._limits_record(limits),
            "originals": None, "genesis_physical": None, _managed_key(_protocol): managed,
            "native_work_bytes": runner._limits.max_work_bytes, "schedule_sha256": runner._schedule.digest}
        if combined:
            handle._header["event_kinds"] = [event.kind for event in runner._schedule.events]
        _admit(handle, [], runner._limits.max_work_bytes, initial=True)
        # Both memory and combined free/disk/inodes were admitted before recapture.
        captured = c.oi.capture_original_inputs(runner, source_identity=originals.source_identity)
        if (captured.content_sha256 != originals.content_sha256
                or captured.static_sha256 != originals.static_sha256
                or originals.runner_limits.max_work_bytes != runner._limits.max_work_bytes
                or originals.schedule.digest != runner._schedule.digest):
            c._fail("original_identity", "independently recaptured original/limits/schedule differ")
        del captured
        original_path = handle._root / "originals" / ("o-" + uuid.uuid4().hex)
        c.oi.save_original_inputs(original_path, originals,
            limits=c.oi.OriginalPackLimits(max_cells=limits.image.max_cells,
                max_classes=limits.image.max_classes, max_voxels=limits.image.max_voxels,
                max_cohorts=limits.image.max_cohorts, max_events=limits.image.max_events))
        handle._header["originals"] = c._reference(original_path, c.oi.SCHEMA,
                                                   originals.content_sha256, handle._root)
        operation = c._operation(store_id, None, 0, runner._schedule.digest)
        request = c._sha(c.oi._json([_schema(_protocol) + ".genesis", store_id, runner._schedule.digest]))
        directory, intent, halt = _reserve_attempt(handle, None, None, None, operation, request)
        physical = handle._save_physical(directory / "physical", runner, originals)
        handle._header["genesis_physical"] = physical
        handle._header = _seal_control(handle._header, "store_sha256", 128 << 10, "active STORE")
        c._sealed_write(handle._root, "STORE.json", handle._header, 128 << 10)
        handle._store_raw = c.oi._json(handle._header)
        image = handle._save_image(directory / "image", runner, physical)
        head = _seal_generation(handle, directory, intent, None, None, image, "genesis", None)
        prepared = _prepare_anchor(handle, head)
        c._compare_owners(runner, c._owner_for(handle._header, prepared), limits.image)
        generation = c._record(directory / "generation.json", limits.max_generation_bytes,
                               links=c._published_links(directory, "generation.json"))[0]
        state = _idle_state(handle, head, prepared, generation)
        del prepared
        _inventory(handle)
        c._sync_tree(_managed(handle))
        c._sync_tree(handle._root)
        handle._state = halt
        handle._entered = False
        handle._publish(head, state, None, halt)
        return handle
    except BaseException as primary:
        try:
            handle._entered = False
            handle.close()
        except BaseException as closing:
            primary.add_note("closing unpublished active store refused: " + type(closing).__name__[:64])
        raise


def _load_selected_water(handle, context, runner, generation):
    from masy import dense_active_checkpoint as a
    water = generation[_ref_key(handle._header)]
    path = _managed(handle) / water["attempt_name"]
    token = a.load_water_pause(context, runner, attempt=path,
        expected_export_sha256=water["export_sha256"], runtime_reference=handle._header["runtime_reference"])
    from masy import dense_water_pause as pause
    lease = pause._lease(runner, token)
    if (lease.ordinal != water["ordinal"] or lease.pauses != water["pauses"]
            or lease.origin.hex() != water["origin_s"]
            or float(lease.progress.result.state.t_s).hex() != water["native_time_s"]
            or c._sha(c.oi._json([x.hex() for x in lease.plan])) != water["plan_sha256"]):
        c._fail("active_water_reference", "genuine Native LOAD progress differs from selected metadata")
    return token


def _load_selected_wind(handle, context, runner, generation):
    from masy import dense_active_wind_checkpoint as wind, dense_wind_pause as pause
    from masy.clock import decode_time
    ref = generation["active_ref"]
    token = wind.load_wind_pause(context, runner, attempt=_managed(handle) / ref["attempt_name"],
        expected_export_sha256=ref["export_sha256"], runtime_reference=handle._header["runtime_reference"])
    lease = pause._lease(runner, token)
    if (lease.ordinal != ref["ordinal"] or lease.pauses != ref["pauses"]
            or lease.origin.hex() != ref["origin_s"]
            or decode_time(lease.progress["native_time"]).hex() != ref["native_time_s"]
            or c._sha(c.oi._json([x.hex() for x in lease.prepared.plan])) != ref["plan_sha256"]):
        c._fail("active_wind_reference", "genuine Native restore progress differs from selected metadata")
    return token


def _load_selected(handle, context, runner, generation):
    if _combined(handle._header) and generation["active_ref"]["process_kind"] == "wind":
        return _load_selected_wind(handle, context, runner, generation)
    return _load_selected_water(handle, context, runner, generation)


def _recover_active_store(path, *, expected_store_file_sha256, limits=c._DEFAULT, combined=False):
    """Validate exactly selected revision, with real wet LOAD and zero transport."""
    c.BoundaryStoreLimits.__post_init__(limits)
    c._digest(expected_store_file_sha256, "external exact active STORE SHA")
    # Unknown stored W is not used until typed header admission; this fixed
    # metadata prefix is admitted against the original independent C bound.
    _protocol = {"schema": SCHEMA_PROCESS if combined else SCHEMA,
                 "profile": PROFILE_PROCESS if combined else PROFILE}
    _reservation(limits, 0, combined=combined)
    _declare_for(_protocol)
    handle = c.DenseBoundaryStore._open(path, limits, create=False)
    context = None
    try:
        _inventory_at(handle._root, None, limits, handle._root_identity, sibling=False)
        header, raw = c._record(handle._root / "STORE.json", 128 << 10,
                                links=c._published_links(handle._root, "STORE.json"))
        if c._sha(raw) != expected_store_file_sha256:
            c._fail("store_identity", "actual externally selected STORE differs")
        _header_fields(header, limits)
        if _combined(header) is not combined:
            c._fail("store_identity", "explicit recovery format differs from stored profile")
        handle._header, handle._store_raw = header, raw
        handle._stable()
        head, head_raw = handle._head()
        handle._state = dataclasses.replace(handle._state, head=copy.deepcopy(head),
            token=c._token(head), sequence=head["accepted_sequence"],
            max_work_bytes=header["native_work_bytes"], schedule_sha256=header["schedule_sha256"])
        prepared, chain = _load(handle, head)
        state = _idle_state(handle, head, prepared, chain[0][1])
        selected_generation = chain[0][1]
        del prepared, chain
        if _paused(head):
            handle._entered = True
            c._new_runtime_validation(handle)
            context = _borrow_parent(handle, None)
            context.__enter__()
            runner = context.take_runner()
            _load_selected(handle, context, runner, selected_generation)
            _close_parent(context)
            c._drop_runtime_validation(handle)
            context = None
            del runner
            handle._entered = False
        _inventory(handle)
        c._sync_tree(_managed(handle))
        c._sync_tree(handle._root)
        handle._stable()
        if handle._head()[1] != head_raw:
            c._fail("parent_drift", "selected HEAD changed during real recovery")
        os.fsync(handle._rootfd)
        handle._state = state
        return handle
    except BaseException as primary:
        c._drop_runtime_validation(handle)
        unclosed = False
        if context is not None:
            try:
                _close_parent(context)
            except BaseException as closing:
                primary.add_note("active recovery context close refused: " + type(closing).__name__[:64])
            unclosed = context._phase != "closed"
        handle._entered = unclosed
        try:
            handle.close()
        except BaseException as closing:
            primary.add_note("active recovery handle close refused: " + type(closing).__name__[:64])
        raise


def _reserve_attempt(handle, selected, anchor, request_intent, operation, request_id):
    attempt = uuid.uuid4().hex
    revision = c._integer(0 if selected is None else selected["revision"] + 1,
                          _MAX_REVISION, "attempt revision")
    sequence = 0 if selected is None else selected["accepted_sequence"]
    name = f"g-{revision:04d}-{attempt}"
    old = handle._state
    busy = dataclasses.replace(old, phase="busy_before_head", attempt_id=attempt,
                               sequence=sequence, intended=None)
    halt = dataclasses.replace(busy, phase="halted_before_head")
    handle._attempt_halt = halt
    handle._state = busy
    try:
        path = handle._root / "generations" / name
        directory = c.oi._open_dir(handle._root / "generations")
        try:
            os.mkdir(name, 0o700, dir_fd=directory)
            os.fsync(directory)
        finally:
            os.close(directory)
        intent = _seal_control({"schema": _schema(handle._header) + ".intent", "store_id": handle._header["store_id"],
            "attempt_id": attempt, "revision": revision, "accepted_sequence": sequence,
            "expected_parent_head_file_sha256": None if selected is None else c._sha(c.oi._json(selected)),
            "expected_parent_generation": None if selected is None else selected["generation_name"],
            "entry_anchor_head_file_sha256": None if anchor is None else c._sha(c.oi._json(anchor)),
            "event_operation_id": operation, "request_id": request_id,
            "request_intent": copy.deepcopy(request_intent),
            "source_identity": handle._header["source_identity"],
            "environment_sha256": handle._header["environment_sha256"]}, "intent_sha256",
            handle._limits.max_intent_bytes, "active reserved intent")
        _intent_fields(intent, handle._limits, header=handle._header)
        c._write(path / "intent.json", c.oi._json(intent))
        fd = c.oi._open_dir(path)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return path, intent, halt
    except BaseException:
        handle._state = halt
        raise


def _seal_generation(handle, directory, intent, selected, anchor, image, kind, water_ref):
    header, limits = handle._header, handle._limits
    manifest, _raw = c._image_for(header, handle._root, image, limits)
    receipt = c._receipt(manifest)
    sequence = manifest["state"]["next_index"]
    if (kind not in _kinds(header)
            or intent["revision"] != (0 if selected is None else selected["revision"] + 1)
            or intent["expected_parent_generation"] != (None if selected is None else selected["generation_name"])
            or intent["expected_parent_head_file_sha256"] != (None if selected is None else c._sha(c.oi._json(selected)))
            or intent["entry_anchor_head_file_sha256"] != (None if anchor is None else c._sha(c.oi._json(anchor)))
            or type(sequence) is not int
            or sequence != intent["accepted_sequence"] + (kind == "event_accepted")):
        c._fail("active_generation", "sealed intent/selected/anchor/output cursor")
    store_sha = c._sha(handle._store_raw)
    ack = {"schema": _schema(header) + ".ack", "store_id": header["store_id"],
        "event_operation_id": intent["event_operation_id"], "request_id": intent["request_id"],
        "request_intent": copy.deepcopy(intent["request_intent"]), "returned_kind": kind,
        "revision": intent["revision"], "accepted_sequence": sequence,
        "selected_parent_token": None if selected is None else c._token(selected),
        "pause_ordinal": None if water_ref is None else water_ref["ordinal"],
        "pause_count": None if water_ref is None else water_ref["pauses"],
        "completion_receipt_sha256": (c._sha(receipt) if kind == "event_accepted" and receipt is not None
                                      else None)}
    generation = _seal_control({"schema": _schema(header) + ".generation", "store_id": header["store_id"],
        "store_file_sha256": store_sha, "attempt_id": intent["attempt_id"],
        "revision": intent["revision"], "accepted_sequence": sequence, "kind": kind,
        "parent_head_record": copy.deepcopy(selected),
        "parent_head_file_sha256": None if selected is None else c._sha(c.oi._json(selected)),
        "parent_generation": None if selected is None else selected["generation_name"],
        "entry_anchor_head_record": copy.deepcopy(anchor),
        "event_operation_id": intent["event_operation_id"], "request_id": intent["request_id"],
        "request_intent": copy.deepcopy(intent["request_intent"]),
        "intent_file_sha256": c._sha(c.oi._json(intent)), "image": image,
        "accepted_summary": c._summary(manifest),
        "accepted_receipt_sha256": None if receipt is None else c._sha(receipt),
        _ref_key(header): copy.deepcopy(water_ref), "ack_core": ack,
        "ack_core_sha256": c._sha(c.oi._json(ack))}, "generation_sha256",
        limits.max_generation_bytes, "active generation")
    raw = _metadata_bytes(generation, limits.max_generation_bytes, "active generation")
    head = _head_record(header, store_sha, directory.name, generation, c._sha(raw))
    _generation_fields(generation, head, header, limits)
    file_sha = c._sealed_write(directory, "generation.json", generation, limits.max_generation_bytes)
    if file_sha != head["generation_file_sha256"]:
        c._fail("metadata_readback", "sealed generation bytes")
    return head


def _lookup(handle, expected_parent, intent, chain):
    for index, (head, generation, manifest) in enumerate(chain):
        if c._token(head) != expected_parent:
            continue
        if index == 0:
            c._fail("request_not_published", "current selected parent has no direct selected child")
        child_head, child_generation, child_manifest = chain[index - 1]
        if (c.oi._json(child_generation["parent_head_record"]) != c.oi._json(head)
                or c.oi._json(child_generation["request_intent"]) != c.oi._json(intent)):
            c._fail("request_conflict", "different direct-child request; no Native replay")
        return _acknowledgement(handle._header, child_head, child_generation, child_manifest)
    c._fail("stale_parent", "token is not selected ancestry; no orphan/fallback adoption")


def acknowledge(handle, expected_parent, *, request_kind, boundary_ordinal=None):
    """Metadata-only lost-ACK query; no Native LOAD, transport or implicit retry."""
    handle._binding()
    c._digest(expected_parent, "expected selected parent token")
    intent = _intent_for(request_kind, boundary_ordinal, handle._header)
    if handle._entered:
        c._fail("handle_reentrant", "one store operation at a time")
    if handle._state.phase != "idle":
        c._fail("store_halted", handle._state.phase)
    handle._entered = True
    try:
        handle._stable()
        head, _raw = handle._head()
        if head != handle._state.head:
            c._fail("parent_drift", "actual selected HEAD differs from facade")
        chain = _generation_chain(handle._root, handle._header, c._sha(handle._store_raw), head, handle._limits)
        return _lookup(handle, expected_parent, intent, chain)
    finally:
        handle._entered = False


def _paused_state(handle, head, work, diagnostics_raw, generation):
    # Actual B diagnostics were captured from the real loaded accepted A before
    # Native work. Actual export checks reverified A before/after writer IO.
    diagnostics = c.oi._strict_json(diagnostics_raw, c._JSON)
    diagnostics[_progress_key(handle._header)] = copy.deepcopy(generation[_ref_key(handle._header)])
    if _combined(handle._header):
        diagnostics["next_event_kind"] = _next_kind(work)
    report_ = work.report()
    if c.oi._json(report_["accepted"]) != c.oi._json(generation["accepted_summary"]):
        c._fail("readback_identity", "unaccepted water changed actual accepted A")
    ack = _acknowledgement(handle._header, head, generation,
        c._image_for(handle._header, handle._root, generation["image"], handle._limits)[0])
    base = c._State("idle", copy.deepcopy(head), c._token(head), report_, diagnostics, ack,
        sequence=head["accepted_sequence"], intended=c._token(head),
        max_work_bytes=work._limits.max_work_bytes, schedule_sha256=work._schedule.digest)
    halt = dataclasses.replace(base, phase="halted_before_head", intended=None)
    return dataclasses.replace(base, identity_halt=halt)


def _operate(handle, expected_parent, request_intent):
    from masy import dense_active_checkpoint as a, dense_water_pause as pause
    handle._binding()
    c._digest(expected_parent, "expected selected parent token")
    request_intent = _check_intent_for(request_intent, handle._header)
    if handle._entered:
        c._fail("handle_reentrant", "one store operation at a time")
    if handle._state.phase != "idle":
        c._fail("store_halted", handle._state.phase)
    old = handle._state
    halt = dataclasses.replace(old, phase="halted_before_head", intended=None)
    handle._entered = True
    handle._publication_uncertain = handle._attempt_halt = None
    context = reader_context = None
    directory = None
    work = reader = prepared = lease = None
    policy = False  # Error-origin classification only; never Native authority.
    try:
        c._new_runtime_validation(handle)
        handle._stable()
        selected, selected_raw = handle._head()
        if selected != old.head:
            c._fail("parent_drift", "durable selected R differs from facade")
        chain = _generation_chain(handle._root, handle._header, c._sha(handle._store_raw),
                                  selected, handle._limits)
        policy = True
        if expected_parent != old.token:
            c._drop_runtime_validation(handle)
            answer = _lookup(handle, expected_parent, request_intent, chain)
            handle._entered = False
            return answer
        if old.report["status"] == "complete" or selected["accepted_sequence"] >= handle._limits.image.max_events:
            c._fail("schedule_complete", "all original events accepted")
        if _combined(handle._header):
            expected_kind = old.diagnostics["next_event_kind"]
            requested_kind = {"checkpoint_water": "rain", "checkpoint_wind": "wind"}.get(request_intent["kind"])
            if requested_kind is not None and requested_kind != expected_kind:
                c._fail("active_process_kind", "request differs from the actual next accepted-anchor event")
            if _paused(selected) and request_intent["kind"] != "finish":
                if selected["pause_count"] >= _MAX_PAUSES:
                    c._fail("wind_pause_limit" if selected["kind"] == "wind_paused" else "water_pause_limit",
                            "unchanged at most two pauses per event")
                if request_intent["boundary_ordinal"] <= selected["pause_ordinal"]:
                    c._fail("wind_pause_ordinal" if selected["kind"] == "wind_paused" else "water_pause_ordinal",
                            "original ordinal must strictly increase")
        if request_intent["kind"] == "checkpoint_water" and _paused(selected):
            if selected["pause_count"] >= _MAX_PAUSES:
                c._fail("water_pause_limit", "unchanged at most two wet pauses per event")
            if request_intent["boundary_ordinal"] <= selected["pause_ordinal"]:
                c._fail("water_pause_ordinal", "original ordinal must strictly increase")
        policy = False
        anchor, anchor_generation, _manifest = _anchor_row(chain)
        operation, request_id = _request_id(handle._header, anchor, selected, selected_raw,
                                             request_intent, old.schedule_sha256)
        policy = True
        _admit(handle, [r[0]["generation_name"] for r in chain], old.max_work_bytes)
        policy = False
        directory, intent, halt = _reserve_attempt(handle, selected, anchor, request_intent, operation, request_id)
        context = _borrow_parent(handle, intent)
        context.__enter__()
        work = context.take_runner()
        diagnostics_raw = _selection(context).diagnostics_raw
        handle._failed_owner = work
        token = (_load_selected(handle, context, work, chain[0][1])
                 if _paused(selected) else None)
        # Drop old full metadata rows before Native work; all accepted versions
        # remain retained on disk and revalidated by actual parent checks.
        chain = None
        wind = None
        if _combined(handle._header):
            from masy import dense_wind_pause as wind
            actual_kind = _next_kind(work)
            if actual_kind != old.diagnostics["next_event_kind"]:
                c._fail("active_process_kind", "actual loaded anchor schedule differs from preflight metadata")
            if actual_kind == "wind" and request_intent["kind"] == "finish" and token is None:
                # Checkpoint/resume already perform their unchanged owning
                # profile/runtime checks. Ordinary step has no pause entry.
                _wind_profile(work, handle._header["runtime_reference"])
        if request_intent["kind"] == "checkpoint_wind":
            ordinal = request_intent["boundary_ordinal"]
            result = (wind.pause_wind(work, runtime_reference=handle._header["runtime_reference"],
                                      boundary_ordinal=ordinal) if token is None
                      else wind.resume_wind(work, token, boundary_ordinal=ordinal))
        elif request_intent["kind"] == "checkpoint_water":
            ordinal = request_intent["boundary_ordinal"]
            result = (pause.pause_water(work, runtime_reference=handle._header["runtime_reference"],
                                      boundary_ordinal=ordinal) if token is None
                      else pause.resume_water(work, token, boundary_ordinal=ordinal))
        else:
            if wind is not None and selected["kind"] == "wind_paused":
                result = wind.resume_wind(work, token)
            else:
                result = work.step() if token is None else pause.resume_water(work, token)
        if type(result) is pause._WaterToken:
            lease = pause._lease(work, result)
            export_path = _managed(handle) / directory.name
            export_sha = a.export_water_pause(context, work, result, attempt=export_path)
            water = {"attempt_name": directory.name, "export_sha256": export_sha,
                     "ordinal": lease.ordinal, "pauses": lease.pauses,
                     "origin_s": lease.origin.hex(), "native_time_s": float(lease.progress.result.state.t_s).hex(),
                     "plan_sha256": c._sha(c.oi._json([x.hex() for x in lease.plan]))}
            if _combined(handle._header):
                water = {"process_kind": "water", **water}
            # A second real accepted-A parent and actual anchored Native LOAD
            # prove the new child; metadata-only hash checks never replace it.
            reader_context = _borrow_parent(handle, intent)
            reader_context.__enter__()
            reader = reader_context.take_runner()
            _load_selected_water(handle, reader_context, reader, {_ref_key(handle._header): water})
            _close_parent(reader_context)
            reader_context = None
            reader = None
            image = anchor_generation["image"]
            head = _seal_generation(handle, directory, intent, selected, anchor,
                                     image, "water_paused", water)
            generation = c._record(directory / "generation.json", handle._limits.max_generation_bytes,
                                    links=c._published_links(directory, "generation.json"))[0]
            state = _paused_state(handle, head, work, diagnostics_raw, generation)
        elif wind is not None and type(result) is wind._WindToken:
            from masy import dense_active_wind_checkpoint as checkpoint
            from masy.clock import decode_time
            lease = wind._lease(work, result)
            export_sha = checkpoint.export_wind_pause(context, work, result,
                                                       attempt=_managed(handle) / directory.name)
            active = {"process_kind": "wind", "attempt_name": directory.name,
                "export_sha256": export_sha, "ordinal": lease.ordinal, "pauses": lease.pauses,
                "origin_s": lease.origin.hex(),
                "native_time_s": decode_time(lease.progress["native_time"]).hex(),
                "plan_sha256": c._sha(c.oi._json([x.hex() for x in lease.prepared.plan]))}
            # Exactly the same fresh accepted-A context discipline as wet LOAD:
            # real codec restore/progress, then genuine abandon before HEAD.
            reader_context = _borrow_parent(handle, intent)
            reader_context.__enter__()
            reader = reader_context.take_runner()
            _load_selected_wind(handle, reader_context, reader, {"active_ref": active})
            _close_parent(reader_context)
            reader_context = None
            reader = None
            head = _seal_generation(handle, directory, intent, selected, anchor,
                                     anchor_generation["image"], "wind_paused", active)
            generation = c._record(directory / "generation.json", handle._limits.max_generation_bytes,
                                    links=c._published_links(directory, "generation.json"))[0]
            state = _paused_state(handle, head, work, diagnostics_raw, generation)
        elif (type(result) is dict and result.get("status") == "accepted"
                and work._next == selected["accepted_sequence"] + 1):
            # Genuine original finish/ordinary step only; no copied finish tail.
            c._admit_owner_for(handle._header, work, handle._limits.image)
            anchor_manifest, wrapper_raw = c._image_for(handle._header, handle._root,
                                                         anchor_generation["image"], handle._limits)
            originals = handle._extra_original(anchor_manifest, wrapper_raw)
            physical = handle._save_physical(directory / "physical", work, originals)
            image = handle._save_image(directory / "image", work, physical)
            head = _seal_generation(handle, directory, intent, selected, anchor,
                                     image, "event_accepted", None)
            prepared = _prepare_anchor(handle, head)
            c._compare_for(handle._header, work, c._owner_for(handle._header, prepared), handle._limits.image)
            generation = c._record(directory / "generation.json", handle._limits.max_generation_bytes,
                                    links=c._published_links(directory, "generation.json"))[0]
            state = _idle_state(handle, head, prepared, generation)
            prepared = originals = None
        else:
            c._fail("event_incomplete", "actual Native limit/error is not accepted completion")
        # All real contexts/Native leases close before the unchanged publisher.
        _close_parent(context)
        context = None
        work = token = result = lease = None
        handle._failed_owner = None
        _inventory(handle)
        c._sync_tree(_managed(handle))
        c._sync_tree(handle._root)
        c._final_runtime_validation(handle)
        handle._stable()
        c._drop_runtime_validation(handle)
        if handle._head()[1] != selected_raw:
            c._fail("parent_drift", "selected parent changed before publication")
        handle._state = halt
        handle._entered = False
    except BaseException as primary:
        c._drop_runtime_validation(handle)
        unclosed = False
        for temporary in (reader_context, context):
            if temporary is not None:
                try:
                    _close_parent(temporary)
                except BaseException as closing:
                    primary.add_note("prepublication context close refused: " + type(closing).__name__[:64])
                unclosed = unclosed or temporary._phase != "closed"
        if handle._publication_uncertain is not None:
            handle._state = handle._publication_uncertain
        elif handle._attempt_halt is not None:
            handle._state = handle._attempt_halt
        elif (policy and directory is None and context is None and reader_context is None
                and handle._state is old and type(primary) is c.BoundaryStoreError
                and primary.code in ("schedule_complete", "water_pause_limit",
                    "water_pause_ordinal", "request_conflict", "request_not_published",
                    "stale_parent", "memory_cap", "store_cap", "storage_cap",
                    "orphan_cap", "metadata_cap") + (("active_process_kind", "wind_pause_limit",
                    "wind_pause_ordinal") if _combined(handle._header) else ())):
            # Closed known policy refusal: no reservation/Native work/file
            # mutation occurred. Identity/source/IO/unknown failures still halt.
            handle._state = old
        else:
            handle._state = halt
        handle._entered = unclosed
        if directory is not None:
            try:
                handle._failure(directory, primary)
            except BaseException as evidence_error:
                primary.add_note("optional failure evidence interrupted: " + type(evidence_error).__name__[:64])
        raise
    # Nothing with a close/scan/validation/finally remains around this call.
    # Original _publish handles replace/fsync uncertainty and returns prepared ACK.
    return handle._publish(head, state, selected, halt)


def checkpoint_water(handle, expected_parent, *, boundary_ordinal):
    return _operate(handle, expected_parent, _intent("checkpoint_water", boundary_ordinal))


def step(handle, expected_parent):
    return _operate(handle, expected_parent, _intent("finish"))


def checkpoint_wind(handle, expected_parent, *, boundary_ordinal):
    if not _combined(handle._header):
        c._fail("store_profile", "wind pause requires the explicit active-process format")
    return _operate(handle, expected_parent, _intent_for("checkpoint_wind", boundary_ordinal, handle._header))


def create_active_water_boundary_store(path, *, runner, originals, runtime_reference,
                                       limits=c._DEFAULT, storage_contract):
    return _create_active_store(path, runner=runner, originals=originals,
        runtime_reference=runtime_reference, limits=limits, storage_contract=storage_contract)


def recover_active_water_boundary_store(path, *, expected_store_file_sha256, limits=c._DEFAULT):
    return _recover_active_store(path, expected_store_file_sha256=expected_store_file_sha256, limits=limits)


def create_active_process_boundary_store(path, *, runner, originals, runtime_reference,
                                         limits=c._DEFAULT, storage_contract):
    return _create_active_store(path, runner=runner, originals=originals,
        runtime_reference=runtime_reference, limits=limits, storage_contract=storage_contract, combined=True)


def recover_active_process_boundary_store(path, *, expected_store_file_sha256, limits=c._DEFAULT):
    return _recover_active_store(path, expected_store_file_sha256=expected_store_file_sha256,
                                  limits=limits, combined=True)


def _reservation_for_handle(handle, work_bytes, chain_charge=0):
    original = _reservation(handle._limits, work_bytes, chain_charge,
                            combined=_combined(handle._header))
    owner = c._checked_runtime_validation(handle)
    if owner is None:
        return original
    charge = c._runtime_validation_charge()
    if charge is None:
        c._drop_runtime_validation(handle)
        return original
    # Independent reference admission and a conservative finite union of
    # actual modified/new old-water metadata frame roles. Existing native
    # arrays/codec admission is unchanged; this is not a heap/CPU-fit law.
    extra = (charge['bytes'] + 2 * _parse_charge(c._RUNTIME_REUSE_REFERENCE_BYTES)
             + c.rw._compiled_frame_charge((
                 _reservation_for_handle, _reservation, _stable, _admit,
                 _load, _prepare_anchor, _operate, _recover_active_store,
                 recover_active_water_boundary_store, _load_selected, _load_selected_water, _combined,
                 _schema, _profile_name, _kinds, _store_fields,
                 _generation_fields_list, _water_fields, _ref_key, _managed_key,
                 _progress_key, _sibling_prefix, _paused, _same_format,
                 _pause_fields_for, _intent_for, _check_intent_for, _active_reference,
                 _generation_chain, _generation_fields, _head, _head_fields,
                 _head_record, _header_fields, _idle_state, _intent_fields,
                 _inventory_at, _managed_at, _metadata_stat_charge, _parent_light,
                 _paused_state, _request_id, _reserve_attempt, _seal_generation,
                 _temporary_heads, _water_reference, _borrow_parent, _ack_fields,
                 _acknowledgement, report, acknowledge,
             )))
    if original['memory_bytes'] + extra > handle._limits.max_memory_bytes:
        # Optional optimization only: no cache rows, full original validation.
        c._drop_runtime_validation(handle)
        return original
    owner._charge = extra
    # Admission is separate from the existing report/consumer record.
    return original
