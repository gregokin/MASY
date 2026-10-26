"""Explicit windowed-only qualification quotas and the unchanged Linux guard.

The quota rows are engineering declarations, not runtime/heap/fit predictions.
No Native imports, transport, limit mutation, generic binder or Dense override.
"""
from __future__ import annotations

import dataclasses
import resource
import types
from typing import Any

from masy import dense_production as _original

SCOPE = "windowed_cpu_paired_protocol_qualification_v1"
DenseProductionError = _original.DenseProductionError
PROFILES = (("ownership", 120, 135, 180), ("two_events", 720, 735, 1080),
            ("four_events", 1440, 1470, 2160))
_THREAD_ITEMS = (("MALLOC_ARENA_MAX", "1"), ("OMP_NUM_THREADS", "1"),
                 ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
                 ("NUMEXPR_NUM_THREADS", "1"), ("NUMBA_NUM_THREADS", "1"),
                 ("CUDA_VISIBLE_DEVICES", ""), ("JAX_PLATFORMS", "cpu"))


def profile_row(profile: str) -> tuple[int, int, int]:
    if (PROFILES != (("ownership", 120, 135, 180), ("two_events", 720, 735, 1080),
                     ("four_events", 1440, 1470, 2160))
            or SCOPE != "windowed_cpu_paired_protocol_qualification_v1"):
        raise DenseProductionError("windowed_process_binding_changed", "Source policy declaration changed")
    if type(profile) is str:
        for name, soft, hard, wall in PROFILES:
            if profile == name:
                return soft, hard, wall
    raise DenseProductionError("invalid_windowed_process_limit", "unknown windowed process profile")


@dataclasses.dataclass(frozen=True)
class WindowedProductionLimits:
    profile: str = "ownership"
    address_space_bytes: int = 4 << 30
    cpu_soft_s: int = 120
    cpu_hard_s: int = 135
    controller_wall_s: int = 180
    max_cells: int = 64
    max_classes: int = 1
    max_voxels: int = 16

    def __post_init__(self) -> None:
        if type(self) is not WindowedProductionLimits:
            raise DenseProductionError("invalid_windowed_process_limit", "exact WindowedProductionLimits required")
        soft, hard, wall = profile_row(self.profile)
        values = (self.address_space_bytes, self.cpu_soft_s, self.cpu_hard_s, self.controller_wall_s)
        if any(type(value) is not int for value in values) or values != (4 << 30, soft, hard, wall):
            raise DenseProductionError("invalid_windowed_process_limit", "process fields must match selected Source row")
        for value, maximum in ((self.max_cells, 64), (self.max_classes, 1), (self.max_voxels, 16)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise DenseProductionError("invalid_windowed_process_limit", "dimension declaration outside original domain")

    @classmethod
    def for_profile(cls, profile: str) -> WindowedProductionLimits:
        soft, hard, wall = profile_row(profile)
        return cls(profile=profile, cpu_soft_s=soft, cpu_hard_s=hard, controller_wall_s=wall)

    def as_record(self) -> dict[str, Any]:
        self.__post_init__()
        return {**dataclasses.asdict(self), "schema": "masy.windowed_process_policy.v1", "scope": SCOPE,
                "controller_wall_is_declared_not_observed": True, "core_soft_hard_bytes": [0, 0],
                "file_size_soft_hard_bytes": [128 << 20] * 2, "nofile_soft_hard": [1024, 1024],
                "native_peak_estimate": None,
                "native_peak_estimate_missing": ["flat_transport", "FFT_and_separation_scratch",
                    "avalanche_and_commit_scratch", "native_output_lifetimes"]}


# Only exact Source-bound Original declarations. Capture before any copies.
if len(vars(_original)) > 512 or any(type(key) is not str or len(key) > 512 for key in vars(_original)):
    raise DenseProductionError("windowed_process_source_domain", "Original namespace domain exceeded")
_ORIGINAL_FUNCTION = _original.check_process
if (type(_ORIGINAL_FUNCTION) is not types.FunctionType
        or _ORIGINAL_FUNCTION.__globals__ is not vars(_original)
        or _ORIGINAL_FUNCTION.__closure__ is not None or vars(_ORIGINAL_FUNCTION)
        or _ORIGINAL_FUNCTION.__defaults__ != (None,)
        or _ORIGINAL_FUNCTION.__kwdefaults__ != {"upcoming_wrapper_bytes": 0, "checkpoint": "unspecified"}
        or _original.THREAD_ENV != dict(_THREAD_ITEMS)):
    raise DenseProductionError("windowed_process_binding_changed", "Original guard header/Source declaration changed")
_ORIGINAL_ITEMS = tuple(vars(_original).items())
_ORIGINAL_CODE = _ORIGINAL_FUNCTION.__code__
_ORIGINAL_DEFAULTS = _ORIGINAL_FUNCTION.__defaults__
_ORIGINAL_KW_ITEMS = tuple(_ORIGINAL_FUNCTION.__kwdefaults__.items())
_ORIGINAL_ANNO_ITEMS = tuple(_ORIGINAL_FUNCTION.__annotations__.items())
_ORIGINAL_TYPE_PARAMS = getattr(_ORIGINAL_FUNCTION, "__type_params__", ())
_BOUND_GLOBALS = dict(_ORIGINAL_ITEMS)
_BOUND_GLOBALS["DenseProductionLimits"] = WindowedProductionLimits
_BOUND_GLOBALS["SCOPE"] = SCOPE
_BOUND_CHECK = types.FunctionType(_ORIGINAL_CODE, _BOUND_GLOBALS, _ORIGINAL_FUNCTION.__name__,
                                 tuple(list(_ORIGINAL_DEFAULTS)), _ORIGINAL_FUNCTION.__closure__)
_BOUND_CHECK.__kwdefaults__ = dict(_ORIGINAL_KW_ITEMS)
_BOUND_CHECK.__annotations__ = dict(_ORIGINAL_ANNO_ITEMS)
_BOUND_CHECK.__qualname__ = _ORIGINAL_FUNCTION.__qualname__
_BOUND_CHECK.__module__ = _ORIGINAL_FUNCTION.__module__
_BOUND_CHECK.__doc__ = _ORIGINAL_FUNCTION.__doc__
_BOUND_CHECK.__type_params__ = _ORIGINAL_TYPE_PARAMS

# These complete owned holders have a dedicated positive projection below.
BINDING_FIELDS = ("_ORIGINAL_FUNCTION", "_ORIGINAL_ITEMS", "_ORIGINAL_CODE", "_ORIGINAL_DEFAULTS",
                  "_ORIGINAL_KW_ITEMS", "_ORIGINAL_ANNO_ITEMS", "_ORIGINAL_TYPE_PARAMS",
                  "_BOUND_GLOBALS", "_BOUND_CHECK")


def _check_binding() -> None:
    if (len(vars(_original)) != len(_ORIGINAL_ITEMS) or len(_BOUND_GLOBALS) != len(_ORIGINAL_ITEMS)
            or _original.check_process is not _ORIGINAL_FUNCTION
            or _ORIGINAL_FUNCTION.__code__ is not _ORIGINAL_CODE
            or _ORIGINAL_FUNCTION.__globals__ is not vars(_original)
            or _ORIGINAL_FUNCTION.__defaults__ is not _ORIGINAL_DEFAULTS
            or _ORIGINAL_FUNCTION.__closure__ is not None or vars(_ORIGINAL_FUNCTION)
            or type(_ORIGINAL_FUNCTION.__kwdefaults__) is not dict
            or type(_ORIGINAL_FUNCTION.__annotations__) is not dict
            or tuple(_ORIGINAL_FUNCTION.__kwdefaults__.items()) != _ORIGINAL_KW_ITEMS
            or tuple(_ORIGINAL_FUNCTION.__annotations__.items()) != _ORIGINAL_ANNO_ITEMS
            or getattr(_ORIGINAL_FUNCTION, "__type_params__", ()) is not _ORIGINAL_TYPE_PARAMS
            or _original.THREAD_ENV != dict(_THREAD_ITEMS)):
        raise DenseProductionError("windowed_process_binding_changed", "Original Source/header/environment changed")
    for key, value in _ORIGINAL_ITEMS:
        expected = WindowedProductionLimits if key == "DenseProductionLimits" else SCOPE if key == "SCOPE" else value
        if vars(_original).get(key) is not value or _BOUND_GLOBALS.get(key) is not expected:
            raise DenseProductionError("windowed_process_binding_changed", "closed two-substitution namespace changed")
    if (type(_BOUND_CHECK) is not types.FunctionType or _BOUND_CHECK.__code__ is not _ORIGINAL_CODE
            or _BOUND_CHECK.__globals__ is not _BOUND_GLOBALS or _BOUND_CHECK.__closure__ is not None
            or _BOUND_CHECK.__defaults__ != _ORIGINAL_DEFAULTS
            or type(_BOUND_CHECK.__kwdefaults__) is not dict or type(_BOUND_CHECK.__annotations__) is not dict
            or tuple(_BOUND_CHECK.__kwdefaults__.items()) != _ORIGINAL_KW_ITEMS
            or tuple(_BOUND_CHECK.__annotations__.items()) != _ORIGINAL_ANNO_ITEMS
            or _BOUND_CHECK.__name__ != _ORIGINAL_FUNCTION.__name__
            or _BOUND_CHECK.__qualname__ != _ORIGINAL_FUNCTION.__qualname__
            or _BOUND_CHECK.__module__ != _ORIGINAL_FUNCTION.__module__
            or _BOUND_CHECK.__doc__ != _ORIGINAL_FUNCTION.__doc__ or vars(_BOUND_CHECK)
            or _BOUND_CHECK.__type_params__ is not _ORIGINAL_TYPE_PARAMS):
        raise DenseProductionError("windowed_process_binding_changed", "bound Original metadata changed")


def binding_metadata_terms() -> tuple[tuple[str, int], ...]:
    """Complete owned clone/current-next refs/text/error projections, NOT ABI heap bytes.

    Pre-existing original implementation values remain real guarded anchors.
    No Native array/backing or new owned dictionary/container is exempted.
    """
    _check_binding()
    g, d, k, a = len(_ORIGINAL_ITEMS), len(_ORIGINAL_DEFAULTS), len(_ORIGINAL_KW_ITEMS), len(_ORIGINAL_ANNO_ITEMS)
    text = 0
    for key, unused in _ORIGINAL_ITEMS:
        text += len(key)
    for pairs in (_ORIGINAL_KW_ITEMS, _ORIGINAL_ANNO_ITEMS):
        for key, value in pairs:
            if type(key) is not str or not (type(value) is str or type(value) is int):
                raise DenseProductionError("windowed_process_source_domain", "unexpected owned guard metadata")
            text += len(key) + (len(value) if type(value) is str else 42)
    code = _ORIGINAL_CODE
    raw = len(code.co_code) + len(code.co_linetable) + len(code.co_exceptiontable)
    # G: captured tuple+G pairs (3G) and the copied dictionary (2G).
    # K/A: captured pairs+dictionary+two current comparison forests (11 each).
    # Two13-field function headers, both default tuples and real typeparams;
    # two8-entry environment dictionaries and three added result key/value pairs.
    # Current+possible-next/error holders are separate; no alias/release credit.
    frames = (code.co_nlocals + code.co_stacksize + len(code.co_cellvars) + len(code.co_freevars))
    words = 2 * (5*g + 11*k + 11*a + 2*d + len(_ORIGINAL_TYPE_PARAMS)
                 + 26 + 4*len(_THREAD_ITEMS) + 6 + frames)
    for name in ("__name__", "__qualname__", "__module__", "__doc__"):
        value = getattr(_ORIGINAL_FUNCTION, name)
        if value is not None:
            if type(value) is not str:
                raise DenseProductionError("windowed_process_source_domain", "plain guard text required")
            text += len(value)
    text += len(SCOPE)
    return (("windowed_guard_current_next_global_header_container_refs", 8 * words),
            ("windowed_guard_code_and_raw_buffers", 4 * raw),
            ("windowed_guard_text_and_retained_error_frontiers", 12 * 2 * text))


def check_process(limits: WindowedProductionLimits, expected_identity=None, *,
                  upcoming_wrapper_bytes=0, checkpoint="unspecified") -> dict[str, Any]:
    """Every windowed checkpoint goes through this wrapper; Original body private."""
    if type(limits) is not WindowedProductionLimits:
        raise DenseProductionError("invalid_windowed_process_limit", "exact WindowedProductionLimits required")
    limits.__post_init__()
    _check_binding()
    if resource.getrlimit(resource.RLIMIT_NOFILE) != (1024, 1024):
        raise DenseProductionError("windowed_process_fd_mismatch", "NOFILE must already be exactly1024")
    additional = sum(value for unused, value in binding_metadata_terms())
    # All original arguments/defaults are forwarded WITHOUT coercion or substitution.
    answer = _BOUND_CHECK(limits, expected_identity, upcoming_wrapper_bytes=upcoming_wrapper_bytes,
                          checkpoint=checkpoint)
    if additional > answer["virtual_headroom_bytes"] - upcoming_wrapper_bytes:
        raise DenseProductionError("address_space_headroom", "known additional windowed guard owners cannot fit")
    _check_binding()  # success only; never replace an escaping Original/BaseException
    answer["windowed_process_profile"] = limits.profile
    answer["controller_wall_s_declared"] = limits.controller_wall_s
    answer["additional_windowed_control_bytes"] = additional
    return answer
