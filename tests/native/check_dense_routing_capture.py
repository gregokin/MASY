"""Bounded native writer control and closed routing-carrier ownership checks.

Actual native arrays/receipts remain local. Synthetic arrays below test wire
layout, ownership, and caps only; they do not represent calibrated physics.
"""

from __future__ import annotations

if not __debug__:
    raise SystemExit("Routing capture checks require assertions")

import contextlib
import dataclasses
import hashlib
import json
import os
import resource
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fixture_dense_alternation as f
import numpy as np

from masy import dense_bed as db
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as b

MODE = os.environ["MASY68_MODE"]
STAGE = Path(os.environ["MASY68_STAGE"])
REQUEST = json.loads(Path(os.environ["MASY68_RUNTIME"]).read_text())
SOURCES = REQUEST["source_identity"]
LIMITS = c.BoundaryStoreLimits()
COUNTS = {"wind_blocks": 0, "wind_finalization": 0, "water_completion": 0}
dr._native()
oi._registry()


def write(path, value):
    raw = oi._json(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def runtime_reference():
    roots = {
        "masy": REQUEST["import_prefix"][0],
        "maple": REQUEST["import_prefix"][2],
        "maple_syrup": REQUEST["import_prefix"][1],
    }
    tables = {
        "masy": {
            name: row
            for name, row in REQUEST["candidate_files"].items()
            if name.startswith("src/") or name == "pyproject.toml"
        },
        "maple": REQUEST["native_files"]["MAPLE"],
        "maple_syrup": REQUEST["native_files"]["SYRUP"],
    }
    record = {
        "schema": c.SCHEMA + ".sources",
        "source_identity": SOURCES,
        "module_roots": roots,
        "files": tables,
        "environment": c._environment_identity(),
    }
    path = STAGE / "runtime_sources.json"
    digest = write(path, record)
    native = Path(REQUEST["native_bundle_manifest_path"])
    return {
        "schema": c.SCHEMA + ".runtime",
        "kind": "source_manifest_prototype",
        "source_manifest_path": str(path),
        "source_manifest_file_sha256": digest,
        "native_bundle_manifest_path": str(native),
        "native_bundle_manifest_file_sha256": hashlib.sha256(
            native.read_bytes()
        ).hexdigest(),
        "module_roots": roots,
        "cpu_store": None,
        "cpu_lock_id": None,
        "cpu_lock_file_sha256": None,
    }


def array_record(array):
    return {
        "id": id(array),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": array.nbytes,
        "C": bool(array.flags.c_contiguous),
        "writable": bool(array.flags.writeable),
        "finite": bool(np.isfinite(array).all()),
        "digest": db._sha256(db._array_digest(array)),
    }


def owner_record(owner):
    arrays = []
    for name in ("_pool", "_soil", "_inventory0", "_totals"):
        db._leaves(getattr(owner, name), name, arrays)
    db._leaves(owner._bed.state, "bed", arrays)
    if owner._routing is not None:
        arrays.append(("routing_source", owner._routing.source))
        velocity = (
            owner._routing.velocity
            if type(owner._routing) is b._RestoredRouting
            else owner._routing.template.sediment_velocity_m_s
        )
        arrays.append(("routing_velocity", velocity))
    return {
        "arrays": {name: array_record(array) for name, array in arrays},
        "routing_id": id(owner._routing),
        "bed_id": id(owner._bed),
        "report": owner.report(),
        "ledger": owner.ledger_record(),
        "bed_identity": owner._bed.identity_sha256(),
        "private_identity": owner._accepted_private,
    }


def owner_arrays(owner):
    leaves = []
    for name in ("_pool", "_soil"):
        db._leaves(getattr(owner, name), name, leaves)
    db._leaves(owner._bed.state, "bed", leaves)
    leaves.extend(
        (name, array)
        for name, array in zip(
            ("routing_source", "routing_velocity"),
            b._capture_wrapper_arrays(owner, LIMITS.image)[5:],
        )
    )
    return [array for _, array in leaves]


@contextlib.contextmanager
def forbid_transport_and_construction():
    forbidden = AssertionError(
        "transport, constructor or fresh bed during passive save/load"
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(dr.DenseRunner, "__init__", side_effect=forbidden)
        )
        stack.enter_context(mock.patch.object(f, "build_bed", side_effect=forbidden))
        for name in ("_advance_wind_block", "_finalize_wind_event", "_complete_event"):
            stack.enter_context(mock.patch.object(dr, name, side_effect=forbidden))
        for module in (db, dr, b, c, f):
            if hasattr(module, "build_dense_bed"):
                stack.enter_context(
                    mock.patch.object(module, "build_dense_bed", side_effect=forbidden)
                )
        yield


def actual_native_control():
    def counted(name, actual):
        def call(*args, **kwargs):
            COUNTS[name] += 1
            return actual(*args, **kwargs)

        return call

    runner = f.make_runner(f.build_bed(), f.schedule(f.wind(), f.rain()))
    originals = oi.capture_original_inputs(runner, source_identity=SOURCES)
    handle = c.create_boundary_store(
        STAGE / "component_store",
        runner=runner,
        originals=originals,
        runtime_reference=runtime_reference(),
        limits=LIMITS,
        storage_contract=c.StorageContract(True, True),
    )
    try:
        head_before = (handle._root / "HEAD").read_bytes()
        with contextlib.ExitStack() as stack:
            for name, label in (
                ("_advance_wind_block", "wind_blocks"),
                ("_finalize_wind_event", "wind_finalization"),
                ("_complete_event", "water_completion"),
            ):
                stack.enter_context(
                    mock.patch.object(dr, name, new=counted(label, getattr(dr, name)))
                )
            assert runner.step()["status"] == "accepted"
            assert runner.step()["status"] == "accepted"
        assert COUNTS == {
            "wind_blocks": 1,
            "wind_finalization": 1,
            "water_completion": 1,
        }
        assert runner._next == 2 and runner._routing is not None
        before = owner_record(runner)
        arrays = b._capture_wrapper_arrays(runner, LIMITS.image)
        assert len(arrays) == 7
        shape, nc = (f.NY, f.NX), len(f.CLASS_IDS)
        assert b._array(arrays[5], shape, "live source") is arrays[5]
        assert b._array(arrays[6], (*shape, nc), "live velocity") is arrays[6]
        assert arrays[5].flags.writeable and arrays[6].flags.writeable
        physical = handle._save_physical(
            handle._root / "component_physical", runner, originals
        )
        arguments = dict(
            artifact_root=handle._root,
            originals=handle._header["originals"],
            genesis=handle._header["genesis_physical"],
            physical=physical,
            source_identity=SOURCES,
            environment_sha256=handle._header["environment_sha256"],
            limits=LIMITS.image,
        )
        image = STAGE / "native_rain_image"
        evidence = {
            "native_counts": dict(COUNTS),
            "live_routing_arrays": [array_record(array) for array in arrays[5:]],
            "routing_copy_bytes": sum(array.nbytes for array in arrays[5:]),
            "maximum_routing_copy_bytes": LIMITS.image.max_numeric_bytes,
            "component_reservation_ceiling": 2 << 30,
            "C_reservation": c._reservation(LIMITS, runner._limits.max_work_bytes),
            "native_time_hex": float(runner._bed.time_s).hex(),
            "observed_maxrss_kib_before_readback": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "ledger": runner.ledger_record(),
            "physics_inputs": {
                "wind_s": f.WIND_S,
                "rain_s": f.RAIN_S,
                "rain_mm_h": f.RAIN_MM_H,
                "recession_guard_s": f.GUARD_S,
            },
        }
        with forbid_transport_and_construction():
            if MODE == "baseline":
                try:
                    b.save_passive_boundary(image, runner, **arguments)
                except b.BoundaryCheckpointError as error:
                    assert (
                        error.code == "array_layout"
                        and "saved routing source" in error.detail
                    )
                    assert not image.exists()
                    evidence["baseline_refusal"] = {
                        "code": error.code,
                        "detail": error.detail,
                    }
                    evidence["failed_predicate"] = (
                        "readonly=True and value.flags.writeable"
                    )
                else:
                    raise AssertionError(
                        "accepted64 no longer reproduces writer readonly mismatch"
                    )
            else:
                actual_prepare, actual_write = (
                    b._prepare_material,
                    np.lib.format.write_array,
                )
                weak_staging, preparations, serializations = [], [], []

                def prepare(manifest, carriers, *args, **kwargs):
                    if not preparations:
                        admitted = kwargs["artifacts"]
                        assert type(admitted) is b._Artifacts
                        evidence["actual_artifact_reservation_bytes"] = (
                            admitted.reservation_bytes
                        )
                        evidence["combined_writer_reservation_bytes"] = (
                            admitted.reservation_bytes + evidence["routing_copy_bytes"]
                        )
                        assert evidence["combined_writer_reservation_bytes"] <= 2 << 30
                        for index in (5, 6):
                            assert carriers[index] is not arrays[index]
                            assert not np.shares_memory(carriers[index], arrays[index])
                            assert not carriers[index].flags.writeable
                            assert carriers[index].tobytes() == arrays[index].tobytes()
                            weak_staging.append(weakref.ref(carriers[index]))
                        assert all(
                            carriers[index] is arrays[index] for index in range(5)
                        )
                    preparations.append(len(carriers))
                    return actual_prepare(manifest, carriers, *args, **kwargs)

                def serialize(stream, array, *args, **kwargs):
                    assert all(reference() is None for reference in weak_staging)
                    assert array is arrays[len(serializations)]
                    serializations.append(id(array))
                    return actual_write(stream, array, *args, **kwargs)

                with (
                    mock.patch.object(b, "_prepare_material", new=prepare),
                    mock.patch.object(np.lib.format, "write_array", new=serialize),
                ):
                    seal = b.save_passive_boundary(image, runner, **arguments)
                assert preparations == [7, 7] and len(serializations) == 7
                loaded = b.prepare_passive_boundary(
                    image,
                    artifact_root=handle._root,
                    expected_manifest_sha256=seal,
                    expected_source_identity=SOURCES,
                    expected_environment_sha256=handle._header["environment_sha256"],
                    limits=LIMITS.image,
                )
                restored = c._owner(loaded)
                c._compare_owners(runner, restored, LIMITS.image)
                assert not restored._routing.source.flags.writeable
                assert not restored._routing.velocity.flags.writeable
                assert not any(
                    np.shares_memory(a, z)
                    for a in owner_arrays(runner)
                    for z in owner_arrays(restored)
                )
                restored_before = owner_record(restored)
                resave_seal = b.save_passive_boundary(
                    STAGE / "readonly_owner_resave", restored, **arguments
                )
                assert oi._json(restored_before) == oi._json(owner_record(restored))
                reloaded = b.prepare_passive_boundary(
                    STAGE / "readonly_owner_resave",
                    artifact_root=handle._root,
                    expected_manifest_sha256=resave_seal,
                    expected_source_identity=SOURCES,
                    expected_environment_sha256=handle._header["environment_sha256"],
                    limits=LIMITS.image,
                )
                second = c._owner(reloaded)
                c._compare_owners(runner, second, LIMITS.image)
                assert not any(
                    np.shares_memory(a, z)
                    for a in owner_arrays(restored)
                    for z in owner_arrays(second)
                )
                failed = STAGE / "failed_serializer_image"
                injected = OSError("injected first NPY failure")
                with mock.patch.object(
                    np.lib.format,
                    "write_array",
                    side_effect=injected,
                ):
                    try:
                        b.save_passive_boundary(failed, runner, **arguments)
                    except OSError as error:
                        assert error is injected and error.args == (
                            "injected first NPY failure",
                        )
                        assert any(
                            "unsealed boundary image retained" in note
                            for note in getattr(error, "__notes__", ())
                        )
                    else:
                        raise AssertionError("injected serializer failure was accepted")
                assert failed.exists() and not (failed / "manifest.json").exists()
                assert not (failed / "manifest.tmp").exists()
                tiny = dataclasses.replace(LIMITS.image, max_numeric_bytes=1)
                with mock.patch.object(
                    np,
                    "isfinite",
                    side_effect=AssertionError("scan before numeric cap"),
                ):
                    try:
                        b.save_passive_boundary(
                            STAGE / "over_numeric_cap",
                            runner,
                            **{**arguments, "limits": tiny},
                        )
                    except b.BoundaryCheckpointError as error:
                        assert error.code == "array_cap"
                    else:
                        raise AssertionError("numeric cap did not refuse")
                assert not (STAGE / "over_numeric_cap").exists()
                evidence.update(
                    {
                        "native_roundtrip_exact": True,
                        "reader_readonly_unchanged": True,
                        "source_disjoint_readback": True,
                        "copy_released_before_serialization": True,
                        "actual_serialization_uses_live_capture": True,
                        "readonly_owner_resave": True,
                        "serializer_failure_preserves_owner": True,
                        "cap_before_scan_and_copy": True,
                    }
                )
        assert oi._json(before) == oi._json(owner_record(runner))
        assert (handle._root / "HEAD").read_bytes() == head_before
        evidence.update(
            {
                "accepted_owner_unchanged": True,
                "genesis_HEAD_unchanged": True,
                "scope": "native two-event passive component; no four-event/HEAD continuation qualification",
            }
        )
        write(STAGE / "native_capture_evidence.json", evidence)
    finally:
        handle.close()


class RoutingCarrierContract(unittest.TestCase):
    def test_reader_still_refuses_writable_source_and_velocity(self):
        for shape in ((1, 6), (1, 6, 6)):
            array = np.zeros(shape, dtype="<f8")
            with self.assertRaises(b.BoundaryCheckpointError) as caught:
                b._array(array, shape, "saved routing", readonly=True)
            self.assertEqual(caught.exception.code, "array_layout")
            array.flags.writeable = False
            self.assertIs(b._array(array, shape, "saved routing", readonly=True), array)

    def test_reader_layout_controls_remain_strict(self):
        good = np.zeros((2, 3), dtype="<f8")
        for array in (
            good.astype("<f4"),
            np.asfortranarray(good),
            np.zeros((2, 6), dtype="<f8")[:, ::2],
            good[:, :2],
            np.full((2, 3), np.nan),
            np.full((2, 3), np.inf),
        ):
            array.flags.writeable = False
            with self.subTest(dtype=array.dtype.str, shape=array.shape):
                with self.assertRaises(b.BoundaryCheckpointError):
                    b._array(array, (2, 3), "saved routing", readonly=True)

    def test_private_reservation_refuses_before_prepare_or_output(self):
        arrays = [np.zeros((1,), dtype="<f8") for _ in range(7)]
        target = STAGE / "reservation_refusal"
        with mock.patch.object(
            b, "_prepare_material", side_effect=AssertionError("prepare before reserve")
        ):
            with self.assertRaises(b.BoundaryCheckpointError) as caught:
                b._write_passive_boundary(
                    target,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    LIMITS.image,
                    {},
                    arrays,
                    SimpleNamespace(reservation_bytes=2 << 30),
                )
        self.assertEqual(caught.exception.code, "memory_cap")
        self.assertFalse(target.exists())

    def test_copies_preserve_signed_zero_bits_without_changing_source(self):
        # Same ndarray copy primitive as writer; explicit bit fixture, no physics.
        original = np.array([0x8000000000000000, 0], dtype="<u8").view("<f8")
        before = original.tobytes()
        staged = original.copy(order="C")
        staged.flags.writeable = False
        self.assertEqual(staged.tobytes(), before)
        self.assertTrue(original.flags.writeable)
        self.assertFalse(np.shares_memory(staged, original))

    def test_reservation_exact_ceiling_passes_guard_but_plus_one_refuses(self):
        # Refusal-only whitebox control. No fake successful native preparation.
        class StopBeforeNativePreparation(Exception):
            pass

        arrays = [np.zeros((1,), dtype="<f8") for _ in range(7)]
        routing_bytes = sum(array.nbytes for array in arrays[5:])
        target = STAGE / "reservation_exact_control"
        values = (None, None, None, None, None, None, None, LIMITS.image, {}, arrays)
        with mock.patch.object(
            b, "_prepare_material", side_effect=StopBeforeNativePreparation
        ) as prepare:
            with self.assertRaises(StopBeforeNativePreparation):
                b._write_passive_boundary(
                    target,
                    *values,
                    SimpleNamespace(reservation_bytes=(2 << 30) - routing_bytes),
                )
            self.assertEqual(prepare.call_count, 1)
            with self.assertRaises(b.BoundaryCheckpointError) as caught:
                b._write_passive_boundary(
                    target,
                    *values,
                    SimpleNamespace(reservation_bytes=(2 << 30) - routing_bytes + 1),
                )
            self.assertEqual(caught.exception.code, "memory_cap")
            self.assertEqual(prepare.call_count, 1)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    if MODE in ("baseline", "native"):
        actual_native_control()
        print(json.dumps({"passed": True, "mode": MODE, "native_counts": COUNTS}))
    else:
        unittest.main(verbosity=2)
