"""Synthetic genesis/passive components only. Never event/transport qualification."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fixture_dense_alternation as f
import numpy as np
from maple.aeolian.scheduler import physical_checkpoint as cp

from masy import dense_bed as db
from masy import dense_checkpoint as bridge
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as c

SOURCES = json.loads(os.environ["MASY43_SOURCE_IDS"])
ENVIRONMENT = os.environ["MASY43_ENV_SHA"]


def reference(path, fmt, identity):
    return {
        "format": fmt,
        "relative_path": path.name,
        "manifest_file_sha256": hashlib.sha256(
            (
                path / ("manifest.json" if fmt == oi.SCHEMA else "checkpoint.json")
            ).read_bytes()
        ).hexdigest(),
        "identity": identity,
    }


def reseal(path, value):
    value["manifest_sha256"] = c._sha(
        {k: v for k, v in value.items() if k != "manifest_sha256"}
    )
    (path / "manifest.json").chmod(0o600)
    (path / "manifest.json").write_bytes(oi._json(value))
    return value["manifest_sha256"]


def leaves(value):
    if type(value) is np.ndarray:
        yield value
    elif dataclasses.is_dataclass(value):
        for fld in dataclasses.fields(value):
            yield from leaves(getattr(value, fld.name))
    elif type(value) is dict:
        for item in value.values():
            yield from leaves(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from leaves(item)


class PassiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.bed = f.build_bed()
        cls.runner = f.make_runner(cls.bed, f.schedule(f.wind(), f.rain(), f.wind()))
        cls.original = oi.capture_original_inputs(cls.runner, source_identity=SOURCES)
        oi.save_original_inputs(cls.root / "original", cls.original)
        cls.oref = reference(
            cls.root / "original", oi.SCHEMA, cls.original.content_sha256
        )
        caller = {
            "schema": "masy.dense_runner_boundary_native.v1",
            "source_identity": SOURCES,
            "environment_sha256": ENVIRONMENT,
            "static_sha256": cls.runner.static_sha256,
        }
        packed = bridge.pack_boundary_physical(
            cls.bed,
            cls.runner._pool,
            cls.runner._wind.config,
            caller,
            inputs=cls.original.capsule,
            max_bytes=64 << 20,
        )
        cp.save_exact_physical_checkpoint(
            cls.root / "genesis",
            packed.physical,
            cls.runner._wind.config,
            packed.identity,
            max_bytes=64 << 20,
        )
        cls.pref = reference(cls.root / "genesis", cp.SCHEMA, packed.identity)
        cls.seal = c.save_passive_boundary(
            cls.root / "boundary",
            cls.runner,
            artifact_root=cls.root,
            originals=cls.oref,
            genesis=cls.pref,
            physical=cls.pref,
            source_identity=SOURCES,
            environment_sha256=ENVIRONMENT,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def prepare(self, path=None, seal=None, **kwargs):
        return c.prepare_passive_boundary(
            path or self.root / "boundary",
            artifact_root=self.root,
            expected_manifest_sha256=seal or self.seal,
            expected_source_identity=SOURCES,
            expected_environment_sha256=ENVIRONMENT,
            **kwargs,
        )

    def mutated(self, mutate):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "boundary"
        shutil.copytree(self.root / "boundary", path)
        record = json.loads((path / "manifest.json").read_bytes())
        mutate(record)
        return path, reseal(path, record)

    def test_01_genesis_exact_inert_roundtrip(self):
        prepared = self.prepare()
        owner = (
            prepared._PreparedBoundary__owner
        )  # test-only private inspection, not product API
        self.assertEqual(set(vars(owner)), set(c.RUNNER_FIELDS))
        self.assertFalse(hasattr(prepared, "step"))
        self.assertFalse(hasattr(prepared, "owner"))
        self.assertEqual(owner.accepted.as_record(), self.runner.accepted.as_record())
        self.assertEqual(owner.ledger_record(), self.runner.ledger_record())
        self.assertEqual(owner.receipts, [])
        self.assertIs(owner._wind.config.geometry, owner._bed.context.geometry)
        self.assertIs(
            owner._wind.config.initial_available_fraction,
            owner._bed.context.initial_available_fraction,
        )
        self.assertIsNot(owner._pool, owner._wind.initial_mobile_state)
        self.assertTrue(
            oi._same(owner._pool, self.runner._pool, oi.OriginalPackLimits())
        )
        self.assertFalse(
            np.shares_memory(owner._inventory0, self.original.initial_inventory_kg)
        )
        self.assertEqual(prepared.diagnostics()["n_wrapper_arrays"], 5)

    def test_02_no_constructor_transport_commit_or_reset(self):
        dr._native()
        forbidden = AssertionError("passive restore called forbidden operation")
        with (
            mock.patch.object(dr.DenseRunner, "__init__", side_effect=forbidden),
            mock.patch.object(db, "build_dense_bed", side_effect=forbidden),
            mock.patch.object(dr, "_advance_wind_block", side_effect=forbidden),
            mock.patch.object(dr, "_finalize_wind_event", side_effect=forbidden),
            mock.patch.object(dr, "_complete_event", side_effect=forbidden),
            mock.patch.object(
                bridge, "load_boundary_physical", wraps=bridge.load_boundary_physical
            ) as load,
            mock.patch("maple_syrup.sediment_bed.commit_bed", side_effect=forbidden),
            mock.patch(
                "maple_syrup.sediment_event.initial_event_state", side_effect=forbidden
            ),
        ):
            self.prepare()
        self.assertEqual(load.call_count, 1)

    def test_03_source_owner_disjoint_native_fields(self):
        owner = self.prepare()._PreparedBoundary__owner
        actual = (
            list(leaves(owner._bed.state)) + list(leaves(owner._pool)) + [owner._soil]
        )
        source = (
            list(leaves(self.runner._bed.state))
            + list(leaves(self.runner._pool))
            + [self.runner._soil]
        )
        self.assertTrue(all(not np.shares_memory(a, b) for a in actual for b in source))
        self.assertTrue(all(not a.flags.writeable for a in actual))

    def test_04_source_refusal_before_native(self):
        path, seal = self.mutated(
            lambda r: r["source_identity"].__setitem__("masy", "a" * 64)
        )
        with (
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("native called"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "source_identity"),
        ):
            self.prepare(path, seal)

    def test_05_native_raw_cap_before_hash_and_load(self):
        with (
            mock.patch(
                "masy.runtime._hash_path", side_effect=AssertionError("hash before cap")
            ),
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("native before cap"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "serialized_cap"),
        ):
            self.prepare(limits=c.BoundaryImageLimits(native_max_bytes=1))

    def test_06_wrapper_raw_cap_before_header(self):
        with (
            mock.patch.object(
                oi,
                "_read_npy_header",
                side_effect=AssertionError("header before raw cap"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "serialized_cap"),
        ):
            self.prepare(limits=c.BoundaryImageLimits(max_serialized_bytes=1))

    def test_07_resource_changes_refuse_before_native(self):
        for key, value in [
            ("version", 1),
            ("update_cursor", True),
            ("update_receipts", [{}]),
        ]:
            with self.subTest(key=key):
                path, seal = self.mutated(
                    lambda r, key=key, value=value: r["shared_resources"].__setitem__(
                        key, value
                    )
                )
                with (
                    mock.patch.object(
                        bridge,
                        "load_boundary_physical",
                        side_effect=AssertionError("native called"),
                    ),
                    self.assertRaises(c.BoundaryCheckpointError),
                ):
                    self.prepare(path, seal)

    def test_08_state_counter_types_and_fields(self):
        for key, value in [
            ("next_index", True),
            ("n_accepted", 1),
            ("n_classes", 7),
            ("extra", 0),
        ]:
            with self.subTest(key=key):
                path, seal = self.mutated(
                    lambda r, key=key, value=value: r["state"].__setitem__(key, value)
                )
                with self.assertRaises(c.BoundaryCheckpointError):
                    self.prepare(path, seal)

    def test_09_no_overwrite_and_quiescent_refusals(self):
        with self.assertRaises(FileExistsError):
            c.save_passive_boundary(
                self.root / "boundary",
                self.runner,
                artifact_root=self.root,
                originals=self.oref,
                genesis=self.pref,
                physical=self.pref,
                source_identity=SOURCES,
                environment_sha256=ENVIRONMENT,
            )
        for name, value in [
            ("_stop_receipt", {}),
            ("_active_evidence", object()),
            ("_boundary_binding", object()),
        ]:
            with self.subTest(name=name):
                had = name in vars(self.runner)
                old = vars(self.runner).get(name)
                setattr(self.runner, name, value)
                try:
                    with self.assertRaises(c.BoundaryCheckpointError):
                        c._runner_admission(self.runner, c.BoundaryImageLimits())
                finally:
                    if had:
                        setattr(self.runner, name, old)
                    else:
                        delattr(self.runner, name)

    def test_10_private_ticket_rejects_raw_owner(self):
        with self.assertRaises(c.BoundaryCheckpointError):
            dr.DenseRunner._from_passive_boundary_load(self.runner)

    def held(self):
        n = dr._native()
        # Illustrative retained-source component; never claimed to be an accepted rain run.
        source = np.array(
            self.runner._water.terrain.initial_committed_elevation_m, copy=True
        )
        source[:, 0] += np.linspace(0.001, 0.002, source.shape[0])
        graph = n.routing_graph_from_elevation(
            self.runner._water.terrain, source, xp=np
        )
        network, grid = n.transport_network(graph), n.physics_grid_from_graph(graph)
        velocity = np.zeros((*source.shape, self.runner._n_classes), dtype=np.float64)
        from maple_syrup.sediment_event import SedimentEventState

        template = SedimentEventState(
            n.initial_state(
                graph,
                self.runner._bed.state.water.depth_m,
                self.runner._soil,
                t_s=self.runner._bed.time_s,
            ),
            self.runner._bed.state,
            velocity,
            graph,
            network,
            grid,
            self.runner._water.terrain,
            None,
            source,
        )
        return SimpleNamespace(
            graph=graph,
            terrain=self.runner._water.terrain,
            network=network,
            grid=grid,
            source=source,
            template=template,
        )

    def projected(self):
        held = self.held()
        record = c._routing_record(
            held,
            self.runner._water,
            held.graph.shape,
            self.runner._n_classes,
            c.BoundaryImageLimits(),
        )
        source = np.array(held.source, copy=True)
        source.flags.writeable = False
        velocity = np.array(held.template.sediment_velocity_m_s, copy=True)
        velocity.flags.writeable = False
        projected = c._load_routing(
            record,
            self.original,
            [None] * 5 + [source, velocity],
            c.BoundaryImageLimits(),
        )
        return held, projected, record

    def test_11_native_saved_source_full_cache_roundtrip(self):
        held, projection, record = self.projected()
        self.assertIs(projection.terrain, self.original.water.terrain)
        self.assertFalse(
            np.array_equal(
                projection.source,
                self.runner._water.terrain.initial_committed_elevation_m,
            )
        )
        self.assertEqual(
            c._routing_record(
                projection,
                self.runner._water,
                held.graph.shape,
                self.runner._n_classes,
                c.BoundaryImageLimits(),
            ),
            record,
        )
        self.assertFalse(projection.source.flags.writeable)
        self.assertFalse(projection.velocity.flags.writeable)
        self.assertFalse(np.shares_memory(projection.source, held.source))
        self.assertFalse(
            np.shares_memory(projection.velocity, held.template.sediment_velocity_m_s)
        )

    def test_12_projection_poison_before_fingerprint(self):
        _held, p, _r = self.projected()
        for value in (
            np.zeros((100000,), dtype=np.float64),
            [],
            np.zeros(p.velocity.shape, dtype=np.float32),
        ):
            bad = dataclasses.replace(p, velocity=value)
            with (
                mock.patch.object(
                    c,
                    "_cache_fingerprint",
                    side_effect=AssertionError("fingerprint before cap"),
                ),
                self.assertRaises((c.BoundaryCheckpointError, dr._Stop)),
            ):
                c._restored_memory(
                    bad, self.runner._water, p.graph.shape, self.runner._n_classes
                )

    def test_13_projection_deleted_slot_and_extra_native_attr(self):
        _h, p, _r = self.projected()
        missing = dataclasses.replace(p)
        object.__delattr__(missing, "phase")
        with self.assertRaises(dr._Stop):
            c._restored_memory(
                missing, self.runner._water, p.graph.shape, self.runner._n_classes
            )
        graph = copy.copy(p.graph)
        object.__setattr__(graph, "unknown", 1)
        bad = dataclasses.replace(p, graph=graph)
        with (
            mock.patch.object(
                c,
                "_cache_fingerprint",
                side_effect=AssertionError("fingerprint before layout"),
            ),
            self.assertRaises(dr._Stop),
        ):
            c._restored_memory(
                bad, self.runner._water, p.graph.shape, self.runner._n_classes
            )

    def test_14_deleted_default_native_field_refuses(self):
        _h, p, _r = self.projected()
        graph = copy.copy(p.graph)
        object.__delattr__(graph, "flat_storage")
        with self.assertRaises(dr._Stop):
            c._restored_memory(
                dataclasses.replace(p, graph=graph),
                self.runner._water,
                p.graph.shape,
                self.runner._n_classes,
            )

    def test_15_poisoned_cache_fingerprint_refuses(self):
        _h, p, _r = self.projected()
        a = np.array(p.graph.slope, copy=True)
        a.flat[0] += 0.1
        a.flags.writeable = False
        graph = dataclasses.replace(p.graph, slope=a)
        with self.assertRaisesRegex(dr._Stop, "restored_routing_changed"):
            c._restored_memory(
                dataclasses.replace(p, graph=graph),
                self.runner._water,
                p.graph.shape,
                self.runner._n_classes,
            )

    def test_16_original_terrain_identity_binding(self):
        _h, p, _r = self.projected()
        bad = dataclasses.replace(p, terrain=copy.copy(p.terrain))
        with self.assertRaises(dr._Stop):
            c._restored_memory(
                bad, self.runner._water, p.graph.shape, self.runner._n_classes
            )

    def test_17_velocity_host_kind_legacy_code(self):
        _h, p, _r = self.projected()
        with self.assertRaises(dr._Stop) as caught:
            c._restored_memory(
                dataclasses.replace(p, velocity=[]),
                self.runner._water,
                p.graph.shape,
                self.runner._n_classes,
            )
        self.assertEqual(caught.exception.code, "transport_memory_unsupported")

    def test_18_native_none_distinction(self):
        _h, p, _r = self.projected()
        graph = dataclasses.replace(
            p.graph, flat_storage=np.zeros(p.graph.shape, dtype=bool)
        )
        graph.flat_storage.flags.writeable = False
        with self.assertRaises(dr._Stop):
            c._restored_memory(
                dataclasses.replace(p, graph=graph),
                self.runner._water,
                p.graph.shape,
                self.runner._n_classes,
            )

    def test_19_three_reader_component_parity_no_native_event(self):
        legacy, projection, _r = self.projected()
        event = self.runner._schedule.events[1]
        from masy.dense_schedule import resolve_event

        resolved = resolve_event(self.runner._schedule, 1, self.runner._bed.time_s)
        observations = []
        for held in (legacy, projection):
            runner = copy.copy(self.runner)
            runner._routing = held
            n = dr._native()
            boundary = SimpleNamespace(
                state=runner._bed.state,
                graph=held.graph,
                did_commit=True,
                rerouted=False,
                routing_source_m=held.source,
            )

            def observe(entry, *args, **kwargs):
                observations.append(entry)
                raise ValueError("passive spy stops before native complete_event")

            commit = mock.Mock(return_value=boundary)
            n.commit_bed = commit
            with (
                mock.patch.object(dr, "_native", return_value=n),
                mock.patch.object(
                    runner,
                    "_stage",
                    return_value=SimpleNamespace(state=runner._bed.state),
                ),
                mock.patch.object(runner, "_work_bytes_check") as work,
                mock.patch.object(dr, "_complete_event", side_effect=observe),
            ):
                with self.assertRaises(dr._Stop) as caught:
                    runner._rain_event(1, event, resolved)
                self.assertEqual(caught.exception.code, "rain_native_failure")
                self.assertIs(commit.call_args.kwargs["routing_source_m"], held.source)
                self.assertTrue(commit.call_args.kwargs["force"])
                budget = work.call_args.args[0]
                if held is legacy:
                    legacy_budget = budget
                else:
                    self.assertEqual(budget, legacy_budget)
        a, b = observations
        self.assertEqual(
            tuple(v.name for v in dataclasses.fields(a)),
            tuple(v.name for v in dataclasses.fields(b)),
        )
        self.assertTrue(oi._same(a, b, oi.OriginalPackLimits()))
        self.assertFalse(
            np.shares_memory(
                a.sediment_velocity_m_s, legacy.template.sediment_velocity_m_s
            )
        )
        self.assertFalse(np.shares_memory(b.sediment_velocity_m_s, projection.velocity))
        self.assertIs(b.routing_source_elevation_m, projection.source)
        self.assertIs(b.bed, self.runner._bed.state)

    def test_20_projection_phase_guard_no_commit(self):
        _h, p, _r = self.projected()
        with self.assertRaises(dr._Stop) as caught:
            c._restored_guard(
                dataclasses.replace(p, phase=object()),
                self.runner._water,
                p.graph.shape,
                self.runner._n_classes,
            )
        self.assertEqual(caught.exception.code, "transport_memory_not_empty")

    def test_21_wrapper_unlisted_and_symlink_refuse(self):
        for kind in ("extra", "link"):
            with self.subTest(kind=kind):
                path, seal = self.mutated(lambda r: None)
                if kind == "extra":
                    (path / "x.npy").write_bytes(b"x")
                else:
                    a = path / "a_0000.npy"
                    data = a.read_bytes()
                    a.unlink()
                    target = path.parent / "else.npy"
                    target.write_bytes(data)
                    a.symlink_to(target)
                with self.assertRaises(
                    (c.BoundaryCheckpointError, oi.OriginalInputsError, OSError)
                ):
                    self.prepare(path, seal)

    def test_22_corrupt_numeric_refuses(self):
        path, seal = self.mutated(lambda r: None)
        a = path / "a_0001.npy"
        a.chmod(0o600)
        raw = bytearray(a.read_bytes())
        raw[-1] ^= 1
        a.write_bytes(raw)
        with self.assertRaises((c.BoundaryCheckpointError, oi.OriginalInputsError)):
            self.prepare(path, seal)

    def test_23_own_leftover_seal_hardlink_recovers(self):
        path, seal = self.mutated(lambda r: None)
        os.link(path / "manifest.json", path / "manifest.tmp")
        self.assertEqual(self.prepare(path, seal).diagnostics()["n_wrapper_arrays"], 5)

    def test_24_child_path_escape_refuses_before_expansion(self):
        for value in ("../outside", "/absolute", "original/../x", "original//x"):
            with self.subTest(value=value):
                path, seal = self.mutated(
                    lambda r, value=value: r["originals"].__setitem__(
                        "relative_path", value
                    )
                )
                with (
                    mock.patch.object(
                        oi,
                        "_read_npy_header",
                        side_effect=AssertionError("header before path admission"),
                    ),
                    self.assertRaises(c.BoundaryCheckpointError),
                ):
                    self.prepare(path, seal)

    def test_25_tiny_global_raw_budget_before_expansion(self):
        with (
            mock.patch.object(
                oi,
                "_read_npy_header",
                side_effect=AssertionError("header before global raw cap"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "serialized_cap"),
        ):
            self.prepare(limits=c.BoundaryImageLimits(max_global_raw_bytes=1))

    def test_26_signed_zero_total_not_normalized(self):
        path, seal = self.mutated(lambda r: None)
        a = path / "a_0001.npy"
        a.chmod(0o600)
        values = np.load(a, allow_pickle=False)
        values[0] = -0.0
        with a.open("wb") as stream:
            np.lib.format.write_array(stream, values, allow_pickle=False)
        record = json.loads((path / "manifest.json").read_bytes())
        row = record["arrays"][1]
        row["sha256"] = hashlib.sha256(a.read_bytes()).hexdigest()
        row["file_bytes"] = a.stat().st_size
        row["data_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
        seal = reseal(path, record)
        with self.assertRaisesRegex(c.BoundaryCheckpointError, "content_mismatch"):
            self.prepare(path, seal)

    def test_27_manifest_duplicate_json_refuses(self):
        path, seal = self.mutated(lambda r: None)
        m = path / "manifest.json"
        raw = m.read_bytes()
        m.write_bytes(b'{"schema":"duplicate",' + raw[1:])
        with self.assertRaises((c.BoundaryCheckpointError, oi.OriginalInputsError)):
            self.prepare(path, seal)

    def test_28_preseal_failure_preserves_source_and_no_seal(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "failed"
        before = self.runner.accepted.as_record()
        real = c._prepare_material
        calls = 0

        def fail_late(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise MemoryError("bounded injected private readback failure")
            return real(*args, **kwargs)

        with (
            mock.patch.object(c, "_prepare_material", side_effect=fail_late),
            self.assertRaises(MemoryError),
        ):
            c.save_passive_boundary(
                path,
                self.runner,
                artifact_root=self.root,
                originals=self.oref,
                genesis=self.pref,
                physical=self.pref,
                source_identity=SOURCES,
                environment_sha256=ENVIRONMENT,
            )
        self.assertFalse((path / "manifest.json").exists())
        self.assertEqual(self.runner.accepted.as_record(), before)
        self.assertTrue((self.root / "boundary/manifest.json").exists())

    def test_29_postlink_failure_truthful_uncertainty_and_recovery(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "uncertain"
        real = os.unlink

        def fail_unlink(name, *args, **kwargs):
            if name == "manifest.tmp":
                raise OSError("injected postlink interruption")
            return real(name, *args, **kwargs)

        with (
            mock.patch.object(c.os, "unlink", side_effect=fail_unlink),
            self.assertRaises(c.BoundaryPublicationUncertain),
        ):
            c.save_passive_boundary(
                path,
                self.runner,
                artifact_root=self.root,
                originals=self.oref,
                genesis=self.pref,
                physical=self.pref,
                source_identity=SOURCES,
                environment_sha256=ENVIRONMENT,
            )
        record = json.loads((path / "manifest.json").read_bytes())
        self.assertTrue((path / "manifest.tmp").exists())
        self.prepare(path, record["manifest_sha256"])

    def test_30_diagnostics_copied_private_owner_not_exposed(self):
        prepared = self.prepare()
        value = prepared.diagnostics()
        value["source_identity"]["masy"] = "b" * 64
        self.assertEqual(prepared.diagnostics()["source_identity"], SOURCES)
        self.assertFalse(hasattr(prepared, "run"))
        self.assertFalse(hasattr(prepared, "restore"))

    def synthetic_history(self, count):
        """Bookkeeping templates ONLY; values are not asserted to be native event outcomes."""
        from fractions import Fraction

        from maple.aeolian.scheduler.finalization import FINALIZATION_ALGORITHM_VERSION

        from masy.clock import ClockLedger, encode_time
        from masy.dense_schedule import resolve_event

        n = dr._native()
        nc = self.runner._n_classes
        inventory = np.array(self.original.initial_inventory_kg, copy=True)
        ledger = ClockLedger(self.bed.time_s)
        before = self.bed.identity_sha256()
        time = self.bed.time_s
        receipts = []
        totals = {
            k: np.zeros(nc)
            for k in ("export", "input", "bound", "internal_terminal_deposition")
        }
        totals["water"] = {k: 0.0 for k in c.WATER_FIELDS}
        totals["n_entry_reroutes"] = 0
        totals["n_event_reroutes"] = 0
        for index in range(count):
            event = self.original.schedule.events[index]
            kind = event.kind
            duration = (
                event.duration_s
                if kind == "wind"
                else sum(v.duration_s for v in event.intervals)
            )
            end = time + duration
            after = ("f" if index == count - 1 else "e") * 64
            clock = ledger.append(
                "event",
                native_start=time,
                native_end=end,
                commit_before=0.0,
                commit_after=0.0,
                returned_elapsed=duration if kind == "wind" else None,
            )
            body = {
                "schema": dr.RECEIPT_SCHEMA,
                "index": index,
                "kind": kind,
                "status": "accepted",
                "code": "accepted",
                "static_sha256": self.original.static_sha256,
                "resolved": resolve_event(self.original.schedule, index, time),
                "clock": clock,
                "bed_identity_before": before,
                "bed_identity_after": after,
            }
            classes = {key: [0.0] * nc for key in c.RECEIPT_BODIES[kind]["classes"]}
            classes["inventory_before_kg"] = inventory.tolist()
            classes["inventory_after_kg"] = inventory.tolist()
            if kind == "wind":
                native = {
                    "block": {
                        "start_time_s": encode_time(time),
                        "end_time_s": encode_time(end),
                        "duration_s": encode_time(duration),
                    },
                    "advanced_end_time_s": encode_time(end),
                    "advanced_duration_s": encode_time(duration),
                    "physical_time_s": encode_time(end),
                    "process_mode": dr.SUPPORTED_WIND_MODE,
                    "integration_path": "baseline_direct",
                    "n_accounting_steps": 2,
                    "accounting_step_upper_bound": 3,
                    "n_commits_this_block": 0,
                    "conservation_residual_kg": 0.0,
                }
                classes.update(
                    native_bound_kg=0.125,
                    event_bound_kg=0.125,
                    event_summation_terms=6,
                    event_arithmetic_scale_kg=1.0,
                    inventory_bound_kg=0.125,
                    detachment_kg=None,
                    boundary_input_kg=[1.0] * nc,
                    boundary_export_kg=[1.0] * nc,
                )
                record = n.resolve_wind_forcing_record(
                    {
                        "blocks": [
                            {
                                "shear_velocity_m_s": event.shear_velocity_m_s,
                                "direction_deg": event.direction_deg_cw_from_north,
                                "duration_s": event.duration_s,
                            }
                        ]
                    }
                )
                canonical = dr._canonical_input(np, record, "completed_record")
                final = {key: 0.0 for key in c.RECEIPT_BODIES["wind"]["finalization"]}
                for key in final:
                    if key.endswith("by_class_kg"):
                        final[key] = [0.0] * nc
                final.update(
                    status="already_empty",
                    algorithm_version=FINALIZATION_ALGORITHM_VERSION,
                    representation="flat",
                    placement_rule="empty",
                    record_origin_s=encode_time(time),
                    record=canonical,
                    record_sha256=db._sha256(canonical),
                    completed_block_index=1,
                    completed_block_offset_s=0.0,
                    physical_time_s=encode_time(end),
                    n_settled=0,
                    n_exported=0,
                    deposited_mass_by_class_kg=[0.25] * nc,
                )
                body.update(
                    native=native,
                    classes=classes,
                    wind_end_pool="genuinely_empty",
                    finalization=final,
                    coverage={
                        "requested_s": encode_time(duration),
                        "native_plan_steps": 2,
                        "native_plan_sum_s": f"{Fraction(duration).numerator}/{Fraction(duration).denominator}",
                        "plan_discrepancy_s": "0/1",
                        "plan_rounding_bound_s": "0/1",
                        "elapsed_discrepancy_s": "0/1",
                    },
                )
                additions = ([1.0] * nc, [1.0] * nc, np.full(nc, 0.125), [0.25] * nc)
            else:
                classes.update(
                    inventory_bound_kg=[0.25] * nc,
                    internal_terminal_deposition_kg=[0.5] * nc,
                    flow_end_mobile_before_settlement_kg=None,
                    # Native reservoir_bound_kg is a scalar; DenseRunner encodes it as list1.
                    terminal_sediment_tolerance_kg=[0.0],
                )
                native = {key: 1 for key in c.RECEIPT_BODIES["rain"]["native"]}
                native.update(
                    dry_time_s=encode_time(end),
                    rainfall_end_s=encode_time(end),
                    recession_guard_end_s=encode_time(end + event.recession_guard_s),
                    rainfall_edges_s=[encode_time(time), encode_time(end)],
                    intensity_mm_h=[event.intervals[0].intensity_mm_h],
                    transport_scheme="upwind",
                    phase=None,
                )
                water = {key: 0.0 for key in c.RECEIPT_BODIES["rain"]["water"]}
                water.update(
                    rain_m3=5.0,
                    drainage_m3=2.0,
                    export_m3=1.0,
                    dry_surface_removed_m3=0.25,
                    dry_soil_removed_m3=0.5,
                    dry_reset="external dry-again removal (native); soil water is zero after the reset",
                )
                body.update(
                    native=native,
                    classes=classes,
                    water=water,
                    routing={
                        "first_graph_construction": True,
                        "entry_rerouted": False,
                        "entry_net_displacement_m": None,
                        "graph_retained_by_identity": True,
                        "n_reroutes_in_event": 1,
                    },
                )
                delta = {
                    "rain_m3": 5.0,
                    "drainage_m3": 2.0,
                    "storm_export_m3": 1.0,
                    "dry_surface_removed_m3": 0.25,
                    "dry_soil_removed_m3": 0.5,
                }
                totals["water"] = {
                    key: totals["water"][key] + value for key, value in delta.items()
                }
                totals["n_event_reroutes"] += 1
                additions = ([0.0] * nc, [0.0] * nc, np.full(nc, 0.25), [0.5] * nc)
            for key, value in zip(
                ("export", "input", "bound", "internal_terminal_deposition"),
                additions,
                strict=True,
            ):
                totals[key] = totals[key] + np.asarray(value)
            terms = 2 * self.runner._inventory_elements + 2 * (index + 1)
            scale = max(
                float(np.abs(inventory).sum()),
                float(np.abs(inventory).sum()),
                float(totals["export"].sum()),
                float(totals["input"].sum()),
            )
            body["aggregate"] = {
                "joint_residual_kg": (
                    inventory + totals["export"] - totals["input"] - inventory
                ).tolist(),
                "joint_bound_kg": (
                    totals["bound"] + n.summation_error_bound_kg(terms, scale)
                ).tolist(),
                "n_summation_terms": terms,
            }
            receipts.append(body)
            time = end
            before = after
        manifest = json.loads((self.root / "boundary/manifest.json").read_bytes())
        rows = c._receipt_rows(receipts, c.BoundaryImageLimits())
        manifest["receipts"] = rows
        manifest["ledger"] = ledger.as_record()
        manifest["totals"] = c._totals_wire(totals)
        manifest["state"].update(
            next_index=count,
            n_accepted=count,
            native_time=encode_time(time),
            accepted_bed_sha256="f" * 64,
            receipt_chain_sha256=c._chain(self.oref, self.pref, rows),
        )
        arrays = [
            np.zeros((f.NY, f.NX)),
            *[
                totals[k]
                for k in ("export", "input", "bound", "internal_terminal_deposition")
            ],
        ]
        if count == 2:
            held = self.held()
            manifest["routing"] = c._routing_record(
                held, self.runner._water, held.graph.shape, nc, c.BoundaryImageLimits()
            )
            arrays.extend([held.source, held.template.sediment_velocity_m_s])
        current = SimpleNamespace(
            bed=SimpleNamespace(
                time_s=time, state=self.bed.state, identity_sha256=lambda: "f" * 64
            ),
            mobile_state=self.runner._pool,
        )
        genesis = SimpleNamespace(bed=self.bed, mobile_state=self.runner._pool)
        return manifest, arrays, genesis, current

    def test_31_synthetic_wind_receipt_nonzero_fold_component(self):
        m, a, g, p = self.synthetic_history(1)
        _soil, totals, ledger, _receipts, _state = c._validate_history(
            m, self.original, g, p, a, c.BoundaryImageLimits()
        )
        self.assertTrue(np.array_equal(totals["input"], np.ones(6)))
        self.assertTrue(np.array_equal(totals["bound"], np.full(6, 0.125)))
        self.assertTrue(
            np.array_equal(totals["internal_terminal_deposition"], np.full(6, 0.25))
        )
        self.assertEqual(
            ledger.records()[0]["commit_after"],
            self.runner.ledger_record()["native_epoch"],
        )

    def test_32_synthetic_rain_water_mapping_fold_component(self):
        m, a, g, p = self.synthetic_history(2)
        _soil, totals, _ledger, _receipts, _state = c._validate_history(
            m, self.original, g, p, a, c.BoundaryImageLimits()
        )
        self.assertEqual(
            totals["water"],
            {
                "rain_m3": 5.0,
                "drainage_m3": 2.0,
                "storm_export_m3": 1.0,
                "dry_surface_removed_m3": 0.25,
                "dry_soil_removed_m3": 0.5,
            },
        )
        self.assertEqual(totals["n_event_reroutes"], 1)
        self.assertTrue(np.array_equal(totals["bound"], np.full(6, 0.375)))

    def test_33_last_aggregate_bits_refuse(self):
        m, a, g, p = self.synthetic_history(1)
        m["receipts"][-1]["body"]["aggregate"]["joint_bound_kg"][0] = np.nextafter(
            m["receipts"][-1]["body"]["aggregate"]["joint_bound_kg"][0], np.inf
        ).item()
        m["receipts"] = c._receipt_rows(
            [r["body"] for r in m["receipts"]], c.BoundaryImageLimits()
        )
        m["state"]["receipt_chain_sha256"] = c._chain(
            self.oref, self.pref, m["receipts"]
        )
        with self.assertRaisesRegex(c.BoundaryCheckpointError, "content_mismatch"):
            c._validate_history(m, self.original, g, p, a, c.BoundaryImageLimits())

    def test_34_receipt_closed_types_and_optional_none(self):
        for section, key, value in [
            ("classes", "inventory_bound_kg", True),
            ("native", "n_accounting_steps", 1.0),
            ("finalization", "n_settled", True),
            ("classes", "detachment_kg", {}),
        ]:
            with self.subTest(key=key):
                m, _a, _g, _p = self.synthetic_history(1)
                body = m["receipts"][0]["body"]
                body[section][key] = value
                with self.assertRaises(c.BoundaryCheckpointError):
                    c._receipt_shape(body, 6, c.BoundaryImageLimits())

    def test_35_native_loaded_pool_exact_alias(self):
        loaded = []
        real = bridge.load_boundary_physical

        def observe(*args, **kwargs):
            value = real(*args, **kwargs)
            loaded.append(value)
            return value

        with mock.patch.object(bridge, "load_boundary_physical", side_effect=observe):
            prepared = self.prepare()
        owner = prepared._PreparedBoundary__owner
        self.assertEqual(len(loaded), 1)
        self.assertIs(owner._pool, loaded[0].mobile_state)
        self.assertIs(owner._bed, loaded[0].bed)
        for name in db.SHARED_FIELDS:
            self.assertIs(
                getattr(owner._bed.state, name), getattr(loaded[0].bed.state, name)
            )

    def test_36_tight_file_cap_refuses_save_before_directory(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "uncreated"
        with self.assertRaises(c.BoundaryCheckpointError):
            c.save_passive_boundary(
                path,
                self.runner,
                artifact_root=self.root,
                originals=self.oref,
                genesis=self.pref,
                physical=self.pref,
                source_identity=SOURCES,
                environment_sha256=ENVIRONMENT,
                limits=c.BoundaryImageLimits(max_files=4),
            )
        self.assertFalse(path.exists())

    def test_37_durable_seal_fsync_ambiguity(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "uncertain"
        real = os.fsync

        def failing(fd):
            if (path / "manifest.json").exists():
                raise OSError("postpublication fsync injected")
            return real(fd)

        with (
            mock.patch.object(c.os, "fsync", side_effect=failing),
            self.assertRaises(c.BoundaryPublicationUncertain),
        ):
            c.save_passive_boundary(
                path,
                self.runner,
                artifact_root=self.root,
                originals=self.oref,
                genesis=self.pref,
                physical=self.pref,
                source_identity=SOURCES,
                environment_sha256=ENVIRONMENT,
            )
        m = json.loads((path / "manifest.json").read_bytes())
        self.prepare(path, m["manifest_sha256"])

    def test_38_original_terrain_and_soil_epochs_not_rebound(self):
        owner = self.prepare()._PreparedBoundary__owner
        self.assertIsNot(owner._soil, owner._water.initial_soil_water_m)
        self.assertIsNot(owner._inventory0, self.original.initial_inventory_kg)
        self.assertTrue(
            oi._same(
                owner._soil, owner._water.initial_soil_water_m, oi.OriginalPackLimits()
            )
        )
        self.assertEqual(
            owner._bed._initialization_receipt,
            self.original.capsule.initialization_receipt,
        )

    def test_39_array_role_shape_refuses_before_child_load(self):
        path, seal = self.mutated(
            lambda r: r["arrays"][0].__setitem__("shape", [f.NY * f.NX])
        )
        with (
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("native before role shape"),
            ),
            self.assertRaises(c.BoundaryCheckpointError),
        ):
            self.prepare(path, seal)

    def test_40_runtime_refusal_reason_preserved(self):
        from masy.runtime import RuntimeRefusal

        with (
            mock.patch(
                "masy.runtime._hash_path",
                side_effect=RuntimeRefusal("source_changed_during_read", "injected"),
            ),
            self.assertRaises(c.BoundaryCheckpointError) as caught,
        ):
            self.prepare()
        self.assertEqual(caught.exception.code, "source_changed_during_read")

    def test_41_fraction_exponents_zero_denominator_and_noncanonical_refuse(self):
        for value in ("1e999999999", "1/0", "x", "2/4", "-0/1", "0/2", "1/-2"):
            with self.subTest(value=value):
                m, _a, _g, _p = self.synthetic_history(1)
                m["receipts"][0]["body"]["coverage"]["native_plan_sum_s"] = value
                with self.assertRaises(c.BoundaryCheckpointError):
                    c._receipt_shape(
                        m["receipts"][0]["body"], 6, c.BoundaryImageLimits()
                    )

    def test_42_limits_exact_type_and_tightening_only(self):
        class Child(c.BoundaryImageLimits):
            pass

        for value in (SimpleNamespace(**vars(c.BoundaryImageLimits())),):
            with self.assertRaises(c.BoundaryCheckpointError):
                self.prepare(limits=value)
        with self.assertRaises(c.BoundaryCheckpointError):
            Child()
        with self.assertRaises(c.BoundaryCheckpointError):
            c.BoundaryImageLimits(max_cells=65)

    def test_43_original_manifest_growth_bound_before_parse(self):
        limits = c.BoundaryImageLimits()
        art = c._Artifacts(self.root, self.oref, self.pref, self.pref, limits)
        self.addCleanup(art.close)
        real = oi._open_file

        def changed(directory, name):
            fd, s = real(directory, name)
            if name == "manifest.json":
                values = list(s)
                values[6] = s.st_size + 1
                s = os.stat_result(values)
            return fd, s

        with (
            mock.patch.object(oi, "_open_file", side_effect=changed),
            mock.patch.object(
                oi,
                "_strict_json",
                side_effect=AssertionError("parse before bounded identity"),
            ),
            self.assertRaises(c.BoundaryCheckpointError),
        ):
            c._original_manifest_seal(art, self.oref, limits)

    def test_44_cross_format_child_path_alias_refuses(self):
        native = copy.deepcopy(self.pref)
        native["relative_path"] = self.oref["relative_path"]
        with self.assertRaisesRegex(c.BoundaryCheckpointError, "artifact_reference"):
            c._Artifacts(self.root, self.oref, native, native, c.BoundaryImageLimits())

    def test_45_native_manifest_cohort_cap_before_physical_load(self):
        temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        target = Path(temporary.name) / "native"
        shutil.copytree(self.root / "genesis", target)
        m = json.loads((target / "checkpoint.json").read_bytes())
        for name, row in m["files"]["native_state.npz"]["members"].items():
            if name.startswith("cohort_"):
                row["shape"] = [16001]
        m["manifest_sha256"] = hashlib.sha256(
            cp._json({k: v for k, v in m.items() if k != "manifest_sha256"}).encode()
        ).hexdigest()
        (target / "checkpoint.json").chmod(0o600)
        (target / "checkpoint.json").write_text(cp._json(m))
        new = reference(target, cp.SCHEMA, self.pref["identity"])
        new["relative_path"] = str(target.relative_to(self.root))
        path, seal = self.mutated(lambda r: r.update(genesis=new, physical=new))
        with (
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("native expansion before cohort cap"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "profile_cap"),
        ):
            self.prepare(path, seal)

    def test_46_wrapper_object_header_no_pickle_expansion(self):
        path, seal = self.mutated(lambda r: None)
        a = path / "a_0000.npy"
        a.chmod(0o600)
        import io

        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "|O", "fortran_order": False, "shape": (f.NY, f.NX)}
        )
        a.write_bytes(stream.getvalue() + bytes(8 * f.NY * f.NX))
        m = json.loads((path / "manifest.json").read_bytes())
        m["arrays"][0].update(
            file_bytes=a.stat().st_size,
            sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
        )
        seal = reseal(path, m)
        with (
            mock.patch.object(
                np, "load", side_effect=AssertionError("pickle/np.load forbidden")
            ),
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("native before header guard"),
            ),
            self.assertRaises((c.BoundaryCheckpointError, oi.OriginalInputsError)),
        ):
            self.prepare(path, seal)

    def test_47_header_declared_size_cap_before_numpy_parser(self):
        path, seal = self.mutated(lambda r: None)
        a = path / "a_0000.npy"
        a.chmod(0o600)
        a.write_bytes(
            b"\x93NUMPY\x01\x00"
            + (4097).to_bytes(2, "little")
            + b" " * 4097
            + bytes(8 * f.NY * f.NX)
        )
        m = json.loads((path / "manifest.json").read_bytes())
        m["arrays"][0].update(
            file_bytes=a.stat().st_size,
            sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
        )
        seal = reseal(path, m)
        with (
            mock.patch.object(
                np.lib.format,
                "read_array_header_1_0",
                side_effect=AssertionError("numpy parse before prefix cap"),
            ),
            self.assertRaises((c.BoundaryCheckpointError, oi.OriginalInputsError)),
        ):
            self.prepare(path, seal)

    def test_48_preserves_wet_genesis_and_nonzero_original_soil(self):
        actual = db.build_dense_bed

        def wet(**kwargs):
            kwargs.update(
                initial_time_s=7.25, initial_water_depth_m=np.full((f.NY, f.NX), 0.001)
            )
            return actual(**kwargs)

        with mock.patch.object(f, "build_dense_bed", side_effect=wet):
            bed = f.build_bed()
        water = dataclasses.replace(
            f.water_inputs(bed), initial_soil_water_m=np.full((f.NY, f.NX), 0.02)
        )
        runner = f.make_runner(bed, f.schedule(f.rain()), water=water)
        original = oi.capture_original_inputs(runner, source_identity=SOURCES)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        oi.save_original_inputs(root / "original", original)
        oref = reference(root / "original", oi.SCHEMA, original.content_sha256)
        caller = {
            "schema": "masy.dense_runner_boundary_native.v1",
            "source_identity": SOURCES,
            "environment_sha256": ENVIRONMENT,
            "static_sha256": runner.static_sha256,
        }
        packed = bridge.pack_boundary_physical(
            bed,
            runner._pool,
            runner._wind.config,
            caller,
            inputs=original.capsule,
            max_bytes=64 << 20,
        )
        cp.save_exact_physical_checkpoint(
            root / "genesis",
            packed.physical,
            runner._wind.config,
            packed.identity,
            max_bytes=64 << 20,
        )
        pref = reference(root / "genesis", cp.SCHEMA, packed.identity)
        seal = c.save_passive_boundary(
            root / "boundary",
            runner,
            artifact_root=root,
            originals=oref,
            genesis=pref,
            physical=pref,
            source_identity=SOURCES,
            environment_sha256=ENVIRONMENT,
        )
        prepared = c.prepare_passive_boundary(
            root / "boundary",
            artifact_root=root,
            expected_manifest_sha256=seal,
            expected_source_identity=SOURCES,
            expected_environment_sha256=ENVIRONMENT,
        )
        owner = prepared._PreparedBoundary__owner
        self.assertEqual(owner._bed.time_s.hex(), (7.25).hex())
        self.assertTrue(
            np.array_equal(owner._bed.state.water.depth_m, np.full((f.NY, f.NX), 0.001))
        )
        self.assertTrue(np.array_equal(owner._soil, np.full((f.NY, f.NX), 0.02)))
        self.assertEqual(owner.ledger_record()["native_epoch"], c.encode_time(7.25))

    def test_49_terrain_raw_cap_before_fingerprint(self):
        _h, p, _r = self.projected()
        terrain = copy.copy(p.terrain)
        object.__setattr__(terrain, "initial_full_elevation_m", np.zeros((100000, 1)))
        bad = dataclasses.replace(p, terrain=terrain)
        water = dataclasses.replace(self.runner._water, terrain=terrain)
        with (
            mock.patch.object(
                c,
                "_cache_fingerprint",
                side_effect=AssertionError("terrain fingerprint before cap"),
            ),
            self.assertRaises(dr._Stop),
        ):
            c._restored_memory(bad, water, p.graph.shape, self.runner._n_classes)

    def test_50_reader_reconciles_second_raw_admission(self):
        real = c._read_wrapper
        calls = 0

        def changed(*args, **kwargs):
            nonlocal calls
            result = real(*args, **kwargs)
            calls += 1
            if calls == 2:
                return result[0], result[1], result[2] + 1
            return result

        with (
            mock.patch.object(c, "_read_wrapper", side_effect=changed),
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("native after mismatched raw basis"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "file_changed"),
        ):
            self.prepare()

    def test_51_projection_changed_across_commit_preserves_refusal(self):
        _held, p, _record = self.projected()
        runner = copy.copy(self.runner)
        runner._routing = p
        n = dr._native()
        event = runner._schedule.events[1]
        from masy.dense_schedule import resolve_event

        resolved = resolve_event(runner._schedule, 1, runner._bed.time_s)

        def commit(*args, **kwargs):
            object.__setattr__(p, "phase", object())
            return SimpleNamespace(
                state=runner._bed.state,
                graph=p.graph,
                did_commit=True,
                rerouted=False,
                routing_source_m=p.source,
            )

        n.commit_bed = commit
        with (
            mock.patch.object(dr, "_native", return_value=n),
            mock.patch.object(
                runner, "_stage", return_value=SimpleNamespace(state=runner._bed.state)
            ),
            mock.patch.object(
                dr,
                "_complete_event",
                side_effect=AssertionError("native event forbidden"),
            ),
            self.assertRaises(dr._Stop) as caught,
        ):
            runner._rain_event(1, event, resolved)
        self.assertEqual(caught.exception.code, "transport_memory_not_empty")
        self.assertEqual(caught.exception.guarantee, dr._EXTERNAL)

    def test_52_tiny_wrapper_budget_before_any_hash(self):
        with (
            mock.patch.object(
                c,
                "_cache_fingerprint",
                side_effect=AssertionError("cache hash before byte cap"),
            ),
            mock.patch.object(
                self.runner,
                "_precheck",
                side_effect=AssertionError("private hash before byte cap"),
            ),
            self.assertRaises(c.BoundaryCheckpointError),
        ):
            c._snapshot(
                self.runner,
                self.oref,
                self.pref,
                self.pref,
                SOURCES,
                ENVIRONMENT,
                c.BoundaryImageLimits(max_numeric_bytes=1),
            )

    def test_53_seven_array_wire_and_own_leftover_seal(self):
        # Closed wire codec component only. These synthetic bookkeeping templates
        # do not establish a native accepted wind/rain boundary or event continuation.
        m, arrays, _g, _p = self.synthetic_history(2)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "wire"
        path.mkdir()
        rows = []
        for index, array in enumerate(arrays):
            name = f"a_{index:04d}.npy"
            with (path / name).open("wb") as stream:
                np.lib.format.write_array(stream, array, allow_pickle=False)
            rows.append(
                {
                    "name": name,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "data_bytes": array.nbytes,
                    "data_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                    "file_bytes": (path / name).stat().st_size,
                    "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
                }
            )
        m["arrays"] = rows
        m["manifest_sha256"] = c._sha(
            {k: v for k, v in m.items() if k != "manifest_sha256"}
        )
        (path / "manifest.json").write_bytes(oi._json(m))
        os.link(path / "manifest.json", path / "manifest.tmp")
        decoded, restored, _raw = c._read_wrapper(
            path, m["manifest_sha256"], c.BoundaryImageLimits()
        )
        self.assertEqual(len(restored), 7)
        self.assertEqual(decoded, m)
        for original, actual in zip(arrays, restored, strict=True):
            self.assertTrue(oi._same(original, actual, oi.OriginalPackLimits()))
            self.assertFalse(actual.flags.writeable)
            self.assertFalse(np.shares_memory(original, actual))

    def test_54_native_pending_shape_before_physical_expand(self):
        temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        target = Path(temporary.name) / "native"
        shutil.copytree(self.root / "genesis", target)
        m = json.loads((target / "checkpoint.json").read_bytes())
        m["files"]["pending_state.npz"]["members"][
            "ledger__pending_bed_mass_change_kg"
        ]["shape"] = [65, 1, 6]
        m["manifest_sha256"] = hashlib.sha256(
            cp._json({k: v for k, v in m.items() if k != "manifest_sha256"}).encode()
        ).hexdigest()
        (target / "checkpoint.json").chmod(0o600)
        (target / "checkpoint.json").write_text(cp._json(m))
        new = reference(target, cp.SCHEMA, self.pref["identity"])
        new["relative_path"] = str(target.relative_to(self.root))
        path, seal = self.mutated(lambda r: r.update(genesis=new, physical=new))
        with (
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("physical expand before pending cap"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "profile_cap"),
        ):
            self.prepare(path, seal)

    def test_55_artifact_recheck_enumeration_is_bounded(self):
        art = c._Artifacts(
            self.root, self.oref, self.pref, self.pref, c.BoundaryImageLimits()
        )
        self.addCleanup(art.close)

        class Many:
            def __init__(self):
                self.count = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def __iter__(self):
                return self

            def __next__(self):
                self.count += 1
                if self.count > 260:
                    raise AssertionError("unbounded enumeration")
                return SimpleNamespace(name=f"x{self.count}")

        with (
            mock.patch.object(oi.os, "scandir", side_effect=lambda *args: Many()),
            self.assertRaises(oi.OriginalInputsError),
        ):
            art.check()

    def test_56_reference_invalid_os_names_closed(self):
        for value in ("a\x00b", "a\nb", "x" * 256, "é" * 128):
            with self.subTest(path=value):
                r = copy.deepcopy(self.oref)
                r["relative_path"] = value
                with self.assertRaises(c.BoundaryCheckpointError):
                    c._reference(r, oi.SCHEMA)

    def changed_returned_clock(self, count, index, returned):
        from masy.clock import ClockLedger, decode_time

        m, a, g, p = self.synthetic_history(count)
        ledger = ClockLedger(self.bed.time_s)
        for i, old in enumerate(m["ledger"]["entries"]):
            value = (
                returned
                if i == index
                else (
                    None
                    if old["returned_elapsed"] is None
                    else decode_time(old["returned_elapsed"])
                )
            )
            ledger.append(
                "event",
                native_start=decode_time(old["native_start"]),
                native_end=decode_time(old["native_end"]),
                commit_before=decode_time(old["commit_before"]),
                commit_after=decode_time(old["commit_after"]),
                returned_elapsed=value,
            )
        m["ledger"] = ledger.as_record()
        bodies = [r["body"] for r in m["receipts"]]
        for body, clock in zip(bodies, ledger.records(), strict=True):
            body["clock"] = clock
        m["receipts"] = c._receipt_rows(bodies, c.BoundaryImageLimits())
        m["state"]["receipt_chain_sha256"] = c._chain(
            self.oref, self.pref, m["receipts"]
        )
        return m, a, g, p

    def test_57_returned_elapsed_matches_owning_finish_kind(self):
        for count, index, returned in ((1, 0, None), (1, 0, 1.5), (2, 1, 1.0)):
            with self.subTest(count=count, index=index, returned=returned):
                m, a, g, p = self.changed_returned_clock(count, index, returned)
                with self.assertRaisesRegex(c.BoundaryCheckpointError, "receipt_clock"):
                    c._validate_history(
                        m, self.original, g, p, a, c.BoundaryImageLimits()
                    )

    def test_58_intermediate_aggregate_term_count_exact(self):
        m, a, g, p = self.synthetic_history(2)
        m["receipts"][0]["body"]["aggregate"]["n_summation_terms"] += 2
        m["receipts"] = c._receipt_rows(
            [r["body"] for r in m["receipts"]], c.BoundaryImageLimits()
        )
        m["state"]["receipt_chain_sha256"] = c._chain(
            self.oref, self.pref, m["receipts"]
        )
        with self.assertRaisesRegex(c.BoundaryCheckpointError, "aggregate_identity"):
            c._validate_history(m, self.original, g, p, a, c.BoundaryImageLimits())

    def test_59_native_pending_metadata_bool_and_B_preflight_both_refuse(self):
        temporary = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(temporary.cleanup)
        target = Path(temporary.name) / "native"
        shutil.copytree(self.root / "genesis", target)
        m = json.loads((target / "checkpoint.json").read_bytes())
        row = m["files"]["pending_state.npz"]["members"][
            "ledger__pending_bed_mass_change_kg"
        ]
        row["shape"] = [f.NY, True, 6]
        m["manifest_sha256"] = hashlib.sha256(
            cp._json({k: v for k, v in m.items() if k != "manifest_sha256"}).encode()
        ).hexdigest()
        (target / "checkpoint.json").chmod(0o600)
        (target / "checkpoint.json").write_text(cp._json(m))
        # Independently test the suspected normalization: owning15 correctly
        # rejects via canonical JSON header-spec comparison before loading arrays.
        with self.assertRaisesRegex(
            cp.PhysicalCheckpointError, "payload member inventory/spec mismatch"
        ):
            cp.load_exact_physical_checkpoint(
                target,
                self.runner._wind.config,
                self.pref["identity"],
                max_bytes=64 << 20,
            )
        new = reference(target, cp.SCHEMA, self.pref["identity"])
        new["relative_path"] = str(target.relative_to(self.root))
        path, seal = self.mutated(lambda r: r.update(genesis=new, physical=new))
        with (
            mock.patch.object(
                bridge,
                "load_boundary_physical",
                side_effect=AssertionError("B load before typed pending metadata"),
            ),
            self.assertRaisesRegex(c.BoundaryCheckpointError, "profile_cap"),
        ):
            self.prepare(path, seal)

    def test_60_relative_root_anchored_before_cwd_change(self):
        old_cwd = Path.cwd()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        real = c._Artifacts.__init__

        def admitted(instance, *args, **kwargs):
            real(instance, *args, **kwargs)
            os.chdir(temporary.name)

        try:
            os.chdir(self.root.parent)
            with mock.patch.object(c._Artifacts, "__init__", new=admitted):
                prepared = c.prepare_passive_boundary(
                    self.root / "boundary",
                    artifact_root=self.root.name,
                    expected_manifest_sha256=self.seal,
                    expected_source_identity=SOURCES,
                    expected_environment_sha256=ENVIRONMENT,
                )
            self.assertEqual(
                prepared.diagnostics()["accepted"], self.runner.accepted.as_record()
            )
        finally:
            os.chdir(old_cwd)

    def test_61_admission_context_limits_raw_and_refs_bound(self):
        m = json.loads((self.root / "boundary/manifest.json").read_bytes())
        limits = c.BoundaryImageLimits()
        art = c._Artifacts(self.root, self.oref, self.pref, self.pref, limits)
        self.addCleanup(art.close)
        arrays = [
            self.runner._soil,
            *[
                self.runner._totals[k]
                for k in ("export", "input", "bound", "internal_terminal_deposition")
            ],
        ]
        for supplied, raw in ((dataclasses.replace(limits), 0), (limits, 1)):
            with (
                mock.patch.object(
                    bridge,
                    "load_boundary_physical",
                    side_effect=AssertionError("load before context binding"),
                ),
                self.assertRaisesRegex(c.BoundaryCheckpointError, "restore_ticket"),
            ):
                c._prepare_material(
                    m,
                    arrays,
                    self.root,
                    SOURCES,
                    ENVIRONMENT,
                    supplied,
                    wrapper_raw=raw,
                    artifacts=art,
                )

    def test_62_postlink_fstat_failure_stays_uncertain(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "uncertain"
        real = os.fstat

        def failing(fd):
            if (path / "manifest.json").exists():
                raise OSError("postlink fstat fault")
            return real(fd)

        with (
            mock.patch.object(c.os, "fstat", side_effect=failing),
            self.assertRaises(c.BoundaryPublicationUncertain),
        ):
            c.save_passive_boundary(
                path,
                self.runner,
                artifact_root=self.root,
                originals=self.oref,
                genesis=self.pref,
                physical=self.pref,
                source_identity=SOURCES,
                environment_sha256=ENVIRONMENT,
            )
        m = json.loads((path / "manifest.json").read_bytes())
        self.prepare(path, m["manifest_sha256"])

    def test_63_same_native_path_requires_identical_closed_reference(self):
        m, _a, _g, _p = self.synthetic_history(1)
        m["genesis"] = copy.deepcopy(m["genesis"])
        m["genesis"]["identity"]["bed_identity_sha256"] = "a" * 64
        with self.assertRaisesRegex(c.BoundaryCheckpointError, "artifact_reference"):
            c._preflight_metadata(m, SOURCES, ENVIRONMENT, c.BoundaryImageLimits())

    def test_64_malformed_receipt_container_kinds_closed(self):
        with self.assertRaises(c.BoundaryCheckpointError):
            c._receipt_shape([], 6, c.BoundaryImageLimits())
        m, _a, _g, _p = self.synthetic_history(2)
        body = m["receipts"][1]["body"]
        body["native"]["rainfall_edges_s"] = 0
        with self.assertRaises(c.BoundaryCheckpointError):
            c._receipt_shape(body, 6, c.BoundaryImageLimits())

    def test_65_raw_boolean_npy_shape_refuses_before_expansion(self):
        # The ambiguity is True == 1: make the native fixture precondition explicit.
        self.assertEqual(f.NX, 1)
        import io

        path, seal = self.mutated(lambda r: None)
        a = path / "a_0000.npy"
        a.chmod(0o600)
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "<f8", "fortran_order": False, "shape": (f.NY, True)}
        )
        a.write_bytes(stream.getvalue() + bytes(8 * f.NY * f.NX))
        m = json.loads((path / "manifest.json").read_bytes())
        m["arrays"][0].update(
            file_bytes=a.stat().st_size,
            sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
        )
        seal = reseal(path, m)
        with (
            mock.patch.object(
                np,
                "fromfile",
                side_effect=AssertionError(
                    "numeric expansion before raw dimension kind"
                ),
            ),
            self.assertRaisesRegex(
                c.BoundaryCheckpointError, "array_layout: NPY prefix differs"
            ),
        ):
            self.prepare(path, seal)

    def test_66_body_schedule_kind_mismatch_is_coded_refusal(self):
        m, a, g, p = self.synthetic_history(1)
        other, _a, _g, _p = self.synthetic_history(2)
        body = copy.deepcopy(other["receipts"][1]["body"])
        self.assertEqual(body["kind"], "rain")
        self.assertEqual(self.original.schedule.events[0].kind, "wind")
        body.update(
            index=0,
            clock=m["ledger"]["entries"][0],
            bed_identity_before=self.bed.identity_sha256(),
        )
        m["receipts"] = c._receipt_rows([body], c.BoundaryImageLimits())
        m["state"]["receipt_chain_sha256"] = c._chain(
            self.oref, self.pref, m["receipts"]
        )
        with self.assertRaisesRegex(
            c.BoundaryCheckpointError, "source/event/clock/bed chain differs"
        ):
            c._validate_history(m, self.original, g, p, a, c.BoundaryImageLimits())


if __name__ == "__main__":
    unittest.main(verbosity=2)
