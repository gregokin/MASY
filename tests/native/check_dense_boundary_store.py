"""StageC structural CPU genesis/passive/fault checks; no event execution."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fixture_dense_alternation as f
import numpy as np
from maple.aeolian.scheduler import physical_checkpoint as cp

from masy import dense_bed as db
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as b

REQUEST = json.loads(Path(os.environ["MASY53_RUNTIME"]).read_text())
STAGE = Path(os.environ["MASY53_STAGE"])
SOURCES = REQUEST["source_identity"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cpu_declarations():
    """Same literal pinned CPU declarations as the harness, before identity capture."""
    for name in ("jaxlib.mlir.dialects.mhlo", "jaxlib.mlir._mlir_libs._mlirHlo", "readline", "resource"):
        importlib.import_module(name)


def held(runner):
    """Illustrative native saved-source graph; no rain or hydraulic reroute."""
    n = dr._native()
    source = np.array(runner._water.terrain.initial_committed_elevation_m, copy=True)
    source[:, 0] += np.linspace(0.001, 0.002, source.shape[0])
    graph = n.routing_graph_from_elevation(runner._water.terrain, source, xp=np)
    network, grid = n.transport_network(graph), n.physics_grid_from_graph(graph)
    from maple_syrup.sediment_event import SedimentEventState

    template = SedimentEventState(
        n.initial_state(
            graph, runner._bed.state.water.depth_m, runner._soil, t_s=runner._bed.time_s
        ),
        runner._bed.state,
        np.zeros((*source.shape, runner._n_classes), dtype=np.float64),
        graph,
        network,
        grid,
        runner._water.terrain,
        None,
        source,
    )
    return SimpleNamespace(
        graph=graph,
        network=network,
        grid=grid,
        terrain=runner._water.terrain,
        source=source,
        template=template,
    )


def arrays(value):
    if type(value) is np.ndarray:
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from arrays(getattr(value, field.name))
    elif type(value) is dict:
        for item in value.values():
            yield from arrays(item)
    elif type(value) in (tuple, list):
        for item in value:
            yield from arrays(item)


class StoreChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu_declarations()
        # Authored component initialization, not a restored/fabricated event.
        actual = db.build_dense_bed

        def wet(**kwargs):
            return actual(
                **{
                    **kwargs,
                    "initial_time_s": 7.25,
                    "initial_water_depth_m": np.full((f.NY, f.NX), 0.001),
                }
            )

        with mock.patch.object(f, "build_dense_bed", side_effect=wet):
            cls.bed = f.build_bed()
        water = dataclasses.replace(
            f.water_inputs(cls.bed), initial_soil_water_m=np.full((f.NY, f.NX), 0.02)
        )
        cls.runner = f.make_runner(
            cls.bed, f.schedule(f.wind(), f.rain(), f.wind(), f.rain()), water=water
        )
        cls.originals = oi.capture_original_inputs(cls.runner, source_identity=SOURCES)
        cls.temporary = tempfile.TemporaryDirectory(dir=STAGE)
        cls.root = Path(cls.temporary.name)
        tables = {
            "masy": {
                rel: row
                for rel, row in REQUEST["candidate_files"].items()
                if rel.startswith("src/") or rel == "pyproject.toml"
            },
            "maple": REQUEST["native_files"]["MAPLE"],
            "maple_syrup": REQUEST["native_files"]["SYRUP"],
        }
        roots = {
            "masy": REQUEST["import_prefix"][0],
            "maple": REQUEST["import_prefix"][2],
            "maple_syrup": REQUEST["import_prefix"][1],
        }
        source_manifest = {
            "schema": c.SCHEMA + ".sources",
            "source_identity": SOURCES,
            "module_roots": roots,
            "files": tables,
            "environment": c._environment_identity(),
        }
        source_path = STAGE / "retained_source_manifest.json"
        source_path.write_bytes(oi._json(source_manifest))
        source_path.chmod(0o444)
        native_manifest = Path(REQUEST["native_bundle_manifest_path"])
        cls.reference = {
            "schema": c.SCHEMA + ".runtime",
            "kind": "source_manifest_prototype",
            "source_manifest_path": str(source_path),
            "source_manifest_file_sha256": sha(source_path.read_bytes()),
            "native_bundle_manifest_path": str(native_manifest),
            "native_bundle_manifest_file_sha256": sha(native_manifest.read_bytes()),
            "module_roots": roots,
            "cpu_store": None,
            "cpu_lock_id": None,
            "cpu_lock_file_sha256": None,
        }
        (STAGE / "runtime_reference.json").write_bytes(oi._json(cls.reference))
        bootstrap = """import json,os,runpy,sys
from pathlib import Path
r=json.loads(Path(os.environ['MASY53_RUNTIME']).read_text())
sys.path[:0]=r['import_prefix']+[str(Path(r['candidate_root'])/'tests/native')]
module=runpy.run_path(str(Path(r['candidate_root'])/'tests/native/check_dense_boundary_store.py'),run_name='child_fixture')
module['_child_main'](sys.argv[1],Path(sys.argv[2]),sys.argv[3])
"""
        (STAGE / "child_bootstrap.py").write_text(bootstrap)
        cls.counter = 0
        cls.main = cls.new()
        counts = {
            "original_decode": 0,
            "native_physical_load": 0,
            "saved_source_graph_factory": 0,
        }

        def counted(name, actual):
            def call(*args, **kwargs):
                counts[name] += 1
                return actual(*args, **kwargs)

            return call

        n = dr._native()
        with (
            mock.patch.object(
                oi,
                "load_original_inputs",
                new=counted("original_decode", oi.load_original_inputs),
            ),
            mock.patch.object(
                c.bridge,
                "load_boundary_physical",
                new=counted("native_physical_load", c.bridge.load_boundary_physical),
            ),
            mock.patch.object(
                n,
                "routing_graph_from_elevation",
                new=counted(
                    "saved_source_graph_factory", n.routing_graph_from_elevation
                ),
            ),
        ):
            handle = cls.create(cls.main)
        storage = json.loads((cls.main / "STORE.json").read_bytes())["storage_contract"]
        (STAGE / "storage_observation.json").write_bytes(
            oi._json(
                {
                    "storage_contract": storage,
                    "probe_receipt": json.loads(
                        (cls.main / ".storage_probe/result.json").read_bytes()
                    ),
                    "scope": "actual metadata-only POSIX observation/probes, not power-loss proof",
                }
            )
        )
        (STAGE / "component_cost_observation.json").write_bytes(
            oi._json(
                {
                    "actual_genesis_calls": counts,
                    "computed_attempt_reservation": handle._reservation_record,
                    "scope": "actual passive/genesis counts; no actual event scratch, peak or scale claim",
                }
            )
        )
        cls.store_sha = handle.report()["store_file_sha256"]
        cls.token = handle.head_token()
        cls.report0 = handle.report()["accepted"]
        handle.close()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def new(cls):
        cls.counter += 1
        return cls.root / f"store_{cls.counter:03d}"

    @classmethod
    def create(cls, path, **kwargs):
        return c.create_boundary_store(
            path,
            runner=cls.runner,
            originals=cls.originals,
            runtime_reference=cls.reference,
            storage_contract=c.StorageContract(True, True),
            **kwargs,
        )

    def recover(self, path=None, expected=None, **kwargs):
        return c.recover_boundary_store(
            path or self.main,
            expected_store_file_sha256=expected or self.store_sha,
            **kwargs,
        )

    def test_01_complete_genesis_metadata_facade(self):
        handle = self.recover()
        self.addCleanup(handle.close)
        self.assertEqual(handle.head_token(), self.token)
        report = handle.report()
        self.assertEqual(report["accepted"], self.report0)
        self.assertEqual(report["accepted"]["accepted"]["time_s"], b.encode_time(7.25))
        self.assertEqual(
            report["accepted"]["clock_ledger"]["native_epoch"], b.encode_time(7.25)
        )
        self.assertFalse(hasattr(handle, "runner"))
        self.assertIsNone(handle._failed_owner)
        self.assertIsNone(handle._state.ack.receipt_bytes)
        cost = json.loads((STAGE / "component_cost_observation.json").read_bytes())
        self.assertEqual(cost["actual_genesis_calls"]["original_decode"], 3)
        self.assertEqual(cost["actual_genesis_calls"]["native_physical_load"], 3)
        self.assertEqual(cost["actual_genesis_calls"]["saved_source_graph_factory"], 0)
        report["accepted"].clear()
        self.assertEqual(handle.report()["accepted"], self.report0)

    def test_02_restore_forbidden_calls(self):
        forbidden = AssertionError("forbidden reconstruction/transport/reset/commit")
        dr._native()
        with (
            mock.patch.object(dr.DenseRunner, "__init__", side_effect=forbidden),
            mock.patch.object(db, "build_dense_bed", side_effect=forbidden),
            mock.patch.object(dr, "_advance_wind_block", side_effect=forbidden),
            mock.patch.object(dr, "_finalize_wind_event", side_effect=forbidden),
            mock.patch.object(dr, "_complete_event", side_effect=forbidden),
            mock.patch("maple_syrup.sediment_bed.commit_bed", side_effect=forbidden),
        ):
            handle = self.recover()
            handle.close()

    def test_03_quiescent_owner_alias_and_disjointness(self):
        handle = self.recover()
        self.addCleanup(handle.close)
        prepared, _chain = handle._load(handle._state.head)
        owner = c._owner(prepared)
        self.assertEqual(set(vars(owner)), set(b.RUNNER_FIELDS))
        self.assertIs(owner._wind.config.geometry, owner._bed.context.geometry)
        native = owner._bed.fresh_scheduler_state(owner._pool, time_s=owner._bed.time_s)
        self.assertIs(native.physical.mobile_state, owner._pool)
        source = (
            list(arrays(self.runner._bed.state))
            + list(arrays(self.runner._pool))
            + [self.runner._soil]
        )
        loaded = (
            list(arrays(owner._bed.state)) + list(arrays(owner._pool)) + [owner._soil]
        )
        self.assertTrue(all(not a.flags.writeable for a in loaded))
        self.assertTrue(all(not np.shares_memory(a, z) for a in source for z in loaded))
        self.assertTrue(np.array_equal(owner._soil, np.full((f.NY, f.NX), 0.02)))

    def test_04_head_canonical_token_and_complete_identity(self):
        raw = (self.main / "HEAD").read_bytes()
        head = json.loads(raw)
        self.assertFalse(raw.endswith(b"\n"))
        self.assertEqual(raw, oi._json(head))
        self.assertEqual(
            self.token,
            sha(oi._json({"store_id": head["store_id"], "head_file_sha256": sha(raw)})),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.BoundaryAcknowledgement("x", "s", "o", 0, "t", None).sequence = 1

    def test_05_lease_second_process_and_exec_cloexec(self):
        handle = self.recover()
        self.addCleanup(handle.close)
        self.assertFalse(os.get_inheritable(handle._lockfd))
        result, text = self.child("lease", self.main, self.store_sha)
        self.assertEqual(result, 0)
        self.assertEqual(text, "lease_busy")

    def child(self, mode, path, expected="-"):
        name = f"child-{mode}-{self.counter:03d}"
        stdout, stderr = STAGE / (name + ".stdout"), STAGE / (name + ".stderr")
        command = [
            sys.executable,
            "-I",
            "-B",
            str(STAGE / "child_bootstrap.py"),
            mode,
            str(path),
            expected,
        ]
        with stdout.open("x") as out, stderr.open("x") as err:
            child = subprocess.Popen(
                command,
                env=dict(os.environ),
                stdout=out,
                stderr=err,
                close_fds=True,
                start_new_session=False,
            )
            ticks = (
                Path(f"/proc/{child.pid}/stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[19]
            )
            launch = {
                "command": command,
                "pid": child.pid,
                "start_ticks": ticks,
                "timeout_s": 60,
                "close_fds": True,
                "inherited_GPU_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            }
            (STAGE / (name + ".launch.json")).write_bytes(oi._json(launch))
            started = time.monotonic()
            reason = None
            while child.poll() is None:
                if (
                    time.monotonic() - started > 60
                    or stdout.stat().st_size + stderr.stat().st_size > 1 << 20
                ):
                    reason = "wall_or_output_cap"
                    child.kill()
                    break
                time.sleep(0.05)
            child.wait(timeout=10)
        (STAGE / (name + ".exit.json")).write_bytes(
            oi._json(
                {
                    "pid": child.pid,
                    "start_ticks": ticks,
                    "exit_code": child.returncode,
                    "reaped": True,
                    "termination_reason": reason,
                    "elapsed_s": time.monotonic() - started,
                }
            )
        )
        return child.returncode, stdout.read_text().strip()

    def test_06_thread_and_reentrant_refusal(self):
        handle = self.recover()
        self.addCleanup(handle.close)
        errors = []

        def other():
            try:
                handle.head_token()
            except c.BoundaryStoreError as exc:
                errors.append(exc.code)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["handle_binding"])
        handle._entered = True
        with self.assertRaisesRegex(c.BoundaryStoreError, "handle_reentrant"):
            handle.step(self.token)
        with self.assertRaisesRegex(c.BoundaryStoreError, "handle_reentrant"):
            handle.close()
        handle._entered = False

    def test_07_stale_refusal_writes_nothing_and_remains_idle(self):
        handle = self.recover()
        self.addCleanup(handle.close)
        before = sorted(str(p.relative_to(self.main)) for p in self.main.rglob("*"))
        with (
            mock.patch.object(
                dr.DenseRunner,
                "step",
                side_effect=AssertionError("event on stale token"),
            ),
            self.assertRaisesRegex(c.BoundaryStoreError, "stale_parent"),
        ):
            handle.step("a" * 64)
        self.assertEqual(
            before, sorted(str(p.relative_to(self.main)) for p in self.main.rglob("*"))
        )
        self.assertEqual(handle.report()["phase"], "idle")

    def test_08_reservation_interrupt_no_event_or_head_change(self):
        path = self.new()
        handle = self.create(path)
        self.addCleanup(handle.close)
        token, raw = handle.head_token(), (path / "HEAD").read_bytes()
        with (
            mock.patch.object(
                dr.DenseRunner,
                "step",
                side_effect=KeyboardInterrupt("abort before event"),
            ) as event,
            self.assertRaises(KeyboardInterrupt),
        ):
            handle.step(token)
        self.assertEqual(event.call_count, 1)
        self.assertEqual((path / "HEAD").read_bytes(), raw)
        self.assertEqual(handle.report()["phase"], "halted_before_head")
        self.assertIsNone(handle.report()["intended_token"])
        with self.assertRaisesRegex(c.BoundaryStoreError, "store_halted"):
            handle.step(token)
        self.assertIsNotNone(handle._failed_owner)
        store_sha = handle.report()["store_file_sha256"]
        handle.close()
        restored = self.recover(path, store_sha)
        restored.close()

    def test_09_raw_and_aggregate_cap_before_metadata_hash(self):
        with (
            mock.patch.object(
                c.runtime,
                "_read_bounded",
                side_effect=AssertionError("hash before raw cap"),
            ),
            self.assertRaisesRegex(c.BoundaryStoreError, "metadata_cap"),
        ):
            c._read(self.main / "HEAD", 1)
        with (
            mock.patch.object(
                c.runtime,
                "_hash_path",
                side_effect=AssertionError("hash before aggregate cap"),
            ),
            self.assertRaisesRegex(c.BoundaryStoreError, "store_raw_cap"),
        ):
            self.recover(
                limits=dataclasses.replace(c.BoundaryStoreLimits(), max_store_bytes=1)
            )

    def test_10_noncanonical_head_refuses_no_fallback(self):
        path = self.main / "HEAD"
        raw = path.read_bytes()
        try:
            path.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(c.BoundaryStoreError, "metadata_noncanonical"):
                self.recover()
        finally:
            path.write_bytes(raw)

    def test_11_unknown_root_and_symlink_refused(self):
        extra = self.main / "unexpected"
        try:
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(c.BoundaryStoreError, "path_refused"):
                self.recover()
        finally:
            extra.unlink()
        try:
            extra.symlink_to(self.main / "HEAD")
            with self.assertRaisesRegex(c.BoundaryStoreError, "path_refused"):
                self.recover()
        finally:
            extra.unlink()

    def test_12_missing_head_incomplete_not_promoted(self):
        path = self.main / "HEAD"
        raw = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                self.recover()
        finally:
            path.write_bytes(raw)

    def test_13_unsupported_runtime_kind_before_native(self):
        reference = copy.deepcopy(self.reference)
        reference["kind"] = "unsupported_runtime_kind"
        with (
            mock.patch.object(
                c,
                "_environment_identity",
                side_effect=AssertionError("runtime expanded"),
            ),
            self.assertRaisesRegex(c.BoundaryStoreError, "runtime_kind_unsupported_v1"),
        ):
            c._runtime_binding(reference, c.BoundaryStoreLimits())

    def test_14_mountinfo_longest_boundary_device_and_caps(self):
        dev = os.makedev(8, 48)
        raw = b"1 0 8:48 / / rw - ext4 /dev/x rw\n2 1 8:48 / /a rw - xfs /dev/x rw\n"
        self.assertEqual(
            c._mount_observation(Path("/ab/c"), dev, raw)["fs_type"], "ext4"
        )
        self.assertEqual(c._mount_observation(Path("/a/c"), dev, raw)["fs_type"], "xfs")
        with self.assertRaisesRegex(c.BoundaryStoreError, "storage_observation"):
            c._mount_observation(Path("/a/c"), os.makedev(9, 0), raw)
        with self.assertRaisesRegex(c.BoundaryStoreError, "storage_observation"):
            c._mount_observation(Path("/"), dev, b"x" * ((2 << 20) + 1))

    def test_15_passive_resave_none_and_held_genesis_refusal(self):
        for routing in (False, True):
            with self.subTest(routing=routing):
                runner = copy.copy(self.runner)
                original = oi.capture_original_inputs(runner, source_identity=SOURCES)
                path = self.new()
                handle = c.create_boundary_store(
                    path,
                    runner=runner,
                    originals=original,
                    runtime_reference=self.reference,
                    storage_contract=c.StorageContract(True, True),
                )
                self.addCleanup(handle.close)
                prepared, chain = handle._load(handle._state.head)
                owner = c._owner(prepared)
                if routing:
                    owner._routing = held(owner)
                parent = chain[0][2]
                destination = path / "passive-control"
                # Outside retained store namespace: this control uses unchanged B authority.
                destination = self.root / f"passive-{self.counter:03d}"
                kwargs = {
                    "artifact_root": path,
                    "originals": parent["originals"],
                    "genesis": parent["genesis"],
                    "physical": parent["physical"],
                    "source_identity": SOURCES,
                    "environment_sha256": handle._header["environment_sha256"],
                }
                if routing:
                    # Root-approved source correction: held routing exists iff
                    # accepted rain; successful held re-save is PENDING real Q.
                    with self.assertRaisesRegex(
                        b.BoundaryCheckpointError, "routing_history"
                    ):
                        b.save_passive_boundary(destination, owner, **kwargs)
                    self.assertFalse(destination.exists())
                    continue
                seal = b.save_passive_boundary(destination, owner, **kwargs)
                reloaded = b.prepare_passive_boundary(
                    destination,
                    artifact_root=path,
                    expected_manifest_sha256=seal,
                    expected_source_identity=SOURCES,
                    expected_environment_sha256=handle._header["environment_sha256"],
                )
                c._compare_owners(owner, c._owner(reloaded), handle._limits.image)
                self.assertEqual(
                    reloaded.diagnostics()["n_wrapper_arrays"], 7 if routing else 5
                )
                self.assertEqual(owner._totals["n_entry_reroutes"], 0)
                self.assertEqual(owner._totals["n_event_reroutes"], 0)

    def test_16_partial_intent_interrupt_preallocated_halt(self):
        path = self.new()
        handle = self.create(path)
        self.addCleanup(handle.close)
        token, old = handle.head_token(), (path / "HEAD").read_bytes()
        write = c._write

        def partial(destination, raw):
            if Path(destination).name == "intent.json":
                write(destination, raw[:17])
                raise KeyboardInterrupt("partial intent before work")
            return write(destination, raw)

        with (
            mock.patch.object(c, "_write", side_effect=partial),
            mock.patch.object(
                dr.DenseRunner,
                "step",
                side_effect=AssertionError("native work after partial intent"),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            handle.step(token)
        self.assertEqual((path / "HEAD").read_bytes(), old)
        self.assertEqual(handle._state.phase, "halted_before_head")
        self.assertRegex(handle._state.attempt_id, r"^[0-9a-f]{32}$")
        self.assertIsNone(handle._state.intended)
        self.assertTrue(
            any(
                p.stat().st_size == 17
                for p in (path / "generations").glob("g-0001-*/intent.json")
            )
        )

    def test_17_four_retained_orphans_exhaust_before_event(self):
        path = self.new()
        handle = self.create(path)
        self.addCleanup(handle.close)
        for index in range(4):
            (path / "generations" / f"g-0001-{index + 1:032x}").mkdir()
        names = sorted(p.name for p in (path / "generations").iterdir())
        with (
            mock.patch.object(
                dr.DenseRunner,
                "step",
                side_effect=AssertionError("event before orphan cap"),
            ),
            self.assertRaisesRegex(c.BoundaryStoreError, "orphan_cap"),
        ):
            handle.step(handle.head_token())
        self.assertEqual(
            names, sorted(p.name for p in (path / "generations").iterdir())
        )
        self.assertEqual(handle.report()["phase"], "idle")

    def test_18_disk_and_inode_refusal_before_reservation(self):
        handle = self.recover()
        self.addCleanup(handle.close)
        real = os.fstatvfs
        fields = (
            "f_bsize",
            "f_frsize",
            "f_blocks",
            "f_bfree",
            "f_bavail",
            "f_files",
            "f_ffree",
            "f_favail",
            "f_flag",
            "f_namemax",
            "f_fsid",
        )
        for missing in ("f_bavail", "f_favail"):
            with self.subTest(missing=missing):

                def low(fd, missing=missing):
                    value = real(fd)
                    return SimpleNamespace(
                        **{
                            name: 0 if name == missing else getattr(value, name)
                            for name in fields
                        }
                    )

                with (
                    mock.patch.object(c.os, "fstatvfs", side_effect=low),
                    mock.patch.object(
                        handle.__class__,
                        "_reserve",
                        side_effect=AssertionError("reservation before quota"),
                    ),
                    self.assertRaisesRegex(c.BoundaryStoreError, "storage_cap"),
                ):
                    handle.step(handle.head_token())
                self.assertEqual(handle.report()["phase"], "idle")

    def test_19_foreign_metadata_hardlink_refuses(self):
        outside = self.root / "foreign-link"
        try:
            os.link(self.main / "HEAD", outside)
            with self.assertRaisesRegex(c.BoundaryStoreError, "manifest_leftover"):
                self.recover()
        finally:
            outside.unlink()

    def test_20_bool_sequence_resealed_refuses_before_native(self):
        path = self.main / "HEAD"
        raw = path.read_bytes()
        try:
            head = json.loads(raw)
            head.pop("head_sha256")
            head["sequence"] = False
            path.write_bytes(oi._json(c._seal(head, "head_sha256")))
            with (
                mock.patch.object(
                    b,
                    "prepare_passive_boundary",
                    side_effect=AssertionError("native before plain wire kinds"),
                ),
                self.assertRaisesRegex(c.BoundaryStoreError, "metadata_integer"),
            ):
                self.recover()
        finally:
            path.write_bytes(raw)

    def test_21_same_size_wrapper_corruption_refuses_before_expansion(self):
        image = next((self.main / "generations").glob("*/image"))
        path = image / "a_0000.npy"
        raw = path.read_bytes()
        try:
            path.chmod(0o600)
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with (
                mock.patch.object(
                    b,
                    "prepare_passive_boundary",
                    side_effect=AssertionError("expand before ancestor bytes"),
                ),
                self.assertRaisesRegex(c.BoundaryStoreError, "image_reference"),
            ):
                self.recover()
        finally:
            path.write_bytes(raw)
            path.chmod(0o444)

    def test_22_exact_store_generation_manifest_leftovers(self):
        generation = next((self.main / "generations").iterdir())
        pairs = [
            (self.main / "STORE.json", self.main / "STORE.tmp"),
            (generation / "generation.json", generation / "generation.tmp"),
        ]
        try:
            for source, temporary in pairs:
                os.link(source, temporary)
            handle = self.recover()
            handle.close()
            pairs[1][1].unlink()
            pairs[1][1].write_bytes(pairs[1][0].read_bytes())
            with self.assertRaisesRegex(c.BoundaryStoreError, "manifest_leftover"):
                self.recover()
        finally:
            for _source, temporary in pairs:
                if temporary.exists():
                    temporary.unlink()

    def test_23_no_head_after_actual_native_seal_interrupt(self):
        path = self.new()
        actual = cp.save_exact_physical_checkpoint

        def save(*args, **kwargs):
            actual(*args, **kwargs)
            raise KeyboardInterrupt("native target already visible")

        with (
            mock.patch.object(cp, "save_exact_physical_checkpoint", side_effect=save),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.create(path)
        self.assertFalse((path / "HEAD").exists())
        self.assertTrue(list((path / "generations").glob("*/physical/checkpoint.json")))
        with self.assertRaises(FileNotFoundError):
            self.recover(path, "0" * 64)

    def test_24_outer_seal_readback_failure_no_head(self):
        path = self.new()
        actual = c._sealed_write

        def seal(directory, name, *args):
            result = actual(directory, name, *args)
            if name == "generation.json":
                raise OSError("after outer seal before readback")
            return result

        with (
            mock.patch.object(c, "_sealed_write", side_effect=seal),
            self.assertRaises(OSError),
        ):
            self.create(path)
        self.assertFalse((path / "HEAD").exists())
        self.assertTrue(list((path / "generations").glob("*/generation.json")))

    def test_25_genesis_replace_error_is_indeterminate(self):
        path = self.new()
        observed = []
        actual_open = c.DenseBoundaryStore._open

        def opened(*args, **kwargs):
            value = actual_open(*args, **kwargs)
            observed.append(value)
            return value

        actual_replace = os.replace

        def replace(source, destination, **kwargs):
            if destination == "HEAD":
                raise OSError("unknown replace effect")
            return actual_replace(source, destination, **kwargs)

        with (
            mock.patch.object(c.DenseBoundaryStore, "_open", side_effect=opened),
            mock.patch.object(c.os, "replace", side_effect=replace),
            self.assertRaises(OSError),
        ):
            self.create(path)
        self.assertEqual(observed[0]._state.phase, "publication_indeterminate")
        self.assertTrue(observed[0]._closed)
        self.assertFalse((path / "HEAD").exists())
        self.assertTrue(list(path.glob("HEAD.*.tmp")))

    def test_26_genesis_postreplace_rootfsync_error_recover(self):
        path = self.new()
        replaced = False
        observed = []
        actual_replace, actual_sync, actual_open = (
            os.replace,
            os.fsync,
            c.DenseBoundaryStore._open,
        )

        def opened(*args, **kwargs):
            value = actual_open(*args, **kwargs)
            observed.append(value)
            return value

        def replace(source, destination, **kwargs):
            nonlocal replaced
            result = actual_replace(source, destination, **kwargs)
            if destination == "HEAD":
                replaced = True
            return result

        def sync(fd):
            if replaced and observed and fd == observed[0]._rootfd:
                raise OSError("rootfsync uncertain")
            return actual_sync(fd)

        with (
            mock.patch.object(c.DenseBoundaryStore, "_open", side_effect=opened),
            mock.patch.object(c.os, "replace", side_effect=replace),
            mock.patch.object(c.os, "fsync", side_effect=sync),
            self.assertRaises(OSError),
        ):
            self.create(path)
        self.assertEqual(observed[0]._state.phase, "publication_indeterminate")
        self.assertTrue((path / "HEAD").exists())
        recovered = self.recover(path, sha((path / "STORE.json").read_bytes()))
        self.assertEqual(recovered.report()["accepted"], self.report0)
        recovered.close()

    def test_27_genesis_sigkill_after_replace_recover(self):
        path = self.new()
        status, _text = self.child("kill_replace", path)
        self.assertEqual(status, -signal.SIGKILL)
        recovered = self.recover(path, sha((path / "STORE.json").read_bytes()))
        self.assertEqual(recovered.report()["accepted"], self.report0)
        recovered.close()

    def test_28_interrupt_after_reserve_return_before_binding(self):
        path = self.new()
        handle = self.create(path)
        self.addCleanup(handle.close)
        old = (path / "HEAD").read_bytes()
        reserve = c.DenseBoundaryStore._reserve

        def interrupt(instance, *args):
            reserve(instance, *args)
            raise KeyboardInterrupt("after reserve return before caller binding")

        with (
            mock.patch.object(c.DenseBoundaryStore, "_reserve", new=interrupt),
            mock.patch.object(
                dr.DenseRunner,
                "step",
                side_effect=AssertionError("event after interrupt"),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            handle.step(handle.head_token())
        self.assertEqual(handle._state.phase, "halted_before_head")
        self.assertRegex(handle._state.attempt_id, r"^[0-9a-f]{32}$")
        self.assertEqual((path / "HEAD").read_bytes(), old)

    def test_29_partial_close_error_never_recloses_descriptor(self):
        handle = self.recover()
        actual = os.close
        calls = []
        lock = handle._lockfd

        def close(fd):
            calls.append(fd)
            actual(fd)
            if fd == lock:
                raise OSError("observed post-close error")

        with mock.patch.object(c.os, "close", side_effect=close):
            with self.assertRaises(OSError):
                handle.close()
            handle.close()
        self.assertTrue(handle._closed)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(set(calls)), 2)

    def test_30_late_identity_drift_halts_even_if_path_restored(self):
        path = self.new()
        handle = self.create(path)
        self.addCleanup(handle.close)
        lock, moved = path / ".writer.lock", path / "held-original-lock"
        lock.rename(moved)
        lock.write_bytes(b"")
        try:
            with self.assertRaisesRegex(c.BoundaryStoreError, "lease_identity"):
                handle.head_token()
        finally:
            lock.unlink()
            moved.rename(lock)
        self.assertEqual(handle.report()["phase"], "halted_before_head")
        with self.assertRaisesRegex(c.BoundaryStoreError, "store_halted"):
            handle.step(self.token)

    def test_31_failure_metadata_error_preserves_primary_and_head(self):
        path = self.new()
        handle = self.create(path)
        self.addCleanup(handle.close)
        old = (path / "HEAD").read_bytes()
        actual_write = c._write

        def failing_write(destination, raw, **kwargs):
            if Path(destination).name == "failure.json":
                raise OSError("controlled evidence storage failure")
            return actual_write(destination, raw, **kwargs)

        with (
            mock.patch.object(c, "_write", side_effect=failing_write),
            mock.patch.object(
                dr.DenseRunner,
                "step",
                side_effect=RuntimeError("controlled pre-transport abort"),
            ),
            self.assertRaisesRegex(RuntimeError, "controlled pre-transport abort"),
        ):
            handle.step(handle.head_token())
        self.assertEqual((path / "HEAD").read_bytes(), old)
        self.assertEqual(handle.report()["phase"], "halted_before_head")
        self.assertIsNotNone(handle.report()["failure_evidence_error"])
        self.assertFalse(list((path / "generations").glob("g-0001-*/failure.json")))

    def test_32_accepted_generation_cap_refuses_before_reservation(self):
        path = self.new()
        limits = dataclasses.replace(c.BoundaryStoreLimits(), max_accepted=1)
        handle = self.create(path, limits=limits)
        self.addCleanup(handle.close)
        old = (path / "HEAD").read_bytes()
        names = sorted(entry.name for entry in (path / "generations").iterdir())
        with (
            mock.patch.object(
                c.DenseBoundaryStore,
                "_reserve",
                side_effect=AssertionError("reserved beyond accepted cap"),
            ),
            mock.patch.object(
                dr.DenseRunner,
                "step",
                side_effect=AssertionError("event beyond accepted cap"),
            ),
            self.assertRaisesRegex(c.BoundaryStoreError, "accepted_cap"),
        ):
            handle.step(handle.head_token())
        self.assertEqual(handle.report()["phase"], "idle")
        self.assertEqual((path / "HEAD").read_bytes(), old)
        self.assertEqual(
            names, sorted(entry.name for entry in (path / "generations").iterdir())
        )

    def test_33_genesis_source_readback_mismatch_never_publishes_head(self):
        path = self.new()
        with (
            mock.patch.object(
                c,
                "_compare_owners",
                side_effect=c.BoundaryStoreError(
                    "readback_identity", "controlled source/readback disagreement"
                ),
            ) as compare,
            self.assertRaisesRegex(c.BoundaryStoreError, "readback_identity"),
        ):
            self.create(path)
        self.assertEqual(compare.call_count, 1)
        self.assertIs(compare.call_args.args[0], self.runner)
        self.assertIsInstance(compare.call_args.args[1], dr.DenseRunner)
        self.assertFalse((path / "HEAD").exists())
        self.assertTrue(list((path / "generations").glob("*/generation.json")))


def _child_main(mode, path, expected):
    cpu_declarations()
    if mode == "lease":
        try:
            c.recover_boundary_store(path, expected_store_file_sha256=expected)
        except c.BoundaryStoreError as exc:
            print(exc.code, flush=True)
            return
        raise AssertionError("second process unexpectedly acquired lease")
    if mode != "kill_replace":
        raise AssertionError("unknown child control")
    real = db.build_dense_bed
    with mock.patch.object(
        f,
        "build_dense_bed",
        side_effect=lambda **kwargs: real(
            **{
                **kwargs,
                "initial_time_s": 7.25,
                "initial_water_depth_m": np.full((f.NY, f.NX), 0.001),
            }
        ),
    ):
        bed = f.build_bed()
    water = dataclasses.replace(
        f.water_inputs(bed), initial_soil_water_m=np.full((f.NY, f.NX), 0.02)
    )
    runner = f.make_runner(
        bed, f.schedule(f.wind(), f.rain(), f.wind(), f.rain()), water=water
    )
    original = oi.capture_original_inputs(runner, source_identity=SOURCES)
    reference = json.loads((STAGE / "runtime_reference.json").read_bytes())
    actual = os.replace

    def kill_after(source, destination, **kwargs):
        result = actual(source, destination, **kwargs)
        if destination == "HEAD":
            os.kill(os.getpid(), signal.SIGKILL)
        return result

    with mock.patch.object(c.os, "replace", side_effect=kill_after):
        c.create_boundary_store(
            path,
            runner=runner,
            originals=original,
            runtime_reference=reference,
            storage_contract=c.StorageContract(True, True),
        )
    raise AssertionError("kill point not reached")


if __name__ == "__main__":
    group = os.environ.get("MASY53_TEST_GROUP", "all")
    names = unittest.defaultTestLoader.getTestCaseNames(StoreChecks)
    if group == "foundation":
        names = [name for name in names if int(name.split("_")[1]) <= 15]
    elif group == "faults":
        names = [name for name in names if int(name.split("_")[1]) > 15]
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(StoreChecks(name) for name in names)
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
