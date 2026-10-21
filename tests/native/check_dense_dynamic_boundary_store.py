"""Authored tiny manual-vegetation C-v2 checks; transport is explicitly separate.

The gated worker supplies immutable source/runtime metadata. No saved case or
simulation payload is imported. A real native genesis is constructed in memory.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import check_dense_boundary_store as old
import check_dense_dynamic_physical_checkpoint as fixture
from masy import dense_boundary_store as c
from masy import dense_original_inputs as oi
from masy import dense_runner as dr
from masy import dense_runner_checkpoint as b
from masy import landscape_vegetation as lv


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def environment(request):
    """Same actual closed prototype source/environment contract as static C."""
    old.cpu_declarations()
    runner = fixture.dynamic_genesis()
    originals = oi.capture_original_inputs_v2(runner, source_identity=old.SOURCES)
    roots = {"masy": old.REQUEST["import_prefix"][0],
        "maple_syrup": old.REQUEST["import_prefix"][1], "maple": old.REQUEST["import_prefix"][2]}
    tables = {"masy": {key: value for key, value in old.REQUEST["candidate_files"].items()
        if key.startswith("src/") or key == "pyproject.toml"},
        "maple": old.REQUEST["native_files"]["MAPLE"],
        "maple_syrup": old.REQUEST["native_files"]["SYRUP"]}
    record = {"schema": c.SCHEMA + ".sources", "source_identity": old.SOURCES,
        "module_roots": roots, "files": tables, "environment": c._environment_identity()}
    path = old.STAGE / ("dynamic_retained_source_manifest_" + digest(request.node.name.encode())[:16] + ".json")
    # Each gated worker has a fresh stage. Never overwrite an earlier attempt.
    with path.open("xb") as stream:
        stream.write(oi._json(record))
    path.chmod(0o444)
    native = Path(old.REQUEST["native_bundle_manifest_path"])
    reference = {"schema": c.SCHEMA + ".runtime", "kind": "source_manifest_prototype",
        "source_manifest_path": str(path), "source_manifest_file_sha256": digest(path.read_bytes()),
        "native_bundle_manifest_path": str(native),
        "native_bundle_manifest_file_sha256": digest(native.read_bytes()), "module_roots": roots,
        "cpu_store": None, "cpu_lock_id": None, "cpu_lock_file_sha256": None}
    source, actual_environment = c._runtime_binding(reference, c.BoundaryStoreLimits())
    assert source == old.SOURCES and actual_environment == fixture.ENVIRONMENT
    return runner, originals, reference


def create(path, environment, **kwargs):
    runner, originals, reference = environment
    return c.create_dynamic_boundary_store(path, runner=runner, originals=originals,
        runtime_reference=reference, storage_contract=c.StorageContract(True, True), **kwargs)


def recover(path, store_sha, **kwargs):
    return c.recover_dynamic_boundary_store(path,
        expected_store_file_sha256=store_sha, **kwargs)


@pytest.fixture(scope="module")
def genesis(tmp_path_factory, environment):
    path = tmp_path_factory.mktemp("dynamic-store") / "store"
    handle = create(path, environment)
    try:
        result = path, handle.report()["store_file_sha256"], handle.head_token()
    finally:
        handle.close()
    return result


def manifest_chain(handle):
    head, _ = handle._head()
    return c._generation_chain(handle._root, handle._header,
        digest(handle._store_raw), head, handle._limits)


def test_dynamic_genesis_uses_distinct_schema_and_same_atomic_facade(tmp_path, environment):
    # The actual create admission records its reservation. Recover validates
    # that bound again without manufacturing a new recorded admission report.
    path = tmp_path / "positive-genesis"
    with contextlib.closing(create(path, environment)) as handle:
        report = handle.report()
        header = json.loads((path / "STORE.json").read_bytes())
        assert header["schema"] == c.SCHEMA_V2 and header["profile"] == c.PROFILE_V2
        assert set(header) == set(c._STORE)
        assert report["phase"] == "idle" and handle.head_token() == report["head_token"]
        assert report["accepted"]["accepted"]["next_index"] == 0
        assert report["reservation"]["max_total_passive_graph_builds"] == 16
        assert report["reservation"]["max_original_terrain_graph_builds"] == 10
        assert not hasattr(handle, "runner")


def test_fresh_initial_native_copy_matches_content_without_prescribed_alias(environment):
    runner = environment[0]
    env = runner._bed._dynamic
    initial = env._native_borrow(initial=True)
    prescribed = runner._bed._veg_initial
    assert initial is not prescribed
    for field in dataclasses.fields(initial.raster):
        assert not np.shares_memory(getattr(initial.raster, field.name),
            getattr(prescribed.raster, field.name))
    b._same(initial, prescribed, c._DEFAULT.image)
    assert env._native_borrow() is runner._bed._vegetation
    assert env.plan is runner._vegetation_plan
    c._compare_owners_v2(runner, runner, c._DEFAULT.image)


@pytest.mark.parametrize("corruption", ("initial_content", "initial_dtype", "current_alias", "plan_alias"))
def test_owner_comparison_preserves_initial_content_and_current_plan_alias_contract(environment, corruption):
    runner = environment[0]
    bed, env = runner._bed, runner._bed._dynamic
    held_initial, held_current, held_plan = bed._veg_initial, bed._vegetation, runner._vegetation_plan
    try:
        if corruption in ("initial_content", "initial_dtype"):
            height = np.array(held_initial.raster.height_m, copy=True)
            if corruption == "initial_content":
                height.flat[0] = np.nextafter(height.flat[0], np.inf)
            else:
                height = height.astype(np.float32)
            bed._veg_initial = dataclasses.replace(held_initial,
                raster=dataclasses.replace(held_initial.raster, height_m=height))
            error, code = b.BoundaryCheckpointError, "content_mismatch"
        elif corruption == "current_alias":
            bed._vegetation = dataclasses.replace(held_current)
            assert bed._vegetation is not env._native_borrow()
            b._same(bed._vegetation, env._native_borrow(), c._DEFAULT.image)
            error, code = c.BoundaryStoreError, "readback_identity"
        else:
            runner._vegetation_plan = lv.build_vegetation_plan(held_plan.deltas,
                shape=held_plan.shape, n_classes=held_plan.n_classes,
                n_voxels=held_plan.n_voxels,
                event_kinds=tuple(event.kind for event in runner._schedule.events),
                schedule_sha256=runner._schedule.digest, limits=held_plan.limits)
            assert runner._vegetation_plan is not env.plan
            assert runner._vegetation_plan.plan_record() == held_plan.plan_record()
            error, code = c.BoundaryStoreError, "readback_identity"
        with pytest.raises(error) as caught:
            c._compare_owners_v2(runner, runner, c._DEFAULT.image)
        assert caught.value.code == code
    finally:
        bed._veg_initial, bed._vegetation = held_initial, held_current
        runner._vegetation_plan = held_plan
    assert bed._veg_initial is held_initial and bed._vegetation is held_current
    assert runner._vegetation_plan is held_plan
    c._compare_owners_v2(runner, runner, c._DEFAULT.image)


def test_dynamic_recovery_never_initializes_transports_or_applies_deltas(genesis):
    path, sha, _ = genesis
    fail = AssertionError("passive recovery replayed a model/update")
    with (mock.patch.object(fixture.db, "build_dense_bed", side_effect=fail),
          mock.patch.object(dr.DenseRunner, "step", side_effect=fail),
          mock.patch.object(lv, "prepare_vegetation", side_effect=fail),
          contextlib.closing(recover(path, sha)) as handle):
        assert handle.report()["accepted"]["accepted"]["next_index"] == 0


def test_dynamic_recovery_preserves_owned_current_soil_and_full_future_plan(genesis, environment):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        prepared, _ = handle._load(handle._state.head)
        owner = c._owner_v2(prepared)
        runner = environment[0]
        assert owner._vegetation_plan.plan_record() == runner._vegetation_plan.plan_record()
        assert owner._bed._dynamic.as_record() == runner._bed._dynamic.as_record()
        assert owner._bed._dynamic.plan is owner._vegetation_plan
        assert owner._soil is not owner._water.initial_soil_water_m
        assert not owner._soil.flags.writeable
        assert not np.shares_memory(owner._soil, runner._soil)
        assert fixture.signature(owner._soil)[:3] == fixture.signature(runner._soil)[:3]
        for name in ("_veg_initial", "_vegetation"):
            actual = getattr(owner._bed, name)
            assert actual is owner._bed._dynamic._native_borrow(initial=name == "_veg_initial")
        assert owner._vegetation_plan.deltas[-1].after_event_index == 3


def test_v1_recovery_explicitly_refuses_v2_before_physical_load(genesis):
    path, sha, _ = genesis
    with (mock.patch.object(c.bridge, "load_boundary_physical", side_effect=AssertionError("expanded")),
          pytest.raises(c.BoundaryStoreError, match="metadata_schema")):
        c.recover_boundary_store(path, expected_store_file_sha256=sha)


def test_dynamic_create_refuses_static_runner_before_path_creation(tmp_path, environment):
    static = old.f.make_runner(old.f.build_bed(), old.f.schedule(old.f.wind()))
    before = fixture.signature(static._bed.state), fixture.signature(static._soil), static.report()
    with pytest.raises(b.BoundaryCheckpointError, match="runner_layout"):
        c.create_dynamic_boundary_store(tmp_path / "wrong", runner=static,
            originals=environment[1], runtime_reference=environment[2],
            storage_contract=c.StorageContract(True, True))
    assert not (tmp_path / "wrong").exists()
    assert before == (fixture.signature(static._bed.state), fixture.signature(static._soil), static.report())


def test_added_memory_admission_precedes_original_capture(tmp_path, environment):
    # Base v1 admission fits; the independently computed dynamic increment does
    # not. The native/capture entry must never be reached in this refusal.
    limits = c.BoundaryStoreLimits()
    base = c._reservation(limits, 0)["memory_bytes"]
    narrow = dataclasses.replace(limits, max_memory_bytes=base)
    assert c._reservation(narrow, 0)["memory_bytes"] == base
    with (mock.patch.object(c, "_reservation_v2", wraps=c._reservation_v2) as admission,
          mock.patch.object(oi, "capture_original_inputs_v2", side_effect=AssertionError("late admission")),
          mock.patch.object(b, "_runner_admission_v2", side_effect=AssertionError("late admission")),
          pytest.raises(c.BoundaryStoreError, match="computed dynamic owner")):
        create(tmp_path / "refused", environment, limits=narrow)
    assert admission.call_args_list == [mock.call(narrow, 0)]
    assert not (tmp_path / "refused").exists()


def test_resource_reservation_keeps_caps_and_full_probe_units():
    limits = c.BoundaryStoreLimits()
    result = c._reservation_v2(limits, 1 << 28)
    assert result["memory_bytes"] == c._reservation(limits, 1 << 28)["memory_bytes"] + sum(result["v2_increments"].values())
    assert result["disk_bytes"] == c._reservation(limits, 1 << 28)["disk_bytes"]
    assert result["v2_increments"]["additional_dynamic_array_bytes"] == 416 * limits.image.max_cells
    assert limits.max_accepted == 9
    assert result["max_native_header_parses"] == 8 * limits.max_accepted
    assert result["max_streamed_soil_checks"] == 4 * limits.max_accepted
    assert result["max_passive_runner_constructions"] == 4
    assert limits.image.max_cache_bytes == 1 << 20


def test_raw_manifest_binding_uses_admitted_canonical_bytes_and_preserves_count(genesis):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        image = manifest_chain(handle)[0][1]["image"]
        actual = c.runtime._hash_path

        def no_second_manifest_read(value, **kwargs):
            assert Path(value).name != "manifest.json"
            return actual(value, **kwargs)

        with mock.patch.object(c.runtime, "_hash_path", side_effect=no_second_manifest_read):
            manifest, total = c._image_metadata_v2(path, image, handle._limits)
        assert type(total) is int and total > 0
        assert digest(oi._json(manifest)) == image["manifest_file_sha256"]
        bad = {**image, "manifest_file_sha256": "f" * 64}
        actual_reader = b._read_wrapper_v2
        completed = []

        def marked_reader(*args, **kwargs):
            result = actual_reader(*args, **kwargs)
            completed.append(True)
            return result

        def no_hash_after_reader(value, **kwargs):
            assert not completed, "raw manifest mismatch must precede row hashes"
            return actual(value, **kwargs)

        with (mock.patch.object(b, "_read_wrapper_v2", side_effect=marked_reader),
              mock.patch.object(c.runtime, "_hash_path", side_effect=no_hash_after_reader),
              pytest.raises(c.BoundaryStoreError, match="image_reference")):
            c._image_metadata_v2(path, bad, handle._limits)
        assert completed == [True]


@pytest.mark.parametrize("schema,profile", (
    (c.SCHEMA, c.PROFILE_V2), (c.SCHEMA_V2, c.PROFILE),
    (True, c.PROFILE_V2), (c.SCHEMA_V2, "unknown")))
def test_mixed_store_profiles_never_select_a_codec(schema, profile):
    with pytest.raises(c.BoundaryStoreError, match="store_profile"):
        c._store_schema({"schema": schema, "profile": profile})


@pytest.mark.parametrize("key", tuple(sorted(lv._ENVELOPE_KEYS)))
def test_missing_ancestor_envelope_key_refuses_before_soil_read(genesis, key):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        chain = manifest_chain(handle)
        # Standalone prefix-validator probe against a real admitted genesis.
        # This duplicate shadow row is NOT claimed to be a valid C ancestry.
        shadow = copy.deepcopy(chain[0])
        del shadow[2]["shared_resources"]["vegetation"][key]
        with (mock.patch.object(oi, "_verify_written", side_effect=AssertionError("late key admission")),
              pytest.raises(c.BoundaryStoreError, match="metadata_layout")):
            # Missing key on the first row must refuse before touching any soil.
            c._history_v2(path, handle._header, [shadow], handle._limits)


def test_extra_ancestor_envelope_key_refuses(genesis):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        chain = copy.deepcopy(manifest_chain(handle))
        chain[0][2]["shared_resources"]["vegetation"]["extra"] = None
        with pytest.raises(c.BoundaryStoreError, match="metadata_layout"):
            c._history_v2(path, handle._header, chain, handle._limits)


def test_cross_header_metadata_seal_and_resource_binding_refuse(genesis):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        manifest = copy.deepcopy(manifest_chain(handle)[0][2])
        manifest["shared_resources"]["vegetation"]["metadata_sha256"] = "f" * 64
        with pytest.raises(b.BoundaryCheckpointError, match="resource_identity"):
            b._preflight_metadata_v2(manifest, old.SOURCES, fixture.ENVIRONMENT, handle._limits.image)
        manifest = copy.deepcopy(manifest_chain(handle)[0][2])
        manifest["shared_resources"]["accepted_run_copy_bytes"] += 8
        with pytest.raises(b.BoundaryCheckpointError, match="resource_identity"):
            b._preflight_metadata_v2(manifest, old.SOURCES, fixture.ENVIRONMENT, handle._limits.image)


@pytest.mark.parametrize("key,value,code", (
    ("accepted_run_copy_bytes", True, "scalar_kind"),
    ("water_exchange", [{"water_m3": 1}], "resource_profile")))
def test_bool_copy_counter_and_nonempty_water_exchange_refuse(genesis, key, value, code):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        manifest = manifest_chain(handle)[0][2]
        changed = copy.deepcopy(manifest)
        changed["shared_resources"][key] = value
        with pytest.raises(b.BoundaryCheckpointError, match=code):
            b._preflight_metadata_v2(changed, old.SOURCES, fixture.ENVIRONMENT, handle._limits.image)


def test_native_current_metadata_mismatch_is_refused_by_real_preflight(genesis):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        chain = manifest_chain(handle)
        manifest = copy.deepcopy(chain[0][2])
        # At genesis both roles reference the same immutable native child.
        assert manifest["genesis"]["relative_path"] == manifest["physical"]["relative_path"]
        assert manifest["shared_resources"]["vegetation_metadata_sha256"] != "e" * 64
        manifest["physical"]["identity"]["vegetation_metadata_sha256"] = "e" * 64
        with (mock.patch.object(c.bridge, "load_boundary_physical_v2", side_effect=AssertionError("archive expansion")),
              pytest.raises(b.BoundaryCheckpointError, match="artifact_reference")):
            b._preflight_metadata_v2(manifest, old.SOURCES, fixture.ENVIRONMENT, handle._limits.image)
        # Preserve that earlier same-path invariant while targeting the real
        # native-current metadata/resource mismatch, still before expansion.
        manifest["genesis"]["identity"]["vegetation_metadata_sha256"] = "e" * 64
        assert oi._json(manifest["genesis"]) == oi._json(manifest["physical"])
        with (mock.patch.object(c.bridge, "load_boundary_physical_v2", side_effect=AssertionError("archive expansion")),
              pytest.raises(b.BoundaryCheckpointError, match="resource_identity")):
            b._preflight_metadata_v2(manifest, old.SOURCES, fixture.ENVIRONMENT, handle._limits.image)


def test_dynamic_lease_is_permanent_and_exclusive(genesis):
    path, sha, _ = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        inode = (path / ".writer.lock").stat().st_ino
        with pytest.raises(c.BoundaryStoreError, match="lease_busy"):
            recover(path, sha)
        assert handle.head_token() == genesis[2]
    assert (path / ".writer.lock").stat().st_ino == inode


def test_stale_parent_refuses_without_step(genesis):
    path, sha, token = genesis
    with contextlib.closing(recover(path, sha)) as handle:
        before = (path / "HEAD").read_bytes()
        with (mock.patch.object(dr.DenseRunner, "step", side_effect=AssertionError("unexpected event")),
              pytest.raises(c.BoundaryStoreError, match="stale_parent")):
            handle.step("a" * 64)
        assert (path / "HEAD").read_bytes() == before and handle.head_token() == token
