"""Synthetic stat/clock-format fixtures, never Native capsules or store authority."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import _support  # noqa: F401 - selects this checkout's MASY source

from masy import dense_active_water_store as active
from masy.clock import encode_time


class ActiveChainMetadataBudget(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        outer = Path(self.temp.name)
        root = outer / "store"
        root.mkdir()
        (root / "generations").mkdir()
        self.root = root
        self.inventory = {"water_attempts": set()}

    def handle(self, combined=True):
        store_id = "0" * 32
        managed = self.root.parent / ((".masy-active-" if combined else ".masy-water-") + store_id)
        managed.mkdir(exist_ok=True)
        sibling, parent = managed.stat(), managed.parent.stat()
        fs_id = os.statvfs(managed).f_fsid
        header = {
            "schema": active.SCHEMA_PROCESS if combined else active.SCHEMA,
            "store_id": store_id,
            "storage_contract": {"observed_fs_id": fs_id},
            "managed_active" if combined else "managed_water": {
                "path": str(managed), "dev": sibling.st_dev, "ino": sibling.st_ino,
                "parent_path": str(managed.parent), "parent_dev": parent.st_dev,
                "parent_ino": parent.st_ino, "fs_id": fs_id,
            },
        }
        return SimpleNamespace(_root=self.root, _header=header, _limits=active.c._DEFAULT,
                               _root_identity=(self.root.stat().st_dev, self.root.stat().st_ino))

    def capsule(self, handle, size, export_size=0):
        name = "synthetic-stat-" + str(len(self.inventory["water_attempts"]))
        path = active._managed(handle) / name
        (path / "entry").mkdir(parents=True)
        # Sparse raw extents intentionally are NOT parsed or used as capsules.
        with (path / "entry/capsule.json").open("wb") as stream:
            stream.truncate(size)
        if export_size:
            with (path / "export.json").open("wb") as stream:
                stream.truncate(export_size)
        self.inventory["water_attempts"].add(name)
        return path / "entry/capsule.json"

    def charge(self, handle):
        return active._metadata_stat_charge(handle, self.inventory)

    def test_empty_has_no_prospective_constructor(self):
        self.assertEqual(self.charge(self.handle()), 0)

    def test_more_smaller_capsules_keep_current_occurrences_without_extra_future(self):
        handle = self.handle()
        self.capsule(handle, 1000)
        first = self.charge(handle)
        self.capsule(handle, 200)
        self.capsule(handle, 200)
        # Both identical CURRENT graphs stay charged; future selection is still
        # the original larger candidate, not another capture per old child.
        self.assertEqual(self.charge(handle) - first, 6 * active._parse_charge(200))

    def test_new_largest_capsule_expands_future_once(self):
        handle = self.handle()
        self.capsule(handle, 7)
        first = self.charge(handle)
        self.capsule(handle, 91)
        self.assertEqual(self.charge(handle) - first,
                         3 * active._parse_charge(91) + 6 * active._parse_charge(91 - 7))

    def test_histories_descriptors_and_repeated_image_receivers_remain_additive(self):
        handle = self.handle()
        generation = self.root / "generations" / ("g-0000-" + "1" * 32)
        (generation / "image").mkdir(parents=True)
        (generation / "intent.json").write_bytes(b"x" * 11)
        (generation / "generation.json").write_bytes(b"x" * 17)
        (generation / "image/manifest.json").write_bytes(b"x" * 19)
        (self.root / "generations" / ("g-0001-" + "2" * 32)).mkdir()
        self.capsule(handle, 23, export_size=29)
        # Two B receiver occurrences pay the largest image separately even
        # though only one such file is physically present in this stat fixture.
        current = sum(active._parse_charge(n) for n in (11, 17, 19, 19, 23, 29))
        self.assertEqual(self.charge(handle), 3 * current + 6 * active._parse_charge(23 + 2933))

    def test_legacy_water_reservation_is_unchanged(self):
        handle = self.handle(combined=False)
        sizes = (1, 17, 11, 1 << 20)
        for size in sizes:
            self.capsule(handle, size)
        expected = sum(6 * active._parse_charge(min(1 << 20, n + 3)) for n in sizes)
        self.assertEqual(self.charge(handle), expected)

    def test_wire_cap_exact_and_future_constructor_not_clamped(self):
        handle = self.handle()
        path = self.capsule(handle, 1 << 20)
        expected = (3 * active._parse_charge(1 << 20)
                    + 6 * active._parse_charge((1 << 20) + 2933))
        self.assertEqual(self.charge(handle), expected)
        with path.open("r+b") as stream:
            stream.truncate((1 << 20) + 1)
        with self.assertRaises(active.c.BoundaryStoreError) as caught:
            self.charge(handle)
        self.assertEqual(caught.exception.code, "active_memory")

    def test_nonregular_capsule_still_refuses_before_parse(self):
        handle = self.handle()
        path = self.capsule(handle, 3)
        target = path.with_name("synthetic-target")
        path.rename(target)
        path.symlink_to(target.name)
        with self.assertRaises(active.c.BoundaryStoreError) as caught:
            self.charge(handle)
        self.assertEqual(caught.exception.code, "active_memory")

    def test_growth_bound_covers_independent_binary64_fraction_formats(self):
        # General signed differences challenge the 2099-bit proof; native_time
        # itself remains nonnegative, as required by the actual wind clock.
        values = (0.0, math.ulp(0.0), -math.ulp(0.0), sys.float_info.min,
                  1.0, -1.0, sys.float_info.max, -sys.float_info.max)
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode("ascii")
        rational = lambda value: f"{value.numerator}/{value.denominator}"
        original = {"ordinal": 1, "pauses": 1, "progress": {
            "native_time": encode_time(0.0), "elapsed_local_s": "0/1",
            "global_elapsed_s": "0/1", "next_step_index": 0,
            "advanced_hex": [0.0.hex(), 1.0.hex(), 2.0.hex()], "n_steps": 4096}}
        for left, right in itertools.product(values, repeat=2):
            difference = Fraction(left) - Fraction(right)
            self.assertLessEqual(len(rational(difference)), 958)
            updated = {"ordinal": 4096, "pauses": 2, "progress": {
                **original["progress"], "native_time": encode_time(abs(left)),
                "elapsed_local_s": rational(difference),
                "global_elapsed_s": rational(-difference), "next_step_index": 4096}}
            with self.subTest(left=left.hex(), right=right.hex()):
                self.assertLessEqual(len(canonical(updated)) - len(canonical(original)),
                                     active._COMBINED_CAPSULE_GROWTH_BYTES)
        self.assertEqual(active._COMBINED_CAPSULE_GROWTH_BYTES, 2933)


class SyntheticCompactMetadataInterface(unittest.TestCase):
    """Plain synthetic metadata/interfaces ONLY, never Root/Native authority."""

    def row(self, last=b'{"synthetic":true}'):
        return active._CompactChainRow(active._COMPACT_ROW_KEY,
            b'[{"synthetic":true}]', b'[]', last)

    def test_projection_has_exact_four_slots_and_only_byte_payloads(self):
        row = self.row()
        self.assertEqual(row.__slots__, ('_key', 'full_receipts_raw',
            'parent_receipts_raw', 'last_receipt_raw'))
        with self.assertRaises(TypeError):
            vars(row)
        for name in row.__slots__[1:]:
            self.assertIs(type(getattr(row, name)), bytes)
        self.assertIs(active._checked_compact_row(row), row)

    def test_projection_rejects_a_parsed_graph_in_any_payload(self):
        graph = {'receipts': [{'body': {'synthetic': True}}]}
        for payloads in ((graph, b'[]', None), (b'[]', graph, None),
                         (b'[]', b'[]', graph)):
            with self.subTest(payload=payloads):
                with self.assertRaises(active.c.BoundaryStoreError):
                    active._CompactChainRow(active._COMPACT_ROW_KEY, *payloads)

    def test_projection_is_immutable_and_cannot_pickle(self):
        row = self.row()
        with self.assertRaises(TypeError):
            row.last_receipt_raw = b'changed'
        with self.assertRaises(TypeError):
            del row.full_receipts_raw
        with self.assertRaises(TypeError):
            row.__reduce_ex__(5)

    def test_foreign_projection_key_refuses(self):
        with self.assertRaises(active.c.BoundaryStoreError):
            active._CompactChainRow(object(), b'[]', b'[]', None)

    def test_paused_ack_never_borrows_the_anchors_last_receipt(self):
        header = {'schema': active.SCHEMA_PROCESS, 'store_id': '0' * 32}
        head = {'store_id': header['store_id'], 'kind': 'wind_paused', 'event_operation_id': '1' * 64,
                'request_id': '2' * 64, 'revision': 1, 'accepted_sequence': 1}
        generation = {'ack_core': {'synthetic': 'metadata-only'}}
        ack = active._acknowledgement(header, head, generation, self.row())
        self.assertIsNone(ack.receipt_bytes)

    def test_accepted_ack_keeps_identical_existing_completion_bytes(self):
        header = {'schema': active.SCHEMA_PROCESS, 'store_id': '0' * 32}
        head = {'store_id': header['store_id'], 'kind': 'event_accepted', 'event_operation_id': '1' * 64,
                'request_id': '2' * 64, 'revision': 1, 'accepted_sequence': 1}
        generation = {'ack_core': {'synthetic': 'metadata-only'}}
        last = b'{"synthetic":"unchanged"}'
        ack = active._acknowledgement(header, head, generation, self.row(last))
        self.assertIs(ack.receipt_bytes, last)

    def test_unknown_root_reference_cannot_enter_or_publish_a_context(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(active._MIXED_OPERATION_SLOT)
            with self.assertRaises(active.c.BoundaryStoreError):
                active.mixed_metadata_operation(root=root, runtime_reference={},
                                                proof_reference=None)
            self.assertIsNone(active._MIXED_OPERATION_SLOT)

    def test_raw_parser_law_and_original_three_six_families_stay_positive(self):
        self.assertEqual(active._parse_charge(0), 0)
        self.assertEqual(active._parse_charge(19), 19 * active._parse_charge(1))
        census = dict(active._STATIC_CENSUS)
        self.assertEqual(census['current_receiver_families'], 3)
        self.assertEqual(census['future_constructor_families'], 6)
        self.assertEqual(census['artifact_deepcopy_calls_per_row'], 79)

    def test_all_parser_cycle_occurrences_are_additive(self):
        # Unit1 is synthetic algebra input, NOT actual ABI/admission evidence.
        units = {name: 1 for name in ('pointer', 'integer', 'function', 'tuple',
                 'cell', 'limit_holder', 'set', 'string')}
        strict = active._mixed_cycle_charge(units, 1, 0)
        plain = active._mixed_cycle_charge(units, 0, 1)
        self.assertGreater(strict, 0)
        self.assertGreater(plain, 0)
        self.assertEqual(active._mixed_cycle_charge(units, 7, 11),
                         7 * strict + 11 * plain)

    def test_static_wire_theorems_are_not_a_single_default_allowance(self):
        expected = (87166, 358472, 629780, 901088, 1172396, 1443704,
                    1715012, 1986320, 2257628)
        self.assertEqual(tuple(active.c.b._static_metadata_wire_upper(i)
                               for i in range(9)), expected)
        for outside in (-1, 9, True, 1.5, None):
            with self.subTest(value=outside):
                with self.assertRaises(active.c.b.BoundaryCheckpointError):
                    active.c.b._static_metadata_wire_upper(outside)

    def test_wire_source_literal_guard_handles_escape_and_control_boundaries(self):
        for value in ('', 'source', 'a"b\\c', ' '):
            self.assertEqual(active.c.b._wire_string(value), len(json.dumps(value).encode()))
        for value in ('\x00', '\n', '\x7f', '\u00e9', 'x' * 129):
            with self.subTest(value=value):
                with self.assertRaises(active.c.b.BoundaryCheckpointError):
                    active.c.b._wire_string(value)



class SyntheticPublicationReservation(unittest.TestCase):
    """Metadata-only controls; these are NOT Native/Root proof authorities."""

    def scalar_original(self, receipt=128, kinds=('wind', 'rain', 'wind', 'rain')):
        # Deliberately synthetic, NON-loadable Original control projection.
        # The planner is permitted to read scalar controls only; actual Original
        # class/payload construction remains an independent unchanged guard.
        nodes = []
        def node(tag, value=None, **extra):
            row = {'t': tag, **extra}
            if tag != 'none':
                row['v'] = value
            nodes.append(row)
            return len(nodes) - 1
        def mapping(fields):
            return node('dict', [[key, value] for key, value in fields.items()])
        source = {'kind': 'source_manifest_prototype', 'masy': '1' * 64,
                  'maple': '2' * 64, 'syrup': '3' * 64}
        source_index = mapping({name: node('str', value) for name, value in source.items()})
        events = node('list', [mapping({'kind': node('str', kind)}) for kind in kinds])
        schedule = mapping({'schema': node('str', 'masy.dense_schedule.v1'), 'events': events})
        values = {'max_events': 8, 'max_wind_accounting_steps': 16,
                  'max_storm_steps': 16, 'max_report_rows': 16,
                  'max_work_bytes': 1, 'max_receipt_bytes': receipt}
        limits = node('class', [[name, node('int', value)] for name, value in values.items()],
                      c='masy.dense_runner:DenseRunnerLimits')
        fields = {name: node('none') for name in active.c.oi._ROOT_FIELDS
                  if name != 'content_sha256'}
        fields.update(source_identity=source_index, schedule=schedule, runner_limits=limits)
        body = mapping(fields)
        manifest = {'schema': active.c.oi.SCHEMA, 'profile': active.c.oi.PROFILE,
                    'content_sha256': '4' * 64,
                    'wire': {'anchors': [], 'body': body, 'nodes': nodes}, 'files': []}
        canonical = active.c.oi._json
        manifest['manifest_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
        raw = canonical(manifest)
        reference = {'format': active.c.oi.SCHEMA, 'relative_path': 'originals/o-synthetic',
                     'manifest_file_sha256': hashlib.sha256(raw).hexdigest(),
                     'identity': manifest['content_sha256']}
        return raw, reference, source, list(kinds)

    def test_static_plan_reads_scalar_controls_without_snapshot_or_native_construction(self):
        from unittest.mock import patch
        raw, reference, source, kinds = self.scalar_original()
        with patch.object(active.c.b, '_snapshot', side_effect=AssertionError('unexpected physical snapshot')):
            plan = active.c.b._static_original_scalar_plan(raw, reference, source, kinds,
                                                         active.c.b.BoundaryImageLimits(), 4)
        self.assertEqual((plan.events, plan.accepted, plan.max_receipt_bytes), (4, 4, 128))
        self.assertEqual(plan.wire_upper, 1172396)
        self.assertEqual(plan.original_manifest_bytes, len(raw))
        self.assertFalse(hasattr(plan, '__dict__'))
        with self.assertRaises(AttributeError):
            plan.accepted = 0
        with self.assertRaises(TypeError):
            plan.__reduce_ex__(5)

    def test_producer_aggregate_receipts_refuse_before_future_publication(self):
        raw, reference, source, kinds = self.scalar_original(receipt=1024)
        limits = active.c.b.BoundaryImageLimits(max_receipts_bytes=3 * 1024)
        with self.assertRaises(active.c.b.BoundaryCheckpointError):
            active.c.b._static_original_scalar_plan(raw, reference, source, kinds, limits, 4)

    def test_original_source_and_event_kind_mismatch_refuse(self):
        raw, reference, source, kinds = self.scalar_original()
        with self.assertRaises(active.c.b.BoundaryCheckpointError):
            active.c.b._static_original_scalar_plan(raw, reference, source,
                ['rain', *kinds[1:]], active.c.b.BoundaryImageLimits(), 4)
        changed = dict(source, maple='5' * 64)
        with self.assertRaises(active.c.b.BoundaryCheckpointError):
            # This scalar Source admission refuses; the call does not
            # create or load a physical Original or fabricate any Root proof.
            active.c.b._static_original_scalar_plan(raw, reference, changed, kinds,
                                                   active.c.b.BoundaryImageLimits(), 4)

    def test_warm_pause_pays_current_capsule_and_next_uncapped_constructor(self):
        descriptor, capsule, peak = active._mixed_next_pause_sizes(0, False)
        self.assertEqual((descriptor, capsule), (1 << 20, 1 << 20))
        self.assertEqual(peak, active._parse_charge((1 << 20) + 2933))
        self.assertGreater(peak, active._parse_charge(capsule))

    def test_cold_pause_advances_six_max_without_dropping_old_orphans(self):
        old = active._parse_charge(1234 + 2933)
        descriptor, capsule, peak = active._mixed_next_pause_sizes(old, True)
        self.assertEqual(descriptor, 1 << 20)
        self.assertEqual(capsule, 1234 + 2933)
        self.assertEqual(peak, active._parse_charge(capsule + 2933))
        orphan_peak = active._parse_charge((1 << 20) + 2933)
        self.assertEqual(active._mixed_next_pause_sizes(orphan_peak, True)[2], orphan_peak)

    def test_non_linear_or_unknown_pause_extent_refuses(self):
        for value, cold in ((1, True), (-1, False), (True, False), (0, None)):
            with self.subTest(value=value, cold=cold):
                with self.assertRaises(active.c.BoundaryStoreError):
                    active._mixed_next_pause_sizes(value, cold)

    def test_unknown_context_cleanup_preserves_primary_and_removes_provisional_facts(self):
        owner = object.__new__(active._MixedMetadataOperation)
        for name in owner.__slots__:
            object.__setattr__(owner, name, object())
        primary = RuntimeError('synthetic primary')
        self.assertIs(owner.__exit__(RuntimeError, primary, None), False)
        self.assertEqual(owner._phase, 'closed')
        self.assertEqual(owner._fixed, 0)
        for name in ('_nonce', '_ordinal', '_record', '_raw', '_image', '_handle',
                     '_plan', '_ticket', '_counts', '_entry_kind', '_request_kind',
                     '_selected', '_files', '_observer', '_header_raw'):
            self.assertIsNone(getattr(owner, name))

    def test_reference_reader_returns_only_exact_readonly_synthetic_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic-source-evidence.json'
            raw = b'{"synthetic":true}'
            path.write_bytes(raw)
            path.chmod(0o444)
            ref = {'path': str(path.resolve()), 'size': len(raw),
                   'sha256': hashlib.sha256(raw).hexdigest()}
            self.assertEqual(active._mixed_read_ref(ref, 'synthetic metadata'), raw)
            path.chmod(0o644)
            with self.assertRaises(active.c.BoundaryStoreError):
                active._mixed_read_ref(ref, 'synthetic metadata')

    def test_reference_reader_refuses_changed_digest_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'one.json'
            path.write_bytes(b'{}')
            path.chmod(0o444)
            ref = {'path': str(path.resolve()), 'size': 2, 'sha256': '0' * 64}
            with self.assertRaises(active.c.BoundaryStoreError):
                active._mixed_read_ref(ref, 'synthetic changed digest')
            alias = Path(directory) / 'alias.json'
            alias.symlink_to(path)
            ref.update(path=str(alias), sha256=hashlib.sha256(b'{}').hexdigest())
            with self.assertRaises(active.c.BoundaryStoreError):
                active._mixed_read_ref(ref, 'synthetic alias')

    def test_planner_refuses_a_forged_or_raised_owning_limit_before_json(self):
        raw, reference, source, kinds = self.scalar_original()
        good = active.c.b._static_original_scalar_plan(raw, reference, source, kinds,
                                                      active.c.b.BoundaryImageLimits(), 4)
        self.assertEqual((good.events, good.accepted), (4, 4))
        for limits in (SimpleNamespace(max_receipt_bytes=256 << 10),
                       SimpleNamespace(max_receipt_bytes=257 << 10)):
            with self.subTest(limits=limits):
                with self.assertRaises(active.c.b.BoundaryCheckpointError) as refused:
                    active.c.b._static_original_scalar_plan(raw, reference, source, kinds, limits, 4)
                self.assertEqual(refused.exception.code, 'limit_layout')


if __name__ == "__main__":
    unittest.main()
