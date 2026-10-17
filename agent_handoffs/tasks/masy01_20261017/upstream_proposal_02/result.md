# MASY-01 revised: SYRUP dense-bed classification (design only)

## Recommendation

Make a single owning-SYRUP change to `is_windowed_bed` in `SYRUP/src/maple_syrup/sediment_bed.py`. This is the Dense-first priority. Adding real Windowed support to MAPLE would also supply the missing class, but it is a broader storage and interface qualification task, so it is deferred rather than rejected. No upstream edit, adoption, transport or commit is authorized here.

## Proposed function (replaces lines 610–613)

```python
_MISSING = object()


def _maple_voxel_class(module, name: str, *, required: bool):
    value = module.__dict__.get(name, _MISSING)  # true absence only; no module __getattr__ synthesis
    if value is _MISSING:
        if required:
            raise AttributeError(f'{module.__name__} lacks required {name}')
        return None
    if not (isinstance(value, type) and value.__module__ == module.__name__ and value.__qualname__ == name):
        raise BedIntegrationError(f'{module.__name__}.{name} is present but is not the MAPLE class: {value!r}')
    return value


def is_windowed_bed(state: BedState) -> bool:
    """True for an exact MAPLE WindowedVoxelColumnState, False for an exact MAPLE VoxelColumnState; any other
    column type (subclass, duck type, foreign) raises. Only true absence of the optional windowed class is tolerated."""
    from maple.core.types import voxel as maple_voxel  # ImportError propagates

    windowed_type = _maple_voxel_class(maple_voxel, 'WindowedVoxelColumnState', required=False)
    dense_type = _maple_voxel_class(maple_voxel, 'VoxelColumnState', required=True)
    column_type = type(state.voxel_column)
    if windowed_type is not None and column_type is windowed_type:
        return True
    if column_type is dense_type:
        return False
    raise BedIntegrationError(f'unsupported bed representation {column_type.__module__}.{column_type.__qualname__}')
```

### Point-by-point resolution

1. **Absence versus invalid presence.**
   - Absence is detected through the module `__dict__` with a sentinel.
   - A present `None`, a present non-class, or a foreign alias raises `BedIntegrationError` before any classification. This applies to both symbols.
   - Nothing catches `ImportError` or `AttributeError`, so a broken present module propagates.
   - Both symbols are resolved before the column is inspected. A Dense bed therefore cannot be accepted beside a corrupt windowed symbol.

2. **Compatibility.**
   - Windowed is checked first.
   - There is one function and no second helper. Foreign columns raise inside `is_windowed_bed` itself, so no caller can fall through to its dense branch.
   - Genuine exact Dense and Windowed behaviour is intended to stay unchanged, and regression tests below verify it.
   - Behaviour is deliberately tightened in two places. Subclasses and foreign representations, previously treated as non-windowed (that is, Dense), now raise. `pending_journal_records` raises for them instead of returning 0.
   - Callers audited: `apply_bed_demand`, `apply_bed_pickup_transport_deposit`, `bed_inventory`, `bed_by_cell_class`, `bed_voxel_count`, `bed_already_committed` and `pending_journal_records`. All of them call `is_windowed_bed` before any MAPLE physics, accounting or commit call.

3. **Minimal scope.** Add only the required Dense entries to `dependency.py`, and skip any that are already declared:
   ```python
   ApiRequirement("maple.core.types.voxel", "VoxelColumnState"),
   ApiRequirement("maple.aeolian.flux.step", "bed_inventory_by_class_kg"),
   ApiRequirement("maple.surface.topographic_commit.commit", "is_already_committed"),
   ```
   There is no optional-API registry and no whole-program static test. These are untouched: `paired_transaction_entry`, `check_deep_storage_source`, `journal_recognizer_source`, `require_typed_journal_capacity_error`, both recognizer digests, and the refusal of non-windowed beds in the paired transaction.

## Focused SYRUP tests

Monkeypatching is test-only, on the bound voxel module.

1. **Dense regression (real MAPLE).**
   - A `zeros_voxel_column_state` bed returns `False`.
   - `bed_inventory`, `bed_by_cell_class`, `bed_voxel_count` and `bed_already_committed` match their pre-change values.
   - `pending_journal_records` is 0.
   - `apply_bed_pickup_transport_deposit` raises `BedIntegrationError`.
2. **Windowed symbol absent.** The Dense bed classifies as Dense. All windowed-only entries still refuse.
3. **Windowed symbol present and valid.** This runs only where the real class exists and is otherwise skipped with a stated reason. Classification, paired-entry use and digest refusals are unchanged.
4. **Windowed symbol present but invalid.**
   - The value is set to `None`, `3`, a function, or a class with a different `__module__`.
   - `BedIntegrationError` is raised even for a genuine Dense bed.
5. **Required failures.**
   - `sys.modules['maple.core.types.voxel'] = None` makes `ImportError` propagate.
   - Deleting `VoxelColumnState` raises `AttributeError`.
   - A non-class `VoxelColumnState` raises `BedIntegrationError`.
   - `check_required_api()` reports the missing `VoxelColumnState`.
6. **Foreign columns.**
   - Cases: `object()`, `SimpleNamespace(mass_kg=...)`, and subclasses of each MAPLE type.
   - Every audited caller raises `BedIntegrationError`.
   - A spy confirms that no MAPLE apply, inventory or commit function is called.

## Source and qualification policy

- Default new runs still refuse dirty runtime inputs and require qualified committed sources.
- Under AGENTS.md, an explicitly selected, labelled development snapshot is an allowed alternative.
- No current dirty source was selected, and this proposal authorizes no actual transport.
- The excerpts are the originally captured committed pair (MAPLE `f1555e09…` / SYRUP `9fabdca6…`), used diagnostically and not as a fallback runtime.
- Implementation needs its own owning-SYRUP scope and an independent review.
- This proposal does not address metadata origin, wind settlement, routing reentry, restart, or physical or numerical qualification of the Dense path.
