# MASY-01 final: SYRUP dense-bed classification (design only)

This is an owning-SYRUP change to `SYRUP/src/maple_syrup/sediment_bed.py`. It replaces `is_windowed_bed` (lines 610–613) and adds one small private helper so the class validation isn't duplicated. It authorizes no upstream edit, adoption, transport or commit.

```python
_MISSING = object()


def _maple_voxel_class(module, name: str, *, required: bool):
    value = module.__dict__.get(name, _MISSING)  # true absence only; no module __getattr__ synthesis
    if value is _MISSING:
        if required:
            raise AttributeError(f'{module.__name__} lacks required {name}')
        return None
    if not isinstance(value, type):
        raise BedIntegrationError(f'{module.__name__}.{name} is present but not a class: {value!r}')
    return value


def is_windowed_bed(state: BedState) -> bool:
    """True for exactly the WindowedVoxelColumnState object exposed by maple.core.types.voxel, False for exactly its
    VoxelColumnState; any other column type (subclass, duck type, foreign) raises. Only true absence of the optional
    windowed class is tolerated."""
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

**Authority** is the identity of the class object exposed by the verified module. Name strings are not checked, so legitimate re-exports work. The check is not a defence against monkeypatching.

**Unchanged:** existing callers, the `dependency.py` Dense `ApiRequirement` entries from the prior proposal, and all windowed, digest and transaction guards.

## Focused tests (monkeypatching is test-only, on the bound voxel module)

1. **Dense regression (real MAPLE).**
   - A `zeros_voxel_column_state` bed returns `False`.
   - Inventory, by-class, voxel-count and committed values match pre-change.
   - `pending_journal_records` is 0.
   - Paired transport raises `BedIntegrationError`.
2. **Windowed absent.** Dense still classifies as Dense, and windowed-only entries still refuse.
3. **Windowed real.** Exact classification is unchanged. Skip with a stated reason if the class is unavailable.
4. **Re-export.**
   - Bind `WindowedVoxelColumnState` (or `VoxelColumnState`) to a class defined in another module, so its `__module__` and `__qualname__` differ.
   - An exact instance of that class classifies correctly.
   - A subclass of it raises.
5. **Present-invalid.**
   - Windowed set to `None`, `3`, or a function raises `BedIntegrationError`, even for a genuine Dense bed.
   - A non-class `VoxelColumnState` also raises.
6. **Required failures.**
   - `sys.modules['maple.core.types.voxel'] = None` makes `ImportError` propagate.
   - Deleting `VoxelColumnState` raises `AttributeError`.
   - `check_required_api()` reports the missing symbol.
7. **Unknown and subclass columns.**
   - Cases: `object()`, `SimpleNamespace`, and subclasses of either MAPLE type.
   - Every audited caller raises `BedIntegrationError`.
   - A spy confirms no MAPLE apply, inventory or commit call is made.

Refusing subclasses and unknown representations is an intentional boundary on supported representations. It is pinned by the caller regressions above.

## Source and qualification limit

- New runs refuse dirty runtime inputs and require qualified committed sources. A development snapshot is allowed only if it is explicitly selected and labelled, per AGENTS.md. None is selected.
- MASY cannot synthesize a missing MAPLE class or run dirty sources.
- The cited excerpts are the captured committed pair (MAPLE `f1555e09…` / SYRUP `9fabdca6…`). They are diagnostic only, not a fallback runtime.
- Accepting this design does not qualify the Dense path physically or numerically.
- Implementation requires a separate owning-SYRUP scope and an independent review.
