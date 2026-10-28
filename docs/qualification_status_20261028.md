# Execution qualification status — October 28, 2026

MAPLE-SYRUP runs actual alternating MAPLE wind and SYRUP water on the same evolving bed. The following results have separate, bounded qualification subjects. They do not establish a general landscape size or production calibration.

| Path | Accepted result | User interface and limit |
| --- | --- | --- |
| Dense CPU | The MAPLE-case workflow completed a labelled synthetic 9 × 8, two-class case, including a fresh explicit water-policy run. | `tools/masy_case.py run`; supply an authored landscape and scientific inputs. Exact revisions and limits are in [CPU case qualification](case_cpu_qualification.md) and the additional CPU authored-policy qualification below. |
| Windowed CPU | Native reduced wind and positive water alternated on a shared 4 × 4, one-class, six-voxel bed with three resident voxels. | Bounded Python interface; see [CPU windowed integration](windowed_cpu_integration.md). |
| Windowed GPU wind and water | Wind → water → wind → water passed on an 8 × 8, one-class bed with 20 logical voxels and three resident voxels. Water-only, entry and rejected-input recovery checks also passed. | Closed C64 Python/qualification interfaces; [details](windowed_device_c64_integration.md). No arbitrary-landscape GPU CLI bridge is provided. |
| GPU saved boundary restart | Three fresh processes saved after wind, recovered through water, then recovered and finished the remaining wind and water events. | Closed Small 4 × 4, one-class, six-voxel/three-resident-voxel profile. [Restart interface](windowed_device_boundary_restart.md). |

The CPU case workflow accepts user-authored MAPLE landscapes within its documented admission rules. A successful synthetic case is not validation of another landscape's boundaries, forcing, class mixture, coefficients or resource fit. The windowed case-adoption command establishes storage adoption only; it does not grant GPU or event-execution eligibility.

## Latest GPU evidence and source identities

The C64 result executed MAPLE `40d1a598b09929b2d0f831513fb6d5c015d88772`, SYRUP `919c82c702f3a1b485a35bd3d5c39960c3b577b8` and MASY `5cab8272d8df18b58d492dfeba9c1074b882544f`. Four physical checks passed: complete water, water entry, four-event alternation and rejection after an accepted wind boundary. Root's acceptance receipt SHA-256 is `83170ce84c604382a6c7e88eee0175497ca25799c9dc1c36e8131d03fe9e25a4`.

The Small restart result executed MAPLE `40d1a598b09929b2d0f831513fb6d5c015d88772`, SYRUP `53b9385edf7b04d815320be178e47bef6b73d7ba` and MASY `6b783693c95dc6c06d2878ec5734b0fb3ff3d7a2`. The save, recovered-water and recovered-finish checks took 84.118, 113.444 and 148.267 seconds, respectively. Every accepted boundary matched the retained uninterrupted GPU reference exactly. That reference remains attributed to its original MASY `9a48220865a2dd5ca6cdede7b3d29f4ee38238fc` execution; source equivalence permits the comparison, and no new continuous run is claimed. Duplicate-token replay did not repeat transport or accounting. Referenced old deep-store versions were retained. Root's final restart acceptance receipt SHA-256 is `aa3db4006c239277bbabb2bbf6dbaa81f14aa4a18e76e9a528f28661738aead6`.

All three restart processes exited and were reaped. Source, original inputs and compiled-runtime files were independently rechecked after each role. Sampled host/device usage stayed within the original controls. Samples establish neither peak allocator usage nor capacity for a different landscape; opaque external, primitive and constructor allocation terms remain unpriced.

Masy source composition `04f765a1a76e3e8f1da3de8e5bd390461d5d61dd` combines the accepted C64 and corrected checkpoint files while preserving the existing CPU case workflow and current policy documentation. It is an exact source composition, not a new physical run or permission to migrate an existing checkpoint. Retain the original result identities above. New runs still need an explicitly resolved compatible source pair and a fresh immutable runtime; an old checkpoint must retain its original source lock. Qualification branches remain distinct from upstream main, and unrelated uncommitted upstream work is excluded.

## Additional CPU authored-policy qualification

A fresh real `tools/masy_case.py run` completed wind → rain and recession → wind on the unchanged synthetic 9 × 8, two-class, 20-voxel case. It executed MASY `ebc4b1745807df81f8d4f2b250b6179b1dbc10c5`, MAPLE `40d1a598b09929b2d0f831513fb6d5c015d88772` and SYRUP `919c82c702f3a1b485a35bd3d5c39960c3b577b8`, using fresh lock `24ded59b0510d7acb1007eecea9df907045cb1f618a0f0d29c195ec1ec15d55e`.

The complete authored request explicitly selected pit storage, terminal-storage completion and mobile-sediment deposition at flow end. Its values survived preparation and reconstruction by the real case worker. Native water-removal and residual accounting matched the single rain receipt and aggregate. This unchanged CPU case removed zero surface water and positive soil water; it establishes policy forwarding and accounting, while positive terminal ponded-water removal remains the separate C64 GPU result above. No terrain, boundary, forcing, scientific coefficient or ordinary default was changed.

Root independently joined all six required roles in the full 20-process record, checked all process exits and restored affinity/subreaper state, rehashed original inputs and the locked source/runtime, and read back all 16 final arrays, 20 scalars and five prescribed vegetation fields exactly. The original case worker took 10.357 seconds. Existing resource limits were retained; this observation is not a fit or performance guarantee. Final acceptance receipt SHA-256: `4626354000c8521a1369237b9fb149be6302b80f8dd8bc23842209acad58914a`.

## Interim ponded-water policy

For the qualified C64 profile, native routing permits water to collect in closed depressions. Its explicit completion policy selects `terminal_storage_water=True` and retains `deposit_mobile_at_flow_end=True`; flat storage remains disabled. Native completion must still satisfy its other gates and conservation checks. A time/resource limit remains incomplete.

Native completion settles the remaining mobile sediment, then externally removes and separately accounts for the remaining surface and soil water before returning the dry handoff. MASY records that native accounting once and carries forward the returned evolving bed. Wind starts only from an accepted dry handoff. This removal is not ET, infiltration or a drainage calculation. The ordinary authored-case defaults and calibration were not changed by this qualification. The prescribed shared vegetation remains intact; growth, mortality, uptake and ET await separately specified physics. See the [vegetation information plan](vegetation_information_plan.md).

## Remaining qualification limits

Production reduced-wind calibration on evolving multi-class beds remains pending. The current fixtures use clearly labelled synthetic calibration. General landscape dimensions, larger class counts, arbitrary-landscape GPU CLI execution and capacity/performance scaling remain separate work. C64 persisted restart, GPU mid-event restart and cross-backend checkpoint migration are not established by the Small boundary-restart result. The C64 report continues to declare its own restart support as false. Future vegetation/soil-water updates must use explicit versioned transactions and named water-budget exchanges; this status does not introduce those response laws.
