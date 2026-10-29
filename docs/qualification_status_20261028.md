# Execution qualification status — October 29, 2026

MAPLE-SYRUP runs actual alternating MAPLE wind and SYRUP water on the same evolving bed. The following results have separate, bounded qualification subjects. They do not establish a general landscape size or production calibration.

| Path | Accepted result | User interface and limit |
| --- | --- | --- |
| Dense CPU | The MAPLE-case workflow completed a labelled synthetic 9 × 8, two-class case, including a fresh explicit water-policy run. | `tools/masy_case.py run`; supply an authored landscape and scientific inputs. Exact revisions and limits are in [CPU case qualification](case_cpu_qualification.md) and the additional CPU authored-policy qualification below. |
| Windowed CPU | Native reduced wind and positive water alternated on a shared 4 × 4, one-class, six-voxel bed with three resident voxels. | Bounded Python interface; see [CPU windowed integration](windowed_cpu_integration.md). |
| Windowed GPU wind and water | Wind → water → wind → water passed on an 8 × 8, one-class bed with 20 logical voxels and three resident voxels. Water-only, entry and rejected-input recovery checks also passed. | Closed C64 interfaces; [details](windowed_device_c64_integration.md). The supplied-case command below retains the same bounded domain. |
| Imported NPY landscape on windowed GPU | The imported-case API matched direct native state/accounting after every event. The public command separately completed all four events and final output/provenance checks. | Exact 8 × 8, one-class, 20-logical/three-resident-voxel profile; [API evidence](user-guide/case-device-c64.md) and [GPU command](user-guide/case-gpu.md). Fresh operator-issued authority is required. |
| GPU saved boundary restart | Three fresh processes saved after wind, recovered through water, then recovered and finished the remaining wind and water events. | Closed Small 4 × 4, one-class, six-voxel/three-resident-voxel profile. [Restart interface](windowed_device_boundary_restart.md). |
| Factory C64 GPU saved boundary restart | A fresh four-event reference and three save/recovery processes passed, with exact scientific state at all five boundaries and duplicate-operation handling. | Separate factory 8 × 8, one-class, 20-logical/three-resident-voxel store API. [C64 restart evidence](windowed_device_c64_boundary_restart.md). Imported-case budgets remain unsupported. |

The CPU case workflow accepts user-authored MAPLE landscapes within its documented admission rules. A successful synthetic case is not validation of another landscape's boundaries, forcing, class mixture, coefficients or resource fit. The windowed case-adoption API establishes storage adoption only; it does not grant GPU or event-execution eligibility.

## Earlier C64 and Small restart evidence and source identities

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

Production reduced-wind calibration on evolving multi-class beds remains pending. The current fixtures use clearly labelled synthetic calibration. General landscape dimensions, larger class counts, GPU execution beyond the closed qualified profiles and capacity/performance scaling remain separate work. Imported-case persisted restart, GPU mid-event restart and cross-backend checkpoint migration remain unsupported. The separate factory C64 store qualification does not change the imported-case report's `restart_qualified=false` and `general_domain_qualified=false` or the common runner's conservative restart flag. Future vegetation/soil-water updates must use explicit versioned transactions and named water-budget exchanges; this status does not introduce those response laws.

## Imported GPU case comparison

A fresh imported NPY case completed wind → water → wind → water using MASY
`9e7af84aecf46d4b7f6de7d0218534c03e8230c2`, MAPLE
`40d1a598b09929b2d0f831513fb6d5c015d88772` and SYRUP
`61dcf61de14ba302148a4fb1bfdbe08ddf4df10c`. The test-only optional-accounting
correction is separately attributed to `fc39319ef5d04156eec037bbeaf71108b8c572ba`;
the original verified runtime profile and probe were reused without retagging.
The case had exactly 8 × 8 cells, one class, 20 logical voxels and three resident
voxels. Both real initializations, all four wrapper events and all four direct
native events finished within the unchanged original controls in 387.636 seconds.

The affected test passed exact state comparisons and native sediment/water
budget checks, preserved prescribed vegetation and availability, and verified
wind finalization, event clocks and dry water handoffs. Native terminal split
accounting is conditional on an actual terminal mask; this result does not
replace the separate positive-ponding result above. Root rehashed 507 runtime
source files, 133 compiled files, 625 input-reference visits and 590 distinct
input files, plus all 565 assembly origins. Actual controller/worker processes
were reaped and the selected GPU returned to idle. Final acceptance receipt
SHA-256: `f054cd53b1f52710281a487a4230f0d34ab438bf0a0e6b3c2aef5228bd44ecad`.

This result qualifies the bounded imported Python API. Public-command execution,
imported-case saved-boundary restart, broader dimensions/classes and measured
production calibration have separate acceptance requirements. The factory C64
store result below does not qualify imported-case restart. Neither result
establishes a general capacity or peak allocator bound.

## Factory C64 saved-boundary restart

The separate factory-profile store passed its real type-graph gate and four
fresh GPU processes on MASY `efaed2ab08ab77d7e92f7476559de61989551c1f`,
MAPLE `40d1a598b09929b2d0f831513fb6d5c015d88772` and SYRUP
`61dcf61de14ba302148a4fb1bfdbe08ddf4df10c`. Profile
`7bf62f70e46d6ba80340f97a7b61258acef49d4fdb93b56db4f39b227eda844d`
retains its actual immutable source/environment identity.

The uninterrupted reference took 250.405 seconds; first-wind save, recovered
water and recovered finish took 114.797, 182.336 and 257.891 seconds. Every
scientific boundary matched exactly, including sediment, shared vegetation,
native resources, routing, clocks and accounting. Saved receipts and nine water
counters survived recovery. A duplicate old-parent submission caused no Native
restore, transport or counter replay. Referenced old deep versions were retained. All four roles
passed source/runtime checks, cleanup and original resource controls. Root's
final acceptance receipt SHA-256 is
`4ca45e77d71d97f354183f539e7e8d2d5e84a216608e0f10227cbd0d3f1b7cfe`.

This is the [factory C64 store API](windowed_device_c64_boundary_restart.md).
It rejects imported `DeviceWaterC64CaseBudget`, active events and source,
backend or device migration. It supplies no public save/recover command and
does not imply production calibration or larger capacity.

## Public supplied-case GPU command

The real `tools/masy_case_gpu.py run` command separately completed the imported
synthetic NPY 8 × 8/K1/V20/W3 case on MASY
`efaed2ab08ab77d7e92f7476559de61989551c1f`, MAPLE
`40d1a598b09929b2d0f831513fb6d5c015d88772` and SYRUP
`61dcf61de14ba302148a4fb1bfdbe08ddf4df10c`, using the same actual immutable
`7bf62f70e46d6ba80340f97a7b61258acef49d4fdb93b56db4f39b227eda844d`
profile. It compiled and adopted the supplied case once, ran all four events
on the evolving bed, published its Native physical diagnostic and passed final
source, distribution RECORD, environment and compiled-runtime checks.

The case worker took 287.903 seconds; the controller completed in 306.781
seconds. All seven required primary process roles were independently observed
and reaped. Root rechecked 2,683 input-reference visits covering 2,154 distinct
files, verified cleanup and released the selected GPU. The finite approved
runtime roster contained 461 exact binary identities within the original
4 GiB/4,000-file limits; it is specific to this run, not a future wildcard.
Original execution budgets remained unchanged. Root's final acceptance receipt
SHA-256 is `662134cfa8cffc938aaf1a0e12284f0ca4cae99f008e1ac0babe92bcbbadaffb`.

Earlier attempts completed their events but refused final publication on
unlisted lazy-loaded libraries. They remain failed attempts. This acceptance
belongs only to the fresh operation with its complete verified inventory.
It qualifies the [closed GPU command](user-guide/case-gpu.md), with fresh
operator-issued authority and explicit scientific inputs. Its Native diagnostic
is not a combined checkpoint. Supplied-case restart, automatic authority
issuance, other dimensions/classes and production calibration remain separate
work. The separate factory restart result above does not change those limits.
