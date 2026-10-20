The passive boundary writer validates live native routing through a private
readback before creating the wrapper image. Native completed rain retains a
writable source surface and freshly allocated zero sediment velocity. The disk
reader requires private read-only routing arrays. The writer now copies only
those two admitted routing arrays for its transient validation; the actual NPY
serializer still receives the original accepted references.

Closed five/seven-array inventory, native dtype, shape, C layout and the numeric
byte cap are admitted by the unchanged snapshot path before either copy. The
component pre-checks its existing static readback reservation plus the two copies'
exact combined bytes against the existing 2 GiB ceiling before allocation. There
is no mutable allocation/charge ledger or new reservation API. The additional
copies cannot exceed the existing 1 MiB wrapper numeric cap. The prepared owner
and both copies are released before
serialization. Accepted arrays, flags, aliases, clocks and receipts do not change.
Loaded read-only/disjoint ownership checks and the native physical codec remain
unchanged. This is a bounded Dense CPU source-manifest prototype correction.

The focused check has separate old-code native failing control, corrected native
wind/rain/save/load, and synthetic layout/cap modes. Native fixtures keep the
existing six-cell, six-class, three-voxel, 16000-cohort limits, wind 2 s, rain 60 s
at 72 mm/h, and 180 s recession guard. The saved routing graph is reconstructed
from its retained actual source; that validation is separate from hydraulic
rerouting. No transport runs during save/readback/resave. Raw payloads stay local.
Successful two-event passive save/load does not establish four-event durable
continuation, crash recovery or a genuine committed source lock.
