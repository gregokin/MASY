# Held inventory batch currentness

`_HeldInventory.check()` applies one complete parent-currentness pass, then
checks every admitted file through its exact retained parent descriptor, then
applies another complete parent pass. Each file retains the original comparison
of `(dev, ino, size, mtime_ns, ctime_ns)`, full mode and link count. Parent
identity remains only `(dev, ino)`. No payload, hash, Source, runtime, environment,
compiled-file or Native validator is omitted or cached. Ordinary `_check_file`,
`read()` and `admit()` retain their original bodies and per-file guards.

For P held parents and F files, one successful batch has 2P parent checks and F
relative no-follow file stats. Its predecessor has P+2F pathname-currentness
checks; the new batch has 2P, a difference of 2F-P. An empty inventory stays valid;
P>0/F=0 can add parent checks. Each call repeats all checks from current state.
These are logical call counts, not a timing, whole-heap fit or speedup claim.

This is the existing quiescent owned-Source protocol. Neither implementation
provides hostile-writer atomicity. Transient A-B-A pathname swaps may lose
opportunistic detection relative to per-file polling. Persistent parent drift
before or during the file pass is refused by its prepass or postpass; mutations
after a file's stat remain outside that stat's observation. Delegated reads remain
path based. Owners must keep the admitted Source quiescent for the entire native
provenance operation. Retained directory descriptors and original no-follow
acquisition/currentness policy stay unchanged, including the separately reviewed
ancestor-read permission refinement of the original7fc opener.

The new test Source authors tiny ordinary filesystem fixtures and injects
mutations while retaining the real checks and real held descriptors. It covers
stat identity, mode, hard links, removal, symlinks, directory replacement,
persistent parent drift, after-prepass and between-file mutation, final endpass,
closed ownership, and unchanged standalone read/readmission refusals. These
Source-authored tests are unexecuted until Root admits a new compatible Source
identity. Root must separately qualify the existing23 filesystem and43 Dense
regressions, first publication and duplicate/recovery behavior. Historical
partial profiling is not proof of a complete binding, runtime fit or speedup.
