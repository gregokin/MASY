"""Bounded inventory of actually loaded extension files and CUDA cache."""
from pathlib import Path
import hashlib,stat,sys,types

def inventory(run_dir):
    cap_file=1<<30;cap_total=6<<30;total=0;seen={};rows={}
    paths={types.ModuleType.__getattribute__(m,'__dict__').get('__file__') for m in list(sys.modules.values()) if isinstance(m,types.ModuleType)}
    # Shared libraries used by imported extensions/driver, not anonymous model arrays.
    for line in Path('/proc/self/maps').read_text().splitlines():
        bits=line.split(None,5)
        if len(bits)==6 and bits[5].startswith('/') and '.so' in bits[5] and not bits[5].endswith(' (deleted)'):paths.add(bits[5])
    for name in sorted(x for x in paths if isinstance(x,str) and ('.so' in Path(x).name or Path(x).suffix in ('.pyd','.dll'))):
        p=Path(name);s=p.stat()
        if not stat.S_ISREG(s.st_mode):raise RuntimeError('nonregular compiled dependency')
        ident=(s.st_dev,s.st_ino)
        if ident in seen:
            rows[name]=dict(alias_of=seen[ident]);continue
        if s.st_size>cap_file or total+s.st_size>cap_total:raise RuntimeError(f'compiled dependency inventory exceeds explicit bounds: {p} size={s.st_size} accumulated={total}')
        h=hashlib.sha256();n=0
        with p.open('rb') as f:
            while True:
                chunk=f.read(1<<20)
                if not chunk:break
                n+=len(chunk)
                if n>cap_file:raise RuntimeError('compiled dependency changed size')
                h.update(chunk)
        after=p.stat()
        if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or n!=s.st_size:raise RuntimeError('compiled dependency changed during hash')
        total+=n;seen[ident]=name;rows[name]=dict(bytes=n,sha256=h.hexdigest())
    cache={}
    for p in sorted((Path(run_dir)/'cache').rglob('*')):
        if p.is_file() and p.suffix in ('.cubin','.ptx','.fatbin'):
            raw=p.read_bytes()
            if len(raw)>64<<20:raise RuntimeError('CUDA cache file exceeds inventory bound')
            cache[str(p.relative_to(run_dir))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return dict(schema='masy.actual_compiled_runtime_inventory.v1',compiled_files=rows,total_unique_bytes=total,cuda_cache=cache,scope='actual mapped/imported shared libraries and run-owned compiled CUDA cache; no simulation arrays',limits=dict(file_bytes=cap_file,total_bytes=cap_total,chunk_bytes=1<<20))
