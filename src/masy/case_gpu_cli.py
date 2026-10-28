"""Closed public imported-case GPU operation; parent imports no Native/provider.

A profile is provenance, not execution authority. Root supplies a fresh pinned
preauthorization; one exclusive root and one deadline cover every serial child.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import select
import shutil
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

from . import case_cli as io

AUTH_SCHEMA = 'masy.case_gpu_authority.v1'
REQUEST_SCHEMA = 'masy.case_gpu_request.v1'
WORKER_SCHEMA = 'masy.case_gpu_worker_request.v1'
WORKER_RESULT = 'masy.case_gpu_worker_result.v1'
RESULT_SCHEMA = 'masy.case_gpu_run.v1'
LANE = 'C64_K1_V20_W3_complete_device_water_v1'
OPERATION = 'masy.case_gpu.run'
TOOL = 'tools/masy_case_gpu.py'
CLAIM = 'bounded execution only; known subset; no allocator or general-domain proof'
BUDGETS = dict(leader_rss_abort_bytes=5 << 30, driver_growth_abort_bytes=9 << 30,
    active_pool_limit_bytes=2 << 30, run_storage_abort_bytes=512 << 20,
    per_file_os_limit_bytes=64 << 20, combined_output_abort_bytes=4 << 20,
    wall_abort_seconds=600, cpu_soft_seconds=480, cpu_hard_seconds=540,
    known_host_logical_bytes=5 << 30, known_device_logical_bytes=2 << 30,
    native_safety_fraction=0.5)
FACT_FIELDS = ('cwc_base_slots', 'cwc_k6_slots', 'cwc_slots', 'cwc_current_slots',
    'cwc_operators', 'w1_operators', 'closed_native_operators', 'bounded_cpu_operators',
    'closed_water_operators', 'device_commit_operators', 'device_scheduler_operators', 'device_record_operators')
IO_FIELDS = ('max_path_chars', 'max_water_input_bytes', 'max_water_input_file_bytes',
    'max_report_bytes', 'max_output_bytes', 'max_input_entries', 'max_request_bytes')
AUTH_FIELDS = ('schema', 'operation', 'lane', 'store', 'profile_id', 'profile_file', 'run_id',
    'sources', 'launcher_rel', 'request', 'availability', 'nvidia_smi',
    'native_image_facts', 'case_budget_delegation', 'execution_budgets', 'compiled_additions')
REQUEST_FIELDS = ('schema', 'label', 'calibration', 'case', 'input_limits', 'compile_bed_limits',
    'windowed_storage', 'windowed_limits', 'water_profile', 'water_copy_limits', 'native_limits',
    'bridge_limits', 'bed_limits', 'resource_limits', 'runner_limits', 'placement_limits',
    'io_policy', 'schedule', 'backend_change')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def closed(value, fields, where):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(where + ': exact closed field set required')
    return value


def positive(value, where, high=(1 << 63) - 1, low=1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(where + ': plain bounded integer required')
    return value


def digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('exact lowercase SHA256 required')
    return value


def reference(value):
    closed(value, ('path', 'size', 'sha256'), 'Source reference')
    io.path_value(value['path'], 4096, absolute=True)
    positive(value['size'], 'reference bytes', 64 << 20)
    digest(value['sha256'])
    return value


def pinned(ref, *, immutable=False):
    reference(ref)
    raw, actual = io.read_pin(Path(ref['path']), ref['size'])
    if actual != ref or (immutable and Path(ref['path']).lstat().st_mode & 0o222):
        raise ValueError('pinned file bytes/custody changed')
    return raw


def parse(raw, maximum=2 << 20):
    value = io.strict_json(raw, maximum=maximum, entries=131072, text=maximum, depth=32)
    todo = [value]
    while todo:
        item = todo.pop()
        if type(item) is int and item.bit_length() > 128: raise ValueError('integer domain')
        if type(item) is dict: todo.extend(item.values())
        elif type(item) is list: todo.extend(item)
    return value


def validate_authority(value, *, at_issue, now_ns=None):
    closed(value, AUTH_FIELDS, 'case authority')
    if (value['schema'], value['operation'], value['lane'], value['launcher_rel']) != (
            AUTH_SCHEMA, OPERATION, LANE, TOOL): raise ValueError('unsupported case authority')
    io.path_value(value['store'], 4096, absolute=True)
    digest(value['profile_id'])
    if type(value['run_id']) is not str or re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value['run_id']) is None:
        raise ValueError('unsafe operation run ID')
    for key in ('profile_file', 'request', 'nvidia_smi'): reference(value[key])
    if Path(value['profile_file']['path']) != Path(value['store'])/'profiles'/(value['profile_id']+'.json'):
        raise ValueError('profile reference path join')
    closed(value['sources'], ('masy', 'maple', 'syrup'), 'current sources')
    for name, source in value['sources'].items():
        closed(source, ('root', 'head', 'tree'), name+' Source')
        io.path_value(source['root'], 4096, absolute=True)
        for field in ('head', 'tree'):
            if type(source[field]) is not str or re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', source[field]) is None:
                raise ValueError('current committed Source identity required')
    a = value['availability']; closed(a, ('gpu_uuid', 'pci_bus_id', 'issued_unix_ns', 'expires_unix_ns'), 'availability')
    from .gpu_runtime_worker import physical_uuid, pci_id
    if physical_uuid(a['gpu_uuid']) != a['gpu_uuid'] or pci_id(a['pci_bus_id']) != a['pci_bus_id']:
        raise ValueError('canonical physical UUID/PCI required')
    for key in ('issued_unix_ns', 'expires_unix_ns'): positive(a[key], key)
    if a['issued_unix_ns'] >= a['expires_unix_ns']: raise ValueError('ordered authority timestamps')
    now_ns = time.time_ns() if now_ns is None else now_ns
    if at_issue and not a['issued_unix_ns'] <= now_ns < a['expires_unix_ns']:
        raise ValueError('launch authority is not currently issued')
    if value['execution_budgets'] != BUDGETS or any(type(value['execution_budgets'][k]) is not type(v) for k,v in BUDGETS.items()):
        raise ValueError('original exact execution budgets required')
    additions=value['compiled_additions']
    if type(additions) is not list or len(additions)>4096:raise ValueError('exact bounded compiled additions roster required')
    paths=set()
    for row in additions:
        closed(row,('path','size','sha256'),'compiled addition')
        io.path_value(row['path'],4096,absolute=True);positive(row['size'],'compiled binary bytes');digest(row['sha256'])
        if row['path'] in paths:raise ValueError('duplicate compiled addition')
        paths.add(row['path'])
    closed(value['native_image_facts'], FACT_FIELDS, 'actual Native image facts')
    for key in FACT_FIELDS: positive(value['native_image_facts'][key], key, 1 << 24)
    d = value['case_budget_delegation']
    closed(d, ('issuer', 'native_source_identity', 'current_runtime_ref', 'producer_source',
               'source_bindings', 'source_manifest_sha256', 'grant_path'), 'one-case delegation')
    for key in ('issuer', 'current_runtime_ref', 'producer_source'): reference(d[key])
    for key in ('native_source_identity', 'source_manifest_sha256'): digest(d[key])
    if type(d['source_bindings']) is not list or len(d['source_bindings']) != 22:
        raise ValueError('exact22 owning Source bindings required')
    names = set()
    for row in d['source_bindings']:
        closed(row, ('module', 'path', 'size', 'sha256'), 'Source binding')
        if type(row['module']) is not str or row['module'] in names: raise ValueError('Source module roster')
        names.add(row['module']); reference({k:row[k] for k in ('path','size','sha256')})
    root = Path(value['store'])/'runs'/value['run_id']
    if Path(d['grant_path']) != root/'case'/'case-budget.json': raise ValueError('exclusive owned budget path')
    return value


class IssuedAuthority:
    def __init__(self, ref, raw, value):
        self.ref, self.raw, self.value = dict(ref), raw, value
    def revalidate(self):
        raw = pinned(self.ref, immutable=True)
        if raw != self.raw: raise ValueError('authority bytes changed')
        return validate_authority(parse(raw), at_issue=False)


def read_authority(ref, *, at_issue=True):
    raw = pinned(ref, immutable=True)
    return IssuedAuthority(ref, raw, validate_authority(parse(raw), at_issue=at_issue))


def request_from(raw):
    v = parse(raw); closed(v, REQUEST_FIELDS, 'GPU case request')
    if v['schema'] != REQUEST_SCHEMA or v['calibration'] != 'synthetic_test_fixture':
        raise ValueError('only labelled synthetic C64 imported case lane')
    if type(v['label']) is not str or not 0 < len(v['label']) <= 4096: raise ValueError('explicit bounded label')
    p = v['io_policy']; closed(p, IO_FIELDS, 'IO policy')
    ceilings = (4096,64<<20,64<<20,4<<20,4<<20,4096,2<<20)
    for name, cap in zip(IO_FIELDS, ceilings, strict=True): positive(p[name], name, cap)
    if len(raw) > p['max_request_bytes']: raise ValueError('request declared byte ceiling')
    closed(v['case'], ('directory','authored_source_sha256','copy_mode'), 'authored case')
    io.path_value(v['case']['directory'],p['max_path_chars'],absolute=True);digest(v['case']['authored_source_sha256'])
    if v['case']['copy_mode'] != 'authored_source_then_compile': raise ValueError('case copy mode')
    closed(v['windowed_storage'], ('datum_offset_m','window_voxels','chunk_columns','rows_per_band','with_prefix_packets'), 'storage')
    s=v['windowed_storage']
    if type(s['datum_offset_m']) is not float or s['datum_offset_m'].hex()!=float(0).hex() or s['window_voxels']!=3 or type(s['window_voxels']) is not int or s['with_prefix_packets'] is not True:
        raise ValueError('exact W3/+0.0/E1P8 storage')
    positive(s['chunk_columns'],'chunk width',1024,64);positive(s['rows_per_band'],'row band',8)
    if v['backend_change'] != dict(field='sediment_control.storm.implementation', **{'from':'array','to':'cuda'}):
        raise ValueError('only declared Native derived backend change')
    closed(v['water_profile'], ('spec','control'), 'water profile')
    spec,control=v['water_profile']['spec'],v['water_profile']['control']
    closed(spec,io._SPEC_FIELDS,'water spec');closed(control,('sediment_control','completion_policy','report_every_s'),'water control')
    closed(control['sediment_control'],io._SEDIMENT,'sediment control');closed(control['sediment_control']['storm'],io._STORM,'storm')
    closed(control['completion_policy'],io._COMPLETION,'completion')
    if control['sediment_control']['storm']['implementation']!='array':raise ValueError('authored HOST water control required')
    for name in io._ARRAYS:
        a=spec[name]
        if a is None and name in io._NULL_ARRAYS:continue
        closed(a,('kind','path','size','sha256','dtype','shape'),'water NPY descriptor')
        if a['kind']!='npy' or a['dtype']!=('bool' if name in io._BOOL_ARRAYS else 'float64'):raise ValueError('explicit NPY dtype')
        io.path_value(a['path'],p['max_path_chars'],absolute=False);positive(a['size'],'NPY bytes',p['max_water_input_file_bytes']);digest(a['sha256'])
        if type(a['shape']) is not list or len(a['shape'])!=2:raise ValueError('NPY rank')
        for dim in a['shape']:positive(dim,'NPY dimension',10)
    closed(v['native_limits'],('profile','record_limits'),'Native policies')
    closed(v['schedule'],('schema','events'),'schedule')
    events=v['schedule']['events']
    if v['schedule']['schema']!='masy.case_schedule.v1' or type(events) is not list or len(events)!=4 or [e.get('kind') if type(e) is dict else None for e in events]!=['wind','rain','wind','rain']:
        raise ValueError('closed W/R/W/R schedule')
    for e in events:
        if e['kind']=='wind':
            closed(e,('kind','first_block','stop_block'),'wind group');positive(e['first_block'],'first block',1,0);positive(e['stop_block'],'stop block',2)
        else:
            closed(e,('kind','water_profile_id','intervals','recession_guard_s'),'rain group')
            if type(e['intervals']) is not list or not e['intervals']:raise ValueError('explicit rainfall intervals')
            for interval in e['intervals']:
                closed(interval,('duration_s','intensity_mm_h'),'rain interval')
                io.number(interval['duration_s'],'rain duration',positive=True);io.number(interval['intensity_mm_h'],'rain rate')
            io.number(e['recession_guard_s'],'recession guard',positive=True)
    for key in ('input_limits','compile_bed_limits','windowed_limits','water_copy_limits','bridge_limits','bed_limits','resource_limits','runner_limits','placement_limits'):
        if type(v[key]) is not dict:raise ValueError('explicit complete limit object: '+key)
    if 'run_dir' in v['bridge_limits']:raise ValueError('run_dir is assigned by the owning operation')
    return v


def write_new(path, value, *, raw=False, immutable=False):
    data=value if raw else canonical(value)
    if len(data)>64<<20:raise ValueError('single output file cap')
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    if immutable:Path(path).chmod(0o444)
    return dict(path=str(path),size=len(data),sha256=hashlib.sha256(data).hexdigest())


def census(root):
    """At most3 whole walks; only disappearance retries, never partial credit."""
    for attempt in range(3):
        total=count=0;todo=[Path(root)]
        try:
            while todo:
                p=todo.pop();st=p.lstat()
                if p.is_symlink() or not stat.S_ISDIR(st.st_mode):raise ValueError('run directory custody')
                with os.scandir(p) as entries:
                    for entry in entries:
                        count+=1
                        if count>4096:raise ValueError('whole operation entry cap')
                        q=p/entry.name;info=q.lstat()
                        if stat.S_ISDIR(info.st_mode):todo.append(q)
                        elif stat.S_ISREG(info.st_mode):
                            if info.st_size>64<<20:raise ValueError('per-file cap')
                            total+=info.st_size
                        else:raise ValueError('symlink/special run entry')
                        if total>512<<20:raise ValueError('whole operation storage cap')
            return dict(bytes=total,entries=count)
        except FileNotFoundError:
            if attempt==2:raise
    raise AssertionError('unreachable')


def reserve(root, extra):
    positive(extra,'prospective output bytes',512<<20,0);c=census(root)
    if extra>512*1024**2-c['bytes'] or extra>shutil.disk_usage(root).free:raise ValueError('prospective whole-operation disk admission')
    return dict(**c,reserved_extra_bytes=extra)


class Deadline:
    def __init__(self, started):
        if type(started) not in (float,int) or not math.isfinite(started) or started<=0 or started>time.monotonic():raise ValueError('actual tool start required')
        self.started=float(started);self.end=self.started+600
    def remaining(self):
        left=self.end-time.monotonic()
        if left<=0:raise TimeoutError('whole600s operation deadline')
        return left


def birth(pid):
    data=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=data[19])


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU,(480,540));resource.setrlimit(resource.RLIMIT_FSIZE,(64<<20,64<<20))
    resource.setrlimit(resource.RLIMIT_NOFILE,(1024,1024));resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def reap_owned(child,fd,*,terminate,deadline):
    """Fixed original TERM/5s/KILL/reap; no Native work or renewed deadline."""
    if terminate:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    code=child.wait()
    exited=fd is not None and bool(select.select([fd],[],[],0)[0])
    if fd is not None:os.close(fd)
    return dict(exit_code=code,reaped=True,pidfd_exited=exited,
                reap_after_deadline_s=max(0,time.monotonic()-deadline))


class Monitor:
    def __init__(self,root,clock,authority):
        self.root,self.clock,self.authority=Path(root),clock,authority
        self.stream_bytes=0;self.receipts=[];self.peak_rss=self.peak_driver=self.peak_storage=0
        self.cpu_origin=0.0;self.initial_used=None
    def query(self,args):
        ref=self.authority['nvidia_smi'];pinned(ref)
        result=subprocess.run([ref['path'],*args],env={'PATH':'/usr/bin:/bin','LC_ALL':'C'},stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(2,self.clock.remaining()),check=True)
        self.stream_bytes+=len(result.stdout)+len(result.stderr)
        if len(result.stdout)+len(result.stderr)>65536 or self.stream_bytes>4<<20:raise ValueError('query/combined stream cap')
        if pinned(ref) is None:raise AssertionError('unreachable')
        return result.stdout.decode('ascii')
    def device(self):
        uuid=self.authority['availability']['gpu_uuid']
        row=self.query(['-i',uuid,'--query-gpu=uuid,pci.bus_id,memory.free,memory.used','--format=csv,noheader,nounits']).strip().split(',')
        from .gpu_runtime_worker import pci_id
        if len(row)!=4 or row[0].strip()!=uuid or pci_id(row[1].strip())!=self.authority['availability']['pci_bus_id']:raise ValueError('actual GPU identity changed')
        return int(row[2]),int(row[3])
    def preflight(self):
        free,self.initial_used=self.device()
        apps=self.query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
        if free<9728 or any(l.strip().startswith(self.authority['availability']['gpu_uuid']+',') for l in apps.splitlines()):raise ValueError('authorized GPU is busy or lacks headroom')
        mem=dict(l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0])*1024<6<<30:raise ValueError('HOST headroom below original6GiB')
    def _cpu(self,pid=None):
        own=resource.getrusage(resource.RUSAGE_SELF);done=resource.getrusage(resource.RUSAGE_CHILDREN)
        total=own.ru_utime+own.ru_stime+done.ru_utime+done.ru_stime
        if pid:
            try:
                s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();total+=sum(int(s[i]) for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK')
            except FileNotFoundError:pass
        return total
    def run(self,argv,env,folder):
        self.clock.remaining();folder=Path(folder)
        outpath,errpath=folder/'stdout.txt',folder/'stderr.txt'
        reason=None;child=None;fd=None;started=time.monotonic();identity=None
        # The local loop enforces the SAME deadline while a child exists.
        # Defer the launcher's one-shot alarm until fixed cleanup and its
        # lifecycle receipt are complete; never renew/clear the timer.
        prior_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM})
        def child_limits():
            signal.pthread_sigmask(signal.SIG_SETMASK,prior_mask)
            _limits()
        try:
            with outpath.open('xb') as out,errpath.open('xb') as err:
                try:
                    child=subprocess.Popen(argv,env=env,cwd=folder,stdout=out,stderr=err,start_new_session=True,preexec_fn=child_limits)
                    identity=birth(child.pid);fd=os.pidfd_open(child.pid)
                    # Pending TERM may raise only after all child custody is
                    # assigned; the finally block can now always reap it.
                    signal.pthread_sigmask(signal.SIG_SETMASK,(set(prior_mask)|{signal.SIGALRM})-{signal.SIGTERM})
                    next_device=next_storage=0.0
                    while os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is None:
                        now=time.monotonic()
                        try:self.clock.remaining()
                        except TimeoutError:reason='whole_wall';break
                        if self._cpu(child.pid)>=480:reason='whole_cpu';break
                        try:
                            data=dict(l.split(':',1) for l in Path(f'/proc/{child.pid}/status').read_text().splitlines() if ':' in l)
                            rss=int(data.get('VmRSS','0 kB').split()[0])*1024;self.peak_rss=max(self.peak_rss,rss)
                            if rss>5<<30:reason='leader_rss'
                        except FileNotFoundError:pass
                        if self.initial_used is not None and now>=next_device:
                            _free,used=self.device();growth=max(0,used-self.initial_used)<<20;self.peak_driver=max(self.peak_driver,growth)
                            if growth>9<<30:reason='driver_growth'
                            next_device=now+0.5
                        if now>=next_storage:
                            c=census(self.root);self.peak_storage=max(self.peak_storage,c['bytes'])
                            if self.stream_bytes+outpath.stat().st_size+errpath.stat().st_size>4<<20:reason='combined_streams'
                            next_storage=now+0.2
                        if reason:break
                        time.sleep(min(0.05,max(0.001,self.clock.end-time.monotonic())))
                except BaseException:
                    reason=reason or 'controller_exception';raise
                finally:
                    if child is not None:
                        cleanup_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
                        try:
                            reaped=reap_owned(child,fd,terminate=bool(reason),deadline=self.clock.end)
                            code=reaped['exit_code']
                            exited=reaped['pidfd_exited']
                            self.stream_bytes+=outpath.stat().st_size+errpath.stat().st_size
                            if self._cpu()>=480:reason=reason or 'whole_cpu'
                            row=dict(argv=argv,process=identity,exit_code=code,reaped=True,pidfd_exited=exited,reason=reason,
                                elapsed_s=time.monotonic()-started,reap_after_deadline_s=max(0,time.monotonic()-self.clock.end),
                                combined_stream_bytes=self.stream_bytes,peak_leader_rss=self.peak_rss,peak_driver_growth=self.peak_driver,peak_storage=self.peak_storage)
                            self.receipts.append(row);write_new(folder/'lifecycle.json',row)
                        finally:
                            signal.pthread_sigmask(signal.SIG_SETMASK,cleanup_mask)
            if reason or code!=0 or self.stream_bytes>4<<20:raise ValueError('monitored child refused: '+str(reason or code))
            self.clock.remaining();census(self.root)
            raw=outpath.read_bytes()
            if len(raw)>4<<20:raise ValueError('result stream ceiling')
            return parse(raw,4<<20),row
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,prior_mask)


def _stage(root,name):
    from .gpu_runtime import worker
    p=Path(root)/name;p.mkdir(mode=0o700)
    (p/'cache').mkdir()
    for rel in worker._CACHE_PATHS.values():(p/rel).mkdir(parents=True,exist_ok=True)
    return SimpleNamespace(dir=p,pycache=str(p/'cache/pycache'))


def _worker_bootstrap(path):
    tree=ast.parse(Path(path).read_bytes());matches=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='BOOTSTRAP' and isinstance(n.value,ast.Constant) and type(n.value.value) is str]
    if len(matches)!=1:raise ValueError('one immutable case BOOTSTRAP literal required')
    return matches[0]


class CompletionDeliveryError(RuntimeError):
    """Completed operation/contested publication is never a normal refusal."""
    def __init__(self,ref,detail,*,status='complete'):
        self.record=dict(schema=RESULT_SCHEMA,status=status,code='completion_delivery_failed',
            result=ref,detail=str(detail)[:400])
        super().__init__('authoritative completion publication; notification failed')


def _committed_completion(root,raw):
    """Reconcile the actual pathname even if link succeeded before assignment."""
    if raw is None:return None
    path=Path(root)/'result.json'
    try:data,ref=io.read_pin(path,4<<20)
    except FileNotFoundError:return None
    except (OSError,ValueError) as exc:
        raise CompletionDeliveryError(dict(path=str(path)),exc,status='completion_publication_unverified') from exc
    if data!=raw:raise CompletionDeliveryError(ref,'published bytes differ from staged completion',status='completion_publication_unverified')
    return ref


def _commit_completion(root,raw,clock,launch,stream_bytes):
    """Last acceptance action, two retained names, no postcommit checks."""
    clock.remaining()
    if len(raw)>4<<20 or stream_bytes+len(raw)+1+(64<<10)>4<<20:
        raise ValueError('prospective complete/notification combined stream ceiling')
    # Both hard-link directory entries stay retained and census counts both.
    counts=reserve(root,2*len(raw))
    if counts['entries']+2>4096:raise ValueError('prospective staged/final result entry cap')
    candidate=Path(root)/'result.candidate.json'
    staged=write_new(candidate,raw,raw=True,immutable=True)
    counts=reserve(root,len(raw))
    if counts['entries']+1>4096:raise ValueError('prospective final result entry cap')
    ref=dict(staged,path=str(Path(root)/'result.json'))
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM})
    try:
        clock.remaining()
        if signal.sigpending()&{signal.SIGALRM,signal.SIGTERM}:
            raise InterruptedError('pending cancellation before completion publication')
        os.link(candidate,ref['path'])  # Exclusive atomic publication, no overwrite.
        launch['_published_result']=ref
        return ref
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)


def _emit_completion(raw,deadline):
    """Bounded notification only; nonblocking IO uses the ORIGINAL deadline."""
    fd=sys.stdout.fileno();blocking=os.get_blocking(fd);os.set_blocking(fd,False)
    try:
        view=memoryview(raw+b'\n')
        while view:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('completion notification reached original deadline')
            if not select.select([],[fd],[],min(.1,remaining))[1]:continue
            try:count=os.write(fd,view[:65536])
            except BlockingIOError:continue
            if count<=0:raise BrokenPipeError('completion notification made no progress')
            view=view[count:]
    finally:os.set_blocking(fd,blocking)


def _delivery_notice(error):
    # Bounded best effort control evidence after commit, never new model work.
    try:
        raw=canonical(error.record)+b'\n'
        if len(raw)>64<<10:return
        fd=sys.stderr.fileno();blocking=os.get_blocking(fd)
        try:
            os.set_blocking(fd,False)
            os.write(fd,raw)
        finally:os.set_blocking(fd,blocking)
    except (OSError,ValueError):pass  # Preserve distinct delivery exit even if stderr is closed.


def run_parent(args,launch):
    from . import gpu_runtime as gpu, runtime as rt
    clock=Deadline(launch['started_monotonic']);clock.remaining()
    issued=read_authority(dict(path=args.authority,size=args.authority_size,sha256=args.authority_sha256));a=issued.value
    if (a['store'],a['profile_id'],a['run_id'],a['request']['path'])!=(args.store,args.profile,args.run_id,args.request):raise ValueError('CLI/authority join')
    if launch['root']!=a['sources']['masy']['root'] or launch['path']!=str(Path(launch['root'])/TOOL):raise ValueError('own committed launcher path')
    pinned(a['profile_file'],immutable=True);pinned(a['request'])
    profile,limits,audit_bootstrap=gpu._load_profile(args.store,args.profile);clock.remaining()
    if profile['device']!=dict(uuid=a['availability']['gpu_uuid'],pci_bus_id=a['availability']['pci_bus_id'],visible_ordinal=0):raise ValueError('profile/physical device join')
    if Path(__file__).resolve()!=Path(profile['objects']['masy']['path'])/'source/src/masy/case_gpu_cli.py':raise ValueError('actual parent module outside selected Source')
    python=rt.interpreter_identity(gpu.worker.PYTHON,limits)
    if rt._python_stable(python)!=profile['runtime']['python']:raise ValueError('profile interpreter changed')
    rt.disk_preflight(args.store,512<<20,limits);clock.remaining()
    root=Path(args.store)/'runs'/args.run_id;root.mkdir(mode=0o700)
    consumption=write_new(root/'consumption.json',dict(authority=issued.ref,operation=OPERATION,pid=os.getpid(),started_monotonic=clock.started),immutable=True)
    monitor=Monitor(root,clock,a);completion_raw=None
    def operation(mode):
        run=_stage(root,mode);body=gpu._profile_body(profile,run,python)
        outer=dict(schema=WORKER_SCHEMA,operation=mode,runtime=body,authority=issued.ref,case_request=a['request'],launcher=launch,
                   profile_id=args.profile,operation_root=str(root),deadline_monotonic=clock.end,expected_audit=profile['audit'],consumption=consumption)
        path=run.dir/'request.json';write_new(path,outer)
        argv=[gpu.worker.PYTHON,'-I','-B','-X','pycache_prefix='+run.pycache,'-c',bootstrap,str(path)]
        result,life=monitor.run(argv,body['run']['env'],run.dir)
        if result.get('schema')!=WORKER_RESULT or result.get('operation')!=mode or result.get('status')!='ok' or result.get('process',{}).get('pid')!=life['process']['pid'] or result.get('process',{}).get('start_ticks')!=life['process']['start_ticks'] or result.get('process',{}).get('cwd')!=str(run.dir) or result.get('process',{}).get('pycache_prefix')!=run.pycache or result.get('process',{}).get('sys_path_prefix')!=body['sys_path_prefix'] or any(result.get('process',{}).get('environment',{}).get(k)!=v for k,v in body['run']['env'].items()):
            raise ValueError('actual case worker protocol refused')
        return result['result']
    try:
        # Pure schema validation precedes physical child work. Once consumed,
        # even a malformed request gets a truthful failure record in this root.
        request_from(pinned(a['request']))
        worker_path=Path(profile['objects']['masy']['path'])/'source/src/masy/case_gpu_worker.py'
        if a['case_budget_delegation']['issuer']['path']!=str(worker_path):raise ValueError('delegated immutable worker origin')
        pinned(a['case_budget_delegation']['issuer'],immutable=True)
        bootstrap=_worker_bootstrap(worker_path)
        monitor.preflight()
        before=operation('metadata')
        audit_run=_stage(root,'audit');body=gpu._profile_body(profile,audit_run,python)
        apath=audit_run.dir/'request.json';write_new(apath,dict(schema=gpu.worker.REQUEST_SCHEMA,operation='audit',**body))
        audited,life=monitor.run([gpu.worker.PYTHON,'-I','-B','-X','pycache_prefix='+audit_run.pycache,'-c',audit_bootstrap,str(apath)],body['run']['env'],audit_run.dir)
        closed(audited,('schema','operation','status','result','process','interpretation'),'genuine audit result')
        if (audited['schema']!=gpu.worker.RESULT_SCHEMA or audited['operation']!='audit' or audited['status']!='ok' or audited['result']!=profile['audit'] or audited['interpretation']!=gpu.worker.INTERPRETATION
                or audited['process']['pid']!=life['process']['pid'] or audited['process']['start_ticks']!=life['process']['start_ticks'] or audited['process']['cwd']!=str(audit_run.dir)
                or audited['process']['pycache_prefix']!=audit_run.pycache or audited['process']['sys_path_prefix']!=body['sys_path_prefix']
                or any(audited['process']['environment'].get(k)!=v for k,v in body['run']['env'].items())):
            raise ValueError('genuine cold audit differs from retained profile')
        result=operation('case')
        after=operation('metadata-after')
        if before!=after:raise ValueError('current committed Source/tool/request changed during operation')
        issued.revalidate();pinned(a['request']);gpu._load_profile(args.store,args.profile);clock.remaining();census(root)
        answer=dict(schema=RESULT_SCHEMA,status=result['status'],run_dir=str(root),profile_id=args.profile,authority=issued.ref,
            binding_sha256=result['binding_sha256'],report=result['report'],diagnostic=result['diagnostic'],children=monitor.receipts,
            qualification='this synthetic imported case only; physical acceptance Root-owned',known_subset_only=True,claim=CLAIM,restart_qualified=False,general_domain_qualified=False)
        completion_raw=canonical(answer)
        if monitor._cpu()>=480:raise ValueError('whole CPU ceiling before completion publication')
        _commit_completion(root,completion_raw,clock,launch,monitor.stream_bytes)
        return answer,completion_raw
    except BaseException as exc:
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGTERM})
        try:
            committed=_committed_completion(root,completion_raw)
            if committed is not None:
                launch['_published_result']=committed
                raise CompletionDeliveryError(committed,exc) from exc
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        failure=dict(schema=RESULT_SCHEMA,status='refused',run_dir=str(root),code=getattr(exc,'code',type(exc).__name__),detail=str(exc)[:2000],children=monitor.receipts,authority=issued.ref,cleanup_after_deadline_s=max(0,time.monotonic()-clock.end))
        write_new(root/'failure.json',failure);raise


def main(argv,*,launch):
    p=argparse.ArgumentParser(description='Explicit synthetic imported C64 GPU case; no general capacity or restart claim',allow_abbrev=False)
    p.add_argument('command',choices=('run',))
    for key in ('store','profile','authority','authority-sha256','request','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--authority-size',required=True,type=int)
    args=p.parse_args(argv)
    try:
        answer,raw=run_parent(args,launch)
        try:_emit_completion(raw,launch['started_monotonic']+600)
        except BaseException as exc:
            ref=_committed_completion(Path(answer['run_dir']),raw)
            raise CompletionDeliveryError(ref,exc) from exc
        return 0 if answer['status']=='complete' else 2
    except CompletionDeliveryError as exc:
        _delivery_notice(exc)
        return 3  # Completion publication exists/needs inspection; never refused.
