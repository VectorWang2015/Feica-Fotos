#!/usr/bin/env python3
"""Create exactly one explicitly named research bypass ICC, never install it.

Preserve every non-description tag byte and header byte except declared size.
ICC v2 reserved bytes (including where v4 puts profileID) stay unchanged.
"""
import argparse
import sys
import json
from pathlib import Path
import random
import struct

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.c1 import inspect_profile as audit
SOURCE=ROOT/'inputs/LeicaQTyp116-Generic.icm'
EXPECTED='efed3f9554cb8da91de1fffb0e4a93b2ae9f98acc47d2cd4078133b38f99b969'


def description(text):
    raw=text.encode('ascii')+b'\0'
    if len(raw)>67:raise ValueError('Mac script field too long')
    # ASCII count/data; Unicode language/count both 0; ScriptCode 0;
    # counted MacRoman-compatible ASCII copy in a complete 67-byte field.
    return b'desc'+b'\0'*4+struct.pack('>I',len(raw))+raw+b'\0'*8+struct.pack('>HB',0,len(raw))+raw.ljust(67,b'\0')


def tags(d):
    out=[]
    for i in range(audit.u32(d,128)):
        k=132+12*i;s=d[k:k+4];o=audit.u32(d,k+4);n=audit.u32(d,k+8)
        if o+n>len(d):raise ValueError('bad source tag')
        out.append((s,o,d[o:o+n]))
    return out


def serialize(d,desc):
    original=tags(d);end=132+12*len(original);table=bytearray(struct.pack('>I',len(original)));body=bytearray();pos=end
    for sig,old_off,payload in original:
        if sig==b'desc':payload=description(desc)
        pad=(-pos)%4;body.extend(b'\0'*pad);pos+=pad
        table.extend(sig+struct.pack('>II',pos,len(payload)));body.extend(payload);pos+=len(payload)
    header=bytearray(d[:128]);header[:4]=struct.pack('>I',pos)
    result=bytes(header+table+body)
    if len(result)!=pos or result[4:128]!=d[4:128]:raise AssertionError('unexpected header change')
    return result


def identity_test():
    h=bytearray(52);h[:4]=b'mft2';h[8:11]=bytes([3,3,2])
    # s15Fixed16 unity is 65536, unlike the normalized LUT uint16 max 65535.
    for i in range(9):struct.pack_into('>i',h,12+4*i,65536 if i in (0,4,8) else 0)
    struct.pack_into('>HH',h,48,2,2)
    curves=struct.pack('>6H',0,65535,0,65535,0,65535)
    clut=b''.join(struct.pack('>3H',r*65535,g*65535,b*65535) for r in (0,1) for g in (0,1) for b in (0,1))
    lut=audit.Lut16(bytes(h)+curves+clut+curves)
    rng=random.Random(20261002);pts=[[rng.random() for _ in range(3)] for _ in range(64)]+[[0,0,0],[1,1,1]]
    errors={m:max(abs(a-b) for p in pts for a,b in zip(lut.clut_eval(p,m),p)) for m in ('tetrahedral','trilinear')}
    if max(errors.values())>1e-14:raise AssertionError('self-created identity CLUT test failed')
    return {'points':len(pts),'max_abs_error':errors,'profile_written':False}


def main():
    global SOURCE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, default=SOURCE, help='User-provided original Q1 Generic; exact pinned source required')
    p.add_argument('--out', type=Path, required=True, help='New bypass ICC path')
    p.add_argument('--report', type=Path, required=True)
    a=p.parse_args(); SOURCE, DEST, REPORT=a.profile, a.out, a.report
    if DEST.resolve() == REPORT.resolve():p.error('ICC and report outputs must differ')
    if DEST.exists() or REPORT.exists():raise SystemExit('Refusing overwrite')
    d=SOURCE.read_bytes();before=audit.sha(d)
    if before!=EXPECTED:raise SystemExit('source hash mismatch')
    p=serialize(d,'LocalLooks-Bypass');source_tags={s:b for s,o,b in tags(d)};new_tags={s:b for s,o,b in tags(p)}
    differences=[]
    for s,o,b in tags(p):
        original=source_tags[s]
        differences.append({'signature':s.decode(),'source_sha256':audit.sha(original),'candidate_sha256':audit.sha(b),
          'byte_identical':original==b,'new_offset':o,'new_size':len(b)})
        if s!=b'desc' and original!=b:raise AssertionError('non-desc payload changed')
    oldlut=audit.Lut16(source_tags[b'A2B0']);newlut=audit.Lut16(new_tags[b'A2B0'])
    rng=random.Random(991);pts=[[rng.random() for _ in range(3)] for _ in range(256)]+[[v,v,v] for v in (0,.001,.01,.18,.5,1)]+[[1,0,0],[0,1,0],[0,0,1]]
    maxdiff=max(abs(a-b) for q in pts for a,b in zip(oldlut.evaluate(q)['pcs_Lab'],newlut.evaluate(q)['pcs_Lab']))
    if maxdiff!=0:raise AssertionError('bypass numeric mismatch')
    oldlc=audit.lcms_compare(d,pts);newlc=audit.lcms_compare(p,pts);lccheck={}
    for name in oldlc['intents']:
        x=oldlc['intents'][name];y=newlc['intents'][name]
        err=max(abs(a-b) for xx,yy in zip(x['pcs_Lab'],y['pcs_Lab']) for a,b in zip(xx,yy))
        lccheck[name]={'original_open':x['transform_created'],'candidate_open':y['transform_created'],'max_abs_Lab_component_difference':err}
        if err!=0:raise AssertionError('CMM bypass mismatch')
    identity=identity_test()
    # Only after every in-memory test passes, create one new file in deliverables.
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with DEST.open('xb') as f:f.write(p)
    after=audit.sha(SOURCE.read_bytes())
    if before!=after:raise AssertionError('source changed')
    r={'candidate':DEST.name,'description':'LocalLooks-Bypass','candidate_bytes':len(p),'candidate_sha256':audit.sha(p),
       'source':SOURCE.name,'source_sha256_before':before,'source_sha256_after':after,'source_unchanged':before==after,
       'purpose':'Research bypass only; not official Leica profile; NOT installed or tested in Capture One.',
       'serialization':{'description_changed_only_among_tag_payloads':True,'header_4_127_exact':True,'header_size_old':len(d),'header_size_new':len(p),
       'header_changed_byte_offsets':[i for i in range(128) if d[i]!=p[i]],'v2_reserved_84_127_preserved':d[84:128]==p[84:128],
       'v4_profile_id_not_inserted':True,'tag_count_unchanged':len(source_tags)==len(new_tags),
       'tag_payloads':differences,'new_description_parsed':audit.audit_desc(new_tags[b'desc'])},
       'numeric':{'samples':len(pts),'original_vs_bypass_max_abs_Lab_difference':maxdiff,'non_description_tag_byte_identity_proves_same_stored_transform':True,
       'lcms_original_vs_bypass':lccheck,'synthetic_identity_test':identity},
       'still_required':'Actual C1 original Generic+Auto vs bypass+Auto, with cloned same RAW and identical other settings; Auto resolution is not assumed invariant.'}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'candidate':r['candidate'],'sha256':r['candidate_sha256'],'bytes':len(p),'numeric_maxdiff':maxdiff,'report':str(REPORT)}))

if __name__=='__main__':main()
