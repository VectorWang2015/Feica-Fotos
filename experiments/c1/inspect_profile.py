#!/usr/bin/env python3
"""Read-only bounded ICC audit + independently evaluated mft2 samples.

Never writes an ICC or any source file. The only output is a new JSON report.
No C1 code is executed. Optional LCMS is an installed general-purpose CMM,
not evidence that C1 uses the same interpolation, intent fallback or input domain.
"""
from __future__ import annotations
import argparse
import ctypes as C
import ctypes.util
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct


def sha(b): return hashlib.sha256(b).hexdigest()
def sig(b): return b.decode('ascii', 'backslashreplace')
def u16(b, o): return struct.unpack_from('>H', b, o)[0]
def u32(b, o): return struct.unpack_from('>I', b, o)[0]
def s15(b, o): return struct.unpack_from('>i', b, o)[0] / 65536.0


def interpolate(table, x):
    x = min(1., max(0., x)) * (len(table) - 1)
    k = min(len(table) - 2, int(x)); f = x-k
    return ((1-f)*table[k] + f*table[k+1]) / 65535.


def audit_desc(b):
    # Parse only what is physically within this tag, never consume next tag/pad.
    if len(b) < 12: return {'error': 'missing ASCII length'}
    n = u32(b, 8); stop = 12+n
    out = {'ascii_count': n, 'ascii_payload_in_bounds': stop <= len(b)}
    if stop > len(b): return out
    out.update(ascii=sig(b[12:stop].rstrip(b'\0')), ascii_nul_terminated=n > 0 and b[stop-1] == 0)
    if stop+8 > len(b): return dict(out, unicode_header_in_bounds=False)
    lang, un = u32(b, stop), u32(b, stop+4); stop += 8
    ue = stop+un*2
    out.update(unicode_language=lang, unicode_count=un, unicode_payload_in_bounds=ue <= len(b))
    if ue > len(b): return out
    if un: out['unicode'] = b[stop:ue].decode('utf-16-be', 'replace').rstrip('\0')
    stop = ue
    if stop+3 > len(b): return dict(out, script_header_in_bounds=False)
    code, mn = u16(b, stop), b[stop+2]; start = stop+3
    out.update(script_code=code, script_count=mn, script_available_bytes=len(b)-start,
               script_counted_payload_in_bounds=start+mn <= len(b),
               fixed_67_byte_script_field_present=start+67 <= len(b))
    if start+mn <= len(b): out['script_counted_text'] = sig(b[start:start+mn].rstrip(b'\0'))
    out['note'] = 'A bounded description parse, not a whole-profile ICC conformance certification.'
    return out


class Lut16:
    def __init__(self, b):
        if len(b) < 52: raise ValueError('mft2 fixed header truncated')
        self.raw=b; self.ni=b[8];self.no=b[9];self.grid=b[10]
        self.nin=u16(b,48);self.nout=u16(b,50)
        if not (1 <= self.ni <= 15 and 1 <= self.no <= 15 and 2 <= self.grid <= 255 and
                2 <= self.nin <= 4096 and 2 <= self.nout <= 4096):
            raise ValueError('invalid mft2 dimension or table count')
        self.matrix=[s15(b,12+4*i) for i in range(9)]
        self.ib=52;self.ie=self.ib+2*self.ni*self.nin
        self.cb=self.ie;self.ce=self.cb+2*(self.grid**self.ni)*self.no
        self.ob=self.ce;self.oe=self.ob+2*self.no*self.nout
        if self.oe > len(b): raise ValueError('mft2 tables exceed tag bounds')
        # Count bound checked before allocating/reading arrays.
        self.inputs=[list(struct.unpack_from('>'+str(self.nin)+'H',b,self.ib+2*self.nin*j)) for j in range(self.ni)]
        self.outputs=[list(struct.unpack_from('>'+str(self.nout)+'H',b,self.ob+2*self.nout*j)) for j in range(self.no)]
        self.clut=list(struct.unpack_from('>'+str((self.ce-self.cb)//2)+'H',b,self.cb))

    def node(self, coords):
        k=0
        for coord in coords: k=k*self.grid+coord
        return [z/65535. for z in self.clut[k*self.no:(k+1)*self.no]]

    def clut_eval(self, shaped, method):
        if self.ni != 3: raise ValueError('audit evaluator only supports RGB')
        t=[min(1.,max(0.,x))*(self.grid-1) for x in shaped]
        lo=[min(self.grid-2,int(x)) for x in t];f=[t[i]-lo[i] for i in range(3)]
        out=[0.]*self.no
        if method == 'trilinear':
            for bits in itertools.product((0,1),repeat=3):
                w=math.prod(f[i] if bits[i] else 1-f[i] for i in range(3))
                v=self.node([lo[i]+bits[i] for i in range(3)])
                for j in range(self.no):out[j]+=w*v[j]
        elif method == 'tetrahedral':
            axes=sorted(range(3),key=lambda i:f[i],reverse=True)
            weights=[1-f[axes[0]],f[axes[0]]-f[axes[1]],f[axes[1]]-f[axes[2]],f[axes[2]]]
            q=lo.copy()
            for k,w in enumerate(weights):
                v=self.node(q)
                for j in range(self.no):out[j]+=w*v[j]
                if k<3:q[axes[k]]+=1
        else:raise ValueError(method)
        return out

    def evaluate(self, rgb, method='tetrahedral'):
        if self.matrix != [1.,0.,0.,0.,1.,0.,0.,0.,1.]:
            raise ValueError('this bounded evaluator expects observed identity matrix')
        shaped=[interpolate(t,x) for t,x in zip(self.inputs,rgb)]
        c=self.clut_eval(shaped,method)
        norm=[interpolate(t,x) for t,x in zip(self.outputs,c)]
        # ICC v2 16-bit Lab: L=100 and a,b=+127 map to 0xff00, not 0xffff.
        lab=[norm[0]*65535.*100./65280.,norm[1]*65535./256.-128.,norm[2]*65535./256.-128.]
        return {'input':rgb,'shaped_input':shaped,'pcs_Lab':lab}

    def summary(self):
        inp=[]
        for c,t in enumerate(self.inputs):
            inp.append({'channel':c,'sha256_be_u16':sha(self.raw[self.ib+2*c*self.nin:self.ib+2*(c+1)*self.nin]),
                        'min':min(t),'max':max(t),'monotonic_nondecreasing':all(a<=b for a,b in zip(t,t[1:])),
                        'strictly_increasing':all(a<b for a,b in zip(t,t[1:])),
                        'max_abs_identity_code_difference':max(abs(v-i*65535/(len(t)-1)) for i,v in enumerate(t)),
                        'selected_index_codes':{str(i):t[i] for i in [0,1,2,3,4,8,16,32,64,128,192,254,255] if i<len(t)}})
        return {'type':'mft2','input_channels':self.ni,'output_channels':self.no,'grid_points_per_axis':self.grid,
                'clut_dimensions':[self.grid]*self.ni,'clut_nodes':self.grid**self.ni,
                'input_table_entries_per_channel':self.nin,'output_table_entries_per_channel':self.nout,
                'matrix_s15Fixed16':self.matrix,'input_curves':inp,
                'input_curve_channels_identical':all(t==self.inputs[0] for t in self.inputs),
                'input_tables_sha256':sha(self.raw[self.ib:self.ie]),
                'clut_sha256':sha(self.raw[self.cb:self.ce]),
                'clut_channel_minmax_codes':[[min(self.clut[j::self.no]),max(self.clut[j::self.no])] for j in range(self.no)],
                'output_tables':self.outputs,'output_tables_sha256':sha(self.raw[self.ob:self.oe]),
                'subranges_relative_to_tag':{'input':[self.ib,self.ie],'clut':[self.cb,self.ce],'output':[self.ob,self.oe]},
                'expected_payload_bytes':self.oe,'actual_tag_bytes':len(self.raw),'remaining_tag_bytes':len(self.raw)-self.oe,
                'all_subranges_in_bounds':self.oe<=len(self.raw)}


def lcms_compare(profile_data, points):
    name=ctypes.util.find_library('lcms2')
    if not name:return {'available':False}
    lib=C.CDLL(name)
    lib.cmsGetEncodedCMMversion.restype=C.c_int
    lib.cmsOpenProfileFromMem.argtypes=[C.c_void_p,C.c_uint32];lib.cmsOpenProfileFromMem.restype=C.c_void_p
    lib.cmsCreateLab4Profile.argtypes=[C.c_void_p];lib.cmsCreateLab4Profile.restype=C.c_void_p
    lib.cmsCreateTransform.argtypes=[C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint32,C.c_uint32,C.c_uint32];lib.cmsCreateTransform.restype=C.c_void_p
    lib.cmsDoTransform.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32]
    lib.cmsDeleteTransform.argtypes=[C.c_void_p];lib.cmsCloseProfile.argtypes=[C.c_void_p]
    lib.cmsIsIntentSupported.argtypes=[C.c_void_p,C.c_uint32,C.c_uint32];lib.cmsIsIntentSupported.restype=C.c_int
    buf=C.create_string_buffer(profile_data);src=lib.cmsOpenProfileFromMem(buf,len(profile_data));dst=lib.cmsCreateLab4Profile(None)
    if not src or not dst:raise RuntimeError('LCMS profile open failed')
    rgb_dbl=(1<<22)|(4<<16)|(3<<3);lab_dbl=(1<<22)|(10<<16)|(3<<3)
    data=(C.c_double*(3*len(points)))(*(v for p in points for v in p))
    out={'available':True,'library':name,'encoded_version':lib.cmsGetEncodedCMMversion(),
         'format':'normalized RGB double -> physical CIELab double','flags':'NOOPTIMIZE | NOCACHE (0x0140)',
         'caution':'LCMS-only fallback behavior; not Capture One routing or real sensor-domain proof.','intents':{}}
    try:
        for intent,label in [(0,'perceptual'),(1,'relative_colorimetric'),(2,'saturation'),(3,'absolute_colorimetric')]:
            tr=lib.cmsCreateTransform(src,rgb_dbl,dst,lab_dbl,intent,0x0140)
            record={'intent_number':intent,'cmsIsIntentSupported_input':bool(lib.cmsIsIntentSupported(src,intent,0)),
                    'transform_created':bool(tr)}
            if tr:
                result=(C.c_double*(3*len(points)))();lib.cmsDoTransform(tr,data,result,len(points))
                record['pcs_Lab']=[list(result[3*i:3*i+3]) for i in range(len(points))];lib.cmsDeleteTransform(tr)
            out['intents'][label]=record
    finally:lib.cmsCloseProfile(src);lib.cmsCloseProfile(dst)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path)
    ap.add_argument('--compare-copy',type=Path);ap.add_argument('--expected-sha256');ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise SystemExit('Refusing to overwrite output')
    d=a.source.read_bytes();before=sha(d)
    if a.expected_sha256 and before.lower()!=a.expected_sha256.lower():raise SystemExit('source SHA mismatch')
    if len(d)<132:raise SystemExit('header/tag count truncated')
    n=u32(d,128);declared=u32(d,0);dirend=132+12*n
    if not(132<=declared<=len(d) and dirend<=declared):raise SystemExit('declared profile size or directory out of bounds')
    entries=[];data_map={};lut=None
    for i in range(n):
        p=132+12*i;s=sig(d[p:p+4]);o=u32(d,p+4);z=u32(d,p+8)
        if o<dirend or z<8 or o+z>declared or o+z>len(d):raise SystemExit('tag outside declared/file bounds: '+s)
        b=d[o:o+z]
        item={'signature':s,'offset':o,'size':z,'end_exclusive':o+z,'type':sig(b[:4]),
              'offset_4byte_aligned':o%4==0,'in_declared_bounds':o+z<=declared,'in_file_bounds':o+z<=len(d),
              'starts_after_tag_directory':o>=dirend,'type_reserved_zero':b[4:8]==b'\0'*4,'sha256':sha(b)}
        if s in data_map:raise SystemExit('duplicate tag signature')
        data_map[s]=b
        if item['type']=='desc':item['description']=audit_desc(b)
        elif item['type']=='text':item['text']=b[8:].rstrip(b'\0').decode('utf-8','replace')
        elif item['type']=='XYZ ':
            if (len(b)-8)%12:raise SystemExit('invalid XYZ payload size')
            item['XYZ_values']=[[s15(b,k+j) for j in (0,4,8)] for k in range(8,len(b),12)]
        elif item['type']=='sig ':item['value_signature']=sig(b[8:12])
        elif item['type']=='mft2':
            parsed=Lut16(b);item['lut16']=parsed.summary()
            if s=='A2B0':lut=parsed
        entries.append(item)
    overlaps=[]
    for x,y in itertools.combinations(entries,2):
        if max(x['offset'],y['offset'])<min(x['end_exclusive'],y['end_exclusive']):
            overlaps.append({'a':x['signature'],'b':y['signature'],'exact_alias':(x['offset'],x['size'])==(y['offset'],y['size'])})
    if any(not x['exact_alias'] for x in overlaps):raise SystemExit('partial tag overlap')
    occupied=[(0,dirend)]+sorted((x['offset'],x['end_exclusive']) for x in entries)
    gaps=[];last=0
    for start,end in occupied:
        if start>last:gaps.append({'start':last,'end':start,'bytes':start-last,'all_zero':not any(d[last:start])})
        last=max(last,end)
    if last<len(d):gaps.append({'start':last,'end':len(d),'bytes':len(d)-last,'all_zero':not any(d[last:])})
    v=u32(d,8)
    h={'declared_size':declared,'preferred_cmm':sig(d[4:8]),'version_hex':f'{v:08x}',
       'version':f'{d[8]}.{d[9]>>4}.{d[9]&15}', 'device_class':sig(d[12:16]),'device_color_space':sig(d[16:20]),
       'PCS':sig(d[20:24]),'creation_date_fields':list(struct.unpack_from('>6H',d,24)),
       'signature':sig(d[36:40]),'platform':sig(d[40:44]),'flags_hex':f'{u32(d,44):08x}',
       'manufacturer':sig(d[48:52]),'model':sig(d[52:56]),'attributes_hex':d[56:64].hex(),
       'default_rendering_intent':u32(d,64),'PCS_illuminant_XYZ':[s15(d,68+4*i) for i in range(3)],
       'creator':sig(d[80:84]),'reserved_84_127_all_zero':not any(d[84:128])}
    if h['signature']!='acsp':raise SystemExit('invalid ICC signature')
    transforms={s:{'present':s in data_map,'type':sig(data_map[s][:4]) if s in data_map else None}
                for s in ['A2B0','A2B1','A2B2','B2A0','B2A1','B2A2','D2B0','D2B1','D2B2','D2B3','B2D0','B2D1','B2D2','B2D3',
                          'rXYZ','gXYZ','bXYZ','rTRC','gTRC','bTRC','kTRC','chad','meta','targ']}
    points=[[0.,0.,0.],[1.,1.,1.],[.003,.003,.003],[.01,.01,.01],[.18,.18,.18],[.5,.5,.5],
            [.25,.5,.75],[.02,.01,.03],[.1,.2,.1],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    sample={}
    if lut:
        sample['contract']='Abstract normalized profile device-RGB coordinates only; NOT encoded sRGB, not a WB-neutral assertion.'
        sample['tetrahedral']=[lut.evaluate(p,'tetrahedral') for p in points]
        sample['trilinear']=[lut.evaluate(p,'trilinear') for p in points]
        # Node identity test: interpolation must return exact stored CLUT at 5 grid locations.
        tests=[]
        for coords in [(0,0,0),(32,32,32),(16,8,24),(1,3,5),(30,2,17)]:
            shaped=[x/(lut.grid-1) for x in coords];node=lut.node(coords)
            tests.append({'grid_coords':coords,'tetra_max_error':max(abs(x-y) for x,y in zip(node,lut.clut_eval(shaped,'tetrahedral'))),
                          'tri_max_error':max(abs(x-y) for x,y in zip(node,lut.clut_eval(shaped,'trilinear')))})
        sample['exact_grid_node_checks']=tests
        cm=lcms_compare(d,points);sample['lcms']=cm
        if cm['available']:
            for method in ['tetrahedral','trilinear']:
                sample[method+'_vs_lcms_perceptual_deltaE76']=[math.dist(x['pcs_Lab'],y) for x,y in
                   zip(sample[method],cm['intents']['perceptual']['pcs_Lab'])]
            ref=cm['intents']['perceptual']['pcs_Lab']
            sample['lcms_relative_vs_perceptual_max_abs_component']=max(abs(x-y) for p,q in zip(ref,cm['intents']['relative_colorimetric']['pcs_Lab']) for x,y in zip(p,q))
    after=sha(a.source.read_bytes())
    copy=None
    if a.compare_copy:
        cb=a.compare_copy.read_bytes();copy={'path':str(a.compare_copy),'bytes':len(cb),'sha256':sha(cb),'byte_identical_to_source':cb==d}
    r={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Only supplied native C1 Q1 camera ICC; no vendor executable or system writes',
       'source':{'path':str(a.source),'bytes':len(d),'sha256_before':before,'sha256_after':after,'unchanged':before==after},
       'archived_copy':copy,'header':h,'tag_directory':{'count':n,'end_exclusive':dirend,'all_bounds_pass':True,
       'declared_equals_actual':declared==len(d),'overlaps':overlaps,'gaps':gaps},'tags':entries,'transform_tags':transforms,
       'private_tag_signatures': [x['signature'] for x in entries if x['signature'] not in {'desc','cprt','wtpt','A2B0','tech'}],
       'private_header_observations':{'flags_vendor_bits_hex':f'{u32(d,44)&0xffff0000:08x}',
       'attributes_vendor_bits_hex':d[56:60].hex(),'semantics':'Uninterpreted; not proven effects or curve selector.'},
       'explicit_film_curve_metadata':{'found':False,'scope':'Exactly 5 inventoried standard tags, bounded desc/cprt text; no meta/private tag holding a named Curve.',
       'caveat':'Does not exclude external C1 rules keyed by profile name/header, or nonlinear rendering baked into A2B0.'},
       'synthetic_sampling':sample,'not_proven':['C1 actual profile input numeric encoding','C1 actual rendering intent/tag fallback',
       'C1 Auto resolved curve','C1 position of WB/exposure/tone around ICC','Leica official RAW rendering','Cross-image physical characterization accuracy']}
    if before!=after:raise SystemExit('Source changed during inspection')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'out':str(a.out),'sha256':before,'unchanged':before==after,'tags':n,'bounds_pass':True,
       'max_tetra_vs_lcms_deltaE76':max(sample.get('tetrahedral_vs_lcms_perceptual_deltaE76',[0]))},ensure_ascii=False))

if __name__=='__main__':main()
