#!/usr/bin/env python3
"""Validate independent ICC wrappers against direct CUBE mathematics and FFmpeg.

Uses deterministic random samples, original LUT nodes, gray/dark ramps and RGB
faces. Reports source-profile -> standard sRGB and PCS Lab errors. No C1 engine
is invoked; passing this test is NOT Capture One host certification.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
if __package__:
    from .color_math import Cube,LCMS,TYPE_RGB_16,TYPE_LAB_DBL
else:
    from color_math import Cube,LCMS,TYPE_RGB_16,TYPE_LAB_DBL


def de2000(lab1,lab2):
    L1,a1,b1=np.asarray(lab1,dtype=np.float64).T
    L2,a2,b2=np.asarray(lab2,dtype=np.float64).T
    C1=np.hypot(a1,b1);C2=np.hypot(a2,b2);Cbar=(C1+C2)/2
    G=0.5*(1-np.sqrt(Cbar**7/(Cbar**7+25.0**7)))
    ap1=(1+G)*a1;ap2=(1+G)*a2
    cp1=np.hypot(ap1,b1);cp2=np.hypot(ap2,b2)
    hp1=np.mod(np.degrees(np.arctan2(b1,ap1)),360)
    hp2=np.mod(np.degrees(np.arctan2(b2,ap2)),360)
    product=cp1*cp2
    dh=hp2-hp1
    dh=np.where(dh>180,dh-360,np.where(dh<-180,dh+360,dh))
    dh=np.where(product==0,0,dh)
    dH=2*np.sqrt(product)*np.sin(np.radians(dh/2))
    dL=L2-L1;dC=cp2-cp1;lb=(L1+L2)/2;cb=(cp1+cp2)/2
    hb=(hp1+hp2)/2
    hb=np.where((product!=0)&(np.abs(hp1-hp2)>180),
                np.where(hp1+hp2<360,(hp1+hp2+360)/2,(hp1+hp2-360)/2),hb)
    hb=np.where(product==0,hp1+hp2,hb)
    T=1-0.17*np.cos(np.radians(hb-30))+0.24*np.cos(np.radians(2*hb))+0.32*np.cos(np.radians(3*hb+6))-0.20*np.cos(np.radians(4*hb-63))
    sl=1+0.015*(lb-50)**2/np.sqrt(20+(lb-50)**2)
    sc=1+0.045*cb;sh=1+0.015*cb*T
    rt=-2*np.sqrt(cb**7/(cb**7+25.0**7))*np.sin(np.radians(60*np.exp(-((hb-275)/25)**2)))
    dl=dL/sl;dc=dC/sc;dH=dH/sh
    return np.sqrt(np.maximum(0,dl*dl+dc*dc+dH*dH+rt*dc*dH))


def stats(diff):
    x=np.asarray(diff,dtype=np.float64)
    return {'mean_abs':float(np.abs(x).mean()),'rmse':float(np.sqrt(np.mean(x*x))),
            'p99_abs':float(np.percentile(np.abs(x),99)),'max_abs':float(np.max(np.abs(x)))}


def structure(path):
    raw=path.read_bytes()
    if raw[36:40]!=b'acsp' or struct.unpack_from('>I',raw,0)[0]!=len(raw):
        raise ValueError('ICC signature/declared length mismatch')
    if raw[8:12]!=b'\x02\x10\0\0' or raw[12:24]!=b'scnrRGB Lab ':
        raise ValueError('ICC profile contract header mismatch')
    count=struct.unpack_from('>I',raw,128)[0]
    tags={}
    for i in range(count):
        sig,offset,size=struct.unpack_from('>4sII',raw,132+12*i)
        if offset%4 or offset+size>len(raw) or offset<132+12*count:
            raise ValueError('Invalid tag offset/size')
        tags[sig.decode()]={'offset':offset,'bytes':size,'type':raw[offset:offset+4].decode()}
    if set(tags)!={'desc','cprt','wtpt','A2B0','A2B1','A2B2'}:
        raise ValueError('Unexpected tag set')
    if not tags['A2B0']==tags['A2B1']==tags['A2B2']:
        raise ValueError('Rendering intent tables differ')
    off=tags['A2B0']['offset']; n=raw[off+10]
    if raw[off:off+4]!=b'mft2' or raw[off+8:off+10]!=b'\x03\x03':
        raise ValueError('Invalid LUT16 channels/type')
    if tags['A2B0']['bytes']!=52+12+n**3*6+12:
        raise ValueError('Incorrect CLUT size')
    return {'bytes':len(raw),'grid':n,'tag_structure_valid':True,'tags':tags}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest',type=Path)
    ap.add_argument('cube',type=Path)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    # Known Sharma test pair, deltaE00=2.0425.
    test=de2000(np.array([[50,2.6772,-79.7751]]),np.array([[50,0,-82.7485]]))[0]
    if abs(test-2.0425)>0.0001:raise AssertionError('CIEDE2000 implementation failed standard sample')
    cube=Cube(args.cube);rng=np.random.default_rng(20261001)
    axis=np.linspace(0,1,cube.n)
    nodes=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    face_axis=np.linspace(0,1,65)
    uv=np.stack(np.meshgrid(face_axis,face_axis,indexing='ij'),axis=-1).reshape(-1,2)
    faces=[]
    for fixed_axis in range(3):
        for value in [0,1]:
            face=np.empty((len(uv),3));face[:,fixed_axis]=value
            face[:,[x for x in range(3) if x!=fixed_axis]]=uv;faces.append(face)
    groups={'uniform_random':rng.random((200_000,3)),
            'gray_ramp':np.repeat(np.linspace(0,1,4097)[:,None],3,axis=1),
            'dark_random':rng.random((20000,3))*0.08,
            'original_cube_nodes':nodes,'rgb_faces':np.concatenate(faces)}
    # Make both reference and CMM samples exactly representable as 16-bit inputs.
    groups={k:np.floor(v*65535+0.5)/65535 for k,v in groups.items()}
    manifest=json.loads(args.manifest.read_text())
    result={'sample_seed':20261001,'sample_counts':{k:len(v) for k,v in groups.items()},
            'profiles':[],'c1_executed':False,'black_point_compensation':False,
            'validation_scope':'Standard ICC input interpretation -> ordinary sRGB / D50 Lab, not C1 RAW pipeline'}
    # Isolate the LUT from FFmpeg/swscale packed RGB48 <-> planar float
    # conversion. That conversion itself was observed to introduce tens of
    # uint16 code values, despite an identity (no-LUT) round trip.
    ff_input=(groups['uniform_random'][:16384]*65535).round().astype('<u2')
    ff_x=ff_input.astype(np.float64)/65535
    planar=np.stack([ff_x[:,1],ff_x[:,2],ff_x[:,0]]).astype('<f4')  # G,B,R
    base=['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-f','rawvideo']
    p=subprocess.run(base+['-pixel_format','gbrpf32le','-video_size','256x64','-i','pipe:0',
        '-vf',f'lut3d=file={args.cube.resolve()}:interp=tetrahedral','-frames:v','1',
        '-pix_fmt','gbrpf32le','-f','rawvideo','-threads','1','pipe:1'],input=planar.tobytes(),capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.decode())
    ff=np.frombuffer(p.stdout,dtype='<f4').reshape(3,-1)[[2,0,1]].T
    direct=np.clip(cube.evaluate(ff_x),0,1)
    ff_stats=stats((ff.astype(np.float64)-direct)*65535)
    if ff_stats['max_abs']>0.1:raise AssertionError(f'Isolated CUBE math differs from FFmpeg: {ff_stats}')
    result['cube_math_vs_isolated_ffmpeg_u16_equivalent']=ff_stats
    p=subprocess.run(base+['-pixel_format','rgb48le','-video_size','256x64','-i','pipe:0',
        '-vf','format=gbrpf32le','-frames:v','1','-pix_fmt','rgb48le','-f','rawvideo',
        '-threads','1','pipe:1'],input=ff_input.tobytes(),capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr.decode())
    roundtrip=np.frombuffer(p.stdout,dtype='<u2').reshape(-1,3)
    result['ffmpeg_format_only_roundtrip_error_u16']=stats(roundtrip.astype(np.float64)-ff_input)
    with LCMS() as cms:
        result['lcms_version']=cms.lib.cmsGetEncodedCMMversion()
        srgb,lab=cms.srgb(),cms.lab()
        s_to_lab=cms.transform(srgb,lab,output_format=TYPE_LAB_DBL)
        for profile in manifest['profiles']:
            path=Path(profile['path']);info=structure(path);handle=cms.open(path)
            to_rgb=cms.transform(handle,srgb)
            to_lab=cms.transform(handle,lab,output_format=TYPE_LAB_DBL)
            to_rgb16=cms.transform(handle,srgb,input_format=TYPE_RGB_16,output_format=TYPE_RGB_16)
            alpha=profile['strength']
            record={'profile':str(path),'grid':profile['grid_size'],'strength':alpha,'structure':info,'groups':{}}
            for name,x in groups.items():
                expected=np.clip((1-alpha)*x+alpha*cube.evaluate(x),0,1)
                out=cms.apply(to_rgb,x)
                actual_lab=cms.apply(to_lab,x);expected_lab=cms.apply(s_to_lab,expected)
                de=de2000(actual_lab,expected_lab)
                out16=cms.apply(to_rgb16,np.floor(x*65535+0.5).astype(np.uint16))
                record['groups'][name]={
                    'rgb_code_error_8bit_equivalent':stats((np.clip(out,0,1)-expected)*255),
                    'rgb16_buffer_error_u16':stats(out16.astype(np.float64)-expected*65535),
                    'deltaE00_D50':{'mean':float(de.mean()),'p99':float(np.percentile(de,99)),'max':float(de.max())},
                    'cms_float_output_range':[float(out.min()),float(out.max())]}
            # Also check default optimization and rendering-intent selection.
            short=groups['uniform_random'][:20000]
            optimized=cms.transform(handle,srgb,flags=0)
            perceptual=cms.transform(handle,srgb,intent=0)
            record['default_optimization_difference_8bit']=stats((cms.apply(optimized,short)-cms.apply(to_rgb,short))*255)
            record['perceptual_vs_relative_difference_8bit']=stats((cms.apply(perceptual,short)-cms.apply(to_rgb,short))*255)
            result['profiles'].append(record)
            g=record['groups']['uniform_random']
            print(path.name,json.dumps({'rgb8':g['rgb_code_error_8bit_equivalent'],'de00':g['deltaE00_D50']}),flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
