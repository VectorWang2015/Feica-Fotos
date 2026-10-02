#!/usr/bin/env python3
"""Independently wrap a CUBE as a rendered-sRGB creative INPUT ICC (v2.1).

Contract: A2B(x)=Lab_D50(sRGB_decode((1-a)*x+a*CUBE(x))). x is already
rendered sRGB code values. NOT a camera characterization profile, NOT suitable
for unverified RAW Base Characteristics replacement. No RNI data are used.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import struct
import uuid
import xml.etree.ElementTree as ET
import numpy as np
if __package__:
    from .color_math import Cube,LCMS,TYPE_LAB_DBL,encode_lab_v2
else:
    from color_math import Cube,LCMS,TYPE_LAB_DBL,encode_lab_v2


def s15(value):
    return int(np.floor(value*65536+0.5))


def tag_text(text):
    return b'text'+bytes(4)+text.encode('ascii')+b'\0'


def tag_desc(text):
    data=text.encode('ascii')+b'\0'
    return (b'desc'+bytes(4)+struct.pack('>I',len(data))+data+
            struct.pack('>IIHB',0,0,0,0)+bytes(67))


def tag_xyz(xyz):
    return b'XYZ '+bytes(4)+struct.pack('>3i',*(s15(x) for x in xyz))


def profile_bytes(lab_grid, description):
    n=lab_grid.shape[0]
    if lab_grid.shape!=(n,n,n,3) or not 2<=n<=255:
        raise ValueError('Invalid CLUT grid')
    matrix=(65536,0,0,0,65536,0,0,0,65536)
    identity=struct.pack('>6H',0,65535,0,65535,0,65535)
    encoded=encode_lab_v2(lab_grid).astype('>u2').tobytes(order='C')
    # C order for [R,G,B,channel]: first input varies slowest, as ICC requires.
    mft2=(b'mft2'+bytes(4)+bytes([3,3,n,0])+struct.pack('>9i',*matrix)+
          struct.pack('>HH',2,2)+identity+encoded+identity)
    tags=[(b'desc',tag_desc(description)),
          (b'cprt',tag_text('Independent research wrapper; underlying Leica Look data copyright Leica Camera AG. Personal interoperability study.')),
          (b'wtpt',tag_xyz((0.9642,1.0,0.8249))),
          (b'A2B0',mft2),(b'A2B1',mft2),(b'A2B2',mft2)]
    header=bytearray(128)
    header[8:12]=struct.pack('>I',0x02100000)
    header[12:24]=b'scnrRGB Lab '
    header[24:36]=struct.pack('>6H',2026,10,1,0,0,0)
    header[36:40]=b'acsp';header[40:44]=b'MSFT'
    header[64:68]=struct.pack('>I',1)  # relative colorimetric default
    header[68:80]=struct.pack('>3i',s15(0.9642),65536,s15(0.8249))
    header[80:84]=b'LFRS'  # research wrapper, not Leica manufacturer identity
    offset=128+4+12*len(tags)
    table,blocks=bytearray(),bytearray()
    locations={}
    for signature,block in tags:
        key=hashlib.sha256(block).digest()
        if key in locations:
            position,size=locations[key]
        else:
            position,size=offset+len(blocks),len(block)
            locations[key]=(position,size)
            blocks.extend(block);blocks.extend(bytes((-len(block))%4))
        table.extend(signature+struct.pack('>II',position,size))
    result=header+struct.pack('>I',len(tags))+table+blocks
    result[:4]=struct.pack('>I',len(result))
    return bytes(result)


def build(cube_path, grid, alpha, out_dir, cms, srgb_to_lab):
    cube=Cube(cube_path)
    axis=np.linspace(0,1,grid,dtype=np.float64)
    r,g,b=np.meshgrid(axis,axis,axis,indexing='ij')
    coords=np.stack((r,g,b),axis=-1)
    mapped=cube.evaluate(coords,interpolation='tetrahedral')
    blended=(1-alpha)*coords+alpha*mapped
    outside=int(np.count_nonzero((blended<0)|(blended>1)))
    blended=np.clip(blended,0,1)
    lab=cms.apply(srgb_to_lab,blended)
    strength=f'{round(alpha*100):03d}'
    description=f'EXP TIFF sRGB ONLY Eternal S{strength} G{grid} - NOT RAW'
    stem=f'EXP_Rendered_sRGB_Eternal_S{strength}_G{grid}_v21'
    raw=profile_bytes(lab,description)
    path=out_dir/(stem+'.icc')
    with path.open('xb') as f:f.write(raw)
    opened=cms.open(raw)
    info={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
          'grid_size':grid,'strength':alpha,'strength_definition':'linear blend of encoded sRGB input and LUT result',
          'cube_interpolation':'tetrahedral','profile_pcs':'D50 Lab','profile_version':cms.lib.cmsGetProfileVersion(opened),
          'profile_class':int(cms.lib.cmsGetDeviceClass(opened)).to_bytes(4,'big').decode('ascii'),
          'input_space':int(cms.lib.cmsGetColorSpace(opened)).to_bytes(4,'big').decode('ascii'),
          'pcs_signature':int(cms.lib.cmsGetPCS(opened)).to_bytes(4,'big').decode('ascii'),
          'clamped_grid_channel_values':outside,'camera_profile_included':False,'c1_host_tested':False,
          'source_cube_sha256':hashlib.sha256(Path(cube_path).read_bytes()).hexdigest()}
    return info


def write_style(info,out_dir):
    profile=Path(info['path'])
    pct=round(info['strength']*100)
    name=f'EXPERIMENT - TIFF sRGB ONLY - Eternal {pct}% - NOT RAW'
    root=ET.Element('SL',{'Engine':'1100'})
    for key,value in [('ICCProfile',profile.name),('Name',name),('StyleSource','Styles'),
                      ('UUID','{'+str(uuid.uuid5(uuid.NAMESPACE_URL,'leica-fotos-local-research/'+profile.name)).upper()+'}')]:
        ET.SubElement(root,'E',{'K':key,'V':value})
    ET.indent(root,space=' ')
    path=out_dir/(profile.stem+'.costyle')
    with path.open('xb') as f:f.write(ET.tostring(root,encoding='utf-8',xml_declaration=True)+b'\n')
    return str(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cube',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--grids',type=int,nargs='+',default=[33,65])
    ap.add_argument('--strengths',type=float,nargs='+',default=[0,1])
    ap.add_argument('--styles',action='store_true',help='Emit explicitly experimental rendered-input-only styles')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if any(not 0<=a<=1 for a in args.strengths):raise ValueError('Strength outside [0,1]')
    args.out.mkdir(parents=True)
    with LCMS() as cms:
        standard_to_lab=cms.transform(cms.srgb(),cms.lab(),output_format=TYPE_LAB_DBL)
        profiles=[]
        for grid in args.grids:
            for alpha in args.strengths:
                info=build(args.cube,grid,alpha,args.out,cms,standard_to_lab)
                if args.styles:info['style']=write_style(info,args.out)
                profiles.append(info)
                print(info['path'],info['bytes'],flush=True)
        report={'generated_utc':datetime.now(timezone.utc).isoformat(),
                'littlecms_encoded_version':cms.lib.cmsGetEncodedCMMversion(),
                'contract':'encoded rendered sRGB x -> RGBmix(x,CUBE(x)) -> standard D50 Lab PCS',
                'all_profiles_are_input_only':True,'all_profiles_are_experimental_not_raw':True,
                'no_rni_color_data_used':True,'profiles':profiles,
                'limitations':['Not a Q1 Generic replacement','No live Capture One host validation',
                  'C1 style only selects ICC; it does not disable or configure any other image adjustments',
                  'Do not choose these input profiles as export destination profiles']}
    (args.out/'build-manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
