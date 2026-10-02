#!/usr/bin/env python3
"""Minimal uncompressed RGB uint16 TIFF I/O for numerical test files.

Pillow is used only for metadata. Pixel arrays are read/written directly to avoid
Pillow's regular RGB path reducing precision. Writer makes a new one-page TIFF,
embeds an explicit ICC, and does not claim to preserve source capture metadata.
"""
from pathlib import Path
import struct
import sys
import numpy as np
from PIL import Image


def read_rgb16(path):
    path=Path(path)
    with Image.open(path) as im:
        tags=dict(im.tag_v2);w,h=im.size
        if im.format!='TIFF' or tuple(tags.get(258,()))!=(16,16,16) or tags.get(259)!=1 or tags.get(262)!=2 or tags.get(277)!=3 or tags.get(284,1)!=1:
            raise ValueError('Expected uncompressed contiguous RGB16 TIFF')
        if tags.get(274,1)!=1:raise ValueError('Normalize orientation before reading')
        icc=im.info.get('icc_profile',b'')
        offsets,counts=tags[273],tags[279];rps=tags[278]
        frames=getattr(im,'n_frames',1)
    pixels=np.empty((h,w,3),dtype=np.uint16)
    with path.open('rb') as f:
        byteorder=f.read(2)
        if byteorder not in (b'II',b'MM'):raise ValueError('Invalid TIFF byte order')
        dtype=np.dtype('<u2' if byteorder==b'II' else '>u2')
        y=0
        for off,count in zip(offsets,counts):
            rows=min(rps,h-y)
            required=rows*w*6
            # Some libtiff writers retain a full padded last strip. Read every
            # byte for truncation checks, but only decode rows inside the image.
            if rows<=0 or count<required or count>rps*w*6 or (y+rows<h and count!=required):
                raise ValueError('Unexpected TIFF strip size')
            f.seek(off);block=f.read(count)
            if len(block)!=count:raise ValueError('Short TIFF strip')
            pixels[y:y+rows]=np.frombuffer(block[:required],dtype=dtype).reshape(rows,w,3)
            y+=rows
        if y!=h:raise ValueError('Incomplete TIFF raster')
    return pixels,{'icc':icc,'software':tags.get(305),'frames':frames,'size':[w,h]}


def write_rgb16(path,pixels,icc,description='Local color experiment. Not a RAW capture.'):
    path=Path(path)
    a=np.asarray(pixels)
    if a.dtype!=np.uint16 or a.ndim!=3 or a.shape[2]!=3:raise ValueError('Expected HxWx3 uint16')
    h,w,_=a.shape
    if not icc:raise ValueError('An explicit output ICC is required')
    software=b'Leica FOTOS local interoperability study\0'
    desc=description.encode('ascii')+b'\0'
    # tag: (TIFF type, element count, raw little-endian payload)
    tags={256:(4,1,struct.pack('<I',w)),257:(4,1,struct.pack('<I',h)),
          258:(3,3,struct.pack('<3H',16,16,16)),259:(3,1,struct.pack('<H',1)),
          262:(3,1,struct.pack('<H',2)),270:(2,len(desc),desc),
          273:(4,1,bytes(4)),274:(3,1,struct.pack('<H',1)),277:(3,1,struct.pack('<H',3)),
          278:(4,1,struct.pack('<I',h)),279:(4,1,struct.pack('<I',w*h*6)),
          282:(5,1,struct.pack('<II',300,1)),283:(5,1,struct.pack('<II',300,1)),
          284:(3,1,struct.pack('<H',1)),296:(3,1,struct.pack('<H',2)),
          305:(2,len(software),software),339:(3,3,struct.pack('<3H',1,1,1)),
          34675:(7,len(icc),icc)}
    count=len(tags);header_len=8+2+12*count+4
    external=bytearray();locations={}
    for tag,(typ,n,data) in sorted(tags.items()):
        if len(data)>4:
            locations[tag]=header_len+len(external)
            external.extend(data);external.extend(bytes((-len(external))%4))
    raster_offset=header_len+len(external)
    tags[273]=(4,1,struct.pack('<I',raster_offset))
    directory=bytearray(struct.pack('<H',count))
    for tag,(typ,n,data) in sorted(tags.items()):
        value=data+bytes(4-len(data)) if len(data)<=4 else struct.pack('<I',locations[tag])
        directory.extend(struct.pack('<HHI',tag,typ,n)+value)
    directory.extend(bytes(4))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(b'II'+struct.pack('<HI',42,8));f.write(directory);f.write(external)
        for y in range(0,h,256):f.write(np.ascontiguousarray(a[y:y+256],dtype='<u2').tobytes())
    if path.stat().st_size!=raster_offset+w*h*6:raise AssertionError('Written TIFF length mismatch')
