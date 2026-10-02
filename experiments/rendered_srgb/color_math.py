#!/usr/bin/env python3
"""Small explicit CUBE and LittleCMS numerical helpers for this study.

RGB arguments are normalized encoded values, not implicitly linear-light RGB.
No vendor executable or commercial profile is loaded. LittleCMS runtime is used
only for ordinary profile transforms, with explicit float buffer formats.
"""
from pathlib import Path
import ctypes as C
import ctypes.util
import numpy as np

TYPE_RGB_DBL = (1 << 22) | (4 << 16) | (3 << 3)
TYPE_LAB_DBL = (1 << 22) | (10 << 16) | (3 << 3)
TYPE_RGB_16 = (4 << 16) | (3 << 3) | 2
FLAGS_NOOPTIMIZE_NOCACHE = 0x0100 | 0x0040


class Lab(C.Structure):
    _fields_ = [('L', C.c_double), ('a', C.c_double), ('b', C.c_double)]


class Cube:
    def __init__(self, path):
        self.path = Path(path)
        rows, sizes = [], []
        self.domain_min = np.zeros(3)
        self.domain_max = np.ones(3)
        for line in self.path.read_text(encoding='utf-8-sig').splitlines():
            line = line.split('#',1)[0].strip()
            if not line:
                continue
            p = line.split()
            if p[0] == 'TITLE':
                continue
            if p[0] == 'LUT_3D_SIZE':
                sizes.append(int(p[1])); continue
            if p[0] == 'DOMAIN_MIN':
                self.domain_min = np.array(p[1:], dtype=np.float64); continue
            if p[0] == 'DOMAIN_MAX':
                self.domain_max = np.array(p[1:], dtype=np.float64); continue
            if len(p) != 3:
                raise ValueError('Unexpected CUBE record')
            rows.append([float(v) for v in p])
        if not sizes or len(set(sizes)) != 1:
            raise ValueError('Missing/inconsistent LUT size')
        self.n = sizes[0]
        values = np.array(rows,dtype=np.float64)
        if values.shape != (self.n**3,3) or not np.isfinite(values).all():
            raise ValueError('Invalid CUBE grid')
        if self.domain_min.shape != (3,) or self.domain_max.shape != (3,) or np.any(self.domain_max <= self.domain_min):
            raise ValueError('Invalid CUBE domain')
        self.data = values.reshape(self.n,self.n,self.n,3)  # B,G,R,output
        self.output_min = values.min(axis=0)
        self.output_max = values.max(axis=0)

    def evaluate(self, rgb, interpolation='tetrahedral'):
        original_shape = np.asarray(rgb).shape
        x = np.asarray(rgb,dtype=np.float64).reshape(-1,3)
        scaled = np.clip((x-self.domain_min)/(self.domain_max-self.domain_min),0,1)*(self.n-1)
        lo = np.minimum(np.floor(scaled).astype(np.int32),self.n-2)
        f = scaled-lo
        def at(coords):
            return self.data[coords[:,2],coords[:,1],coords[:,0]]
        if interpolation == 'tetrahedral':
            order = np.argsort(-f,axis=1,kind='stable')
            fs = np.take_along_axis(f,order,axis=1)
            step1 = np.zeros_like(lo); step1[np.arange(len(x)),order[:,0]] = 1
            step2 = step1.copy(); step2[np.arange(len(x)),order[:,1]] = 1
            y = ((1-fs[:,0,None])*at(lo) + (fs[:,0]-fs[:,1])[:,None]*at(lo+step1)
                 + (fs[:,1]-fs[:,2])[:,None]*at(lo+step2) + fs[:,2,None]*at(lo+1))
        elif interpolation == 'trilinear':
            y = np.zeros_like(x)
            for r in (0,1):
                for g in (0,1):
                    for b in (0,1):
                        bit=np.array([r,g,b]); weights=np.prod(np.where(bit,f,1-f),axis=1)
                        y += weights[:,None]*at(lo+bit)
        else:
            raise ValueError('Unknown interpolation')
        return y.reshape(original_shape)


class LCMS:
    def __init__(self):
        library = ctypes.util.find_library('lcms2')
        if not library:
            raise RuntimeError('LittleCMS2 runtime unavailable')
        self.lib = C.CDLL(library)
        self.profiles, self.transforms, self.buffers = [], [], []
        self.transform_formats = {}
        funcs = {
            'cmsOpenProfileFromMem': ([C.c_void_p,C.c_uint32], C.c_void_p),
            'cmsCreate_sRGBProfile': ([], C.c_void_p),
            'cmsCreateLab4Profile': ([C.c_void_p], C.c_void_p),
            'cmsCreateTransform': ([C.c_void_p,C.c_uint32,C.c_void_p,C.c_uint32,C.c_uint32,C.c_uint32],C.c_void_p),
            'cmsDoTransform': ([C.c_void_p,C.c_void_p,C.c_void_p,C.c_uint32],None),
            'cmsDeleteTransform': ([C.c_void_p],None),
            'cmsCloseProfile': ([C.c_void_p],C.c_int),
            'cmsGetEncodedCMMversion': ([],C.c_uint32),
            'cmsGetProfileVersion': ([C.c_void_p],C.c_double),
            'cmsGetColorSpace': ([C.c_void_p],C.c_uint32),
            'cmsGetPCS': ([C.c_void_p],C.c_uint32),
            'cmsGetDeviceClass': ([C.c_void_p],C.c_uint32),
            'cmsFloat2LabEncodedV2': ([C.POINTER(C.c_uint16),C.POINTER(Lab)],None),
            'cmsLabEncoded2FloatV2': ([C.POINTER(Lab),C.POINTER(C.c_uint16)],None),
        }
        for name,(args,ret) in funcs.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=ret

    def open(self, source):
        raw = source if isinstance(source,bytes) else Path(source).read_bytes()
        buffer=C.create_string_buffer(raw); self.buffers.append(buffer)
        p=self.lib.cmsOpenProfileFromMem(buffer,len(raw))
        if not p:
            raise ValueError('LittleCMS cannot open ICC')
        self.profiles.append(p);return p

    def srgb(self):
        p=self.lib.cmsCreate_sRGBProfile()
        if not p: raise ValueError('Cannot create sRGB profile')
        self.profiles.append(p);return p

    def lab(self):
        p=self.lib.cmsCreateLab4Profile(None)
        if not p: raise ValueError('Cannot create Lab profile')
        self.profiles.append(p);return p

    def transform(self, inp, out, input_format=TYPE_RGB_DBL, output_format=TYPE_RGB_DBL,
                  intent=1, flags=FLAGS_NOOPTIMIZE_NOCACHE):
        t=self.lib.cmsCreateTransform(inp,input_format,out,output_format,intent,flags)
        if not t: raise ValueError('LittleCMS cannot build transform')
        self.transforms.append(t)
        self.transform_formats[t] = (input_format, output_format)
        return t

    def apply(self, transform, values, output_dtype=None):
        input_format, output_format = self.transform_formats[transform]
        input_dtype = np.uint16 if input_format == TYPE_RGB_16 else np.float64
        expected_output_dtype = np.uint16 if output_format == TYPE_RGB_16 else np.float64
        if input_format == TYPE_RGB_16 and np.asarray(values).dtype != np.uint16:
            raise TypeError('RGB16 transform requires uint16 input, not normalized floats')
        if output_dtype is not None and np.dtype(output_dtype) != np.dtype(expected_output_dtype):
            raise TypeError('Output dtype must match the LittleCMS buffer format')
        src=np.ascontiguousarray(values,dtype=input_dtype)
        if src.shape[-1] != 3: raise ValueError('Expected last dimension=3')
        dst=np.empty_like(src,dtype=expected_output_dtype)
        x=src.reshape(-1,3);y=dst.reshape(-1,3)
        for start in range(0,len(x),500_000):
            count=min(500_000,len(x)-start)
            self.lib.cmsDoTransform(transform,x[start:].ctypes.data,y[start:].ctypes.data,count)
        return dst

    def lab_v2_encode_one(self, values):
        out=(C.c_uint16*3)();lab=Lab(*values)
        self.lib.cmsFloat2LabEncodedV2(out,C.byref(lab))
        return np.array(list(out),dtype=np.uint16)

    def close(self):
        for t in self.transforms: self.lib.cmsDeleteTransform(t)
        for p in self.profiles: self.lib.cmsCloseProfile(p)
        self.transforms=[];self.profiles=[];self.buffers=[];self.transform_formats={}

    def __enter__(self): return self
    def __exit__(self,*args): self.close()


def encode_lab_v2(lab):
    lab=np.asarray(lab,dtype=np.float64)
    values=np.empty_like(lab)
    values[...,0]=lab[...,0]*(65280.0/100)
    values[...,1:]=(lab[...,1:]+128)*256
    return np.clip(np.floor(values+0.5),0,65535).astype(np.uint16)
