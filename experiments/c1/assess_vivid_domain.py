#!/usr/bin/env python3
"""Assess explicit PCS->sRGB->Vivid->PCS candidate without writing any ICC.

This is a domain-loss gate, not a calibration fit or C1 pipeline claim.
"""
import argparse
import sys
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.c1 import inspect_profile as audit
SOURCE=ROOT/'inputs/LeicaQTyp116-Generic.icm'
CUBE=ROOT/'filters/looks/Leica_Vivid_sRGB_sRGB_Release.cube'
EXPECTED='efed3f9554cb8da91de1fffb0e4a93b2ae9f98acc47d2cd4078133b38f99b969'

# Explicit colorimetric boundary: ICC nominal D50, Bradford adaptation to
# standard D65, sRGB primaries. No gamut mapping, exposure or tone compensation.
WHITE50=np.array([0.9642,1.,0.8249])
WHITE65=np.array([0.3127/0.3290,1.,(1.-0.3127-0.3290)/0.3290])
BRADFORD=np.array([[.8951,.2664,-.1614],[-.7502,1.7135,.0367],[.0389,-.0685,1.0296]])
CAT=np.linalg.inv(BRADFORD)@np.diag((BRADFORD@WHITE65)/(BRADFORD@WHITE50))@BRADFORD
prim=np.array([[.64/.33,.30/.60,.15/.06],[1.,1.,1.],[(1.-.64-.33)/.33,(1.-.30-.60)/.60,(1.-.15-.06)/.06]])
RGB_XYZ65=prim@np.diag(np.linalg.solve(prim,WHITE65))


def lab_xyz(lab):
    y=(lab[:,0]+16.)/116.;x=y+lab[:,1]/500.;z=y-lab[:,2]/200.;f=np.column_stack((x,y,z));d=6/29
    return np.where(f>d,f**3,3*d*d*(f-4/29))*WHITE50


def xyz_lab(xyz):
    t=xyz/WHITE50;d=6/29
    f=np.where(t>d**3,np.cbrt(t),t/(3*d*d)+4/29)
    return np.column_stack((116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])))


def encode(lin):
    # Signed extension only for reporting out-of-domain values; Vivid receives
    # explicit [0,1] clipping, NOT this negative extension.
    a=np.abs(lin)
    return np.sign(lin)*np.where(a<=.0031308,12.92*a,1.055*np.power(a,1/2.4)-.055)


def decode(rgb):
    a=np.abs(rgb)
    return np.sign(rgb)*np.where(a<=.04045,a/12.92,np.power((a+.055)/1.055,2.4))


def lab_rgb(lab):return encode(lab_xyz(lab)@CAT.T@np.linalg.inv(RGB_XYZ65).T)
def rgb_lab(rgb):return xyz_lab(decode(rgb)@RGB_XYZ65.T@np.linalg.inv(CAT).T)


def stats(x):
    a=np.asarray(x)
    return {'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
            'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99))}


def texture(rgb,cube):
    n=cube.shape[0];q=np.clip(rgb*n-.5,0,n-1);lo=np.floor(q).astype(int);hi=np.minimum(lo+1,n-1);f=q-lo
    result=np.zeros_like(rgb)
    for rb,gb,bb in itertools.product((0,1),repeat=3):
        r=hi[:,0] if rb else lo[:,0];g=hi[:,1] if gb else lo[:,1];b=hi[:,2] if bb else lo[:,2]
        w=(f[:,0] if rb else 1-f[:,0])*(f[:,1] if gb else 1-f[:,1])*(f[:,2] if bb else 1-f[:,2])
        result+=cube[b,g,r]*w[:,None]
    return result


def evaluate_set(lab,cube):
    rgb=lab_rgb(lab);mask=np.any((rgb < -1e-6)|(rgb > 1+1e-6),axis=1);mask_loose=np.any((rgb< -1e-4)|(rgb>1+1e-4),axis=1)
    clip=np.clip(rgb,0,1);delta=np.linalg.norm(rgb_lab(clip)-lab,axis=1);roundtrip=np.abs(rgb_lab(rgb)-lab)
    look=texture(clip,cube);outlab=rgb_lab(look)
    codes=np.column_stack((outlab[:,0]*652.8,(outlab[:,1]+128)*256,(outlab[:,2]+128)*256))
    return {'count':len(lab),'out_of_srgb_count_tolerance_1e_6':int(mask.sum()),'out_of_srgb_fraction':float(mask.mean()),
        'out_of_srgb_count_tolerance_1e_4':int(mask_loose.sum()),'encoded_srgb_minmax_per_channel':[[float(rgb[:,j].min()),float(rgb[:,j].max())] for j in range(3)],
        'clipping_only_deltaE76':stats(delta),'clipping_deltaE_gt1_count':int((delta>1).sum()),'clipping_deltaE_gt5_count':int((delta>5).sum()),
        'unclipped_colorimetric_roundtrip_max_abs_Lab':float(roundtrip.max()),
        'vivid_output_rgb_minmax_per_channel':[[float(look[:,j].min()),float(look[:,j].max())] for j in range(3)],
        'vivid_output_lab16_unrepresentable_count':int(np.any((codes<0)|(codes>65535),axis=1).sum()),
        'interpretation':'Synthetic profile-coordinate coverage only; not a fraction of real photos or sensor colors.'}


def main():
    global SOURCE, CUBE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, default=SOURCE, help='User-provided original Q1 Generic; exact pinned source required')
    p.add_argument('--cube', type=Path, default=CUBE)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(); SOURCE, CUBE, OUTPUT=a.profile, a.cube, a.out
    if OUTPUT.exists():raise SystemExit('Refusing overwrite')
    data=SOURCE.read_bytes();before=audit.sha(data)
    if before != EXPECTED:raise SystemExit('source hash mismatch')
    n=audit.u32(data,128);lut=None
    for i in range(n):
        p=132+12*i
        if data[p:p+4]==b'A2B0':
            o=audit.u32(data,p+4);z=audit.u32(data,p+8);lut=audit.Lut16(data[o:o+z])
    if lut is None:raise SystemExit('no A2B0')
    cb=CUBE.read_bytes();cube=np.loadtxt(CUBE,skiprows=6).reshape(64,64,64,3)
    raw=np.array(lut.clut,dtype=float).reshape(-1,3)
    lab=np.column_stack((raw[:,0]/652.8,raw[:,1]/256-128,raw[:,2]/256-128))
    rng=np.random.default_rng(20261002);q=rng.random((4096,3))
    randomlab=np.array([lut.evaluate(list(x))['pcs_Lab'] for x in q])
    grayq=np.linspace(0,1,1025);graylab=np.array([lut.evaluate([float(x)]*3)['pcs_Lab'] for x in grayq])
    samples={'all_native_CLUT_nodes':evaluate_set(lab,cube),'uniform_device_RGB_4096':evaluate_set(randomlab,cube),
             'equal_channel_device_axis_1025':evaluate_set(graylab,cube)}
    # Distinguish color-space roundtrip from texture identity: x*N-.5 mapping
    # on an identity-valued texture intentionally has half-texel behavior.
    synthetic=np.stack(np.meshgrid(np.linspace(0,1,2),np.linspace(0,1,2),np.linspace(0,1,2),indexing='ij'),axis=-1)[...,[2,1,0]]
    test=np.array([[.25,.25,.25],[.75,.75,.75],[.5,.5,.5]])
    error=float(np.max(np.abs(texture(test,synthetic)-np.array([[0,0,0],[1,1,1],[.5,.5,.5]]))))
    if error>1e-12:raise AssertionError('texture coordinate negative control failed')
    r={'source':SOURCE.name,'source_sha256_before':before,'source_sha256_after':audit.sha(SOURCE.read_bytes()),
       'cube':CUBE.name,'cube_sha256_before':audit.sha(cb),'cube_sha256_after':audit.sha(CUBE.read_bytes()),
       'icc_written':False,'proposed_transform':'native A2B0 Lab PCS -> explicit D50/Bradford/D65/encoded sRGB -> clip[0,1] -> Vivid x*64-.5 trilinear texture -> Lab PCS',
       'fixed_policy':{'no_fit':True,'no_exposure_WB_curve_change':True,'preserve_native_input_shaper':True,'gamut_map':'none; hypothetical hard clipping measured before applying Vivid',
                      'candidate_position':'profile PCS, NOT proven after C1 Curve/Auto'},
       'color_math':{'D50_XYZ':WHITE50.tolist(),'D65_XYZ':WHITE65.tolist(),'D50_to_D65_Bradford':CAT.tolist(),'sRGB_to_XYZ_D65':RGB_XYZ65.tolist(),
                     'transfer':'IEC sRGB segmented transfer; signed extension only for diagnostics outside gamut'},
       'samples':samples,'texture_synthetic_coordinate_test_max_abs_error':error,
       'decision':'Do not write Vivid ICC yet: observed irreversible out-of-sRGB clipping under the proposed simple PCS bridge requires a declared accepted gamut policy, and bypass/Auto host gate remains untested.',
       'limits':['Does not estimate real-photo affected fraction','Does not reconstruct Leica RAW renderer','No C1 program executed','Vivid resource is vendor data; redistribution permission unconfirmed']}
    if r['source_sha256_before']!=r['source_sha256_after'] or r['cube_sha256_before']!=r['cube_sha256_after']:raise AssertionError('input changed')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'report':str(OUTPUT),'icc_written':False,'samples':{k:{'n':v['count'],'out_fraction':v['out_of_srgb_fraction'],'clip_max_dE':v['clipping_only_deltaE76']['max'],'clip_p95_dE':v['clipping_only_deltaE76']['p95']} for k,v in samples.items()}},ensure_ascii=False))

if __name__=='__main__':main()
