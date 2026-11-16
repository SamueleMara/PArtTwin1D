#!/usr/bin/env python3
"""Optional reproducible development fixture, NOT a substitute for real object PLYs."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

DC = 0.28209479177387814


def write_ply(path, points, rgb):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    props=['x','y','z','opacity',*[f'f_dc_{i}' for i in range(3)],
           *[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)]]
    header=(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\n'+
            ''.join(f'property float {name}\n' for name in props)+'end_header\n')
    dc=(np.asarray(rgb)-0.5)/DC
    with path.open('w') as out:
        out.write(header)
        for pt in points:
            values=[*pt,4.,*dc,*np.log([0.009,0.014,0.013]),1.,0.,0.,0.]
            out.write(' '.join(f'{v:.8g}' for v in values)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default=Path('examples/hinged_box'),type=Path)
    args=p.parse_args()
    out=args.out
    z=np.linspace(0.,0.65,50)
    x=np.linspace(0.,0.5,50)
    frame=np.concatenate([
      np.stack([np.full_like(z,xi),np.zeros_like(z),z],axis=-1) for xi in (0.,0.5)]+[
      np.stack([x,np.zeros_like(x),np.full_like(x,zi)],axis=-1) for zi in (0.,0.65)])
    xx,zz=np.meshgrid(np.linspace(.035,.49,32),np.linspace(.03,.62,40))
    leaf=np.stack([xx.flatten(),np.full(xx.size,.02),zz.flatten()],axis=-1)
    write_ply(out/'base.ply',frame,[.55,.56,.61])
    write_ply(out/'door.ply',leaf,[.8,.22,.18])
    print(f'Created {out}/base.ply and {out}/door.ply')

if __name__=='__main__':main()
