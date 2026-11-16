#!/usr/bin/env python3
"""Inspect manifest, URDF joints and 3D Gaussian fields without SAPIEN/CUDA."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parttwin1d.sim.assets import load_config, ordered_columns


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    args=p.parse_args()
    cfg=load_config(args.config)
    root=ET.parse(cfg.urdf).getroot()
    links={x.attrib['name'] for x in root.findall('link')}
    joints=[(x.attrib['name'],x.attrib['type']) for x in root.findall('joint')]
    print('URDF:',cfg.urdf)
    print('URDF links:',sorted(links))
    print('URDF joints:',joints)
    for s in cfg.links:
        if s.link not in links:
            raise ValueError(f'PLY mapping for {s.link} has no corresponding URDF link')
        from plyfile import PlyData
        ply=PlyData.read(str(s.path))['vertex']
        names=ply.data.dtype.names
        print('PLY',s.link,s.path,'Gaussians:',len(ply),'SH-rest fields:',len(ordered_columns(names,'f_rest_')))
    print('Camera(s):',[(c.name,c.width,c.height) for c in cfg.cameras])
    print('Manifest: OK')

if __name__=='__main__':main()
