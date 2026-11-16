"""Create a deliberately synthetic two-link Gaussian/URDF fixture for GPU smoke tests.

No external reconstruction or robot model is required. This is NOT a real object test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
C0=0.28209479177387814


def save_ply(path,xyz,rgb):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    n=len(xyz)
    fields=["x","y","z","opacity"]+[f"f_dc_{i}" for i in range(3)]+[f"scale_{i}" for i in range(3)]+[f"rot_{i}" for i in range(4)]
    header="ply\nformat ascii 1.0\nelement vertex %d\n%s\nend_header\n"%(n,"\n".join(f"property float {f}" for f in fields))
    dc=(np.asarray(rgb,dtype=np.float64)-0.5)/C0
    with path.open("w") as f:
        f.write(header)
        for x in xyz:
            vals=list(x)+[4.0]+list(dc)+list(np.log([0.010,0.014,0.014]))+[1,0,0,0]
            f.write(" ".join(f"{v:.7g}" for v in vals)+"\n")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("/tmp/parttwin_fixture"))
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    # Grey cabinet frame: 2 vertical edges + upper/lower horizontal edges.
    z=np.linspace(0.0,0.65,50);x=np.linspace(0.0,0.5,50)
    frame=np.concatenate([
      np.stack([np.full_like(z,xi),np.zeros_like(z),z],axis=1) for xi in (0.,0.5)]+[
      np.stack([x,np.zeros_like(x),np.full_like(x,zi)],axis=1) for zi in (0.,0.65)])
    # Door leaf in its own local frame, hinge z axis through (0,0,0)
    xx,zz=np.meshgrid(np.linspace(.035,.49,32),np.linspace(.03,.62,40))
    leaf=np.stack([xx.ravel(),np.full(xx.size,0.02),zz.ravel()],axis=-1)
    save_ply(out/"base.ply",frame,[0.55,0.56,0.61])
    save_ply(out/"door.ply",leaf,[0.80,0.22,0.18])
    urdf='''<?xml version="1.0"?>
<robot name="smoke_door">
 <link name="base_link"><inertial><mass value="5"/><origin xyz="0.25 0 0.33"/><inertia ixx="0.18" iyy="0.18" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  <collision><origin xyz="0 0 0.33"/><geometry><box size="0.04 0.04 0.68"/></geometry></collision>
 </link>
 <link name="door_link"><inertial><mass value="1"/><origin xyz="0.25 0 0.325"/><inertia ixx="0.04" iyy="0.04" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
  <collision><origin xyz="0.25 0 0.325"/><geometry><box size="0.5 0.02 0.65"/></geometry></collision>
 </link>
 <joint name="door_hinge" type="revolute"><parent link="base_link"/><child link="door_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  <limit lower="0" upper="1.57" effort="20" velocity="1.5"/><dynamics damping="0.1" friction="0.01"/>
 </joint>
</robot>
'''
    (out/"model.urdf").write_text(urdf)
    asset={"urdf":"model.urdf","links":{"base_link":{"ply":"base.ply"},"door_link":{"ply":"door.ply"}}}
    (out/"asset.json").write_text(json.dumps(asset,indent=2))
    print("Generated synthetic fixture:",out)

if __name__=="__main__":main()
