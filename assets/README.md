# assets/ — real object captures (not tracked)

Put real articulated-object exports here, one directory per object, e.g.

```
assets/red_fridge/
  model.urdf          # metres/radians; collision + inertial where known
  base.ply            # Graphdeco 3DGS PLY per URDF link (metric units)
  door.ply
  object.yaml         # copy of configs/real_object.template.yaml with REAL values
```

Everything in `assets/` except this README and `.gitkeep` is gitignored.
Validate with `parttwin inspect --config assets/red_fridge/object.yaml` before
launching the GUI. Never copy the template's placeholder K / transforms.
