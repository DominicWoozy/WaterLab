# Duck model

Original model and texture: © 2006 Sony Computer Entertainment Inc.
Source: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Duck
License: SCEA Shared Source License, Version 1.0 (see LICENSE.txt).
Original author documentation is retained as SOURCE.md; original asset is Duck.glb.

Changes for Water Lab: uniform scale and translation, triangle ordering into a
bounding volume hierarchy, and binary packing for WebGL. No generated geometry,
decimation, texture painting, or normal modification. DuckCM.png is extracted
unchanged from the original GLB. The physical collision hull is an approximation
separate from the displayed mesh. Conversion is reproducible with
scripts/prepare-duck.py in the source repository.
