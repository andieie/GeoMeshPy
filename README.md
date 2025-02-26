## GeoMeshPy

A Python package that converts vertices coming from geo modelling tools like _GemPy_. This package can be linked with the new functionality of _GMSH_ for
creating meshes that match the geometry of your geological model. <br>
The superiority is that this precedure can be achieved without manual treatement of the mesh. At the end, the automated coversion allows for making
water-tight meshes matching the primary geometry.
To install _GeoMeshPy_, just try <br>

```pip install geomeshconv```
<br><br>
or [visit the PyPi page](https://pypi.org/project/geomeshconv/).
<br><br>
The following fig is showing one of the meshes created automatically after generating the gelogical model. <br>

<img width="453" alt="Figure_1" src="https://user-images.githubusercontent.com/62764899/159875216-67d5f557-452f-4721-9e17-1fd123e085a1.png">

In the case of including uncertainty in the simulation, one single deterministic model can not be representative <br>
anymore. The following animation is showing three different possible orientations for one single fault surface.  <br>
It is highly time demanding to genertae mesh for each single scenraion but _GeoMeshPy_ is designed to tackle this problem.

<br><br>

![rotate_elevation_angle_3d_surf](https://user-images.githubusercontent.com/62764899/160632945-b4cdca87-4147-4d4c-a47b-cf3f8a0f7791.gif)

<br><br>
Please see available [examples](https://github.com/Ali1990dashti/GeoMeshPy/tree/main/Examples) to get more familiar with this tool.

## Publications using GeoMeshPy
This repository is part of the following paper. Please cite the following paper if you would like to cite this repository and find its contents useful: <br>

Dashti, A., Korzani, M. G., Geuzaine, C., Egert, R., & Kohl, T. (2023) . Impact of structural uncertainty on tracer test design in faulted geothermal reservoirs. https://doi.org/10.1016/j.geothermics.2022.102607.
