# Fluid structure interaction project
## Authors
- Emma Scalvedi (@emmascalvedi2)
- Giovanni Carpenedo (@gcarpenedo)
- Simone Ferri (@SimoFerri)

## Description
Fluid-Structure Interaction (FSI) represents a class of multiphysics problems in
computational mechanics. It describes the coupling between the behavior of a
fluid and a solid structure, and therefore the connection between laws of fluid
dynamics and laws of elasticity.
FSI is recurring in many real-world applications, such as the flow of blood
through elastic arteries or the wind-induced vibration of bridges, in which the
interaction is bidirectional: the fluid exerts a force on the solid causing its
deformation, while the latter alters the fluid flow.
This project aims to model a 2D steady linear FSI problem using a mono-
lithic solver approach.
For a more detailed description of the problem, the mathematical discussion
and the implementation description please check the [report](/report.pdf).

### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ mpirun -np <nof_processes> FSI
```

The `.vtu` output files will be created inside `build` so the results are
visualizable inside `ParaView`.
