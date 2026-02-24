# Fluid-Structure Interaction Project

## Authors
- Emma Scalvedi ([@emmascalvedi2](https://github.com/emmascalvedi2))
- Giovanni Carpenedo ([@gcarpenedo](https://github.com/gcarpenedo))
- Simone Ferri ([@SimoFerri](https://github.com/SimoFerri))

## Description
Fluid-Structure Interaction (FSI) represents a class of multiphysics problems in
computational mechanics. It describes the coupling between the behavior of a
fluid and a solid structure, and therefore the connection between laws of fluid
dynamics and laws of elasticity.
FSI is recurring in many real-world applications, such as the flow of blood
through elastic arteries or the wind-induced vibration of bridges, in which the
interaction is bidirectional: the fluid exerts a force on the solid causing its
deformation, while the latter alters the fluid flow.
This project aims to model a 2D steady linear FSI problem using a monolithic solver 
approach on deal.II and Trilinos with MPI parallelization.

For a more detailed description of the problem, the mathematical discussion
and the implementation description please check the [report](/report.pdf).

## Compiling
To build the executable, make sure you have loaded the required modules:
```bash
module load gcc-glibc dealii
```

Then build the executable:
```bash
mkdir build
cd build
cmake ..
make
```

The executable `FSI_parallel` will be created inside `build`.

## Running
```bash
mpirun -n <num_of_processes> ./FSI_parallel
```

The `.vtu` and `.pvtu` output files will be created inside `build` and the results can be
visualized in ParaView.