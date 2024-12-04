# I_Fluid
## Introduction
`I_render` is used to learn fluid simulation.

## Learning Process
### Wave 1D
An easy 1D wave simulator built by Taichi.
### Navier-Stokes Equations
The equation is as following: 
$$
\begin{aligned}
&\mathbf{a}=\mathbf{g}-\frac{\nabla p}\rho+\mathbf{\mu}\nabla^2\mathbf{v}\\
&\nabla\cdot\mathbf{v}=0,\end{aligned}
$$
### Particle System
- Conventional particle system realized by SPH has amount of loss. User should tuning the parameters hardly. And the tuning process is such **STUPID**.
- PCISPH solves this problem, giving a better results.
- Taichi has some unknown bugs for implementation.