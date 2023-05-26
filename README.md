### Solution of the acoustic wave equation in 3D using finite difference method.

Equation <br />

$\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2} + \frac{\partial^2u}{\partial z^2} = \frac{1}{c^2} \cdot \frac{\partial^2u}{\partial t^2}$
	
where <br />
	$u = u(x,y,z,t)$ - displacement vector (or acoustic pressure) <br />
	$c$ - speed of sound <br />

This program runs on GPU. So, the nvcc compiler is used.

compile it with

```
nvcc main_proj.cu -o main_proj
```

and run wih

```
./main_proj
```

### Result

This program outputs .vtk file that can be opened in [ParaView](https://www.paraview.org/).

### Visualization

Collected data are presented in the root folder as two animations in .ogv format. You can open it using any video player.
