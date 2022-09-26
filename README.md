## Overview
Signals of Opportunity is an approach for earth observation where reflected signals from other satellites are used to make a measurement.  See [GNSS reflectometry](https://en.wikipedia.org/wiki/GNSS_reflectometry) for further explanation of the specific case of navigation satellites. The purpose of this project is to provide a fast simulation of the overall performance of a receiver constellation for initial receiver orbital parameter and antenna-configuration selection. 

## Approach
1. Select transmitter constellations to use and download their Two Line Element sets from [CelesTrak](https://celestrak.org/NORAD/elements/)
2. Propagate the orbits of selected receiver and transmitter satellites over a given time interval using sgp4.  This is done with the Skyfield python package.
3. Find the specular reflection points between every pair of receiver and transmitter at each time step.  This is done using an [analytical solution](https://www.geometrictools.com/Documentation/SphereReflections.pdf) based on an assumption of a perfectly spherical earth.
4. Construct a [uniform grid on a sphere](https://www.aanda.org/articles/aa/pdf/2010/12/aa15278-10.pdf) and project the specular points onto this grid.
5. Count all the cells touched by the path of the specular point between time samples using a line drawing algorithm. This is done for all specular points for each pair of receiver and transmitter.
6. Divide the number of revisits for each cell by the simulation time to get an estimate of revisit frequency everywhere on the Earth.
7. Optionally, count the number of cells that meet a revisit frequency requirement threshold.

## Demo
Below shows the output of the simulation for an example receiver constellation and a variety of GNSS satellites:
![bulk_globe](https://user-images.githubusercontent.com/43870861/192219641-9ccbfecc-6540-411e-b7e2-2698e4170e65.PNG)

Additionally, on the github-pages for this repository, there is a 3d-visualization of the specular trail between two satellites vs time.  The two thick lines off the sphere represent the position of the satellites.  The thick line on the surface of the sphere represents the position of the specular point.  The thin lines represent the path of the reflected signal at each time step.  The color of the lines varies with time.

[Between a pair of GPS satellites](https://aidanmolnar.github.io/SoOpSimulator/demosite/dist/gps_pair.html)

[Between a Galileo GNSS satellite and an Iridium satellite](https://aidanmolnar.github.io/SoOpSimulator/demosite/dist/galileo_with_iridium.html)

## Project Structure
The python class for running the simulation can be found in the simulator/soopsimulator/model.py. Examples are next to it in examples.py.
The library for calculating the specular points and rasterizing the lines between them is written in rust and located in simulator/src.
An older version of the library written in python with numpy and numba is in simulator/soopsimulator/python_sim_core.  
The code for displaying the github-pages demo is in the demosite directory.

## Future Work / Limitations
* Currently, there is no simulation of the signal degredation due to distance or atmospheric pertubation
* Calculating the specular points using a GPU compute shader would be faster
* The analytical quartic solver used by the specular point calculator suffers from roundoff error that reduces accuracy.  A slower but more accurate solver for comparison would be beneficial.
* Current specular point solver does not account for Earth's ellipsoidal shape or surface features.
* An animated demo would be cool.
