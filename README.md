# asurt_cdt
**ASU Racing Team Computational Dynamics Tool**

_**Note:** This is an initial and very brief introduction of the tool._

## Introduction
**asurt_cdt** is an open-source python package developed by ASU Racing Team -at Ain Shams University- with a main motive to create a free open-source general **multibody systems** python package that can be used by the mechanical engineering community specially the **Formula Student** community to encourage further development of the tool and deeper understanding of the modeling processes and the underlying theories. The tool is built with more focus on vehicle handling and dynamics modeling needs.
### What is Computational Dynamics?
Computational Dynamics is a rapidly developing field that brings together applied mathematics (especially numerical analysis), computer science, and scientific or engineering applications. Multibody Dynamics can be considered as a sub-field of computational dynamics that focuses on the simulation and analysis of engineering systems that consist of interconnected bodies which are highly nonlinear in nature and their analysis requires the use of matrix, numerical, and computer methods.

## Quick brief about the theories.
### What is the problem to be solved?
The primary interest in multibody dynamics is to analyze the system behavior for given inputs. In analogy with control systems; a multi-body system can be thought as a **_system_** subjected to some **_inputs_** producing some **_outputs_** which are dependent on both the system and the inputs characteristics. These three parts of the problem are dependent on the analyst end goal of the analysis and simulation. 
Four main types of analysis are of interest for a given multibody system. These are:
- **Kinematic Analysis:**<br\>
For kinematic analysis of multi-body systems, one of the main end goals is to specify the system configuration as a function of time or the system inputs. In other words, to determine the location and orientation of each body in the system at any time instance throughout the simulation. This does not require the definition of inertia properties of the system bodies nor the system generalized applied forces.
- **Inverse Dynamic Analysis:**
The system in kinematically driven with a prescribed motion path using relevant motion actuators. The output mainly here is the system reactions at the joints' locations resulted from this motion. This takes into consideration forces due to inertia, gravity, centrifugal, force elements and applied external loads.
- **Equilibrium Analysis:**
Starting from the system initial configuration and conditions, the system reaches the steady-state -equilibrium- configuration using high damping ratios to dissipate the system energy. Other methods can be used such as dealing with the system as an optimization problem achieving a configuration with minimal potential energy.
- **Dynamic Analysis:**
The system is dynamically driven using various time-varying force inputs. The system independent accelerations are detected and integrated to obtain the system velocities and position level configuration.

### How is the system physics is abstracted mathematically and computationally?
To keep it simple as possible. A body in space is normally defined using six generalized coordinates defining its location and orientation, so a system of 10 bodies will require 60 unknown generalized coordinates to be fully defined which in turn requires 60 independent equations to be solved for the system coordinates. The basics of the theories will be covered in another document.
For those with some experience in multibody systems, the following points out briefly the mathematical approaches used.
- **Orientation Parameters:**
The tool makes use of the **_euler parameters_** -a 4D unit quaternion- to define rigid bodies orientation, so each body now requires 7 generalized coordinates to be fully defined.
- **Newton-Euler Augmented Formulation of the EOM:**
The equations of motion of the system is assembled in the augmented matrix form of Newton-Euler equations and the second time derivative of the constraint equations in terms of the euler parameters directly resulting in a system of DAEs -differential algebraic equations-. Due to the use of euler parameters, the mass matrix -inertia tensor elements- is not constant.
- **Coordinate Partitioning and Integration:**
Instead of using the implicit BDF integration method to solve the DAEs system directly, the system is partitioned to dependent and independent coordinates numerically based on the jacobian structure of the system constraints then the independent coordinates are re-represented in an **SSODE** -state-space ordinary differential equations- and then integrated using an explicit runge-kutta method of order 8(5,3) implemented in [**_scipy_**](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode). The independent coordinates are then used to evaluate the dependent ones using newton-raphson algorithm.

## Code structure
_**Note:** The code is under rapid changes and frequent modifications. This is an initial structure._
The main modules of the code are as follows:
- **base.py**
This module contains the main classes and functions used to abstract the system physics computationally.
- **constraints.py**
This module contains all the constraint objects used to define the system topology -connectivity conditions-. For every constraint, the analytical jacobian is defined as well as the constraint reaction, acceleration rhs, etc.
- **bodies_inertia.py**
This module defines the rigid body objects with explicitly defining their inertia properties.
- **force_elements.py**
This module defines the basic force elements such as TSDA -translational spring-damper-actuator- element, applied force/torque and a simple vertical tire force model representing the tire/ground contact and tire vertical deflection.
- **preprocessor.py**
This module works as a system constructor. It takes in the system defined topology and forces and construct the system jacobian, constraint equations, mass-matrix, ..etc, in a separate .py file containing the system info in order to avoid nested for loops and function calls. The matrices returned are in compressed sparse columns format making use of the system sparsity.
- **newton_raphson.py**
This module defines the newton-raphson algorithm making use of sparse linear solvers and evaluating the jacobian only when needed to save time.
- **solvers.py**
This module provides the solver functions for kinematically and dynamically driven system and helper functions that analyze the jacobian for selecting the proper independent coordinates.

The modeling process can be done either by coding or using a simple **_gui_** that makes use of **Jupyter Lab**. The project makes use of [**_netwokx_**](https://networkx.github.io/documentation/stable/) in modeling the data structure and dependencies to track and migrate changes over the model components. Also, for a given model, the model topology is modeled as a multigraph, which serves a very good mean to store the topological information of a given model.
**_Further illustrations in progress with a simple tutorial._**

## Current capabilities and my wish-list
The tool is capable of _kinematic, inverse dynamic, dynamic_ and _equilibrium_ analysis of a well posed non-singular system. The main outputs are:
- System configuration / Position level.
- System velocities.
- System accelerations.
- System reactions defined in the global coordinate system.
The data are stored in pandas' DataFrame object that can be easily plotted and extracted to excel files.
### Wish-list:
- Test the tool on several models to better debug the code the concepts used.
- Create more realistic tire and track models.
- Create automated symmetry script to mirror the system automatically.
- Create a variable step-size solver to increase accuracy and solver speed.
- Include bushing joints to account for the system compliances.
- Create a post processor to evaluate the common suspension calculations.
- Obtaining inertia properties from CAD files automatically.
- Animation of the system using vtk as well as vpython.
- Migrate the heavy numerical work to cython.
- .... etc. "Actually it never ends"

### Dependencies
- Scipy
- Numpy
- Pandas
- Matplotlib
- _... To be updated_

## How to use?
_To be updated._.

If interested in participating in the development of the tool or have any inquiries, I can be reached via _khaled.ghobashy@live.com_.
