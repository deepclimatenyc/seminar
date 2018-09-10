# Weekly seminar on deep learning for climate modeling


## 2018-09-04: Reynolds averaged turbulence modelling using deep neural networks with embedded invariance

Ling, J., Kurzawski, A., & Templeton, J. (2016).
_Reynolds averaged turbulence modelling using deep neural networks with embedded invariance_.
Journal of Fluid Mechanics, 807, 155-166. doi:[10.1017/jfm.2016.615](https://doi.org/10.1017/jfm.2016.615)

Context (from Mu): Navier Stokes solution can be solved by 
1. direct numerical simulation
1.	large eddy simulation (spectral cascades)
1.	Reynolds-averaged Navier Stokes (RANS) 
    1.	Eddy viscosity models of order 0, 1, and 2 (k-epsilon). The zero-order solutions are based upon the Boussinesq approximate
        for turbulence. 
    1.	Reynolds stress models that solve the time evolution of the Reynolds stress <ui’uj’>
    1.	Perhaps it is too complicated to explicitly solve the Reynolds stress evolution → development of algebraic stress models (ASM):
        <ui’uj’> = giTi in which Ti are a 10-tensor basis. This serves as a nice test model for machine learning in fluid dynamics as
        conservation laws are satisfied, no physical insight is needed; we simply want the best set of parameters gi.

Overview (from Tom): 
In the article the Navier Stokes equation is solved in the Reynolds decomposition which employs a mean and perturbation to describe
a flow. The shear and rotation tensors are non-dimensionalized with turbulent kinetic energy and turbulent dissipation energy.
The notion of Galilean invariance states the solution of the flow must not be dependent on the orientation of the coordinates.

Article discussion: 
-	The author makes a large assumption in using the 5 invariants, i.e. that the flow is driven by eddies.
-	Nine inputs are used in the MLP, exploiting some symmetries in the rotation and shear tensors.
-	Did they really need 8 hidden layers for this system? How well could one do with just 2 layers?
  Or with a baseline linear or nonlinear model?
-	Some discussion of their Bayesian optimization for 3 hyperparameters
  (2 architectural -- number of nodes per layer + number of layers,1 for learning rate) 
-	They use root mean squared error for their cost function. Is RMSE the right metric?
  Could random search do as well or better? Could certain methods simply learn the cost function?
-	Could they have had the same insight by simply augmenting their training data, i.e. rotating it many times to
  build up the invariance naturally?

General discussion:
-	Generative adversarial networks (GANs) can do better with less data.
-	It could be interesting to come up with certain “climate canonical data sets”.
-	From the discussion of turbulence modeling, it is important to choose your model cleverly. In “softer machine learning” there is an “embedded conservation”. With sufficient data, however, ML tools seem to be able to learn conservation. A means of accelerating this “conservation learning” could be to heavily penalize non-conservation by the model.
-	Spectral energy fluxes could be a useful measure for the cost function. A given architecture may also be more successful in Fourier space.
-	It is worthwhile to think carefully about the best data for these tools.
