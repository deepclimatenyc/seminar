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

## 2018-09-11: A data driven approach to convective parameterization

Pierre Gentine, September 11, 2018

### Convective parametrization (based on Arakawa and Schubert’s idea):

Main objective: getting mass flux profiles

-   Specify mass flux at cloud base(closure)
-   Specify entrainment and detrainment profiles(mixing)
-   Use quasi-equilibrium assumption (which is not right sometimes)
-   Imply strong connection between boundary layers and convection
-   Include trigger function of convection

2.  Problem of convective parametrization in GCMs:

-   Incorrect peak diurnal cycle of Precipitation and Cloud cover, which
    is in correspondent with Peak surface flux
-   Wrong heating sign in heating profiles
-   Incorrect timing of shallow cumulus and deep convection
-   Trigger function and mass closure are not good
-   Quasi-equilibrium is good for longer timescales (CAPE as an
    example), but when forcing is too fast \~ diurnal cycle, it is far
    from equilibrium
-   In a diurnal cycle, memory of convective systems is more important
    than forcing
-   Entrainment depends on environmental conditions, for example, is
    sensitive to humidity of environment, when RH+ mixing+
-   Entrainment is a stochastic representation, not deterministic,
    mixing level is random, state does not depend on cloud base but
    memory
-   A lot of drizzle: not enough moisture (too frequent and too
    little rain) not enough extremes of rainfall
-   Mesoscale convective systems (self-organized, large fraction of
    rainfall and key for extreme rainfall) is not represented
-   Cloud pools: rain evaporation and ice melt generate density current
    and regenerate convection by pushing air back troposphere – but is
    not represented

> Too many biases in diurnal cycles, MCS, organization, precipitation
> extremes, waves, mass flux and entrainment.

### Some improvement ways:


#### ECMWF:

- Departure from quasi-equilibrium using PBL lag/memory
- Relax of PCAPE -&gt; treat the memory

#### Deep convection

- Cold pools in PBL is one way to include triggering and closure
- Modify entrainment -&gt; Wider less entrainment plumes


### Solution A to substitute convective parametrization: CRMs do better
    job(\~10km), we can embed CRMs to GCMs (Super Parametrization)

-   Diurnal cycle is better and intensity is better
-   Order of magnitude of precipitation is right

Explicit convection improves dramatically, SPs are doing well but too
expensive!

Questions: usually we do 1D-2D CRMs and not full 3D structures to avoid
expensive costs, will it cause problems?

Yes, momentum budget and so on… Macroscale statistics can be better
represented in 3D models.

### Solution B to substitute convective parametrization: data driven
    approach machine learning

#### How to do machine learning?

-   Training data: 3-6 months, testing data: one year, long time LES
    might be used as training data
-   Uses T, q, Ps, H, LH, SW~TOA~ to predict their tendency,
    precipitation or TOA radiative flux (Traditional way: dynamical
    core + advection + turbulence + microphysics schemes…. -&gt;
    tendency)
-   SPs can represent some of MCs propagation even with periodic lateral
    boundary conditions
-   Learning rate was the most important variable. Normalization of
    variables was not necessary.

#### Good news in symmetric aquaplanet +4K experiment:

-   Similar to SPs but smoother in machine learning model, less noise
    (fit mean state)
-   Better cloud radiation effect and precipitation
-   Extreme precipitation number is increased
-   MJO is better than cam but not as good as SPs, there is similar wave
    spectra
-   Heating rate in vertical levels is more realistic
-   Only one peak in ITCZ instead of two peaks
-   Integrate heating rate plot: Walker circulation in the right place:
    even though on aqua-planet: shows some generation is possible

#### Limits of machine learning:

-   Does not do well if too much outside of the training data, for
    example, it cannot simulate warming in the future based on past
    data, but, if trained with both past and future data, it is
    improved. And under warming condition, extreme precipitation is
    captured
-   Did not show the variance well

Question: Why we overestimate extreme precipitation in low frequencies
in machine learning model?

###  Discussion

-   A question about conservation.

    It’s very close to be conserved, we plugged the machine learning to
    climate model, the model learned conserve to some extent, it is not
    perfect because we do not have liquid water content, but
    approximately. Condensation diagnostics are needed.

-   Stability of NN model.
-   How much faster is the machine learning model when compared with
    SPs?

    20 times. Training data of 3-6 months could lead to convergence.

-   How many hidden layers?

    256 x 8

-   How many epochs to use? Set line for epoch or set a line for error?
    When to stop the training?

    Stop when reach convergence, not use dropout

-   It is not necessary to normalize the data or dimension analyze,
    because it can figure it out for you.
-   Initial condition for weight is not so important
-   Learning rate is the most important parameter, and sometimes we
    chose one learning rate which still leads to convergence but is
    physically wrong
-   Cost function: RMSE from output vector.
-   Could be useful to check cross correlation of input variables.
-   Start use noise and shape the noise in some way, ask the model to
    learn the distribution not the deterministic relationship.
    Generative model. Gaussian processes might be useful.
-   What is the network architecture? The details of setup.

    To be showed…

-   May train data from 3D cloud resolving models in the future
-   Use GitHub repository and Google Doc to share information.

Yu Huang

