+++
# Date this page was created.
date = "2017-01-18"

# Project title.
title = "Thesis - 4. Dynamics"

# Project summary to display on homepage.
summary = "Thesis - 4. Dynamics"

# Optional image to display on homepage (relative to `static/img/` folder).
#image_preview = "gpue.png"

# Optional image to display on project detail page (relative to `static/img/` folder).
image = ""

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["data","bio", "hpc", "knl"]

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = true

+++
Bose–Einstein condensate dynamics
=================================

\[sec:vortlatt\]

The purpose of this chapter is to introduce the reader to vortex states of BECs, and eventually discuss manipulations of these states. In the following sections we will examine both the static and the dynamical solutions of the Gross–Pitaevskii equation (GPE), using the methods previously discussed. We will first briefly discuss few vortex solutions, with an analysis of their properties via an examination of the velocity field.

We will follow this by providing the details of the model system of a vortex lattice in a rapidly rotating BEC, using the theoretical framework developed in Chapter \[chp:background\]. Next, we will discuss the application of two distinct ways for manipulating and controlling the condensate: (1) applying an external potential to the trapped condensate for a short timescale, and (2) direct manipulation of the wavefunction phase. Both of these are experimentally realistic, and though they can be experimentally implemented by similar means, they serve two very different purposes, and they will be treated as such.

Simulating Bose–Einstein condensate dynamics
--------------------------------------------

In Sections \[sub:gpederiv\] and \[sec:fso\] we have outlined both the analytical Thomas–Fermi (TF) and numerical solutions of the GPE. It is instructive to compare the results from these two methods. For a stationary condensate in the ground state of a harmonic trapping potential the profile and width should be comparable, with the TF solution deviating only in the low-density regions. Fig. \[fig:gpe\_tf\_3\] shows a comparison of the two-dimensional profile for both methods, for condensates of **<sup>87</sup>Rb with *N* = 10<sup>5</sup> atoms, and trapping frequencies of *ω*<sub>**r**</sub> = 2*π* × (1, 1, 16) Hz. Both solutions show good agreement, deviating only where expected, and hence showing a well developed numerical procedure.

<embed src="/ch4_vtx/gpe_tf_3.png" />

While the Thomas–Fermi solution closely agrees with the numerical solution of the Gross–Pitaevskii equation, this solution is only applicable for stationary states with negligible kinetic energy, i.e. when the nonlinear interaction dominates over the kinetic energy term. For more complex problems involving dynamics, full numerical integration of the Gross–Pitaevskii equation is required.

One example where the TF approach fails is when investigating superfluid vortex dynamics. To generate vortices in the condensate angular momentum must be added to the system. Seeding a single vortex in the condensate requires that the frequency of rotation, *Ω*, must be higher than the critical rotation frequency *Ω*<sub>*c*</sub>, as discussed in Sec. \[ss:vorticesinbec\], and numerically, this is most easily simulated in the co-rotating frame as given by Eq. . Experimentally, there are many ways to create vortices in a condensate, such as stirring with a blue-detuned laser beam , carefully inverting the trap bias field potential , or through the use of artificial gauge fields , to name but a few. In the next sections we will concentrate on solutions where the vortices are already present in the condensate and have settled into the lowest energy configuration.

### Few vortex condensates

As discussed in Section \[sec:superfluid\], the study of quantum vortices remains an active area of research. As the condensate kinetic energy scales as *E*<sub>*k*</sub> ∝ *l*<sup>2</sup> (Sec. \[ss:vorticesinbec\]), increased angular momentum leads to the appearance of more singly charged vortices rather than one multiply charged vortex. For a single vortex in a rotating condensate the only stationary solution that exists is if the vortex resides at the exact centre of the trapping potential, making the system radially symmetric. For two vortices the radial symmetry of the system is broken, with the two vortices arranging themselves in the most favourable position to minimise the energy of the system, as with all higher vortex states. Placing vortices of the same sign into a condensate will allow them to behave as identically charged particles, and they will repel, and form ordered lattices with increasing vortex numbers (see Fig. \[fig:few\_rho\]). Similarly, if instead we pair a vortex and an antivortex, they will attract, and remain in constant motion, or disappear altogether in the presence of dissipation.

This stability and dynamical behaviour can be understood by examining the velocity fields of the vortices, as shown in Fig. \[fig:vel\_field\] for like-signed vortices, and Fig. \[fig:vel\_pm\] for a vortex-antivortex pair. The velocity fields of the vortices are additive, which for like-signed windings creates regions of zero flow, and for opposite-windings creates large flows between them respectively. From Eq. \[eqn:1\_over\_r\] the velocity field of the irrotational quantum vortex scales as *v* ∝ 1/*r*, and hence becomes singular at the vortex centre, giving rise to the large velocities close to the singularity. Fig. \[fig:vel\_pm\_contour\] shows a clearer view of the velocity fields as depicted by Fig. \[fig:vel\_field\], where the presence of maxima is easily observed at the cores, with field minima also observed due to the vortex-vortex interactions.

<embed src="/ch4_vtx/fewvortex_rho.png" title="fig:" style="width:48.0%" /> <embed src="/ch4_vtx/fewvortex_theta.png" title="fig:" style="width:48.0%" />

<img src="/ch4_vtx/velocity/velocity_fixedmag.png" alt="Magnitude of the velocity field and the direction of rotation of the field for clockwise circulating vortices. As the velocity follows a 1/r profile the magnitude becomes singular at the centre of the vortices, which is capped here at v=2.4\times 10^{-3} (ms^{-1})." style="width:95.0%" />

<img src="/ch4_vtx/vtx_anti_velfield.png" alt="Velocity field direction and magnitude originating from vortex and antivortex present in a condensate. Colour scale is in units of (ms^{-1})." style="width:55.0%" />

<img src="/ch4_vtx/vtx_anti_velfield.png" title="fig:" alt="(Left) Velocity field originating from vortex and antivortex present in a condensate. (Right) Trajectories of vortex and antivortex in a condensate in the lab frame. Vortices travel on circulating paths which will always intersect through their starting position. Color is used to represent the same times for both vortex and antivortex, which is on the order of seconds." style="width:45.0%" /> <img src="/ch4_vtx/vtx_antivtx_traj.png" title="fig:" alt="(Left) Velocity field originating from vortex and antivortex present in a condensate. (Right) Trajectories of vortex and antivortex in a condensate in the lab frame. Vortices travel on circulating paths which will always intersect through their starting position. Color is used to represent the same times for both vortex and antivortex, which is on the order of seconds." style="width:45.0%" />

<img src="/ch4_vtx/velocity/vel.png" alt="Magnitude of the velocity field for small numbers of like-signed vortices in a condensate. The velocity field can be seen to become singular at the centre, which here has its magnitude capped at a lower value than Figs. [fig:vel_field] and [fig:vel_pm] to aid visibility. Regions of minimal velocity are seen, wherein the opposing field lines compensate one another." style="width:95.0%" />

Rapidly rotating vortex lattice
-------------------------------

### Model system

For the work to follow we assume a standard single component Bose–Einstein condensate in a radially symmetric trap of frequency *ω*<sub>⊥</sub> = 2*π* × 1 Hz. By tightly confining the condensate along the *z*-dimension, with trapping frequency *ω*<sub>*z*</sub> = 2*π* × 16 Hz, such that *ω*<sub>*z*</sub> is greater than *ω*<sub>⊥</sub>, the condensate enters the desired pancake-shaped geometry. The system is then modelled using the mean-field GPE Hamiltonian as
$$\\label{eqn:gpe\_h0}
	H\_{\\mathrm{GP}} = -\\frac{\\hbar^2}{2m}\\nabla^2 + \\frac{1}{2}m\\omega\_{\\perp}^2\\mathbf{r}^2 + g\_{\\textrm{2D}}\\vert\\Psi(\\mathbf{r},t)\\vert^2.$$

Here, we define the two-dimensional effective interaction strength *g*<sub>2D</sub> given by Eq. , as
$$g\_{\\textrm{2D}} = g\\sqrt{\\frac{m\\omega\_z}{2\\pi\\hbar}} = 4g \\sqrt{\\frac{m}{\\hbar}}.$$

To describe the system we enter the co-rotating frame, which is done by including the angular momentum operator *L*<sub>*z*</sub> in the Hamiltonian. The time dependent dynamics of the system are then given by the GPE as
iℏ∂<sub>*t*</sub>*Ψ*(**r**, *t*)=\[*H*<sub>GP</sub>−*Ω**L*<sub>*z*</sub>\]*Ψ*(**r**, *t*).

If the angular rotation frequency approaches the condensate trapping frequency, *Ω* ≈ *ω*<sub>⊥</sub>, the condensate gains a large triangular lattice of vortices. The effect of setting *Ω* = *ω*<sub>⊥</sub>, can be partially understood in a mean-field setting by rewriting the GPE kinetic and rotation terms in the form
$$\\frac{\\mathbf{p}^2}{2m} + \\frac{m\\Omega\\mathbf{r}^2}{2} - \\Omega L\_z = \\frac{\\mathbf{\\left(p -{\\textit{ m}}\\boldsymbol{\\Omega}\\times\\mathbf{r}\\right)^2}}{2m}.$$

The resulting changes to the GPE are then given by
$$\\label{eqn:vector\_potential\_gpe}
    \\textrm{i}\\hbar\\partial\_t \\Psi =
    \\left(\\frac{1}{2m}(-i\\hbar\\nabla - m\\boldsymbol{\\Omega}\\times\\mathbf{r})^2 + \\frac{m}{2}(\\omega\_\\perp^2 - \\Omega^2){r}^2 + g\_{\\textrm{2D}}|\\Psi|^2 \\right)\\Psi.$$
 The above Hamiltonian demonstrates the correspondence between a rotating condensate and that of a non-relativistic charged particle in a magnetic field . One can see that when the rotation and trapping frequencies are equal, the condensate no longer sees a confining potential, due to the centrifugal force experienced, given by the term −*m**Ω*<sup>2</sup>*r*<sup>2</sup>/2.

As stated earlier in Sec. \[sec:sec2\_vtxlatt\] as *Ω* approaches *ω*<sub>⊥</sub>, the use of mean-field theory becomes less justified, and the system enters a strongly correlated regime. If the filling fraction *ν* is in the range of 10 ≤ *ν* ≤ 1000, the system enters the “mean-field quantum Hall” regime, where the Gross–Pitaevskii theory is still working well. The system examined here is well within this regime for a frequency of *Ω* = 0.995*ω*<sub>⊥</sub>. This allows us to work with systems that have a very large vortex lattice, and are still described by mean-field theory .

In the following we will numerically solve Eq. , using the pseudospectral Fourier split operator method and making use of GPU computing as described in Sec. \[sec:GPUE\]. For realistic experimental parameters we assume *N* ≈ 10<sup>6</sup> atoms of **<sup>87</sup>Rb, with an *s*-wave scattering length of *a*<sub>*s*</sub> = 4.76 × 10<sup>−9</sup> m . The numerically evaluated ground-state for the given set of parameters is shown previously in Fig. \[fig:showingoff\] and has a radius of approximately 3.5 × 10<sup>−4</sup> m. For these parameters, the number of vortices within the visible density region is approximately 600, giving a filling factor of *ν* ≈ 800. This places the system within the mean-field quantum Hall regime, and therefore a description using Gross–Pitaevskii theory is adequate .

Following an imaginary time-evolution as outlined in Sec. \[sec:timeev\], we first find the ground state of the condensate in a harmonic potential, starting with a condensate without vortices. We then linearly ramp the rotation frequency to avoid the lattice disordering issue discussed in Sec. \[ss:ang\_mom\_fso\]. This allows us to follow the ground state solution at all times while the rotation frequency is increased, which has the added advantage of returning a ground state solution for any required rotation frequency. Examples of several states obtained during a single simulation are given in Fig. \[fig:inc\_omega\]. The previously discussed resolution considerations become apparent as the rotation rate is increased for both the position and momentum space representations of the wavefunction. A movie of the wavefunction density is available at the following URL , in which the frequency is ramped from *Ω*/*ω*<sub>⊥</sub> = 0.39 → 0.995.

The rapidly rotating vortex lattice is known to exhibit solid-body-like rotation . This can be seen in Fig. \[fig:solidbody\], where the coarse-grained flow of the velocity field increases as a function of the distance from the lattice centre. This rigid-body behaviour allows us to treat the lattice as a solid object. Recalling the Feynman relation for vortex density Eq. , *n*<sub>*v*</sub> = *m**Ω*/(*π*ℏ), we choose an area over which the lattice spacing is almost constant and the vortex density closely matches this relation. In this rapidly rotating regime the vortices close to the centre will have an almost uniform profile ; for the above system parameters this is fulfilled by a hard-walled radial boundary of *r* = 2 × 10<sup>−4</sup> m, in which the number of vortices are calculated to being *N*<sub>*v*</sub> ≈ 342. Employing the vortex detection and tracking methods described in Sec. \[sec:vortrack\], within the same region gives *N*<sub>*v*</sub> = 341 vortices, with a lattice spacing of *a*<sub>*v*</sub> ≈ 2.1 × 10<sup>−5</sup> m, and with a standard deviation of *σ* ≈ 2.7 × 10<sup>−7</sup> m. These values indicate that within this region the lattice is well ordered. While the condensate has more vortices outside this boundary that are initially ordered, during time evolution many of these move more easily due to the large velocity fields closer to the edges. The above boundary gives a well ordered lattice that remains well ordered for several seconds of time evolution.

<embed src="/ch4_vtx/ramp_omega_2.png" style="width:85.0%" />

If, however, the linear ramp is performed too quickly, or an initial state is chosen that already contains a large amount of angular momentum without being the eigenstate, the vortices tend to enter from the boundary all at once, and fail to converge to the well ordered ground state. A demonstration of this such issue is shown in Fig. \[fig:malformed\_lattice\], and indicates the need for a slow ramp of *Ω* that is essentially adiabatic in imaginary time. As higher rotation frequencies are reached, the rate at which vortices enter the condensate increases rapidly. While the rapid entry of vortices was a problem for reaching an ordered lattice without a ramp of rotation frequency, the presence of an existing lattice during ramping allows all newly entered vortices to order more easily. Therefore, a linear ramp is effective and does not require a more complex scaling.

<img src="/ch4_vtx/toofast_099_1e7.png" alt="Disordered lattice resulting from starting in imaginary time evolution at the required rotation rate (here \Omega=0.99\omega_\perp). If the rotation frequency is chosen too large without allowing the lattice to form and order, the resulting vortices all enter instantaneously and compete for their final positions. The system only converges one timescales that exceed reasonable computing times." style="width:65.0%" />

In the following section we will discuss perturbations of the condensate in the presence and absence of vortices. For the above system, we will investigate the effect of perturbations to both the global and local condensate order due to changes in the lattice structure.

Condensate perturbations
------------------------

### Trapping potential control

With the model system outlined in Sec. \[sec:modelsystem\], we will now imagine an abrupt change to the Hamiltonian, such that, *H*(*t*)=*H*<sub>GP</sub> + *f*(*t*)*V*<sub>ext</sub>, where *V*<sub>ext</sub> is an external potential, and *f*(*t*) is some function of time to control the application of *V*<sub>ext</sub>. In this scenario the initial wavefunction, which is a stationary state of *H*<sub>GP</sub>, will no longer remain so provided that *H*<sub>GP</sub> and *f*(*t*)*V*<sub>ext</sub> are non-commuting. Assuming the time of application of the additional term is much shorter than any other timescale of the condensate dynamics, any modification of the Hamiltonian in this way can be viewed as a method for changing the phase of the wavefunction. The resulting effect on the wavefunction can be given as

$$\\begin{aligned}
    \\Psi(t=0) &= |\\Psi(t=0)|e^{\\textrm{i}\\theta\_0} \\\\
    \\Psi(t) &= \\Psi(t=0) e^{ - \\textrm{i} \\frac{ V\_{\\textrm{ext}} \\Delta t}{\\hbar}} \\\\
            &= |\\Psi(t=0)| e^{\\textrm{i}\\left(\\theta\_0 - \\frac{V\_{\\textrm{ext}} \\Delta t}{\\hbar}\\right)} \\nonumber\\end{aligned}$$

where we have made use of Eq. . After application of *V*<sub>ext</sub> the wavefunction phase is given by
$$\\theta^{'} = \\theta\_0 - \\frac{E\_{\\textrm{ext}} \\Delta t}{\\hbar},$$
 where *E*<sub>ext</sub> is the perturbance energy. One commonly used method to manipulate the condensate is through the use of optical potentials, which offer a large degree of control over the respective system’s Hamiltonian . The electric field component of an arbitrary optical field, described by a wavevector **k**, and frequency, *ω* is given by
**E**(**r**, *t*)=*ε*<sub>0</sub>*e*<sup>i(**k**⋅**r**−*ω**t*)</sup> +  *ε*<sub>0</sub><sup>\*</sup>*e*<sup>−i(**k**⋅**r**−*ω**t*)</sup>,
 where *ε*<sub>0</sub> is the field amplitude. Using the dipole approximation, the interaction of an atom with a laser field is given by 
𝒱 = −**d** ⋅ **E**,
 where **d** is the electric dipole moment operator. For a two level atom with a ground state |*g*⟩, and an excited state |*e*⟩ with energy difference ℏ*ω*<sub>0</sub>, the dipole operator can be written as
$$\\begin{aligned}
\\label{eqn:dipole\_approx}
\\mathbf{d} &= \\langle g|\\mathbf{d}|e \\rangle | g \\rangle \\langle e | + \\langle e|\\mathbf{d}|g \\rangle | e \\rangle \\langle g | \\nonumber \\\\
&= \\boldsymbol{\\mu}\_{eg} | g \\rangle \\langle e | + \\boldsymbol{\\mu}\_{eg}^{\*} | e \\rangle \\langle g |,\\end{aligned}$$
 where we have made use of ⟨*g*|**d**|*g*⟩=⟨*e*|**d**|*e*⟩=0, since the atoms have no permanent dipole moment. From Eq.  the Hamiltonian of the two-level system can then be written as
$$\\begin{aligned}
\\label{eqn:dip\_hamiltonian}
    H &= H\_0  + \\mathcal{V} \\nonumber \\\\
      &=  \\hbar\\omega\_0 |e\\rangle\\langle e | - (\\boldsymbol{\\mu}\_{eg} | g \\rangle \\langle e | + \\boldsymbol{\\mu}\_{eg}^{\*} | e \\rangle \\langle g |)\\cdot ( \\varepsilon\_0 e^{\\textrm{i}\\left(\\mathbf{k}\\cdot\\mathbf{r} - \\omega t\\right)} +  \\varepsilon\_0^{\*} e^{-\\textrm{i}\\left(\\mathbf{k}\\cdot\\mathbf{r} - \\omega t\\right)}).\\end{aligned}$$
 After expanding  we can then rewrite **μ**<sub>*e**g*</sub> ⋅ *ε*<sub>0</sub> = ℏ*Ω*<sub>*r*</sub>, where *Ω*<sub>*r*</sub> is the Rabi-oscillation frequency between the states. Assuming that the detuning *Δ* = *ω* − *ω*<sub>0</sub> between the laser field and transition frequency is small, *Δ* ≪ *ω* + *ω*<sub>0</sub>, allows use of the rotating wave approximation. For this we first perform a unitary transformation of the system into the interaction picture rotating with *H*<sub>0</sub> using the operator *U* = *e*<sup>−i*ω*<sub>0</sub>*t*|*e*⟩⟨*e*|</sup> as
$$\\begin{aligned}
    H\_{\\textrm{int}} & = U^{\\dagger} \\mathcal{V} U.\\end{aligned}$$
 All resulting terms featuring *ω* + *ω*<sub>0</sub> can be considered to be rapidly oscillating, and average out to zero. Following this approximation, the Hamiltonian can be transformed back into the Schrödinger picture, giving
$$\\begin{aligned}
 H^{'} = \\hbar\\omega\_0 - \\hbar\\Omega\\left(e^{\\textrm{i}\\omega t}|e\\rangle\\langle g|   + e^{-\\textrm{i}\\omega t}|g\\rangle\\langle e|  \\right).\\end{aligned}$$
 The final shift of the energies is then given as
$$\\label{eqn:acshift}
\\mathcal{V} = -\\frac{\\alpha}{2}\\langle \\mathbf{E}^2\\rangle\_t ,$$
 where *α* is the real component of the atomic polarisability, and ⟨ ⋅ ⟩<sub>*t*</sub> denotes the time average. If the electric field is spatially inhomogeneous, then this leads to a force of the form
$$\\mathbf{F}\_d = \\frac{\\alpha}{2}\\nabla\\langle \\mathbf{E}^2 \\rangle\_t ,$$
 which is known as the dipole force. Assuming counter propagating plane waves, we can then model a standing wave solution of the resulting optical potential as
$$V\_{\\textrm{ext}} \\approx -\\frac{\\Omega\_0^2(\\mathbf{r})}{4\\Delta}  = V\_0 \\cos^2 (\\mathbf{k} \\cdot \\mathbf{r}),$$
 where *V*<sub>0</sub> = −*Ω*<sub>0</sub><sup>2</sup>/4*Δ* is the field intensity, and *Ω*<sub>0</sub> ∝ |*ε*<sub>0</sub>|<sup>2</sup> is the Rabi-frequency of the standing wave. The optical potential forms a highly periodic system given an appropriately chosen **k**, and is known as an *optical lattice*. Optical lattices have become very common in BEC experiments as they allow for control of the kinetic energy term of the atoms to a very high degree .

Different geometric potentials can be formed with optical lattices by using laser fields with different **k** vectors. Assuming standing waves with different polarisation axes or slightly different wavelengths, the interference effects between two or more fields can be ignored, with the resulting optical field given by the summation of lattice potentials with wavevectors **k**<sub>1, ..,*n*</sub>. Creating a 2D lattice with *n*-fold rotational symmetry requires *n*/2 **k**-vectors separated by 2*π*/*n*, and with *n*/2 ∈ ℤ<sup>+</sup>. Taking a square lattice as an example, which has a 4-fold rotational symmetry, it can be created by two **k**-vectors, separated by *π*/2, as
$$\\mathbf{k}\_0 = \\left\[ \\begin{array}{cc}
    1 \\\\
    0
    \\end{array} \\right\],~
    \\mathbf{k}\_1 =
    \\left\[ \\begin{array}{cc}
     0 \\\\
     1
    \\end{array} \\right\].\\label{eqn:sqlatt}$$
 The resulting potential is shown in Fig. \[fig:cos2xy\].

<img src="/ch4_vtx/VOPT/squarelatt.png" alt="Square lattice generation using two orthogonal propagating laser fields with wavevectors \mathbf{k}_1 and \mathbf{k}_2, as defined by Eq. ." style="width:55.0%" />

The time the optical lattice is applied to the condensate can be controlled by choosing the function *f*(*t*). Applying a lattice for a finite, but short, time only will lead to a modification of the wavefunction phase, which then subsequently, and on a much longer time-scale, will have an effect on the density distribution. Of particular interest to us is the use of an optical potential that is pulsed one or several times, which can be described by *f*(*t*) as a periodic delta function. The condensate phase profile is the only quantity immediately modified, and any change in the density distribution appears only in the following evolution.

For the purpose of my system, we intend to create a two-dimensional optical lattice, wherein the structure of the lattice matches that of the triangular Abrikosov vortex pattern. The triangular lattice has 6-fold rotational symmetry, and can be formed with wavevectors

$$\\begin{aligned}
        \\mathbf{k}\_1 &= k\_0\\left\\{\\frac{\\sqrt(3)}{2},\\frac{1}{2}\\right\\} \\\\
        \\mathbf{k}\_2 &= k\_0\\{0,1\\} \\\\
        \\mathbf{k}\_3 &= k\_0\\left\\{\\frac{\\sqrt(3)}{2},-\\frac{1}{2}\\right\\} \\\\
    \\end{aligned}$$

where $k\_0 = 4\\pi/(\\sqrt(3)a\_\\text{O})$, and *a*<sub>O</sub> is the lattice spacing.

### Direct phase manipulation

While ground state condensates will have a flat phase across the system, there are two interesting examples where a spatially dependent phase exists: dark solitons  and vortices . We have previously discussed the 2*π* phase profile of a vortex that leads to the singularities in the wavefunction. In contrast, dark solitons feature a *π* phase jump profile. These excitations are unstable in dimensions higher than one, and will decay via the snake instability to paired vortices and antivortices . Where so far we have only considered the short-term evolution of the wavefunction after being kicked by an optical potential, we will in the following consider what structures can be created in the condensate by careful phase manipulation techniques, besides dark solitons and vortices.

For this we will assume that the BEC allows for a short enough application of potentials so that only the phase is affected, and discuss direct manipulation of the wavefunction, as opposed to modification of the Hamiltonian. Following and taking Eq. , the phase of the condensate can then be written as
*θ* = *θ*<sub>0</sub> + *θ*<sub>*i*</sub>,

where *θ*<sub>0</sub> is the unperturbed condensate phase, and *θ*<sub>*i*</sub> is the phase pattern to be imprinted. Upon solving for the initial condensate ground state with deterministic phase, an additional phase pattern can be imprinted at any time by simply multiplying the wavefunction by *e*<sup>i*θ*<sub>*i*</sub></sup>. However, without careful choice of the phase terms their addition can lead to unwanted dynamics, so care must be taken to choose a well defined initial and imprinted phase pattern.

The advantage of the phase imprinting model is that for topological defects, one can imprint the required winding instantaneously, allowing them to appear at predefined positions. The density also needs to only adjust itself locally to the phase singularity, with the remaining condensate seeing an almost constant shift in phase. The creation of vortices through application of localised ±2*π* phase winding defects in the condensate therefore allows for direct control of the angular momentum and vorticity within the BEC. While discussed in the literature for the creation of vortices, it is worth noting that the phase imprinting method can also be used to annihilate a vortex from the condensate by applying a phase profile of opposite winding, removing the singularity. This will leave the condensate with a density depletion at the prior location of the phase singularity. Without the phase singularity this depletion will fill in and excite phonon modes in the condensate during time evolution. This process will form the basis for further discussions and analysis of vortex carrying condensates.

Experimental realisation of arbitrary potential patterns to achieve the required phase is accessible through the use of spatial light modulators (SLM) . These devices behave as digital displays, through which visual patterns can be expressed in a time dependent manner, allowing the application of a laser field in the required form. We will assume for all future discussions that the potentials we require are experimentally realisable with sufficient resolution, and focus on the resulting effect on the condensate. For the creation of a single vortex the 2*π* phase winding pattern can be created spatially using the two-argument four-quadrant form of arctan as given by
*θ*<sub>*i*</sub>(**x**, **y**; *x*<sub>0</sub>, *y*<sub>0</sub>)=arctan(**y** − *y*<sub>0</sub>, **x** − *x*<sub>0</sub>),
 which locates the singularity at the position (*x*<sub>0</sub>,*y*<sub>0</sub>). The resulting phase is shown in Fig. \[fig:atan2phase\](left), and including the additional phase singularity term *θ*<sub>*i*</sub>, the condensate wavefunction following an imprint is given as
*Ψ*<sup>′</sup>(**r**, *t*)=|*Ψ*(**r**, *t*)|*e*<sup>i(*θ*<sub>0</sub>(**r**, *t*)+*θ*<sub>*i*</sub>(**r**))</sup>.

<embed src="/ch4_vtx/2pi.png" title="fig:" style="width:45.0%" /> <embed src="/ch4_vtx/3_2pi.png" title="fig:" style="width:43.5%" />

Following the imprint this process will create phonons in the condensate density that will radiate outwards from the singularity imprint. As imprinting is directly controlling the condensate phase, it can also be considered a direct manipulation of the kinetic energy since the superfluid velocity depends on the phase gradient (see Eq. ). This, in reverse, means that by applying spatially inhomogeneous phase profiles the atomic velocity can be adjusted to different values in different regions of the condensate. To demonstrate this we consider a simple example of a Gaussian phase profile applied to the condensate. The imprinted profile has the form
$$\\theta\_{i}(\\mathbf{r}) = A\\exp\\left( -\\frac{ |\\mathbf{r}-\\mathbf{r}\_0|^2 }{2\\sigma^2 } \\right) \\mod 2\\pi,$$
 where *A* is the phase profile amplitude, **r**<sub>0</sub> is the centre of the Gaussian curve, and *σ* is adjusted to match the condensate width. The modulo 2*π* ensures that the phase wraps around for amplitudes exceeding the (0, 2*π*) range. The Gaussian profile has large radial gradients in two-dimensions, so that an imprint on the condensate should lead to radial velocities and therefore an expansion or contraction of the cloud (see Fig. \[fig:gaussian\]). This can be expected to lead to interference fringes, as the faster moving atoms have the possibility to overtake the slower ones for sufficiently large amplitudes . In Fig. \[fig:gaussian\_wfc\], where a slice through the condensate centre is given for both position and momentum space, this can be clearly seen.

<img src="/ch4_vtx/velocity/gaussian_imprint.png" alt="Velocity fields and magnitude (left) and phase (right) for a condensate directly following a Gaussian phase imprint. The lengths of the arrows give the magnitude of the respective velocity components, with the color map indicating this also for clarity. The sign of the imprint changes the direction of the respective kinetic components, with a positive imprint initially creating a density contraction (top), and negative creating an expansion (bottom), with the arrows indicating the direction of the flow." style="width:95.0%" />

<img src="/ch4_vtx/gaussian_imprint_B.png" alt="A cut through the condensate wavefunction density following the phase imprinting of a Gaussian with amplitudes A=\pm ( 2\pi, 6\pi, 10\pi). The positive imprints create an initial contraction of the cloud (top), while the negative imprints lead to expansion (bottom). For the larger kicking strengths interference fringes can be observed during expansion and contractions." style="width:95.0%" />

Condensate analysis
-------------------

To analyse the effects of the phase imprinting we will below introduce the decomposition of the kinetic energy to isolate the effect from phonons and vortices. Following this, for the vortex lattice we will introduce two closely linked methods to examine geometric structure — Delaunay triangulation and Voronoi tessellation. These methods are dual to one another, and can be used to easily identify order, structure and local parameters within systems of many particles.

### Kinetic energy decomposition

Given that the phase engineering modifies the condensate kinetic energy profile, it is instructive to quantify this effect. One can apply a spectral decomposition of the kinetic energy of the condensate into contributions solely from the vortices (incompressible), and those from the phonons (compressible) . For this, the wavefunction is again written in terms of amplitude $\\sqrt{\\rho(\\mathbf{r},t)}$ and phase *S*(**r**, *t*), which allows the kinetic part of the Gross–Pitaevskii energy functional Eq.  to be calculated as
$$E\_{\\text{kqp}} = \\int d\\mathbf{r} \\left( \\frac{\\hbar^2}{2m}| \\nabla\\sqrt{\\rho(\\mathbf{r},t)} |^2  + \\frac{m}{2}|\\sqrt{\\rho(\\mathbf{r},t)}\\mathbf{v}(\\mathbf{r},t) |^2\\right).$$
 One can then decompose this into the quantum pressure (first) and kinetic energy (second) terms. The kinetic energy term can be seen as a density-weighted velocity field, $\\mathbf{u}(\\mathbf{r},t) = \\sqrt{\\rho(\\mathbf{r},t)}\\mathbf{v}(\\mathbf{r},t)$, and it can be further decomposed into the sum of compressible and incompressible terms,
**u****(****r**, *t*)=**u**<sup>*c*</sup>(**r**, *t*)+**u**<sup>*i*</sup>(**r**, *t*).
 The two terms can be calculated by performing a Helmholtz decomposition of the field **u**, which separates terms that are longitudinal (**u**<sup>*c*</sup>) and transversal (**u**<sup>*i*</sup>) with

\[eqn:kinterms\]
$$\\begin{aligned}
    \\nabla \\times \\mathbf{u}^c(\\mathbf{r},t) &= 0, \\\\
    \\nabla \\cdot \\mathbf{u}^i(\\mathbf{r},t) &= 0.\\end{aligned}$$

By introducing the vector potential, **A**, and the scalar potential, *B*, such that

$$\\begin{aligned}
    \\mathbf{u}^c = \\nabla B, \\\\
    \\mathbf{u}^i = \\nabla \\times \\mathbf{A},\\end{aligned}$$

we can rewrite Eq.  as
$$\\begin{aligned}
    \\nabla \\times \\mathbf{u}(\\mathbf{r},t) = -\\nabla^2 \\mathbf{A}, \\\\
    \\nabla \\cdot \\mathbf{u}(\\mathbf{r},t) = \\nabla^2 {B}.\\end{aligned}$$

To solve the above equation we begin by seeking a solution for *B* by performing a spectral decomposition of the full density-weighted velocity field as
$$B = \\displaystyle\\sum\\limits\_{j} \\frac{k\_j}{|\\mathbf{k}|^2}\\mathscr{F}\[\\mathbf{u}\],$$
 where *k*<sub>*j*</sub> is the *j*-th component in **k** space, and ℱ is the Fourier transform. The resulting solution for **u**<sup>*c*</sup> is then given by
$$\\mathscr{F}\[\\mathbf{u}\_i^c\] = \\displaystyle\\sum\\limits\_{j} \\frac{k\_i k\_j}{|\\mathbf{k}|^2} \\mathscr{F}\[\\mathbf{u}\],$$
 which after taking note of Eq.  gives
$$\\begin{aligned}
    \\mathscr{F}\[\\mathbf{u}\_i^i\] &= \\mathscr{F}\[\\mathbf{u}\_i\] - \\mathscr{F}\[\\mathbf{u}\_i^c\]. \\\\
    &= \\displaystyle\\sum\\limits\_{j}\\left(\\delta\_{i,j} - \\frac{k\_ik\_j}{|\\mathbf{k}|^2}\\right)\\mathscr{F}\[\\mathbf{u}\_i\]. \\nonumber\\end{aligned}$$

This decomposition separates the energy contribution from phonons and vortex cores, represented by compressible and incompressible terms respectively . By averaging over binned shells in **k**-space, the kinetic energy spectra, *E*<sup>*c*, *i*</sup>(*k*), are calculated as 
$$\\label{eqn:kin\_spec\_ic}
	E^{c,i}(k) = \\frac{mk}{2}\\sum\\limits\_{j\\in\\mathbf{r}} \\int\\limits\_{0}^{2\\pi}d\\phi\_k \\frac{ |\\mathcal{U}\_j^{c,i}(\\mathbf{k},t) |^2}{s\_k},$$
 where
𝒰<sub>*j*</sub><sup>*c*, *i*</sup>(**k**, *t*)=∫*d*<sup>2</sup>**r***e*<sup>−i(**k** ⋅ **r**)</sup>*u*<sub>*j*</sub><sup>*c*, *i*</sup>(**r**, *t*).
 The terms *u*<sub>*j*</sub><sup>*c*, *i*</sup>(**r**, *t*) represent the position-space density-weighted velocity components in the specified shell, where *ϕ*<sub>*k*</sub> is the polar angle, and *s*<sub>*k*</sub> is the number of values in the chosen shell.

### Delaunay triangulation and Voronoi tessellation

A common method for examining the ordering and periodicity of large-scale crystalline structures is to generate a mesh with each vertex being the location of a particle. With this, one can easily observe ordered and disordered regions in a material; well defined straight lines indicate a perfect crystal, with any bends indicating the presence of imperfections. Some of the most widely used methods for this are the dual techniques from computational geometry of Delaunay triangulation and Voronoi tessellation.

The Delaunay triangulation of an arbitrary set of points in Euclidian space, **R**, which we will denote as *D*(**R**), is constructed in the following way:

1.  No point will fall within the interior of any circumcircle of 3 points where **r**<sub>1..3</sub> ⊂ **R**

2.  The Delaunay triangulation will maximise the minimum angle between points.

3.  If four points are on the same circumcircle, then both possible configurations give a Delaunay triangulation.

<img src="/ch4_vtx/deltri.png" alt="Non-Delaunay (a) and Delaunay (b) triangulation of 4 Euclidian points." style="width:45.0%" />

This concept is more easily explained visually. Fig. \[fig:delaun\] shows two different triangulations of four points; situation (*a*) is a non-Delaunay triangulation, as the points **r**<sub>1</sub> and **r**<sub>3</sub> each fall within the circumcircle of the other points. However, by simply flipping the central edge from (**r**<sub>2</sub>, **r**<sub>4</sub>) to (**r**<sub>1</sub>, **r**<sub>3</sub>) we can see in (*b*) that we now have a valid Delaunay triangulation. No point falls within the circumcircle of the other points, and the minimum angle formed is maximised relative to configuration (*a*). Following directly from this, one can see that Delaunay triangulation can be used to connect the closest vertices in a network. A nice side-effect of Delaunay triangulation is that one can examine when the number of edges from a vertex deviates from the expected value in the lattice, which is 6 for triangular lattices. This can be a useful means to locate defects in a crystal lattice, and we will make use of this during later discussions. This is performed using the built-in <span style="font-variant:small-caps;">MATLAB</span> function “delaunayTriangulation”, and counting the number of attachments to each individual vertex. A triangulation of the vortex lattice from Fig. \[fig:showingoff\] within the previously discussed radial boundary of *r* = 2 × 10<sup>−4</sup> m is shown in Fig. \[fig:delaun\_vtxlatt\].

<img src="/ch4_vtx/Del_tr_VTXLATT.png" alt="Delaunay triangulation of the vortex lattice ground state. The vertices away from the condensate boundary have the expected 6-edge structure." style="width:45.0%" />

An alternative representation, using the dual of the Delaunay triangulation, is that of the Voronoi tessellation (diagram). The characteristic of these diagrams is that they are composed of cells each encompassing an individual vertex, within which all enclosed points are closer to that particular vertex than any other. This representation can be generated from the Delaunay triangulation and vice-versa. Taking the centres of the circumcircles describing the Delaunay triangulations, and connecting these forms the boundaries of the Voronoi cells. A simple generation method can be seen as creating and expanding the radius of circles (or *n*-spheres in *n*-dimensions) centred on each vertex. Where the circles intersect with one another defines the boundary of each individual cell. An example of a Voronoi diagram compared with a Delaunay triangulation is given by Fig. \[fig:Voronoi\].

<img src="/ch4_vtx/voronoi.png" alt="Comparison of Delaunay triangulation (a) with a Voronoi diagram (b). These graphs are duals, which means that one can be used to generate the other." style="width:55.0%" />

The area of each cell can be used as a metric of the strength of the interaction between particles in a many-body system, but also we may represent quantities local to each region in a system by the colour-scale of each cell. For the vortex lattice as given by Fig. \[fig:showingoff\], a sample Voronoi diagram is given in Fig. \[fig:voron\_vtxlatt\] with the color representing the area spanned by each Voronoi cell of the lattice.

<img src="/ch4_vtx/Voronoi_area_VTXLATT.png" alt="Voronoi tesselation of the vortex lattice. The area of each cell is represented by the color mapping. To avoid the tessellation tending to infinity, a buffer region of vortices is created close to the boundary. " style="width:55.0%" />