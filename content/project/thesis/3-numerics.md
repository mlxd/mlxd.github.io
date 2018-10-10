+++
# Date this page was created.
date = "2017-01-18"

# Project title.
title = "Thesis - 3. Numerics"

# Project summary to display on homepage.
summary = "Thesis - 3. Numerics"

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
draft = true

+++
Numerical methods
=================

Numerical solutions to problems in quantum physics are important, given the limited availability of exact solutions. Many such models and methods exist when dealing with many-body systems, and have been shown to provide good estimates of physical behaviour . Techniques such as Monte Carlo methods, exact diagonalisation, and DMRG are used to solve a wide variety of many-body problems, but often fail to capture the full system dynamics of such problems . For understanding the behaviour of systems such as Bose–Einstein condensates, the use of these methods is rather limited. For DMRG, the complexity of the problem grows quickly with increasing dimensionality and renders this technique unusable. Exact diagonalisation requires a linearised system to obtain realistic solutions, and also grows significantly in complexity with increased dimensionality. Monte Carlo methods generally do not allow for real-time dynamics, or allow one to calculate the underlying wavefunction.

To obtain solutions to BEC problems we make use of a mean-field approach, outlined previously in Sec. \[sub:gpederiv\]. Using the GPE and performing a numerical integration allows for almost all examinable dynamics that are valid in the mean-field limit. It can be noted though that the computational cost increases significantly with increased dimensions, and is already non-trivial for two-dimensions. The following chapter will introduce the necessary requirements to numerically solve quantum problems using state-of-the-art computational methods.

We will begin with an introduction to the time evolution of a quantum state. We then discuss the use of the time evolution approach to find the ground state of a quantum system using imaginary time evolution. After this necessary mathematical introduction we will discuss the implementation of both real and imaginary time evolution using the Fourier split-operator (split-step) algorithm. We give error bounds for the algorithm, and discuss its use in the context of solving for Hamiltonian dynamics. Though the discussed algorithm is well suited to solving quantum dynamics, the computational cost can be quite high, especially for systems with large grid sizes.

We next discuss ways to overcome this through the use of high performance computing methods, and introduce the concept of graphical processing unit (GPU) computing. We present many of the necessary considerations for mapping a computational problem onto GPUs. Making use of GPUs to numerically solve the Schrödinger equation with the Fourier split-operator method, we present the problem of coherent atomic transport. We introduce the “matter-wave spatial adiabatic passage” technique, with the goal of coherently transporting an atom between trapping potentials with high fidelity. The design of the model system is presented, and the results are shown.

We finish the chapter by introducing the developed algorithms for condensate systems. Performance metrics and considerations are given in the context of solving the Gross–Pitaevskii equation in the presence of vortices.

Time evolution
--------------

Given a quantum state, to examine the dynamics requires an understanding of how it evolves in time. Assuming a quantum state at time *t*<sub>0</sub> to be defined as |*Ψ*(*t*<sub>0</sub>)⟩, and a state at time *t* to be |*Ψ*(*t*)⟩, the two states can be connected with a unitary evolution operator as
|*Ψ*(*t*)⟩ = 𝒰(*t*, *t*<sub>0</sub>)|*Ψ*(*t*<sub>0</sub>)⟩, 
 where we have assumed that *t* &gt; *t*<sub>0</sub>. One can write the wavefunction of a quantum system as the linear superposition of a set of basis states |*Ψ*<sub>*m*</sub>⟩ as
$$\\label{eqn:psicomplete}
    |\\Psi \\rangle = \\displaystyle\\sum\\limits\_{m} C\_m |\\Psi\_m \\rangle,$$
 with coefficients *C*<sub>*m*</sub>. The unitary evolution operator can be written as
$$\\begin{aligned}
   \\mathscr{U}(t,t\_0) &= \\exp\\left(-\\frac{i\\mathcal{H}(t - t\_0)}{\\hbar} \\right) \\\\ &= \\exp\\left(\\frac{-\\text{i}\\mathcal{H}\\delta t}{\\hbar}\\right),\\end{aligned}$$
$$\\begin{aligned}
   \\mathscr{U}(t,t\_0) &= \\exp\\left(-\\frac{i}{\\hbar}\\displaystyle\\int\\limits\_{t\_0}^{t}\\mathcal{H}\\text{d}t\\right),\\end{aligned}$$
 where ℋ is the Hamiltonian of the system, and *δ**t* = *t* − *t*<sub>0</sub>. The effect of 𝒰(*t*, *t*<sub>0</sub>) on any arbitrary state can be examined by expanding it as

$$\\begin{aligned}
        \\mathscr{U}(t,t\_0) &= 1 - \\frac{\\textrm{i}\\mathcal{H}\\delta t}{\\hbar} + \\mathcal{O}(\\delta t^2) \\\\
        &= 1 - \\frac{\\textrm{i}\\delta t}{\\hbar}{\\displaystyle\\sum\\limits\_{n} E\_n |\\Psi\_n\\rangle \\langle \\Psi\_n |} + \\mathcal{O}(\\delta t^2),
    \\end{aligned}$$

where the Hamiltonian operator has been written as a complete set of energy eigenkets, ℋ|*Ψ*<sub>*n*</sub>⟩=*E*<sub>*n*</sub>|*Ψ*<sub>*n*</sub>⟩. Applying this to gives

$$\\begin{aligned}
        |\\Psi (t) \\rangle &= \\left( 1 - \\frac{\\textrm{i}\\delta t}{\\hbar}\\displaystyle\\sum\\limits\_{n}E\_n|\\Psi\_n\\rangle\\langle \\Psi\_n |  \\right)\\displaystyle\\sum\\limits\_{m} C\_m |\\Psi\_m \\rangle \\\\
            &= \\displaystyle\\sum\\limits\_{m} C\_m |\\Psi\_m \\rangle + \\displaystyle\\sum\\limits\_{n}\\left( - \\frac{\\textrm{i}\\delta t}{\\hbar}E\_n \\right) C\_n|\\Psi\_n\\rangle,
    \\end{aligned}$$

where terms of order 𝒪(*δ**t*<sup>2</sup>) and higher have been temporarily left out for notational simplicity, but are still included in the analysis. Time evolving the state from *t*<sub>0</sub> to a final time *t* by applying the evolution operator can then be written as
$$\\mathscr{U}(t,t\_0)|\\Psi(t\_0) \\rangle = \\displaystyle\\sum\\limits\_{n} C\_n \\exp\\left(\\frac{-\\textrm{i}{E\_n}\\delta t}{\\hbar}\\right)|\\Psi\_n \\rangle.$$
 It follows from here that each state oscillates at a different rate, proportional to its eigenenergy; higher energy states will oscillate faster than those of lower energy. For a given set of states the dynamics and evolution of the quantum system can be fully determined for all times using the above evolution operation.

As systems will prefer to reside in the lowest energy state where they are most stable, it is often required to determine the ground state solution of a particular Hamiltonian. A common method for this is by evolving the system in imaginary time. Taking the evolution operator, and applying a Wick rotation rotates the time component through *π*/2 into the imaginary plane, as *t* → −*i**t*. This new evolution operator applied to the wavefunction gives
$$\\mathscr{U^{'}}(t,t\_0)|\\Psi \\rangle = \\displaystyle\\sum\\limits\_{n} C\_n \\exp\\left(\\frac{-{E\_n}\\delta t}{\\hbar}\\right)|\\Psi\_n \\rangle.$$
 This process removes the complex term in the operator, which now takes the form of sums of exponentially decaying states. When applied to the wavefunction the higher energy terms will decay at a rate faster than lower energy components increasing *δ**t*. This process also causes a loss of probability density, and so the wavefunction must be renormalised after each application. Through repeated application of this operator, and a renormalisation afterwards, the simulated quantum system converges to the ground state solution. To begin, however, we must make an initial guess for the wavefunction, which has some finite overlap with the lowest lying state. It should be noted that this method is a mathematical trick used to obtain a simulated ground state, with a real-world system only tending to the ground state in the presence of some form of dissipation. As effective as this technique is, the convergence to the lowest lying energy state becomes less effective as the computation approaches the expected value , and if many eigenstates are lying close to each other. To ensure the system converges to a sufficient degree the resulting energy can be checked after each iteration, and the evolution stopped only when the energy change fluctuates about a stable value.

With the time evolution method introduced, we will next discuss implementing this method. Although many such algorithms exist to implement time evolution, one that is well suited for this task is the Fourier split-operator method. This method works equally for real, as well as imaginary, time evolution.

Fourier split-operator method
-----------------------------

The Gross–Pitaevskii equation is a second order nonlinear partial differential equation, and so very few exact solutions exist; the problem must often be tackled by a numerical approach. Though there are many ways to solve such a system numerically, with the Crank–Nicolson and Trotter–Suzuki algorithms being notable examples, the method we have chosen is the pseudospectral Fourier split-operator method, described below .

If we consider a unitary evolution operator of the form
$$\\label{eqn:1}
\\Psi(\\mathbf{x},t+\\tau) = \\exp\\left( -\\frac{\\text{i}\\hat{H}\\tau}{\\hbar}\\right)\\Psi(\\mathbf{x},t),$$
 where $\\hat{H}$ is the Hamiltonian, composed of momentum, potential, nonlinear interaction, and rotation terms defined in Eq. , we can solve for the wavefunction and its resulting dynamics over a specified timescale, assuming *τ* is a short time increment such that the formalism given in Sec. \[sec:timeev\] is valid. Care must be taken during the implementation of such integration methods, as the loss of precision due to floating-point rounding, as well as the propagation of errors cannot be neglected. If we take $\\hat{H}$ in terms of its components as a combination of position and momentum space operators we obtain
$$\\label{eqn:2}
\\hat{H} = \\hat{H}\_{\\textbf{r}} + \\hat{H}\_{\\textbf{k}} + \\hat{H}\_{\\textbf{L}},$$
 where we first ignore the angular momentum operator, $\\hat{H}\_{\\textbf{L}}$, and consider only the two other non-commuting parts, $\\hat{H}\_{\\textbf{r}}$, containing the operators acting in position space, and $\\hat{H}\_{\\textbf{k}}$, containing the operators acting in momentum space only. The Baker–Campbell–Hausdorf formula gives the relation for non-commuting operators as
$$\\exp\\left( \\tau(A+B) \\right) = \\exp\\left(\\tau A\\right)\\exp\\left(\\tau B\\right)\\exp\\left(-\\frac{\\tau^2}{2}\[A,B\] + \\cdots\\right),$$
 with ⋯ representing higher order commutators. This is directly mappable to the above Hamiltonian for time evolution. Due to the non-commutativity of $\\hat{H}\_{\\textbf{r}}$ and $\\hat{H}\_{\\textbf{k}}$, the above expression cannot be evaluated exactly, and so it is common to Taylor expand and truncate it. The resulting error can be determined as

\[eqn:error\_calc\]
$$\\begin{aligned}
    \\text{err} = \\left\\| \\exp\\left(-\\frac{\\textrm{i}\\hat{H}\_{\\textbf{k}}\\tau}{\\hbar}\\right)\\exp\\left(-\\frac{\\textrm{i}\\hat{H}\_{\\textbf{r}}\\tau}{\\hbar}\\right) - \\exp\\left(-\\frac{\\textrm{i}(\\hat{H}\_{\\textbf{k}} + \\hat{H}\_{\\textbf{r}})\\tau}{\\hbar}\\right) \\right\\| \\\\
    = \\left\\|  \\left(1 + \\left(\\frac{-\\textrm{i}\\hat{H}\_{\\textbf{k}}}{\\hbar}\\right)\\tau + \\left(\\frac{-\\textrm{i}\\hat{H}\_{\\textbf{k}}}{\\hbar}\\right)^2\\frac{\\tau^2}{2}  \\right)\\left(1 + \\left(\\frac{-\\textrm{i}\\hat{H}\_{\\textbf{r}}}{\\hbar}\\right)\\tau + \\left(\\frac{-\\textrm{i}\\hat{H}\_{\\textbf{r}}}{\\hbar}\\right)^2\\frac{\\tau^2}{2}  \\right) \\right. &- \\nonumber \\\\ \\left. \\left(1 + \\left(\\frac{-\\textrm{i}(\\hat{H}\_{\\textbf{r}} + \\hat{H}\_{\\textbf{r}})}{\\hbar}\\right)\\tau + \\left(\\frac{-\\textrm{i}(\\hat{H}\_{\\textbf{r}} + \\hat{H}\_{\\textbf{r}})}{\\hbar}\\right)^2\\frac{\\tau^2}{2}  \\right)  + \\mathcal{O}(\\tau^3) \\right\\|,\\end{aligned}$$

which, upon simplification reduces to
$$\\text{err} = \\left\\| \\frac{\\tau^2\[{\\hat{H}\_{\\textbf{r}}},{\\hat{H}\_{\\textbf{k}}}\]}{2\\hbar^2} + \\mathcal{O}(\\tau^3)\\right\\| = \\mathcal{O}(\\tau^2).$$

The error can be further reduced through the use of 2**<sup>nd</sup> order Strang splitting , taking the error in the numerical integration scheme to 𝒪(*τ*<sup>3</sup>), with the resulting operator implementation given as

$$\\label{eqn:3}
\\exp\\left( -\\frac{ \\textrm{i}\\left(\\hat{H}\_{\\textbf{r}} + \\hat{H}\_{\\textbf{k}}\\right)\\tau}{\\hbar} \\right) = \\exp\\left(- \\frac{\\textrm{i}\\hat{H}\_{\\textbf{r}}\\tau}{2\\hbar} \\right)\\exp\\left(-\\frac{\\textrm{i}\\hat{H}\_{\\textbf{k}}\\tau}{\\hbar}\\right)\\exp\\left( -\\frac{\\textrm{i}\\hat{H}\_{\\textbf{r}}\\tau}{2\\hbar}\\right) + \\mathcal{O}\\left(\\tau^3\\right).$$

In the case of a nonlinear system, such as for solving the GPE, the above scheme attains a second-order error, resulting from the combination of the potential and nonlinear terms, with the respective mapping as

$$\\begin{aligned}
\\hat{H}\_{\\textbf{r}} &= V(\\mathbf{r}) + g\\vert\\Psi(\\mathbf{r},t)\\vert^2, \\\\ \\hat{H}\_{\\textbf{k}} &= \\frac{-\\hbar^2}{2m}\\nabla^2.
    \\end{aligned}$$

Following Bauke *et al*. , we can numerically solve this differential equation as
$$\\label{eqn:baukeetal}
\\Psi\\left(\\textbf{r},t+\\tau\\right) = \\left\[\\hat{U}\_{\\mathbf{r}}\\left(\\frac{\\tau}{2}\\right) \\mathscr{F}^{-1} \\left\[ \\hat{U}\_{\\mathbf{k}}(\\tau) \\mathscr{F} \\left\[ \\hat{U}\_{\\mathbf{r}}\\left(\\frac{\\tau}{2}\\right) \\Psi\\left(\\mathbf{r},t\\right) \\right\] \\right\] \\right\]  \\\\ + \\mathcal{O}\\left(\\tau^2\\right),$$
 where $\\hat{U}\_{\\mathbf{r}}(\\tau)=e^{-\\textrm{i}\\hat{H}\_{\\mathbf{r}}t/\\hbar}$ is the time evolution operator in position space, $\\hat{U}\_{\\mathbf{k}}(\\tau)=e^{-i\\hat{H}\_{\\mathbf{k}}t/\\hbar}$ the time evolution operator in momentum space, and ℱ and ℱ<sup>−1</sup> are the forward and inverse Fourier transform respectively. Taking the Fourier transform of the wavefunction allows the basis to be transformed between position and reciprocal space, wherein the time evolution operators are diagonal in each respective space. Figure \[fig:num\_splitop\] outlines a schematic representation of the method during a single pass of the algorithm.

![A single pass through the Fourier split-operator method.<span data-label="fig:num_splitop"></span>](/ch3_numerics/splitop.png)

The underlying theory of the Fourier split-operator method for the Gross–Pitaevskii equation is given by Javanainen *et al*. , showing how the choice of nonlinearity and operator splitting affects the outcome of the method. By taking the initial step as evolution in momentum space, the choice of the most current wavefunction attains an error of third-order for the algorithm. However, this will require an additional two Fourier transform steps, and as such is rather costly in compute time for large systems. For an initial step in position space, the nonlinear term is best calculated using a linear combination of all available wavefunctions through the algorithm as *Ψ* = *c*<sub>0</sub>*Ψ*<sub>0</sub> + *c*<sub>1</sub>*Ψ*<sub>1</sub> + *c*<sub>2</sub>*Ψ*<sub>2</sub>, where the subscripts denote the wavefunction at each stage of the evolution in position space, as indicated in Fig. \[fig:num\_splitop\], and the *c*<sub>x</sub> are linear coefficients. This gives third order accuracy for the parameters *c*<sub>2</sub> = ±1, *c*<sub>1</sub> = −*c*<sub>0</sub>. However, for simplicity and resource limitations we chose to work with the 𝒪(*τ*<sup>2</sup>) accurate scheme as depicted by Eq. , which was sufficient for the physics we aimed to describe.

An implementation of this method is a straight-forward process using MATLAB, and has been performed for the purpose of this study. However, due to the computational overhead required to time-evolve such a system, the procedure takes a long time to simulate at the required degree of accuracy for any dimension greater than one. Therefore, it is necessary to further develop the methods used, and to improve the implementation of this algorithm to leverage the recent advances in computational acceleration.

### Resolution considerations

As the Fourier split-operator method requires special consideration of resolution in both position and momentum space, care must be taken while choosing numerical grids. The reciprocal relationship between position and momentum space is
$$k\_{\\text{max}} = \\frac{2\\pi}{\\Delta x},$$
 which follows directly from the uncertainty relation; better resolution in one space leads naturally to worse in the other. To allow for a condensate to be simulated efficiently in both spaces, it must fit within the grid on which it is defined, and resolve to at least half the size of the smallest structure. It is easy to estimate a radius for the position space wavefunction, following the Thomas–Fermi approximation. It is also rather easy to know that for a non-rotating condensate the wavefunction should occupy the lowest lying mode (**k** = 0), and those close to it, assuming a harmonic trap. Rotating the condensate, however, has the effect of expanding the wavefunction in position space due to centrifugal forces. Additionally, the momentum space wavefunction also expands with increased angular momentum. With the addition of vortices to the system, there are now small scale structures to resolve. This leads to a system that is difficult to simulate; we have a simultaneously growing position space and momentum space wavefunction.

For a grid to effectively sample the wavefunction and capture all dynamics it will require a sampling rate of at least twice the smallest feature size following the Nyquist sampling theorem . From this it is essential to have a large and finely sampled grid in order to resolve both position and momentum of the wavefunction, with all included features. For the simulations presented below a minimum grid size on the order of 2<sup>8</sup> = 256 for low rotation rates, to 2<sup>11</sup> = 2048 at high rotation rates in 2D for both *X* and *Y* dimensions is necessary to correctly resolve the system dynamics in both position and momentum space with vortices present. One such way of ensuring accurate resolution of the system is to define a sufficient smallest length scale on one such grid (such as position). By ensuring the position grid remains defined with the same lowest increment, it is possible to increase resolution in the reciprocal space with a larger grid. As vortex core sizes are on the order of *μ*m, the above parameters allow between sub-*μ**m* (2<sup>10</sup> and above) to few *μ**m* resolution. This also holds true for features in **k**-space. Computationally, this can be costly, but quite effective when using compute accelerators (GPUs), which we will introduce next.

Parallel computing
------------------

As the dimensionality of a problem increases, so often too does the time required for performing simulations. One method for accelerating numerical solutions involves the use of multiple compute cores on a central processing unit (CPU) operating independently on different data elements in unison. This form of parallel computation can be achieved through the use of the OpenMP (Open Multi-Processing) application programming interface (API), which defines how a program may parallelise certain elements of code. It allows the developer to fully utilise the power of a multicore processor. However, the limit on how much performance can be gained by this method is set by the number of compute cores available to the system, as well as the limited support offered by compilers. It should also be noted that <span style="font-variant:small-caps;">MATLAB</span> has inherent support for such programming paradigms, and fully abstracts the implementation from the developer. In this instance writing a program from scratch in C/C++/Python/etc. for such a means of parallelisation may not be very beneficial due to the cost of diminishing returns; results would likely be obtained much faster from simply using a multicore supported package, provided that one includes the time to write, as well as simulate. However, this is not always the case, given that the size of problems can often require more cores than available on a single machine.

Another widely used programming paradigm that gets around this CPU core limitation is that of MPI (message passing interface). Where OpenMP allows a user to utilise all available processing cores on a single system, MPI allows the use of an (almost) unlimited number of networked computer systems operating in parallel together, each known as a node. This is the method generally preferred in programs written for compute clusters, where a large number of nodes are available to use. It is the preferable choice for distributed computing applications, with the best performance gains given if there is minimal dependence between data. A bottleneck may occur if data spread over multiple nodes is required for an operation, requiring continual transmission of data between individual nodes. At current data rates this would be limited to bus speeds (assuming an Infiniband optical connection) on the order of tens of gigabytes per second. Compared to a local calculation requiring few to no transfers, the memory bandwidth (data quantity transferred between RAM and CPU per second) can be as high as 60 gigabytes per second . It is important to note that transfers should be minimised to avoid bottlenecks, but transfers are often necessary to make use of the large number of processing cores. Therefore, to give a significant performance benefit, a large number of cores, a high memory bandwidth, a high-speed interconnect between cores (nodes), as well as sufficient space to store the problem in memory are required.

One means of achieving the required high performance is through the use of graphics processing units (GPUs). GPUs are signal processing devices created to offload from the central processing unit (CPU) much of the heavy computation required for displaying two and three-dimensional graphics. As a result, GPUs have been given the task of performing operations necessary to update a large number of pixels in a short amount of time, as well as complex 3D mathematics for image rendering. This has been achieved by giving the GPUs a large number of specialised compute cores for floating-point arithmetic, effectively operating in parallel. With the advent of general purpose GPU (GPGPU) computing, the ability to exploit these cores for the purpose of numerical computation has become possible. If a problem can be mapped effectively to the hardware of a GPU, it can reduce the overall compute time required for evaluating results, as well as reducing overall power usage for similar performance as cluster nodes. For the previous generation flagship industry standard GPUs used in computational acceleration (Nvidia M2090) the memory bandwidth for the device global memory (equivalent of RAM) is given as 288 gigabytes per second, with upwards of thousands of cores on demand, yielding a theoretical total of 1.41 × 10<sup>12</sup> floating-point operations per second (FLOPS), following the formula
FLOPS = cores × clock frequency × operations per clock cycle.

For comparison, Intel Xeon CPU throughput values at best yield approximately 10<sup>11</sup> FLOPS. As can be seen, performance of an order of magnitude greater can be gained by using a GPU for calculations, over high-performance (Xeon) CPUs. This has been shown to allow for effective implementation of the previously mentioned Fourier split-operator method . We have also shown that it yields performance exceeding that of CPUs for a modest choice of GPU , of which we will discuss in detail in a later section.

### Parallel operations

\[sub:Parallel operations\] For a calculation to fully utilise all of the available throughput of a parallel-capable compute device, it is necessary to break down the problem into parallelised sub-problems, which can be easily achieved if data and required operations are uncoupled (*embarrassingly parallel*). Considering summation as an example, imagine that we have a large vector of floating point values that are to be summed together. The traditional way to solve this would be to iteratively add values to an accumulator, and return the final value at the end as the sum. This simple algorithm is 𝒪(*n*) complexity in time, as we iterate through each element at a time. Given that summation is associative, we easily parallelise this operation. By dividing our vector amongst a number of available processing cores recursively, we can reduce the computation to 𝒪(log*n*) in time (see Fig. \[fig:prefixsum\]). This algorithm can give a significant benefit when a large number of summations are performed, such as for wavefunction normalisation, and can reduce the accumulated error resulting from floating point addition from 𝒪(*n*) to 𝒪(log*n*) .

<embed src="/ch3_numerics/CUDA/prefixsum.png" style="width:75.0%" />

Another typical algorithm following this is the Hadamard product (element-wise multiplication) of two vectors or matrices. Although in parallel the number of operations and complexity remains the same, the advantage comes from the lack of interdependence between elements. The multiplications can be carried out asynchronously, allowing all freely available cores to work continuously until every element pair is multiplied.

Mapping the above problems onto GPU devices requires an understanding of some of the architecture and hierarchies of the programming model. We will next introduce this model, and discuss some optimal uses of the available memory and structures.

CUDA programming model
----------------------

Although many multicore programming models exist for both CPUs and GPUs, with OpenCL and OpenACC being two such examples, we will concentrate on Nvidia’s CUDA . CUDA is a mature programming model and API for Nvidia GPUs, and has been well-received for high-performance parallel computing. CUDA operates on a single instruction multiple thread (SIMT) architecture, wherein a single operation is mapped to multiple compute threads independently operating on individual units of data. Writing a program using CUDA C/C++ is very similar to standard C/C++ programming, albeit with some minor differences to account for control of GPU compute threads. CUDA manages calculations in a hierarchical structure with differing levels of fine-grained control over these threads. At the finest grained level (*T*) we have compute threads which operate directly on a single datum from memory. The maximum number of threads which can run simultaneously on the device in a single compute unit are known as a *warp*. Threads are grouped into blocks (**B**), which is the next hierarchical level, which will optimally contain multiples of the maximum warp number for the device (32 elements is the standard warp size for Nvidia architecture).

At the coarsest level, the blocks are grouped together into a grid (**G**), which encompasses the entire problem space. Hardware limits are specified limiting the upper-values of how many elements can appear in these units, and thus the hierarchy exists to allow for fine-grained control on memory usage, and hence performance optimisation on calculations. For a given problem it is necessary to find a mapping from data values in memory to execution threads. To ensure optimal use of GPU cores, the number of threads worked on simultaneously can be (for current hardware) up to 1024 threads per block, though keeping this value divisible by the warp size is recommended to allow an exact mapping to device cores. Optimal values can be found for balancing data computation and transfers, giving, depending on system size, necessary values for block size, as well as grid size. The dimensions of each hierarchical layer are independent of one another, and may be up to 3, with a hardware dependent limit only imposed on the maximum number of elements. Figure \[fig:gpu\_threads\] gives a sample layout for a system of 2D threads and 2D blocks within a single grid.

<embed src="/ch3_numerics/CUDA/gputhreads.png" style="width:60.0%" />

Unlike programming for a CPU-based system, GPUs require explicit control of several different types of memory, and so it is necessary to be aware of the different aspects of managing this memory. Nvidia’s CUDA programming model defines these different physical memories into *global*, *constant*, *shared* and *private*. Global memory is analogous to random access memory (RAM) on a CPU-based system, and is the location where we primarily store data for computation on the GPU. This memory block is accessible (readable and writable) to all threads in the computation. This is also the slowest memory on the GPU, with bandwidths of approximately 10<sup>11</sup> bytes per second. Constantly reading and writing to global memory can hinder the performance of a computation. For memory that is statically defined at the start of a program and will only be read thereafter, the constant memory can be used. This is a special area of memory that can be used to store constants and other values that are often read during a computation. Once set, these values cannot be modified. The next memory level is shared memory, which is block accessible only i.e. threads within the same block have access. This allows threads to exchange information with close neighbours, and is the preferred method of inter-thread communication. Although higher performance than global, the size of shared memory is much smaller. An ideal use case is the parallel summation example given in Sec. \[subsec:par\_op\]. Performing this algorithm requires the copying of data from global to shared memory, performing the calculations, then copying the results back. Lastly, private memory which maps directly to device cache, is thread-accessible only i.e. each thread has access only to its own private values. This memory can be used for local variables defined in a function (known as a *kernel* for GPUs). Private memory has the highest performance, but the smallest available size. If possible, copying variables from global memory into private memory, performing all operations on private, then saving back to global can yield the highest performance, as global memory (slow) is read from and written to once each.

Limiting transfers between the CPU/RAM to the GPU/global memory are of high importance, as the slowest line of communication is the PCI-Express bus, connecting the GPU to the host system (≈16 GB/s max). Eliminating unnecessary communication will almost certainly allow for a gain in performance. Mapping a problem to the GPU requires parallelising the calculation and removing transfers where applicable. For maximum performance, a sample model of performing a GPU calculation is:

1.  Define all variables and data on the host system.

2.  Identify the optimal mapping onto the GPU thread execution model.

3.  Send the data from RAM to the GPU global memory.

4.  Perform computation, with necessary elements copied to shared/private memory.

5.  Return final output of computation to host system when completed.

Although idealised and highly simplified, close adherence to such a model should yield significant performance gains compared to multicore CPU-based computation. An important point with memory access in GPUs is that to ensure optimal performance all access should be assumed to be in-order and adjacent. As an example, though the parallel sum method discussed in Sec. \[subsec:par\_op\] is superior to an iterative summation, it is not an optimal example for GPU architectures . With a small improvement, this can be highly optimised, as highlighted by Fig. \[fig:prefixsum2\]. In this case, the memory access is carried out in strided linear chunks, and by ensuring that all memory accesses are performed this way as much as possible an overall higher degree of performance can be achieved . As GPUs operate on blocks of memory simultaneously, minimising the amount of misaligned memory accesses can optimise the access performance. In this instance, these structures can be packed efficiently together and ensure a large number of copies are performed optimally, with coalesced memory access.

While details for the internal workings of the device are often abstracted, an awareness of these can prove useful. If we assume the number of data elements to be summed is larger than the available number of threads, this routine can perform several passes before the threads become sub-optimally used, as in the previous case. In this way, the summation ensures that the available threads are working mostly optimally.

<embed src="/ch3_numerics/CUDA/prefixsum_2.png" style="width:75.0%" />

Given that GPUs are first and foremost image processing devices, their ability to perform Fourier transforms rapidly is a well developed strength. Due to the large number of available cores, fast Fourier transforms (FFTs) can be significantly faster on a GPU than performing the same operation on many CPUs . The CUFFT library allows for a seamless way to take advantage of this performance increase using GPUs. Next, we will discuss an example of GPU-enabled simulations for an experimentally realistic problem making use of the GPU to perform the necessary operations discussed in Secs. \[sec:timeev\] and \[sec:fso\], and offer some realistic performance measurements.

GPU-enabled parallel Schrödinger simulations
--------------------------------------------

Here, we will introduce the problem of atomic transport in cold atomic systems, present the resulting model system, and give all the essential physics. Controlling the centre-of-mass movement of atoms has recently become a popular topic of investigation . One family of techniques that aim to solve this are those of *spatial adiabatic passage*. In the following section we will describe the motivation, physical system, and the numerical implementation for the solution of the problem of transporting a single atom among three trapping potential wells. The fully three-dimensional numerical solution of this problem will then be provided using GPU computing methods, and the results discussed for both the physical and computing aspects.

For performance metrics, we will discuss the use of a GPU-enabled Schrödinger equation integrator developed by myself, based on and compared with the results of a multi-core MPI enabled version by T. Morgan and N. Crowley. The solution of the Schrödinger equation for a fully three-dimensional potential, will demonstrate the effectiveness and improved performance compared to standard HPC methods.

### Spatial adiabatic passage

Controlling the internal degrees of freedom of atoms is well understood and many spectroscopic techniques exist. External degrees of freedom have, however, only recently become interesting due to the advancements in atom trapping, and techniques for controlling the centre of mass state of a single atom are still in development. One promising group of techniques for the generation of spatial superposition states or high fidelity transfer relies on ideas from spatial adiabatic passage (SAP) , which are analogous to STIRAP in optical systems . These techniques are highly robust against variation in the system parameters , but suffer from being slow due to the adiabatic requirement. The use of SAP has recently been experimentally demonstrated in Lieb lattices , and many other accessible systems have been proposed .

To describe the method for SAP, we will first consider the case of a two-state system, which can be realised using two separated harmonic potential traps, with ground states |*L*⟩ and |*R*⟩. A reduction of the distance between the traps will increase the coupling, and hence tunneling rate, between them. This can be modelled with a two-level Hamiltonian as
$$H = -\\frac{\\hbar}{2}
    \\begin{pmatrix}
        0 & J\_{LR} \\\\
        J\_{RL} & -2\\Delta
    \\end{pmatrix},$$
 where *J*<sub>*L**R*</sub> = *J*<sub>*R**L*</sub> are the couplings between states, and *Δ* is the detuning of state |*R*⟩, relative to |*L*⟩. Assuming an atom initially localised in |*L*⟩, and with an increase in coupling strength between the levels, the localised atom will tunnel from |*L*⟩ to |*R*⟩. However, this processes is difficult to control, as Rabi oscillations introduce an explicit time-dependence as

$$\\begin{aligned}
    |c\_L(t)|^2 &\\propto \\sin^2 \\frac{\\omega t}{2} ,\\\\
    |c\_R(t)|^2 &\\propto \\cos^2 \\frac{\\omega t}{2},\\end{aligned}$$

where the |*c*<sub>*L*, *R*</sub>(*t*)|<sup>2</sup> are the populations of the respective states. This time dependence causes the atomic population to continuously tunnel between both traps. It will therefore require precise timing and control to ensure a full, robust transfer of population. From this we see that a double-well potential is a difficult system in which to realise coherent control, though methods such as rapid adiabatic passage (RAP) exist and can allow for this . A more robust method, using three adjacent harmonic traps and the aforementioned matter-wave SAP process, can improve upon the standard double-well system. For this SAP technique, we model the trapping potentials arranged in a single line and coupled with their nearest neighbour only. For three equivalent potentials, *L*, *M*, *R*, with degenerate states {|*L*⟩,|*M*⟩,|*R*⟩}, the system can be described by the Hamiltonian
$$\\label{eqn:sap\_ham}
    H = -\\frac{\\hbar}{2}
    \\begin{pmatrix}
        0 & J\_{LM} & 0 \\\\
        J\_{LM} & 0 & J\_{MR} \\\\
        0 & J\_{MR} & 0
    \\end{pmatrix},$$
 where *J*<sub>*L**M*</sub>,  *J*<sub>*M**R*</sub> describe the left-middle and middle-right couplings respectively. Diagonalising this Hamiltonian gives three distinct eigenstates,

$$\\begin{aligned}
    | \\pm \\rangle &= \\frac{J\_{LM} |L\\rangle \\pm \\sqrt{J\_{LM}^2 + J\_{MR}^2}|M\\rangle + J\_{MR} |R\\rangle  }{\\sqrt{2(J\_{LM}^2 + J\_{MR}^2)}}, \\\\
    | D \\rangle &= \\frac{J\_{MR} |L\\rangle }{\\sqrt{J\_{LM}^2 + J\_{MR}^2}} - \\frac{J\_{LM} |R\\rangle}{\\sqrt{J\_{LM}^2 + J\_{MR}^2}},\\end{aligned}$$

with respective eigenvalues $E\_{\\pm} = \\pm {\\sqrt{J\_{LM}^2 + J\_{MR}^2}}$, *E*<sub>*D*</sub> = 0. Here, only one eigenstate is of interest. For the zero-valued eigenstate of this Hamiltonian, |*D*⟩, known as the *dark state*, the dependence on the middle potential vanishes. The state can be written as
|*D*⟩=cos *Θ*|*L*⟩−sin*Θ*|*R*⟩,
 where tan*Θ* = *J*<sub>*L**M*</sub>/*J*<sub>*M**R*</sub> is the mixing angle, following directly from the trigonometric identities $\\cos (\\arctan (x)) = 1/\\sqrt{1+x^2}$, $\\sin (\\arctan (x)) = x/\\sqrt{1+x^2}$. From the adiabatic theorem of quantum mechanics, it is known that if a system has its Hamiltonian perturbed slowly enough, then we can follow its evolution ensuring that it always remains in an eigenstate of the Hamiltonian. By preparing the system in state |*L*⟩, and varying *Θ* slowly, we can shift the population from the leftmost harmonic potential to the rightmost, without populating the center. To ensure full transfer from |*L*⟩ to |*R*⟩, the mixing angle must change smoothly from *Θ* = 0 → *π*/2. From the properties of arctan, this can be achieved by applying the same spatial variation between |*M*⟩ and |*R*⟩, as between |*L*⟩ and |*M*⟩, with the latter coupling pulse following a delay, *τ*. A diagram of this is given by Fig. \[fig:ch3\_stirap\]. Due to the robustness of this process a variety of coupling profiles can be used , as long as they have a smoothly varying background envelope *p* given by

$$\\begin{aligned}
    J\_{LM}(t) &= J\_0 p(t-\\tau), \\\\
    J\_{MR}(t) &= J\_0 p(t).\\end{aligned}$$

Varying the couplings between the trapping potentials, and hence controlling this mixing angle, can be achieved by either lowering the barrier height of adjacent potentials, or decreasing the distance between them. Here, we will discuss varying the spatial separation of the traps.

<img src="/ch3_numerics/stirap_delay.png" alt="Three trapping potential model for matter-wave SAP. The atom (green) is initially localised in the leftmost potential, |L\rangle, at t=0, with the couplings between adjacent traps controlled by varying the distance dependent parameters J_{LM}, J_{MR}. By varying the couplings between traps in the manner shown, with J_{MR} increasing initially, followed by J_{LM} after a delay \tau, the atom is transferred completely from |L\rangle to | R \rangle." style="width:65.0%" />

Typically, any method to adjust the couplings would be performed with time-dependent potentials. However, a static potential variant can be considered using parallel atomic waveguides, where the separation varies as a function of distance along the parallel axis. If we consider an atom that travels along such a waveguide, the couplings, and hence tunneling rates, seen by the atom in the waveguide are altered as the atom propagates. Such work has been discussed and considered in a realistic system for two spatial dimensions . Although a two dimensional model is effective at describing much of the relevant dynamics, the realism of the model is reduced by the lack of a third dimension. This is due to the lack of effects stemming from dispersion, curvature of the waveguides, as well as the absence of any such eigenstates along this dimension.

### Atom-chip model

As discussed previously, to fully understand ultracold atom dynamics in appropriately shaped waveguide systems we must investigate the fully three-dimensional model. One method of creating the required potential landscape in experiments is through the use of atom-chips . These systems consist of micro-fabricated current-carrying wires, and can be used to create a variety of trapping potential shapes for controlled guidance of the atomic centre-of-mass . The currents produce a magnetic field around individual wires, each of which has a minima at the wire core. Assuming the wire thickness to be negligible, this magnetic field at position **r** can be calculated using the Biot–Savart law
$$\\mathbf{B}(\\mathbf{r}) = \\frac{\\mu\_0}{4\\pi}\\oint I \\frac{\\text{d}\\mathbf{l}\\times \\hat{\\mathbf{r}}^{'}}{|\\mathbf{r^{'}}|^2},$$
 where *I* is the current through the wire, *μ*<sub>0</sub> is the vacuum permeability, d**l** is the differential wire length, and $\\hat{\\mathbf{r}}^{'}$ is the unit vector along **r**<sup>**′**</sup> = **r** − **l**. To be able to trap atoms the minima must be raised to a position above the wire surface. This can be done using an orthogonally applied bias field, *B*<sub>*b*</sub>, which raises the minima to a height of
$$\\mathbf{r}\_0 = \\frac{\\mu\_0 I}{2\\pi {B}\_b},$$
 above the surface. Though, an issue still remains with the presence of the magnetic minima. If the field drops to zero at the centre of the trap, the atoms can be lost due to Majorana spin flips . This can be prevented with the application of an additional field, *B*<sub>*i**p*</sub>, parallel to the wire direction, lifting the degeneracy of the atomic states, and ensuring they remain trapped. Spatial and temporal adjustments of the potentials are possible, with a fine degree of control, either during the production process, or by using time-dependent currents. These have been studied extensively in recent years for highly controllable trapping potentials , and as atomic manipulators .

For this work, we model the system as three adjacent wires on the atom-chip surface. The direction of propagation is along *z*, and an additional harmonic oscillator potential, *V*<sub>*z*</sub> = *m**ω*<sub>*z*</sub><sup>2</sup>(*z* − *z*<sub>0</sub>)<sup>2</sup>/2, is applied in the same direction to impart motion to the atom, which is initially at the *z* = 0 position of the atom-chip. We set *z*<sub>0</sub> = (max*z*)/2 to ensure the oscillator potential is symmetric around the centre of the atomchip. This potential also conveniently guarantees that the wavefunction refocuses after the transition at the opposite side of the atom-chip. A schematic of the atom-chip device and the respective potentials is given by Fig. \[fig:schematic\_atom-chip\].

<img src="/ch3_numerics/MWSTIRAP/Schematic3.png" title="fig:" alt="Schematic of the atom-chip and the resulting potentials. Reprinted from Morgan et al. ." style="width:45.0%" /> <img src="/ch3_numerics/MWSTIRAP/3dpot_schem.png" title="fig:" alt="Schematic of the atom-chip and the resulting potentials. Reprinted from Morgan et al. ." style="width:45.0%" />

The initial state is created by localising the atom with the help of a barrier at one end of the potential (*z* = 0), and in the leftmost waveguide using an additional barrier potential. After finding the ground state, the barriers are removed, and the atom is allowed to propagate along the length of the waveguide. The populations in each waveguide |*c*<sub>*X*</sub>|<sup>2</sup> are tracked at each step of the process as
$$|c\_X|^2 = \\int\\limits\_{\\mathcal{V}\_X} d\\mathbf{r}  \\Psi^{\*} \\Psi$$
 where 𝒱<sub>*X*</sub> is the volume encompassing each waveguide *X* ∈ {*L*, *M*, *R*}, and *Ψ* is the state of the system. The final populations were taken as the atom approached the other classical turning point of the harmonic oscillator along *z*. The fidelity of the process could then be calculated by comparing the initial populations in |*L*⟩ and final ones in |*R*⟩, as well as any ones left in |*M*⟩.

Given that fully three-dimensional simulations of the Schrödinger equation are numerically expensive, the use of GPU computing methods were ideal for accelerating the simulation . At the time of writing, as far as we had been aware, no other work using GPU computing to solve a three dimensional Schrödinger equation had been presented. We will now discuss the data and metadata of the simulations.

### 3D Simulations

Simulations of the proposed system assumed a single **<sup>6</sup>Li atom localised in the left . A localised harmonic oscillator potential was added along *z*, with the transverse guiding potential, and the resulting ground state solution found numerically. The harmonic oscillator width was chosen to closely match the expected size in the transverse direction. The atom was then allowed to propagate along the waveguide potential (*z*).

As the shift in magnetic moment of the atoms is given by *Δ**E* = −*μ* ⋅ **B**, for regions with a larger magnetic field the atom will experience a greater energy shift, and thus the assumption of all traps being on resonance (degenerate) has to be carefully checked. In fact, the simulations showed that the addition of the magnetic fields stemming from the different wires at the center of the atom-chip leads to the central potential moving out of resonance with the outer two. To make SAP work, it therefore required adjusting the current in the central wire, such that the magnetic minima were in resonance within the tunneling region near the atom-chip centre. The resulting potentials for non-optimal (left) and optimal (middle) currents are shown in Fig. \[fig:equaloptcurrent\], which also depicts a three dimensional isosurface of the potential minima along the chip surface for both situations (right).

<embed src="/ch3_numerics/MWSTIRAP/potentials2.png" title="fig:" style="width:55.0%" /> <embed src="/ch3_numerics/MWSTIRAP/3dpot.png" title="fig:" style="width:30.0%" />

The populations for both the direct tunneling case, and the matter-wave SAP processes are shown in Fig. \[fig:mwsVsDT\]. The direct tunneling case can be seen to show Rabi-type oscillations between the waveguides, while the matter-wave SAP process shows a much cleaner transfer, and only a minor occupation of the central potential. The dependence of the transfer probability on the current in the central wire is shown in Fig. \[fig:DIRVSMWSTIRAP\].

<embed src="/ch3_numerics/MWSTIRAP/STIRAP_CINT_POP.png" title="fig:" style="width:47.0%" /> <embed src="/ch3_numerics/MWSTIRAP/STIRAP_INT_POP.png" title="fig:" style="width:47.0%" />

<embed src="/ch3_numerics/MWSTIRAP/SAPvsDirect.png" style="width:45.0%" />

### GPU computing performance

Given the large parameter space over which this system could be evaluated (e.g wire current, spatial separations, trap frequencies), a large number of simulations were required to determine optimal system behaviour. As discussed in Sec. \[sec:cuda\_prog\], one example where GPU computing offers large performance gains are FFTs, which makes the Fourier split operator method an ideal candidate for GPU systems . The body of work for implementing this algorithm was using C, CUDA and Nvidia’s CUFFT libraries for the Fourier transforms, whereas the MPI-enabled code was implemented using C.

To demonstrate the performance offered by GPU computing we compared it to using FFTW with MPI, a well used parallel programming library and paradigm. The MPI implementation allows code to be run across multiple machines, benefiting from the parallelism which may be offered by a supercomputing cluster. Although MPI-enabled FFTW is fast and supports extremely large grid sizes, it requires cluster access of a significant size to be a viable option for this type of system. The MPI work on this project was carried out on the Irish Center for High-End Computing (ICHEC) supercomputer system “Stoney” over the period 2011 to 2012, with all performance metrics data calculated therefrom. This cluster system had 64 available compute nodes, each housing two 2.8GHz Intel Xeon X5560 processors with 4-cores each, and a total of 48GB of RAM per node with inter-node communication using double data rate Infiniband.

Due to the hardware limited memory on the GPU, and because the dynamics along the *x* − *z* plane were of most importance, the grid-size of the simulations were scaled as 256 × 64 × 1024 (*x* × *y* × *z*). Of next importance were the choice of timesteps for the simulations. To ensure minimal loss in accuracy, the timesteps were chosen as *Δ**t* = 10<sup>−6</sup> s. By approximating the waveguides as harmonic oscillators, the relevant timescales of the dynamics are of the order *T*<sub>t</sub> ≈ 10<sup>−4</sup> s, and so we can accurately capture all relevant dynamics in the transverse direction. The timescales in the longitudinal direction, which requires a large oscillation period compared to the transverse plane to ensure the adiabaticity condition (*T*<sub>l</sub> ≈ 10<sup>−1</sup> s), are also fulfilled. For the GPU simulations, the test system was an Intel Core i7 2600K CPU at stock frequency, 8GB DDR3 memory operating at 1600 MHz, 7200 RPM HDD, Nvidia GeForce GTX 580 with 3GB of onboard memory running at 783 MHz GPU core frequency, 1566 MHz shader processor frequency, and 2010 MHz memory frequency. For all simulations the desktop was running Ubuntu 11.10 64-bit operating system and all calculations were performed in double precision (64-bit floating point) where applicable.

Table \[tbl:timing\] shows the approximate timings for the completion of runs using GPU and CPU codes. Not only does GPU computing offer a 6-fold improvement over a single CPU, it also allows us to achieve a performance level which is comparable to an 8-node 8-core (64 cores) core MPI enabled CPU calculation. Even for a modest choice of gaming GPU this offers substantial performance gains. Higher performance was achieved by using specific compute accelerators designed for double precision arithmetic. Making use of eight Nvidia M2090 GPUs available at OIST, terabytes of numerical results were generated, and allowed the problem to become tractable on a short timescale.

|   Device  | Num. Devices |  Timing | Rel. Improvement |
|:---------:|:------------:|:-------:|:----------------:|
| CPU (MPI) |       8      |  ∼6 Hr  |       1.0×       |
|           |      16      |  ∼4 Hr  |       1.5×       |
|           |      32      | ∼1.5 Hr |       4.0×       |
|           |      64      |  ∼1 Hr  |       6.0×       |
|    GPU    |       1      |  ∼1 Hr  |       6.0×       |

GPUE: GPU Gross–Pitaevskii equation solver
------------------------------------------

Given the effectiveness of GPU computing in the simulation of the linear Schödinger equation system for SAP, we next applied the newly-developed techniques to simulating Bose–Einstein condensates, which formed the bulk of work during my thesis. The body of software developed for this project has been released as the tool “GPUE”, available at <https://github.com/mlxd/gpue> . Performance testing of this code was carried out by Peter Wittek, ICFO, Barcelona . A comparison was performed between GPUE, the Trotter–Suzuki (TS) package developed by Wittek *et* al. , and the mature GPELab software suite for MATLAB . The sample results taken for time evolution are given in Fig. \[fig:gpuevsts\]. GPUE and GPU-enabled TS clearly beat MATLAB, and CPU performance by a significant margin. Although TS is a more generalised suite for computing, as far as we are currently aware the GPU computation does not yet allow for Gross–Pitaevskii solutions with angular momentum. GPUE is currently the optimal choice for rotating condensate systems out of the examined software suites.

<img src="/ch3_numerics/GPUEvsTS.png" alt="Performance benchmark of GPUE and other simulation packages for the evolution of a harmonically trapped atom in a superposition state between ground and first excited states. Lower numbers are better and give results in faster times. Data adapted from . " style="width:50.0%" />

Figures  \[fig:profile\_ev\] and \[fig:profile\_im\] demonstrate some of the resulting calls to different segments of the code, with timings given in Tables \[tbl:gpue\_ev\] and \[tbl:gpue\_im\]. The important data of the figures is both the kernel percentage utilisation, and that the operations are mostly saturating the available number of GPU cores. The data shows the average time spent in each individual kernel during both real and imaginary time evolution for 1010 steps at 2<sup>10</sup> × 2<sup>10</sup> resolution with (real) and without (imaginary) angular momentum operators. As can be seen, the inclusion of angular momentum operators lead to a performance hit, compared with an imaginary time evolution for a static condensate. While further optimisations can almost always be provided for such simulations, the performance of the software as a whole is defined by its slowest component. In this case the routines are equally met in performance by the Fourier transforms, which are already fully optimised as an external library. As such, improving performance much beyond this with the other kernels will be wasteful in time and resources.

<img src="/ch3_numerics/CUDA/Profiler_ev_1k_1024.png" alt="Nvidia Nsight performance analysis of GPUE for real time evolution simulation for 1010 steps at 2^{10}\times 2^{10} resolution. The respective kernel calls and total utilisation are listed on the left." style="width:98.0%" />

|     **Kernel**    |                   **Info**                  | **Avg. runtime** | **\# Runs** | **Total time** |
|:-----------------:|:-------------------------------------------:|:----------------:|:-----------:|:--------------:|
| Mem. copy \[H2D\] |         Memory copy from host to GPU        |     2.312 ms     |      11     |    25.432 ms   |
| Mem. copy \[D2H\] |         Memory copy from GPU to host        |      2.18 ms     |      9      |    19.62 ms    |
|       cMult       |          Complex mult. in time ev.          |     0.342 ms     |     3030    |    1.0363 s    |
|    cMultDensity   | Complex mult. in time ev. for nonlinear op. |     0.456 ms     |     2020    |     0.921 s    |
|     scalarDiv     |         Renorm. of *Ψ* following FFT        |     0.2216 ms    |     8080    |     1.791 s    |
|    dpRadix0032B   |           Internal CUFFT operation          |     0.237 ms     |     8080    |     1.915 s    |
|   dpVector1024D   |           Internal CUFFT operation          |     0.252 ms     |     8080    |    2.0362 s    |

<img src="/ch3_numerics/CUDA/Profiler_im_1k_1024.png" alt="Nvidia Nsight performance analysis of GPUE for imaginary time simulation for 1010 steps at 2^{10}\times 2^{10} resolution with angular momentum. The respective kernel calls and total utilisation are listed on the left." style="width:98.0%" />

|     **Kernel**     |                   **Info**                  | **Avg. runtime** | **\# Runs** | **Total time** |
|:------------------:|:-------------------------------------------:|:----------------:|:-----------:|:--------------:|
|  Mem. copy \[H2D\] |         Memory copy from host to GPU        |     2.311 ms     |      3      |    6.933 ms    |
|  Mem. copy \[D2H\] |         Memory copy from GPU to host        |      1.85 ms     |      5      |     9.25 ms    |
|        cMult       |          Complex mult. in time ev.          |     0.342 ms     |     1010    |     0.345 s    |
|    cMultDensity    | Complex mult. in time ev. for nonlinear op. |     0.346 ms     |     2020    |     0.698 s    |
|      multipass     |         Optimised parallel summation        |     0.125 ms     |     3030    |     0.378 s    |
|      scalarDiv     |         Renorm. of *Ψ* following FFT        |     0.221 ms     |     2020    |     0.446 s    |
| scalarDiv\_wfcNorm |   Normalisation of wavefunction during ev.  |     0.226 ms     |     1010    |     0.228 s    |
|    dpRadix0032B    |           Internal CUFFT operation          |     0.237 ms     |     4040    |     0.957 s    |
|    dpVector1024D   |           Internal CUFFT operation          |     0.251 ms     |     2020    |     0.507 s    |

A simplified sequence and state diagram combination is given in Figs. \[fig:gpue\_seq1\] and \[fig:gpue\_seq2\] which describes the operating process for GPUE. A document listing all aspects of component dependencies and intercommunication is available at . For brevity, we will refer the reader to this location for more information.[2].

<img src="/ch3_numerics/GPUE_Seq1.png" alt="Simplified combined sequence and state diagram for GPUE operation (1 of 2). The operation procedure of GPUE is outlined in sequence from top-to-bottom. While much of the setup and analysis takes place on the host (CPU), the device (GPU) is used to offload all the time-evolution calculations. After setup, the wavefunction and all required operators are sent to the GPU." />

<img src="/ch3_numerics/GPUE_Seq2.png" alt="Simplified combined sequence and state diagram for GPUE operation (2 of 2). Following the completion of the time evolution after a predetermined number of steps, the wavefunction is unloaded from the GPU and returned to the CPU for output. Minimising this transfer allows for optimal performance from the device. Further details of dependencies and data flow are given by docs/gpue.pdf." />

### Angular momentum operators using Fourier split-operator method

As discussed earlier, in the presence of large values of angular momentum, the condensate wavefunction will accommodate many vortices. To ensure a well ordered lattice, more consideration is required than to just directly numerically solve the GPE at the required rotation rate. Assuming an initial Gaussian guess, and using the imaginary time evolution algorithm to find the ground state, a large number of vortices will enter the condensate from the edge and compete for lattice sites to form the expected Abrikosov pattern. Due to the highly dense spectrum of the condensate close to the ground state in this regime, only minimal energy shifts will be given for deviations from the perfect Abriksov geometry. As a result, it can take a significantly long time to reach an ordered state for rotation frequencies close to the transverse trapping frequency . To overcome this issue, one can choose to follow the ground state of the condensate with a ramp of the rotation rate. This essentially mimics adiabatic evolution, and allows for the determination of the vortex lattice ground state for all rotation frequencies.

The Fourier split-operator algorithm described earlier works well in handling cases where the individual operators live in position or momentum space respectively. However, the angular momentum operators are a combination of both spaces. Taking the angular momentum operator along the *z*-axis, *L*<sub>*z*</sub> = *x**p*<sub>*y*</sub> − *y**p*<sub>*x*</sub>, and applying it to the wavefunction requires each basis element to be in a different space in the different directions. For applying this operator we must therefore Fourier transform along a single dimension, multiply by the respective **k**-space component, take the inverse, multiply by the respective **r**-space component in the other direction, and then perform this operation along the other dimensions, summing the results.

This accrues an error which is not encountered using methods that are solely in position or momentum space. Following the process given in Sec. \[sec:fso\] the error can be determined by checking the commutativity of the respective components of the angular momentum operator as

$$\\begin{aligned}
 	\\alpha\_1 = \[x p\_y,-y p\_x\] &= \[x p\_y,-y\] p\_x  -  y\[x p\_y,p\_x\] = -\[-y,x p\_y\] p\_x + y \[p\_x, x p\_y\] \\nonumber \\\\
 		&= -\\left( {\\cancelto{0}{\[-y,x\]}} p\_y + x \[-y,p\_y\] \\right) p\_x + y \\left( \[p\_x,x\] p\_y + x {\\cancelto{0}{\[p\_x,p\_y\]}} \\right) \\nonumber \\\\
 		&= -x {\\cancelto{-\\textrm{i}\\hbar}{\[-y, p\_y\]}} p\_x + y {\\cancelto{-\\textrm{i}\\hbar}{\[p\_x,x\]}} p\_y \\nonumber \\\\
        &= \\textrm{i}\\hbar \\left(x p\_x - y p\_y \\right).
 \\end{aligned}$$

The complex error term can be seen as, in the case of the above implemented evolution, allowing the angular momentum operator to change from imaginary time to real-time, and vice-versa in each respective case. To overcome this, we simply swap the application order of the operator components, between odd and even steps during the evolution. Starting with the alternate order we obtain a value of *α*<sub>2</sub> = \[ − *y**p*<sub>*x*</sub>, *x**p*<sub>*y*</sub>\]=iℏ(−*x**p*<sub>*x*</sub>+*y**p*<sub>*y*</sub>). Since we are applying these operators to the condensate we can overcome the error of one term by the application of the other, as
expi*α*<sub>1</sub>expi*α*<sub>2</sub> = 1.

Although alternating will provide a cancellation of this error, it can be assumed that for large timesteps the error will have a significant contribution to the overall dynamics, as the wavefunction evolves during each timestep. For greater accuracy of this method one can perform a decomposition following Eq.  for a third-order error, or using the above splitting for second-order.

An example of the density of the ground state and the associated wavefunction phase at a rotation frequency of *Ω* = 0.995*ω*<sub>*x*</sub> is given in Fig. \[fig:showingoff\] at a resolution of 2<sup>11</sup> × 2<sup>11</sup> (2048 × 2048). Although aliasing may be apparent in the phase, this is due to the limited resolution of the computer monitor (printer). The presence of a well ordered Abrikosov lattice is clearly visible.

<img src="/ch3_numerics/Rho_995.png" title="fig:" alt="Condensate density (left) and phase (right) at a rotation rate of \Omega=0.995\omega_x for a 2^{11}\times 2^{11} grid showing approximately 600 vortices in the visible density regions. For both images the box size is 700~\mu\textrm{m} ~\times ~700~\mu\textrm{m}." style="width:45.0%" /> <img src="/ch3_numerics/Phi_995.png" title="fig:" alt="Condensate density (left) and phase (right) at a rotation rate of \Omega=0.995\omega_x for a 2^{11}\times 2^{11} grid showing approximately 600 vortices in the visible density regions. For both images the box size is 700~\mu\textrm{m} ~\times ~700~\mu\textrm{m}." style="width:45.0%" />

### Vortex tracking

To efficiently follow the dynamics of individual vortices, a robust algorithm is needed to track their positions. One could track regions where the density drops to zero. However, this gives very little information on the topological excitation, and may miss many vortices, as the numerical wavefunction may never truly approach this value. A more effective way is to locate the ±2*π* charge in the wavefunction phase, which is a signature of quantum vortices. For this we examine each 2 × 2 subgrid of the underlying lattice and check if the phase rotates from −*π* to +*π* (or vice versa). After an initial pass to identify vortex locations closest the nearest grid element, a least-squares fit is performed to more accurately determine the vortex core position . Linear least squares is used generally for an overdetermined linear system **A****r** = **b**, where unique solutions are unlikely to exist. Thus, for a solution, we seek the best fit plane that minimises the error, of the form

*S*(**r**)=∑|*b*<sub>*i*</sub> − ∑*A*<sub>*i**j*</sub>*r*<sub>*j*</sub>|<sup>2</sup>

where *S* is the objective function to be minimised, following $\\mathbf{b} = \\operatorname\*{arg\\,min}S(\\mathbf{r})$. The solution of this minimisation problem is given by

$$\\begin{aligned}
    \\mathbf{A} ^{T}\\mathbf{A} \\mathbf{r} &= \\mathbf{A} ^{T}\\mathbf{b}, \\\\
    \\mathbf{r} &= (\\mathbf{A}^{T}\\mathbf{A})^{-1}\\mathbf{A}^{T}\\mathbf{b}.\\end{aligned}$$

The best-fit plane is sought of the form $a\_0 c + \\displaystyle\\sum\\limits\_{i}^{m} a\_i r\_i = f(\\mathbf{r})$ which for a two-dimensional system, **r** = (*x*, *y*), is given by the matrix,
$$\\mathbf{A} = \\left(
    \\begin{array}{ccc}
        0 & 0 & 1 \\\\
        0 & 1 & 1 \\\\
        1 & 0 & 1 \\\\
        1 & 1 & 1
    \\end{array}\\right).$$
 The above matrix is composed of all possible planes that can fit over a square 2 × 2 grid plaquette, and
$$\\mathbf{b} = \\left(
    \\begin{array}{cccc}
        \\Psi(x\_0,y\_0) & \\Psi(x\_0,y\_1) & \\Psi(x\_1,y\_0) & \\Psi(x\_1,y\_1)
    \\end{array} \\right)^{T},$$
 are the wavefunction values around the sampled 2 × 2 grid. Upon evaluating the vector **r** above, one can obtain the best fit plane solution as
$$\\left(
    \\begin{array}{c}
        x \\\\
        y \\\\
        c
    \\end{array}\\right)
    = \\left(
    \\begin{array}{c}
        {\\left( -\\Psi(x\_0,y\_0) + \\Psi(x\_0,y\_1) - \\Psi(x\_1,y\_0) + \\Psi(x\_1,y\_1) \\right)}{/2} \\\\
        {\\left( -\\Psi(x\_0,y\_0) - \\Psi(x\_0,y\_1) + \\Psi(x\_1,y\_0) + \\Psi(x\_1,y\_1) \\right)}{/2} \\\\
        3\\Psi(x\_0,y\_0) + \\Psi(x\_0,y\_1) - \\Psi(x\_1,y\_0) - \\Psi(x\_1,y\_1)
    \\end{array}\\right).$$

The goal is to find where both the real and imaginary components cross through zero, and thus we seek a solution of the form *x* + *y* = −*c*. Rearranging the above equations as
$$\\left(
    \\begin{array}{cc}
        \\Re(x) & \\Re(y) \\\\
        \\Im(x) & \\Im(y) \\\\
    \\end{array}\\right)
    \\left(
    \\begin{array}{c}
        \\delta x \\\\
        \\delta y
    \\end{array}\\right)
    = -
    \\left(
    \\begin{array}{c}
        \\Re(c) \\\\
        \\Im(c)
    \\end{array}\\right),$$
 and again solving the linear system by inverting the left-hand matrix and multiplying across allows one to seek the corrections to the vortex position, *δ***r** = (*δ**x*, *δ**y*).

With this, we can accurately determine the position of the vortices with high precision. To track their motion during the evolution, we create an initial list of positions and give each vortex a unique identifier (UID). Assuming the vortex cores can travel a limited distance (some multiple of the grid resolution) between time steps, we can say at subsequent times which vortex has moved to the newly found positions.

This process is performed by representing the vortices as a graph, each with an assigned unique identifier, associated location, phase winding and on/off flag. Edges are created between vortices that are separated by at most root-two the average of the inter-vortex spacings. A finite boundary is chosen to examine only vortices in areas of significant condensate density, since vortices can easily appear and disappear close to the condensate boundary. Any vortex which appears without association to an initial vortex, or any tracked vortex that crosses the boundary, is switched off and remains so for all analysis. A graph for an initial (*t* = 0) vortex lattice shown in Fig. \[fig:graphit\], with the identifiers indicated on each node, and the neighbouring distances indicated by the edge weights.

<img src="/ch3_numerics/Graph_full.png" alt="Graph of vortex lattice positions indicating the vortex identifier and the intervortex distances in units of grid-spacing \Delta= \Delta x = \Delta y. A hard-walled boundary is chosen such that the vortex distances remain almost uniform, and are shown here with a mean value of \bar{r} = 30.2451 and variance of \sigma^2 = 0.16118." style="width:98.0%" />
