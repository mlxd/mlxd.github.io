+++
title = "Non-equilibrium vortex dynamics in rapidly rotating Bose–Einstein condensates: Introduction"
draft = false
date = "2018-01-14 12:00:00"
tags = []
math = "true"
summary = "Introduction chapter of my thesis."
+++

# Introduction
---------------

The purpose of this work is to understand the dynamics of rapidly rotating Bose–Einstein condensates subjected to perturbations, and to develop techniques to control and engineer specific non-equilibrium states. While it is possible to derive some analytical solutions for rapidly rotating condensates (e.g. lowest Landau level approach), such solutions are rare. This thesis concentrates on the numerical solutions of the Gross–Pitaevskii equation, and the resulting dynamics within this framework. It focuses on gaining an understanding of the dynamical behaviour of quantum vortices in an Abrikosov geometry following a perturbation. This body of work was carried out during my time as a Ph.D student at Okinawa Institute of Science and Technology Graduate University (OIST), and grew out of work and ideas I started to pursue at University College Cork (UCC), Ireland.

Understanding ultracold Abrikosov vortex systems can help with engineering quantum states for future technologies. Ideally, these systems can be used for long-term memory storage in computing applications as individual vortices are topologically protected and, therefore, very robust. They also allow the study of quantum mechanical effects on mesoscopic scales, and the inherent periodicity makes them a promising tool for simulating condensed matter physics. Furthermore, perturbed vortex lattices can be used to investigate turbulent, and possibly chaotic, quantum behaviour. While turbulent classical systems are notoriously hard to understand and control, quantum turbulence is thought to offer a more controllable route to understanding the nature of turbulence, due to the quantisation condition of the circulation. It is therefore of large interest to develop new tools for manipulating and engineering specific states of rotating condensates. In the following work I concentrate on two types of perturbations to the equilibrium state of a rotating condensate: i) the modification of the phonon spectrum of the condensate which does not influence the angular momentum, in particular through the use of a kicked optical potential; ii) the direct control of the topological excitations, and hence the angular momentum, which is performed with direct phase engineering of the condensate wavefunction. I examine both in the above order, and investigate their usefulness in controlling and manipulating condensate dynamics.

For investigating these perturbances I assume a system of a rapidly rotating BEC having a large number of vortices, arranged in a triangular Abrikosov lattice pattern. This requires the solution of a two-dimensional partial differential equation at high grid resolution with a variety of different initial conditions and controllable perturbations. The solution of the proposed system is a non-trivial numerical problem, and requires the use of advanced numerical computing techniques to allow for results in a reasonable time. For this I make use of graphics processing unit (GPU) computing, and I will discuss the development of such tools, my numerical contributions, and compare them against conventional simulation techniques.

The thesis is organised as follows:

Background
----------

I will first give a brief introduction to the field of cold-atomic gases, and discuss the theoretical framework to describe Bose–Einstein condensation. Emphasis will be placed on material and works relevant to the studies I have performed in this thesis. I will present a derivation of the Gross–Pitaevskii equation, used to model Bose–Einstein condensates, as well as a discussion of the Bogoliubov-de Gennes equations. I will then discuss the hydrodynamic description of the condensate, and give the hydrodynamic form of the Gross–Pitaevskii equation. Here I introduce superfluidity, and the nature of quantised vortices in these systems. I will conclude with an outlook on the cutting edge work in the field in the context of condensate trapping and control.

Numerical methods
-----------------

In this chapter I will discuss methods for numerically solving the Gross–Pitaevskii equation for simulating the dynamics of Bose–Einstein condensates. The Fourier split-operator method will be introduced, as well as the need for imaginary time evolution, and considerations required to effectively simulate the condensate. Graphics processing unit (GPU) computing will be introduced here, with the implementation of the Gross–Pitaevskii equation discussed. To demonstrate the power of GPU computing we will present and solve a difficult numerical problem, namely the solution of an experimentally realistic situation for a single, ultracold atom on an atomchip, for which the treatment of the fully three-dimensional Schrödinger equation is required. The use of GPU computing makes this problem tractable in realistic times. The work focuses on the area of adiabatic control techniques, and demonstrates the use of GPU computing to describe the long-time dynamics of a system for observing matter-wave spatial adiabatic passage. This work has been published in Phys. Rev. A **88**, 053618 (2013) .

Bose–Einstein condensate dynamics
---------------------------------

In this chapter I will examine the dynamics of Bose–Einstein condensates under rotation, and discuss the methods used for perturbing the condensate system. I begin by introducing the dynamical behaviour of the condensate in the presence of vortices, and introduce a model system used for further discussions. I present the velocity profiles and discuss some of the dynamics a condensate with many vortices is expected to follow. This will be followed by an introduction to the two main perturbation methods for the condensate that I will later use: optical kicking, and phase imprinting. I will also discuss the techniques that I use to analyse the vortex dynamics, concentrating primarily on the kinetic energy spectrum, Delaunay triangulation and Voronoi tessellation.

Moiré superlattice structures
-----------------------------

Here I investigate effects stemming from the optical kicking of a condensate carrying a vortex lattice. The dynamics of the condensate after a kick with an optical potential of the same geometry as the vortex lattice is demonstrated, and shows little to no deviation of ideal vortex positions. However, the resulting condensate density shows the appearance of a superlattice pattern. I analyse this system, and demonstrate that the resulting superlattice pattern stems from interference between the optical kicking potential and the present vortex lattice in reciprocal space. Moiré interference theory accurately predicts the observed behaviour, and is backed up by examining the kinetic energy spectrum of the condensate. To conclude, I discuss applications of this optical kicking technique and the resulting moiré interference. The results presented in this chapter have been published in Phys. Rev. A **93**, 023609 (2016) .

Defect engineering of the vortex lattice
----------------------------------------

To investigate the robustness of a vortex lattice in a rapidly rotating BEC, I will in this chapter discuss the effect of perturbations induced by adding or removing angular momentum through phase imprinting. This technique creates lattice imperfections, with stable topological lattice defects appearing during time evolution. The behaviour of these resulting defects is investigated over long times. I show that the vortex lattice demonstrates highly robust behaviour, even in the presence of such defects. I discuss the use of this method for creating varying degrees of disorder in the lattice, and propose it as a system for investigating transitions from ordered to disordered lattice geometries. The results presented in this chapter have been published in Phys. Rev. A **94**, 053603 (2016) .

Conclusions and outlook
-----------------------

In this chapter I conclude the work discussed in the thesis, and discuss extensions, and future ideas for the field.