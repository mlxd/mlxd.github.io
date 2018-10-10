+++
# Date this page was created.
date = "2017-01-18"

# Project title.
title = "Thesis - 1. Introduction"

# Project summary to display on homepage.
summary = "Thesis - 1. Introduction"

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

<span>Thesis submitted for the degree </span>
<span>Doctor of Philosophy</span>

------------------------------------------------------------------------

<span>**<span>Non-equilibrium vortex dynamics in rapidly rotating Bose–Einstein condensates</span>**</span>

------------------------------------------------------------------------

by
<span>February, 2017</span>

Declaration of Original and Sole Authorship
===========================================

I, <span>Lee James O’Riordan</span>, declare that this thesis entitled *<span>Non-equilibrium vortex dynamics in rapidly rotating Bose–Einstein condensates</span>* and the data presented in it are original and my own work.

I confirm that:

-   This work was done solely while a candidate for the research degree at the Okinawa Institute of Science and Technology Graduate University, Japan and the University College Cork, Ireland.

-   No part of this work has previously been submitted for a degree at this or any other university.

-   References to the work of others have been clearly attributed. Quotations from the work of others have been clearly indicated, and attributed to them.

-   In cases where others have contributed to part of this work, such contribution has been clearly acknowledged and distinguished from my own work.

Date: <span>February, 2017</span>

Signature:

Abstract
========

### <span>Non-equilibrium vortex dynamics in rapidly rotating Bose–Einstein condensates</span>

This body of work examines the non-equilibrium dynamics of vortex lattice carrying Bose–Einstein condensates. We solve the mean-field Gross–Pitaevskii equation for a two-dimensional pancake geometry, in the co-rotating frame within the limit of high rotation frequencies. The condensate responds to this by creating a large periodic lattice of vortices with 6-fold triangular symmetry. By applying two distinct perturbations to this lattice, we examine the resulting effects on the vortices during time evolution. The first perturbation involves applying an optical potential with matching geometry to the vortex lattice. We observe the appearance of interference fringes, and we show that these can be described by moiré interference theory. This is backed up by a decomposition of the kinetic energy spectra of the condensate. The applied perturbation only modifies the condensate density, with the vortex positions largely unaffected. From this we conclude that the vortex lattice is very stable and robust against phononic disturbances.

Next, by removing vortices at predefined positions in the lattice using phase imprinting techniques, we examine the resulting order of the lattice. By performing this we generate stable topological defects in the crystal structure. The resulting lattice remains highly ordered in the presence of low numbers of these defects, where crystal structure and order of the lattice shows to be highly robust. By varying the type of imprinted phases we can create controllable degrees of disorder in the lattice. This disorder is analysed using orientational correlations, Delaunay triangulation, and Voronoi diagrams of the vortex lattice, and demonstrates a method for examining order and generating disorder in vortex lattices in Bose–Einstein condensates.

All work described makes extensive use of GPU computing techniques, and allows for the simulation of these systems to be realised in short times. The implementation of the calculations using GPU computing are also discussed, where the software is shown to be the fastest of its kind out of the independently tested software suites.

License and copyright information
=================================

The work contained in this thesis includes materials that have been published elsewhere. Notably, Chapters 3, 5 and 6 include figures and text that are published in adapted and original form in the American Physical Society’s journal Physical Review A. Copyright permission has been obtained for use of these figures, and for brevity the copyright notice is given here.

### Chapter 3. Numerical methods

Reprinted excerpts and figures with permission from Tadhg Morgan, Lee James O’Riordan, Neil Crowley, Brian O’Sullivan and Thomas Busch, Physical Review A **88**, 053618 (2013). Copyright (2013) by the American Physical Society.

### Chapter 5. Moiré superlattice structures

Reprinted excerpts and figures with permission from Lee James O’Riordan, Angela White and Thomas Busch, Physical Review A **93**, 023609 (2016). Copyright (2016) by the American Physical Society.

### Chapter 6. Defect engineering of the vortex lattice

Reprinted excerpts and figures with permission from Lee James O’Riordan, and Thomas Busch, Physical Review A **94**, 053603 (2016). Copyright (2016) by the American Physical Society.

Acknowledgments
===============

/\* Firstly, I would like to express my sincerest gratitude to my family back in the ’mel who have always been there for me, and provided amazing support from across the planet since I started my PhD (and before I started too) — most notably, to Mum, Nan, and Maria, who have helped so much over the past few years. Next, my sincerest gratitude and thanks goes to Prof. Thomas Busch, for freeing me from the working world and introducing me to the (ultra)cool world of atomic physics, as well as the guidance he has offered over the years. To past and present members of both the Ultracold Quantum Gases group at University College Cork, and Quantum Systems Unit at OIST, thanks for keeping me sane and helping out through the work and writing over the years — in no particular order: Mossy, Steve, Jeremie, Tara, Albert, Angela, Sawako, Rashi, James, Irina, the magnificent Dave Rea, and of course Tadhg. Next, I would like to thank the Graduate School for all of their help removing bureaucratic dealings from my everyday life here. A special thanks goes to IT & the Scientific Computing Section, for allowing me access to some very amazing toys over these past few years. A special thanks also to Annie, for supplying me with enough caffeine to get to the moon and back. Lastly, but not least-ly, to Christina for her compassion, understanding, and companionship over the past few years. \*/

*i*← ありがとうございます!

<span>
- <span>D</span><span>o</span><span>o</span></span><span>M</span> <span>6) [1]</span>

Introduction
============

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