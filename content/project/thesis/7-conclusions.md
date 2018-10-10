+++
# Date this page was created.
date = "2017-01-18"

# Project title.
title = "Thesis - 7. Conclusions"

# Project summary to display on homepage.
summary = "Thesis - 7. Conclusions"

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

Outlook
-------

Given the current state-of-the-art experimental control of condensate systems through use of SLMs, the perturbation methods discussed within this thesis are expected to be realisable. These perturbations represent two very useful techniques for quantum state control and engineering. For the kicked optical lattice, the creation of moiré interference patterns with wavelengths much greater than the lattice spacing opens the possibility for detecting vortices without time-of-flight expansion in a lattice. We consider this technique to be a unique method for examining the periodicity of a lattice system, where the evolving pattern can also potentially be observed through the *in-situ* imaging techniques, as discussed in \[sec:intro\_super\]. Further extensions of this work can involve investigating the periodicity of large-scale soliton trains in quasi-1D condensates.

Some preliminary work in small-scale zig-zag and linear vortex crystals was carried out in conjunction with A. Barahmi and Th. Busch. This showed that with little periodicity in the system there were negligible peaks in the compressible energy spectrum. As a result, there were no discernible moiré superlattice patterns in the condensate density. It is expected that for these structures to be observed that highly periodic systems with a well defined reciprocal lattice are required. However, given a highly periodic system, any disordering of the system will affect the visibility of the peaks. As a result, this method could potentially allow for an examination of lattice disorder, and can form the basis of a future investigation.

The vortex annihilation/flipping through phase imprinting appears to be a very good candidate to create varying degrees of disorder in a vortex lattice system. The analysis methods discussed and used for this work can easily be applied to real experimental data. A potential use for this is to create controllable routes towards quantum turbulence from a well-ordered system. While the examination presented focussed primarily on the use of phase profiles opposite to that of the lattice, the imprinting of like-signed vortices also remains an interesting choice. Forcing vortices into different locations in the lattice is potentially an additional method to create lattice dislocations, and hence, topological lattice defects. One might consider erasing and adding vortices at different locations to both create and remove topological lattice defects. This can form the basis for a memory storage technique in a quantum computing system. The applicability of this method can potentially be examined in a future work.

Additionally, one can also create multi-charged vortices in the condensate. The effect of the surrounding lattice on the resulting multi-charged vortex would be an interesting problem. One might expect the *l*-charge vortex decay to be suppressed if the energy to move the surrounding lattice vortices is greater than the energy to maintain the *l*-charged vortex. This was briefly investigated by examining the Bogoliubov-de Gennes solutions of the imprinted vortex lattice system, with the aim of observing if the resulting excitation modes were complex. These modes were, however, not found due to the numerical complexity of the problem, and it remains an open question if this suppression exists. This will be investigated in a future work.

While we briefly mentioned the search for a KTHNY hexatic phase transition in this system, this will require further examination. Future work can include an investigation for the existence of this transition, and examine whether dislocation mediated melting of the vortex lattice can occur as a result of the phase imprinting techniques. Though we consider the framework developed and examined for all the above methods to be valid, the consideration of finite temperature effects would ensure that the investigated methods are truly physically realistic. For such finite temperature condensates, one might consider use of the Zaremba–Nikuni–Griffin (ZNG) formalism , or the formalism of Billam *et al.* . An extension of the above works can examine this.

The use of GPU computing for simulating quantum dynamics is currently an under-utilised paradigm. The potential for a significant performance gain exists, given an effective mapping of a numerical algorithm to the GPU hardware. While the code developed and utilised for all the above simulations offers a clear performance advantage, it should be noted that further development and maintenance of such code can be challenging. Rapid changes to the CUDA programming models have introduced many new features to the standard which could potentially be used for solving more complex problems of both linear and nonlinear Schrödinger-type problems. However, such changes often require training, software rewrites, or newer hardware to take advantage of these. An extension of the GPUE codebase to cover one and three dimensional Gross–Pitaevskii systems will allow for this suite to be as feature rich as the currently most capable suites available , whilst still holding the current edge in performance. Solutions using arbitrary gauge fields for these problems will also offer a distinctive advantage. Additionally, the inclusion of a numerical BdG solver for the resulting numerical solutions will allow for this software to become a very general suite for BEC problems.

The methods and works examined in this thesis offer interesting answers, questions and possibilities for the future of controllable quantum systems and technologies.