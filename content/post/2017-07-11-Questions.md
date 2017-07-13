+++
title = "Questions to expand upon"
draft = false
date = "2016-05-11"
tags = ["gpu","uqs","quantum"]
math = "true"
summary = "A place for me to place questions I've considered, and aim to answer or develop further."
+++

# Questions to consider
---
## Q. How do available FFT routines compare over different libraries and accelerator hardwares?
---
This is something I started to investigate, though it can be difficult for an apples-to-apples comparison. 

Firstly, the existing (and soon to be existing) technologies with which computations can be performed are:
$$
d = \left( \textrm{CPU, GPU, KNL, FPGA, IBM_TN, QM} \right)
$$

A representative sample of the widely known and available libraries, both platform independent and not, are:
$$
l = \left( \textrm{CUFFT, clFFT, MKL-FFT, FFTW} \right)
$$

` Generate a table of the support of the hardware with the available software libraries, or alternatives.
`