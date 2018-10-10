+++
# Date this page was created.
date = "2017-01-18"

# Project title.
title = "ExaFEL"

# Project summary to display on homepage.
summary = "Exascale FEL crystallography"

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
# STRUMPACK vs EIGEN for Levenberg-Marquardt nonlinear least squares optimisation

The following documentation and data formed part of the work I carried out at LBNL from 2017 to 2018. The project described herein follows the integration of the [STRUMPACK](https://github.com/pghysels/STRUMPACK) sparse linear solver library into CCTBX. Comparing solver's various backends with EIGEN's gives an indication of the applicability of each individual solver to the given problem (Levenberg-Marquardt nonlinear least-squares minimisation) to the observed data. Scalability comparisons are drawn to showcase which backend offers the best performance, allowing for future design decisions. The given code and documents are all available at [ExaFEL](https://github.com/ExaFEL/exafel_project/tree/master/95-strumpack_cctbx).

Each different test data set receives its own analysis, as well as a distributed test using CORI. The following files represent part of the scalability analysis performed:

- [Single node, MPI+OpenMP, $10^3$ images](/page/exafel/strumpacksolvermpi_1k_data)
- [Single node, MPI+OpenMP, $5\times 10^3$ images](/page/exafel/strumpacksolvermpi_5k_data)
- [Single node, MPI+OpenMP, $10^4$ images](/page/exafel/strumpacksolvermpi_10k_data)
- [Multi node (CORI), MPI, $5\times 10^3$ images](/page/exafel/strumpacksolvermpi_dist_cori)

For my other contributions to ExaFEL project, please visit the [ExaFEL site](https://exafel.github.io/docs).
