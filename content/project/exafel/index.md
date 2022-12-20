+++
# Project title.
title = "ExaFEL"

# Date this page was created.
date = 2017-01-18

# Project summary to display on homepage.
summary = "Exascale FEL crystallography"

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["data", "bio", "hpc", "knl"]

# Optional external URL for project (replaces project detail page).
#external_link = "https://exafel.github.io/docs"
external_link = ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references 
#   `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides = ""

# Links (optional).
url_pdf = ""
url_slides = ""
url_video = ""
url_code = ""

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
# url_custom = [{icon_pack = "fab", icon="twitter", name="Follow", url = "https://twitter.com/mlxd"}]

# Featured image
# To use, add an image named `featured.jpg/png` to your project's folder. 
[image]
  # Caption (optional)
  caption = ""
  
  # Focal point (optional)
  # Options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
  focal_point = "BottomLeft"

  preview_only = true

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

Installation details and additional information can be found [here](/page/exafel/instructions).

For my other contributions to ExaFEL project, please visit the [ExaFEL site](https://exafel.github.io/docs).
