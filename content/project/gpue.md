+++
# Date this page was created.
date = "2017-01-18"

# Project title.
title = "GPUE"

# Project summary to display on homepage.
summary = "GPU enabled Gross-Pitaevskii equation solver"

# Optional image to display on homepage (relative to `static/img/` folder).
image_preview = "gpue.png"

# Optional image to display on project detail page (relative to `static/img/` folder).
image = ""

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["sap","quantum", "bec", "gpu", "uqs"]

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = true

+++
[GPUE](https://github.com/gpue-group/gpue) represents the culmination of work by myself and [James Schloss](https://github.com/leios) into developing a high performance quantum dynamics simulator for investigating superfluid dynamics of Bose-Einstein condensates. This solver allows for studies of 2D and 3D dynamics, quantum vortices, arbitrary potential geometries, artificial gauge fields, as well as a variety of additional features. The solver makes use of CUDA libraries and custom kernels to enable GPUE to simulate quantum systems faster than [competing suites or libraries](https://peterwittek.com/gpe-comparison.html), as well as Python routines for post-processing and data analysis. Documentation and example use-cases are available [here](https://gpue-group.github.io), with the API available [here](https://gpue-group.github.io/development/html/).

Below are three sample videos from the 2D simulation code, showcasing a rapidly rotating BEC vortex lattice, subject to a variety of perturbations. Details of these works are available in the papers on [Moire superlattices](/publication/moire2016) and [defect engineering](/publication/defect2016) respectively.

|   |   |   |
|---|---|---|
|[![Moire superlattice following an optical lattice kick](https://img.youtube.com/vi/ajN40AIq7jA/0.jpg)](https://www.youtube.com/watch?v=ajN40AIq7jA "Superlattice") | [![2D BEC vortex lattice defect engineering](https://img.youtube.com/vi/o-BGZdF1MvI/0.jpg)](https://www.youtube.com/watch?v=o-BGZdF1MvI "Defect engineering") | [![Quantum sharingen](https://img.youtube.com/vi/UA7uVlu7Ykc/0.jpg)](https://www.youtube.com/watch?v=UA7uVlu7Ykc "Quantum sharingen") |

