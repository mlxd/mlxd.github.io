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
[GPUE](https://github.com/gpue-group/gpue) represents the culmination of work by myself and [James Schloss](https://github.com/leios) into investigating superfluid dynamics using Bose-Einstein condensates. This solver allows for studies of 2D and 3D dynamics, quantum vortices, arbitrary potential geometries, artificial gauge fields, as well as a variety of additional features. The solver makes use of CUDA libraries and custom kernels to enable GPUE to simulate quantum systems faster than [competing suites or libraries](https://peterwittek.com/gpe-comparison.html), as well as Python routines for post-processing and data analysis. Documentation and example use-cases are available [here](https://gpue-group.github.io), with the API available [here](https://gpue-group.github.io/development/html/).
