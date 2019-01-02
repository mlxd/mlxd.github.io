+++
# Project title.
title = "GPUE: A GPU enabled Gross–Pitaevskii equation solver."

# Project summary to display on homepage.
summary = "GPU enabled Gross-Pitaevskii equation solver"

# Optional image to display on homepage (relative to `static/img/` folder).
image_preview = ""

# Date this page was created.
date = "2017-01-18"

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["quantum", "hpc", "gpu", "bec"]

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
url_custom = [{icon_pack = "fab", icon="twitter", name="Follow", url = "https://twitter.com/mlxd"}]

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
math = "true"


+++
<img src="/img/gpue.png" width=480/>


[GPUE](https://github.com/gpue-group/gpue) represents the culmination of work by myself and [James Schloss](https://github.com/leios) into developing a high performance quantum dynamics simulator for investigating superfluid dynamics of Bose-Einstein condensates. This solver allows for studies of 2D and 3D dynamics, quantum vortices, arbitrary potential geometries, artificial gauge fields, as well as a variety of additional features. The solver makes use of CUDA libraries and custom kernels to enable GPUE to simulate quantum systems faster than [competing suites or libraries](https://peterwittek.com/gpe-comparison.html), as well as Python routines for post-processing and data analysis. Documentation and example use-cases are available [here](https://gpue-group.github.io), with the API available [here](https://gpue-group.github.io/development/html/).

The work has been [peer-reviewed](https://github.com/openjournals/joss-reviews/issues/1037) and [published](http://joss.theoj.org/papers/10.21105/joss.01037) in the Journal of Open Source Software (JOSS), and is cited as:
$\\textrm{Schloss and O'Riordan, (2018). GPUE: Graphics Processing Unit Gross--Pitaevskii Equation solver.}$ $\\textrm{Journal of Open Source Software, 3(32), 1037,}$ https://doi.org//10.21105/joss.01037.

Below are three sample videos from the 2D simulation code, showcasing a rapidly rotating BEC vortex lattice, subject to a variety of perturbations. Details of these works are available in the papers on [Moire superlattices](/publication/moire2016) and [defect engineering](/publication/defect2016) respectively.

|   |   |   |
|---|---|---|
|[![Moire superlattice following an optical lattice kick](https://img.youtube.com/vi/ajN40AIq7jA/0.jpg)](https://www.youtube.com/watch?v=ajN40AIq7jA "Superlattice") | [![2D BEC vortex lattice defect engineering](https://img.youtube.com/vi/o-BGZdF1MvI/0.jpg)](https://www.youtube.com/watch?v=o-BGZdF1MvI "Defect engineering") | [![Quantum sharingen](https://img.youtube.com/vi/UA7uVlu7Ykc/0.jpg)](https://www.youtube.com/watch?v=UA7uVlu7Ykc "Quantum sharingen") |


---

## GPUE topics
The following pages discuss different aspects of the GPUE functionality. Much of the following can be found in the [documentation](https://gpue-group.github.io).

 - [2D BEC vortex tracking](/post/vortex_2d/): This explains the tracking and determination of vortex core positions from 2D complex scalar field data (aka a wavefunction in 2D).
 - [FFT along n-th dimension](/post/fft_nd/): For implementing the angular momentum operators, performing an FFT along individual dimensions of a multidimensional data set is necessary. This discusses our implementation in GPUE. 