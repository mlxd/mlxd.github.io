+++
# Project title.
title = "Levenberg-Marquardt sparse solver scaling"

# Date this page was created.
date = "2018-10-10"

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

# Notes on using Strumpack within CCTBX
Here are the documented results of using Strumpack on a single node for a variety of data set sizes (`StrumpackSolverMPI_1K`,`StrumpackSolverMPI_5K`,`StrumpackSolverMPI_10K`). All tests were performed on `dials.lbl.gov`, and allow the tests to be repeated at the user's discretion. Example matrices for a variety of different refinement parameters are listed in the given paths, and the times represent a single solution.

# Setting up and running STRUMPACK
To build STRUMPACK alongside a conda cctbx.xfel build, follow the instructions given [here](https://exafel.github.io/docs/psana-cctbx-install)/[here](https://github.com/ExaFEL/exafel_project/tree/master/nks) with the following additional conda packages before building cctbx:

```bash
conda install -y IPython mysql-python matplotlib scipy mpi4py jupyter ipyparallel;
```

Building is now carried out as normal:
```bash
wget https://raw.githubusercontent.com/cctbx/cctbx_project/master/libtbx/auto_build/bootstrap.py;
python bootstrap.py hot update --builder=xfel --sfuser=<USERNAME> --cciuser=<USERNAME>;
python bootstrap.py build --builder=xfel --with-python=`which python` --nproc=<NUM_CORES>;
source ./build/setpaths.sh;
```

The MPI-enabled work requires the *cctbx_project* git repository be checked-out into the *strumpack_solver_backend* branch (eventually all changes will be merged with upstream *master*).
```bash 
cd modules/cctbx_project
git checkout strumpack_solver_backend
cd ../..
```

We can now proceed with building STRUMPACK, and it's required dependencies. The script located at `github.com/exafel/exafel_project/master/strumpack/STRUMPACK_installer_shared.sh` will acquire all dependencies and build the libraries against the conda MPI installation, or build OpenMPI if this is not available (such as may be true under MacOS).

```bash
wget https://raw.githubusercontent.com/ExaFEL/exafel_project/master/strumpack/STRUMPACK_installer_shared.sh
chmod +x STRUMPACK_installer_shared.sh
./STRUMPACK_installer_shared.sh
```

The STRUMPACK (and dependencies) binaries, libraries and headers will be installed into `strumpack_build/{bin,lib,include}`, of which will be added to the dispatcher environment given successful completion of the installation script. With the presence of the new libraries, refreshing the dispatcher and rebuilding the packages will allow any STRUMPACK-enabled Boost.Python extension modules to be built.

```bash
cd build && libtbx.refresh
make
```
To test the new libraries several test scripts are available, for a provided A matrix and b vector solution. The solver will compare both the original Eigen-based solver and the new STRUMPACK solvers for both the OpenMP backend (`cctbx_project/scitbx/examples/bevington/strumpack_eigen_solver.py`) and the distributed MPI-enabled solver (`cctbx_project/scitbx/examples/bevington/strumpack_eigen_solver_mpi_dist.py`).

Sample run commands are given in the provided notebooks.

Full integration with Samosa will be provided shortly to allow the solvers to be used therein. Initial work to provide selective choice of the Eigen solver backend is available in the `eigen_solver_algo` branch of `https://github.com/cctbx/cctbx_project/`.



# Scalability tests on Cori

Scalability testing of the OpenMP and MPI backends are available in Jupyter notebook [StrumpackSolverMPI_dist_Cori.ipynb](https://github.com/ExaFEL/exafel_project/blob/master/95-strumpack_cctbx/StrumpackSolverMPI_dist_Cori.ipynb)

# Experimental spack build
An experimental set of commands to build STRUMPACK using spack is given [here](https://github.com/ExaFEL/exafel_project/tree/master/95-strumpack_cctbx/spack_installation). This is not supported, and not guaranteed to work. It was used as an example environment to build STRUMPACK dependencies.
