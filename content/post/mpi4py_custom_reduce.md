+++
title = "Playing with MPI: custom reduce operations"
draft = true
date = "2019-01-01"
tags = ["math","cs","hpc"]
math = "true"
summary = "An examination of user-defined reduce operations in MPI with python"
+++

For the sake of simplicity, we are going to avoid the use of C/C++, and instead opt for Python. While performance may be higher using C/C++, using python offers a much lower barrier to entry (and we gain plotting tools for free with `matplotlib`).


# Installing MPI & mpi4py

## Option 1: System package manager


## Option 2: Manual


![alt text](/img/ndData.jpg "A sample layout permutation")

The included image above is an example of rotation about the $j$ axis, showing on the right the respective mappings to new index values (for the sake of my interest I've taken slices along $j$ and $i$, but worked with the latter for the rest of this discussion). 

The bottom of the image shows the rearrangements of memory from the original (upper) to the rotated (lower) memory configurations. The indices themselves seem to take the following mappings (orig. $\rightarrow $ rot.):

$$ \begin{align}
i & \rightarrow \textrm{stride}(j)-k, \\\\\\
j & \rightarrow j, \\\\\\
k & \rightarrow i, \\ 
\end{align}$$
