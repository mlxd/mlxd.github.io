+++
title = "Questions to expand upon"
draft = false
date = "2016-05-11"
tags = ["gpu","uqs","quantum"]
math = "true"
summary = "A place for me to place questions I've considered, and aim to answer or develop further."
+++

# Questions to consider

- What is the optimal choice of FFTs over competing HPC software and device architectures for quantum dynamics using pseduospectral methods?
- Do space filling curves have a performance impact on multidimensional data when performing FFTs along the i-th dimension?
- Is there an optimal way to permute an n-D data set so that the elements along a specific basis direction are linear in memory? Reasons for this are optimal data access when performing operations, allowing for higher cache hits (such as with FFTs on GPUs are an example).