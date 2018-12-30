+++
# Project title.
title = "Levenberg-Marquardt sparse solver scaling: Cori Haswell and KNL"

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
math = true

+++

# Performance tests of STRUMPACK using OpenMP and MPI on Cori

We make use of the linear solver backends within cctbx.xfel, and evaluate their performance on a variety of sample A matrices and b vectors for 1k, 5k, 10k, and 32k images each, and for a variety of different refinement parameters. The time to solve a single sparse system will offer insight into the time spent during the Levenberg-Marquardt minimisation process.

For these tests we use EIGEN with the following solver algorithms:

$$
\\textrm{EIGEN} : \\left\\{ \\begin{matrix}
        \\textrm{SimplicialLLT} \\\\\\
        \\textrm{SimplicialLDLT} \\\\\\
        \\textrm{ConjugateGradient} \\\\\\
        \\textrm{BiCGSTAB} \\\\\\
    \\end{matrix} \\right\\}
$$

Additionally, we test against the solvers from STRUMPACK using both the OpenMP and MPI enabled backends. For the single-node (OpenMP) tests, the solvers and reordering methods are:

$$
\\textrm{OpenMP STRUMPACK} : 
\\left\\{
\\begin{matrix}
 \\textrm{METIS} & \\textrm{AUTO} \\\\\\
 \\textrm{METIS} & \\textrm{BiCGSTAB} \\\\\\
 \\textrm{METIS} & \\textrm{PREC\_BICGSTAB} \\\\\\
 \\hline
 \\textrm{SCOTCH} & \\textrm{AUTO} \\\\\\
 \\textrm{SCOTCH} & \\textrm{BiCGSTAB} \\\\\\
 \\textrm{SCOTCH} & \\textrm{PREC\_BICGSTAB} \\\\\\
\\end{matrix}
\\right\\}
$$

For the MPI tests, we can use the parallel versions of the reordering methods, and choose from a variety of solvers as:

$$
\\textrm{MPI STRUMPACK} : 
\\left\\{
\\begin{matrix}
 \\textrm{PARMETIS} & \\textrm{AUTO} \\\\\\
 \\textrm{PARMETIS} & \\textrm{DIRECT} \\\\\\
 \\textrm{PARMETIS} & \\textrm{PREC\_BICGSTAB} \\\\\\
 \\textrm{PARMETIS} & \\textrm{REFINE} \\\\\\
 \\textrm{PARMETIS} & \\textrm{PREC\_GMRES} \\\\\\
 \\textrm{PARMETIS} & \\textrm{GMRES} \\\\\\
 \\textrm{PARMETIS} & \\textrm{PREC\_BICGSTAB} \\\\\\
 \\textrm{PARMETIS} & \\textrm{BICGSTAB} \\\\\\
 \\hline
 \\textrm{PTSCOTCH} & \\textrm{AUTO} \\\\\\\
 \\textrm{PTSCOTCH} & \\textrm{DIRECT} \\\\\\
 \\textrm{PTSCOTCH} & \\textrm{PREC\_BICGSTAB} \\\\\\
 \\textrm{PTSCOTCH} & \\textrm{REFINE} \\\\\\
 \\textrm{PTSCOTCH} & \\textrm{PREC\_GMRES} \\\\\\
 \\textrm{PTSCOTCH} & \\textrm{GMRES} \\\\\\
 \\textrm{PTSCOTCH} & \\textrm{PREC\_BICGSTAB} \\\\\\
 \\textrm{PTSCOTCH} & \\textrm{BICGSTAB} \\\\\\
\\end{matrix}
\\right\\}
$$

For this notebook we will enable the STRUMPACK-enabled conda backend, built following the instructions [here](https://github.com/ExaFEL/exafel_project/tree/master/95-strumpack_cctbx) and setting the Jupyter kernel following [these instructions](https://github.com/ExaFEL/exafel_project/tree/master/jupyter). We also perform job submission using the SlurmMagics Python package, allow direct submission to the Cori queueing system. Additionally, we may also dynamically generate submission scripts, and plot the results upon completion. We begin by verifying the environment is correctly built and set up:

```python
!which libtbx.python
```

    /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/build/bin/libtbx.python

```python
from slurm_magic import SlurmMagics
ip = get_ipython()
ip.register_magics(SlurmMagics)
```


```python
%squeue -u mlxd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>JOBID</th>
      <th>PARTITION</th>
      <th>NAME</th>
      <th>USER</th>
      <th>ST</th>
      <th>TIME</th>
      <th>NODES</th>
      <th>NODELIST(REASON)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



# OpenMP solver performance on Haswell


```python
OMP_SOLVER='''
from __future__ import division
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.development.timers import Profiler
import sys
import numpy as np
import scipy.sparse as sps

A_path=sys.argv[1]
A_mat = np.loadtxt(A_path,dtype={'names':('rows','cols','vals'),'formats':('i8','i8','f8')})

b_path=sys.argv[2] 
b_vec = np.loadtxt(b_path)

n_rows = len(b_vec)
n_cols = n_rows

A_sp = sps.csr_matrix((A_mat['vals'],(A_mat['rows'],A_mat['cols'])))

#Check for triangular matrix. If so, A_sp := A+A^T - diag(A)

tu=sps.triu(A_sp)
tl=sps.tril(A_sp)
sd=sps.diags(A_sp.diagonal())

A_spS = A_sp
if tu.nnz == sd.getnnz() or tl.nnz == sd.getnnz():
    A_spS = A_sp + A_sp.transpose() - sd

A_indptr = flex.int(A_sp.indptr)
A_indices = flex.int(A_sp.indices)
A_values = flex.double(A_sp.data)
b = flex.double(b_vec)

#import time
#timing_dict = {"strum":0, "eigen":0}

import boost.python
ext_omp = boost.python.import_ext("scitbx_examples_strumpack_solver_ext")
ext_mpi = boost.python.import_ext("scitbx_examples_strumpack_mpi_dist_solver_ext")

es   = ext_omp.eigen_solver
ss   = ext_omp.strumpack_solver

eps_tol = 1e-3

def run_solver(n_rows, n_cols, A_indptr ,A_indices, A_values, b, A_spS):

    P = Profiler("EIGEN_LLT_CHOL")
    res_eig_llt_chol = es(0, n_rows, n_cols, A_indptr, A_indices, A_values, b)
    del P
    
    P = Profiler("EIGEN_LDLT_CHOL")
    res_eig_ldlt_chol = es(1, n_rows, n_cols, A_indptr, A_indices, A_values, b)
    del P
    
    A_indptr = flex.int(A_spS.indptr)
    A_indices = flex.int(A_spS.indices)
    A_values = flex.double(A_spS.data)
    
    #Seems to fail for certain data sets.
    #P = Profiler("EIGEN_BICGSTAB")
    #res_eig_bicgstab = es(2, n_rows, n_cols, A_indptr, A_indices, A_values, b)
    #del P
    
    P = Profiler("EIGEN_CG")
    res_eig_cg = es(3, n_rows, n_cols, A_indptr, A_indices, A_values, b)
    del P
    
    P = Profiler("STRUMPACK_SCOTCH_AUTO")
    res_strum_sc_a = ss(n_rows, n_cols, A_indptr, A_indices, A_values, b, ext_omp.scotch, ext_omp.auto)
    del P
    
    P = Profiler("STRUMPACK_METIS_AUTO")
    res_strum_mt_a = ss(n_rows, n_cols, A_indptr, A_indices, A_values, b, ext_omp.metis, ext_omp.auto)
    del P

    P = Profiler("STRUMPACK_SCOTCH_BICGSTAB")
    res_strum_sc_bi = ss(n_rows, n_cols, A_indptr, A_indices, A_values, b, ext_omp.scotch, ext_omp.bicgstab)
    del P
    
    P = Profiler("STRUMPACK_METIS_BICGSTAB")
    res_strum_mt_bi = ss(n_rows, n_cols, A_indptr, A_indices, A_values, b, ext_omp.metis, ext_omp.bicgstab)
    del P
    
    P = Profiler("STRUMPACK_SCOTCH_PRECBICGSTAB")
    res_strum_sc_pr = ss(n_rows, n_cols, A_indptr, A_indices, A_values, b, ext_omp.scotch, ext_omp.prec_bicgstab)
    del P
    
    P = Profiler("STRUMPACK_METIS_PRECBICGSTAB")
    res_strum_mt_pr = ss(n_rows, n_cols, A_indptr, A_indices, A_values, b, ext_omp.metis, ext_omp.prec_bicgstab)
    del P
    num_errors = 0
    err_names = {}
    for i in xrange(len(res_strum_sc_a.x)):
        if not approx_equal(res_eig_llt_chol.x[i], res_eig_ldlt_chol.x[i], eps=eps_tol): 
            num_errors += 1
            print "Error in res_eig_llt_chol:=%f != res_eig_ldlt_chol:=%f @ [%d] "%(res_eig_llt_chol.x[i], res_eig_ldlt_chol.x[i],i)
            
        #if not approx_equal(res_strum_sc_a.x[i], res_eig_bicgstab.x[i], eps=eps_tol):
        #    num_errors += 1
        #    print "Error in res_strum_sc_a:=%f != res_eig_bicgstab:=%f @ [%d] "%(res_strum_sc_a.x[i], res_eig_bicgstab.x[i],i)
            
        if not approx_equal(res_strum_sc_a.x[i], res_eig_cg.x[i], eps=eps_tol):
            num_errors += 1
            print "Error in res_strum_sc_a:=%f != res_eig_cg:=%f @ [%d] "%(res_strum_sc_a.x[i], res_eig_cg.x[i],i)

        if not approx_equal(res_strum_sc_a.x[i], res_eig_ldlt_chol.x[i], eps=eps_tol):
            num_errors += 1
            print "Error in res_strum_sc_a:=%f != res_eig_ldlt_chol:=%f @ [%d] "%(res_strum_sc_a.x[i], res_eig_ldlt_chol.x[i],i)

        if not approx_equal(res_strum_mt_a.x[i], res_strum_sc_a.x[i], eps=eps_tol):
            num_errors += 1
            print "Error in res_strum_mt_a:=%f != res_strum_sc_a:=%f @ [%d] "%(res_strum_mt_a.x[i], res_strum_sc_a.x[i],i)

        if not approx_equal(res_strum_mt_a.x[i], res_strum_sc_bi.x[i], eps=eps_tol):
            num_errors += 1
            print "Error in res_strum_mt_a:=%f != res_strum_sc_bi:=%f @ [%d] "%(res_strum_mt_a.x[i], res_strum_sc_bi.x[i],i)

        if not approx_equal(res_strum_mt_a.x[i], res_strum_mt_bi.x[i], eps=eps_tol):
            num_errors += 1
            print "Error in res_strum_mt_a:=%f != res_strum_mt_bi:=%f @ [%d] "%(res_strum_mt_a.x[i], res_strum_mt_bi.x[i],i)

        if not approx_equal(res_strum_mt_a.x[i], res_strum_sc_pr.x[i], eps=eps_tol):
            num_errors += 1
            print "Error in res_strum_mt_a:=%f != res_strum_sc_pr:=%f @ [%d] "%(res_strum_mt_a.x[i], res_strum_sc_pr.x[i],i)

        if not approx_equal(res_strum_mt_a.x[i], res_strum_mt_pr.x[i], eps=eps_tol):
            num_errors += 1
            print "Error in res_strum_mt_a:=%f != res_strum_mt_pr:=%f @ [%d] "%(res_strum_mt_a.x[i], res_strum_mt_pr.x[i],i)
            
    assert (num_errors == 0)
        
run_solver(n_rows, n_cols, A_indptr ,A_indices, A_values, b, A_spS)
'''
```


```python
OMP_SOLVER_FILE = open("OMP_SOLVER.py", "w")
OMP_SOLVER_FILE.write(OMP_SOLVER)
OMP_SOLVER_FILE.close()
```


```python
DATAPATH="/global/cscratch1/sd/mlxd/feb_sprint/data_matrices/samosa"
A_LIST = !find {DATAPATH} -iname 'A*.csv'
B_LIST = [ii.replace('/A_','/b_') for ii in A_LIST]
```


```python
list_idx={}
for imgs in ['1k','5k','10k','32k']:
    list_idx.update({imgs:[i for i, j in enumerate(A_LIST) if imgs in j]})
list_idx
```




    {'10k': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
     '1k': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
     '32k': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
     '5k': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]}




```python
SBATCH_SCRIPT_HWL=\
"""#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -o <DATNAME>_hwl_out_OMP.log
#SBATCH -e <DATNAME>_hwl_out_OMP.err
#SBATCH -J CCTBX_STRUMPACK
#SBATCH --mail-user=loriordan@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00

#OpenMP settings:
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

#run the application:
cd /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST
source $PWD/miniconda/bin/activate myEnv
source $PWD/build/setpaths.sh
module load darshan
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
mkdir -p OMP_HWL
cd OMP_HWL

mkdir <DATNAME>_hwl_out_OMP
cd <DATNAME>_hwl_out_OMP

#Use FPE mask to avoid floating point exceptions. Need to further investigate reasons for these
for n in 1 2 4 8 16 32 64;
do
    t_c=$((64/${n}));
    echo "OMP_NUM_THREADS=${n} BOOST_ADAPTBX_FPE_DEFAULT=1 time srun -n 1 -c 64 --cpu_bind=cores\
        libtbx.python /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/OMP_SOLVER.py <DATA_A> <DATA_B>";
        
    OMP_NUM_THREADS=${n} BOOST_ADAPTBX_FPE_DEFAULT=1 srun -n 1 -c 64 --cpu_bind=cores libtbx.python \
      /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/OMP_SOLVER.py <DATA_A> <DATA_B>\
      1> <DATNAME>_hwl_out_OMP${n}.log 2> <DATNAME>_hwl_out_OMP${n}.err
      
done
cat <DATNAME>_hwl_out_OMP${n}.log >> <DATNAME>_hwl_out_OMP.log
cat <DATNAME>_hwl_out_OMP${n}.err >> <DATNAME>_hwl_out_OMP.err
"""
```

Generate sbatch submission scripts and send to queue


```python
str_out={}
import os
threads_list = [1,2,4,8,16,32,64]
sub_scripts = []
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        
        A_path = A_LIST[imgs_idx]; b_path = B_LIST[imgs_idx]
        dat_name = A_path.split('/')[-1][2:-4]

        print "Data Set Name:=%s"%(dat_name)

        #Ensure the A and b data are matched correctly
        assert(os.path.dirname(A_path) == os.path.dirname(b_path))
        SUBMIT = SBATCH_SCRIPT_HWL.replace('<DATA_A>',A_path)\
           .replace('<DATA_B>', b_path)\
           .replace('<DATNAME>', dat_name)
        SBATCH_SCRIPT_FILE = open("SBATCH_SCRIPT_OMP_HWL_%s.sh"%(dat_name), "w")
        sub_scripts.append("SBATCH_SCRIPT_OMP_HWL_%s.sh"%(dat_name) )
        SBATCH_SCRIPT_FILE.write(SUBMIT)
        SBATCH_SCRIPT_FILE.close()
        var = !sbatch {sub_scripts[-1]}
        print var
```

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-
    ['Submitted batch job 11713779']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    ['Submitted batch job 11713780']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    ['Submitted batch job 11713781']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11713792']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-
    ['Submitted batch job 11713793']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-
    ['Submitted batch job 11713794']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    ['Submitted batch job 11713795']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Eta-
    ['Submitted batch job 11713797']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11713798']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11713799']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    ['Submitted batch job 11713802']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-
    ['Submitted batch job 11713803']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-
    ['Submitted batch job 11713804']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Deff-
    ['Submitted batch job 11713806']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11713807']
    Data Set Size:=32k
    Skipping 32k


Upon completion of the submitted jobs, we take note of the output logs for each parameter set and OpenMP thread number.


```python
dat_list_hwl_omp = {}
threads_list = [1,2,4,8,16,32,64]
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        for OMP in threads_list:
            dat = A_LIST[imgs_idx].split('/')[-1][2:-4]
            var = !find ./OMP_HWL -iname '{dat}_hwl_out_OMP{OMP}.log' #!ls OMP_KNL/{dat}_knl_out_OMP/ | grep 'log'
            dat_list_hwl_omp.update({dat + str(OMP) : var})
print dat_list_hwl_omp
```

<details>
<summary><p><b><span style="color:#ff3333; border:2px white solid; font-size:20px">---CLICK FOR OUTPUT---</span></b></p></summary>


    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Data Set Size:=32k
    Skipping 32k
    {'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP8.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-16': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP16.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP4.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_OMP64.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_OMP2.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_OMP32.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-4': ['./OMP_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_OMP4.log']}

</details>

```python
time_data_hwl = {}
for key, value in dat_list_hwl_omp.iteritems():
    with open(str(value[0])) as f:
        lines = f.read().splitlines()
        zlist = {}
        [zlist.update({l.split()[-7][:-1]:float(l.split()[-3][:-2])})\
          for l in lines if "#calls" in l]
        time_data_hwl.update({key : zlist})
print time_data_hwl
```

<details>
<summary><p><b><span style="color:#ff3333; border:2px white solid; font-size:20px">---CLICK FOR OUTPUT---</span></b></p></summary>


    {'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.229, 'STRUMPACK_METIS_PRECBICGSTAB': 1.814, 'EIGEN_LLT_CHOL': 38.758, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 3.095, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.707, 'STRUMPACK_METIS_AUTO': 1.835, 'EIGEN_LDLT_CHOL': 38.734, 'STRUMPACK_METIS_BICGSTAB': 0.261}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.245, 'STRUMPACK_METIS_PRECBICGSTAB': 24.702, 'EIGEN_LLT_CHOL': 39.989, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 15.041, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 13.628, 'STRUMPACK_METIS_AUTO': 24.753, 'EIGEN_LDLT_CHOL': 39.248, 'STRUMPACK_METIS_BICGSTAB': 0.338}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.183, 'STRUMPACK_METIS_PRECBICGSTAB': 3.955, 'EIGEN_LLT_CHOL': 12.21, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 7.196, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.612, 'STRUMPACK_METIS_AUTO': 4.0, 'EIGEN_LDLT_CHOL': 10.765, 'STRUMPACK_METIS_BICGSTAB': 0.278}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.13, 'STRUMPACK_METIS_PRECBICGSTAB': 1.445, 'EIGEN_LLT_CHOL': 1.002, 'EIGEN_CG': 0.032, 'STRUMPACK_SCOTCH_AUTO': 3.195, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.119, 'STRUMPACK_METIS_AUTO': 1.464, 'EIGEN_LDLT_CHOL': 0.962, 'STRUMPACK_METIS_BICGSTAB': 0.191}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.184, 'STRUMPACK_METIS_PRECBICGSTAB': 6.299, 'EIGEN_LLT_CHOL': 10.738, 'EIGEN_CG': 0.088, 'STRUMPACK_SCOTCH_AUTO': 7.025, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.301, 'STRUMPACK_METIS_AUTO': 6.201, 'EIGEN_LDLT_CHOL': 10.659, 'STRUMPACK_METIS_BICGSTAB': 0.256}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.23, 'STRUMPACK_METIS_PRECBICGSTAB': 3.623, 'EIGEN_LLT_CHOL': 91.713, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 10.756, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.864, 'STRUMPACK_METIS_AUTO': 3.647, 'EIGEN_LDLT_CHOL': 91.516, 'STRUMPACK_METIS_BICGSTAB': 0.332}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.188, 'STRUMPACK_METIS_PRECBICGSTAB': 11.518, 'EIGEN_LLT_CHOL': 10.797, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 8.643, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.723, 'STRUMPACK_METIS_AUTO': 11.658, 'EIGEN_LDLT_CHOL': 10.794, 'STRUMPACK_METIS_BICGSTAB': 0.279}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.199, 'STRUMPACK_METIS_PRECBICGSTAB': 4.512, 'EIGEN_LLT_CHOL': 38.039, 'EIGEN_CG': 0.088, 'STRUMPACK_SCOTCH_AUTO': 20.027, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.074, 'STRUMPACK_METIS_AUTO': 4.578, 'EIGEN_LDLT_CHOL': 37.941, 'STRUMPACK_METIS_BICGSTAB': 0.296}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.224, 'STRUMPACK_METIS_PRECBICGSTAB': 21.065, 'EIGEN_LLT_CHOL': 10.335, 'EIGEN_CG': 0.085, 'STRUMPACK_SCOTCH_AUTO': 12.95, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 10.517, 'STRUMPACK_METIS_AUTO': 21.087, 'EIGEN_LDLT_CHOL': 10.215, 'STRUMPACK_METIS_BICGSTAB': 0.315}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.224, 'STRUMPACK_METIS_PRECBICGSTAB': 6.378, 'EIGEN_LLT_CHOL': 173.581, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 20.843, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.592, 'STRUMPACK_METIS_AUTO': 6.499, 'EIGEN_LDLT_CHOL': 173.216, 'STRUMPACK_METIS_BICGSTAB': 0.319}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.196, 'STRUMPACK_METIS_PRECBICGSTAB': 2.047, 'EIGEN_LLT_CHOL': 10.454, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 7.549, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.059, 'STRUMPACK_METIS_AUTO': 2.588, 'EIGEN_LDLT_CHOL': 10.311, 'STRUMPACK_METIS_BICGSTAB': 0.292}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.217, 'STRUMPACK_METIS_PRECBICGSTAB': 10.346, 'EIGEN_LLT_CHOL': 174.091, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 9.084, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.237, 'STRUMPACK_METIS_AUTO': 10.247, 'EIGEN_LDLT_CHOL': 172.775, 'STRUMPACK_METIS_BICGSTAB': 0.302}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.255, 'STRUMPACK_METIS_PRECBICGSTAB': 35.295, 'EIGEN_LLT_CHOL': 173.182, 'EIGEN_CG': 0.085, 'STRUMPACK_SCOTCH_AUTO': 19.187, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 15.41, 'STRUMPACK_METIS_AUTO': 35.615, 'EIGEN_LDLT_CHOL': 171.453, 'STRUMPACK_METIS_BICGSTAB': 0.368}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.231, 'STRUMPACK_METIS_PRECBICGSTAB': 18.818, 'EIGEN_LLT_CHOL': 172.775, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 22.139, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 12.081, 'STRUMPACK_METIS_AUTO': 18.83, 'EIGEN_LDLT_CHOL': 172.679, 'STRUMPACK_METIS_BICGSTAB': 0.319}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.243, 'STRUMPACK_METIS_PRECBICGSTAB': 2.194, 'EIGEN_LLT_CHOL': 92.606, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.917, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.549, 'STRUMPACK_METIS_AUTO': 2.203, 'EIGEN_LDLT_CHOL': 92.602, 'STRUMPACK_METIS_BICGSTAB': 0.279}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.24, 'STRUMPACK_METIS_PRECBICGSTAB': 4.242, 'EIGEN_LLT_CHOL': 171.717, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 10.748, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.024, 'STRUMPACK_METIS_AUTO': 4.288, 'EIGEN_LDLT_CHOL': 171.723, 'STRUMPACK_METIS_BICGSTAB': 0.337}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.156, 'STRUMPACK_METIS_PRECBICGSTAB': 3.947, 'EIGEN_LLT_CHOL': 90.136, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.086, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.112, 'STRUMPACK_METIS_AUTO': 3.972, 'EIGEN_LDLT_CHOL': 88.708, 'STRUMPACK_METIS_BICGSTAB': 0.208}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.213, 'STRUMPACK_METIS_PRECBICGSTAB': 2.249, 'EIGEN_LLT_CHOL': 39.94, 'EIGEN_CG': 0.091, 'STRUMPACK_SCOTCH_AUTO': 8.59, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.469, 'STRUMPACK_METIS_AUTO': 2.891, 'EIGEN_LDLT_CHOL': 38.735, 'STRUMPACK_METIS_BICGSTAB': 0.325}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.161, 'STRUMPACK_METIS_PRECBICGSTAB': 2.258, 'EIGEN_LLT_CHOL': 38.509, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.244, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.295, 'STRUMPACK_METIS_AUTO': 2.278, 'EIGEN_LDLT_CHOL': 38.399, 'STRUMPACK_METIS_BICGSTAB': 0.216}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.154, 'STRUMPACK_METIS_PRECBICGSTAB': 1.374, 'EIGEN_LLT_CHOL': 10.668, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 6.206, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.21, 'STRUMPACK_METIS_AUTO': 1.749, 'EIGEN_LDLT_CHOL': 10.579, 'STRUMPACK_METIS_BICGSTAB': 0.232}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.161, 'STRUMPACK_METIS_PRECBICGSTAB': 21.04, 'EIGEN_LLT_CHOL': 89.045, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 3.803, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 2.304, 'STRUMPACK_METIS_AUTO': 21.082, 'EIGEN_LDLT_CHOL': 88.804, 'STRUMPACK_METIS_BICGSTAB': 0.213}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.152, 'STRUMPACK_METIS_PRECBICGSTAB': 11.161, 'EIGEN_LLT_CHOL': 90.764, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.305, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.575, 'STRUMPACK_METIS_AUTO': 11.356, 'EIGEN_LDLT_CHOL': 88.719, 'STRUMPACK_METIS_BICGSTAB': 0.197}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.154, 'STRUMPACK_METIS_PRECBICGSTAB': 6.358, 'EIGEN_LLT_CHOL': 88.402, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 1.522, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 0.952, 'STRUMPACK_METIS_AUTO': 6.34, 'EIGEN_LDLT_CHOL': 88.23, 'STRUMPACK_METIS_BICGSTAB': 0.203}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.152, 'STRUMPACK_METIS_PRECBICGSTAB': 10.649, 'EIGEN_LLT_CHOL': 1.012, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 4.497, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 2.164, 'STRUMPACK_METIS_AUTO': 10.642, 'EIGEN_LDLT_CHOL': 0.997, 'STRUMPACK_METIS_BICGSTAB': 0.184}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.116, 'STRUMPACK_METIS_PRECBICGSTAB': 5.695, 'EIGEN_LLT_CHOL': 1.117, 'EIGEN_CG': 0.032, 'STRUMPACK_SCOTCH_AUTO': 3.562, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.209, 'STRUMPACK_METIS_AUTO': 5.727, 'EIGEN_LDLT_CHOL': 1.101, 'STRUMPACK_METIS_BICGSTAB': 0.165}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.306, 'STRUMPACK_METIS_PRECBICGSTAB': 3.348, 'EIGEN_LLT_CHOL': 176.709, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 12.513, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 10.894, 'STRUMPACK_METIS_AUTO': 3.348, 'EIGEN_LDLT_CHOL': 175.587, 'STRUMPACK_METIS_BICGSTAB': 0.398}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.113, 'STRUMPACK_METIS_PRECBICGSTAB': 3.249, 'EIGEN_LLT_CHOL': 1.112, 'EIGEN_CG': 0.032, 'STRUMPACK_SCOTCH_AUTO': 2.74, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 0.901, 'STRUMPACK_METIS_AUTO': 3.279, 'EIGEN_LDLT_CHOL': 0.996, 'STRUMPACK_METIS_BICGSTAB': 0.163}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.175, 'STRUMPACK_METIS_PRECBICGSTAB': 1.648, 'EIGEN_LLT_CHOL': 37.542, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 3.084, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.529, 'STRUMPACK_METIS_AUTO': 2.188, 'EIGEN_LDLT_CHOL': 37.39, 'STRUMPACK_METIS_BICGSTAB': 0.221}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.12, 'STRUMPACK_METIS_PRECBICGSTAB': 2.08, 'EIGEN_LLT_CHOL': 1.161, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 3.521, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.124, 'STRUMPACK_METIS_AUTO': 2.101, 'EIGEN_LDLT_CHOL': 1.145, 'STRUMPACK_METIS_BICGSTAB': 0.173}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.228, 'STRUMPACK_METIS_PRECBICGSTAB': 6.359, 'EIGEN_LLT_CHOL': 175.108, 'EIGEN_CG': 0.087, 'STRUMPACK_SCOTCH_AUTO': 9.898, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.464, 'STRUMPACK_METIS_AUTO': 6.501, 'EIGEN_LDLT_CHOL': 175.141, 'STRUMPACK_METIS_BICGSTAB': 0.322}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.253, 'STRUMPACK_METIS_PRECBICGSTAB': 29.87, 'EIGEN_LLT_CHOL': 91.683, 'EIGEN_CG': 0.091, 'STRUMPACK_SCOTCH_AUTO': 18.817, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 17.456, 'STRUMPACK_METIS_AUTO': 29.887, 'EIGEN_LDLT_CHOL': 91.628, 'STRUMPACK_METIS_BICGSTAB': 0.362}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.221, 'STRUMPACK_METIS_PRECBICGSTAB': 15.734, 'EIGEN_LLT_CHOL': 92.463, 'EIGEN_CG': 0.091, 'STRUMPACK_SCOTCH_AUTO': 12.448, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 11.199, 'STRUMPACK_METIS_AUTO': 15.747, 'EIGEN_LDLT_CHOL': 92.4, 'STRUMPACK_METIS_BICGSTAB': 0.329}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.265, 'STRUMPACK_METIS_PRECBICGSTAB': 35.425, 'EIGEN_LLT_CHOL': 178.971, 'EIGEN_CG': 0.088, 'STRUMPACK_SCOTCH_AUTO': 19.734, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 15.098, 'STRUMPACK_METIS_AUTO': 35.592, 'EIGEN_LDLT_CHOL': 177.249, 'STRUMPACK_METIS_BICGSTAB': 0.366}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.234, 'STRUMPACK_METIS_PRECBICGSTAB': 18.87, 'EIGEN_LLT_CHOL': 176.605, 'EIGEN_CG': 0.087, 'STRUMPACK_SCOTCH_AUTO': 14.352, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 12.49, 'STRUMPACK_METIS_AUTO': 18.811, 'EIGEN_LDLT_CHOL': 174.187, 'STRUMPACK_METIS_BICGSTAB': 0.326}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.221, 'STRUMPACK_METIS_PRECBICGSTAB': 10.296, 'EIGEN_LLT_CHOL': 174.604, 'EIGEN_CG': 0.084, 'STRUMPACK_SCOTCH_AUTO': 9.122, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.268, 'STRUMPACK_METIS_AUTO': 10.26, 'EIGEN_LDLT_CHOL': 174.727, 'STRUMPACK_METIS_BICGSTAB': 0.306}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.229, 'STRUMPACK_METIS_PRECBICGSTAB': 1.941, 'EIGEN_LLT_CHOL': 89.482, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.642, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.336, 'STRUMPACK_METIS_AUTO': 2.691, 'EIGEN_LDLT_CHOL': 89.321, 'STRUMPACK_METIS_BICGSTAB': 0.246}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.172, 'STRUMPACK_METIS_PRECBICGSTAB': 2.736, 'EIGEN_LLT_CHOL': 89.452, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 15.62, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.249, 'STRUMPACK_METIS_AUTO': 2.791, 'EIGEN_LDLT_CHOL': 89.203, 'STRUMPACK_METIS_BICGSTAB': 0.23}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.212, 'STRUMPACK_METIS_PRECBICGSTAB': 3.112, 'EIGEN_LLT_CHOL': 39.275, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 8.461, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.412, 'STRUMPACK_METIS_AUTO': 3.098, 'EIGEN_LDLT_CHOL': 37.302, 'STRUMPACK_METIS_BICGSTAB': 0.319}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.23, 'STRUMPACK_METIS_PRECBICGSTAB': 1.843, 'EIGEN_LLT_CHOL': 40.711, 'EIGEN_CG': 0.036, 'STRUMPACK_SCOTCH_AUTO': 3.086, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.762, 'STRUMPACK_METIS_AUTO': 1.838, 'EIGEN_LDLT_CHOL': 39.163, 'STRUMPACK_METIS_BICGSTAB': 0.267}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.202, 'STRUMPACK_METIS_PRECBICGSTAB': 1.472, 'EIGEN_LLT_CHOL': 11.808, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 3.378, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.357, 'STRUMPACK_METIS_AUTO': 1.552, 'EIGEN_LDLT_CHOL': 10.476, 'STRUMPACK_METIS_BICGSTAB': 0.252}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.162, 'STRUMPACK_METIS_PRECBICGSTAB': 21.081, 'EIGEN_LLT_CHOL': 90.649, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 4.272, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 2.39, 'STRUMPACK_METIS_AUTO': 21.156, 'EIGEN_LDLT_CHOL': 89.679, 'STRUMPACK_METIS_BICGSTAB': 0.213}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.155, 'STRUMPACK_METIS_PRECBICGSTAB': 11.223, 'EIGEN_LLT_CHOL': 92.11, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.474, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.513, 'STRUMPACK_METIS_AUTO': 11.288, 'EIGEN_LDLT_CHOL': 92.23, 'STRUMPACK_METIS_BICGSTAB': 0.199}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.147, 'STRUMPACK_METIS_PRECBICGSTAB': 6.505, 'EIGEN_LLT_CHOL': 93.545, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.106, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.066, 'STRUMPACK_METIS_AUTO': 6.285, 'EIGEN_LDLT_CHOL': 91.804, 'STRUMPACK_METIS_BICGSTAB': 0.191}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.305, 'STRUMPACK_METIS_PRECBICGSTAB': 3.424, 'EIGEN_LLT_CHOL': 178.467, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 11.982, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 11.122, 'STRUMPACK_METIS_AUTO': 3.42, 'EIGEN_LDLT_CHOL': 178.409, 'STRUMPACK_METIS_BICGSTAB': 0.387}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.156, 'STRUMPACK_METIS_PRECBICGSTAB': 3.954, 'EIGEN_LLT_CHOL': 92.659, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.257, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.283, 'STRUMPACK_METIS_AUTO': 3.992, 'EIGEN_LDLT_CHOL': 92.351, 'STRUMPACK_METIS_BICGSTAB': 0.209}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.154, 'STRUMPACK_METIS_PRECBICGSTAB': 1.418, 'EIGEN_LLT_CHOL': 12.771, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.344, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.201, 'STRUMPACK_METIS_AUTO': 1.852, 'EIGEN_LDLT_CHOL': 10.587, 'STRUMPACK_METIS_BICGSTAB': 0.25}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.225, 'STRUMPACK_METIS_PRECBICGSTAB': 3.612, 'EIGEN_LLT_CHOL': 92.093, 'EIGEN_CG': 0.092, 'STRUMPACK_SCOTCH_AUTO': 10.851, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.609, 'STRUMPACK_METIS_AUTO': 3.633, 'EIGEN_LDLT_CHOL': 91.901, 'STRUMPACK_METIS_BICGSTAB': 0.331}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.145, 'STRUMPACK_METIS_PRECBICGSTAB': 3.196, 'EIGEN_LLT_CHOL': 38.825, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.124, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.129, 'STRUMPACK_METIS_AUTO': 3.278, 'EIGEN_LDLT_CHOL': 38.656, 'STRUMPACK_METIS_BICGSTAB': 0.191}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.186, 'STRUMPACK_METIS_PRECBICGSTAB': 2.668, 'EIGEN_LLT_CHOL': 10.393, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 7.918, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.696, 'STRUMPACK_METIS_AUTO': 2.748, 'EIGEN_LDLT_CHOL': 10.28, 'STRUMPACK_METIS_BICGSTAB': 0.284}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.243, 'STRUMPACK_METIS_PRECBICGSTAB': 4.232, 'EIGEN_LLT_CHOL': 177.511, 'EIGEN_CG': 0.087, 'STRUMPACK_SCOTCH_AUTO': 9.98, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.943, 'STRUMPACK_METIS_AUTO': 4.26, 'EIGEN_LDLT_CHOL': 175.678, 'STRUMPACK_METIS_BICGSTAB': 0.341}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.134, 'STRUMPACK_METIS_PRECBICGSTAB': 2.585, 'EIGEN_LLT_CHOL': 11.807, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 3.319, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 0.955, 'STRUMPACK_METIS_AUTO': 2.591, 'EIGEN_LDLT_CHOL': 10.566, 'STRUMPACK_METIS_BICGSTAB': 0.182}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.242, 'STRUMPACK_METIS_PRECBICGSTAB': 2.96, 'EIGEN_LLT_CHOL': 167.7, 'EIGEN_CG': 0.085, 'STRUMPACK_SCOTCH_AUTO': 10.202, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.069, 'STRUMPACK_METIS_AUTO': 3.651, 'EIGEN_LDLT_CHOL': 167.847, 'STRUMPACK_METIS_BICGSTAB': 0.352}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.125, 'STRUMPACK_METIS_PRECBICGSTAB': 3.995, 'EIGEN_LLT_CHOL': 10.34, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 3.439, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 0.826, 'STRUMPACK_METIS_AUTO': 4.051, 'EIGEN_LDLT_CHOL': 10.193, 'STRUMPACK_METIS_BICGSTAB': 0.168}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.144, 'STRUMPACK_METIS_PRECBICGSTAB': 9.152, 'EIGEN_LLT_CHOL': 39.426, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.76, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.554, 'STRUMPACK_METIS_AUTO': 9.216, 'EIGEN_LDLT_CHOL': 39.281, 'STRUMPACK_METIS_BICGSTAB': 0.191}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.16, 'STRUMPACK_METIS_PRECBICGSTAB': 13.638, 'EIGEN_LLT_CHOL': 10.556, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 3.833, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 2.053, 'STRUMPACK_METIS_AUTO': 13.28, 'EIGEN_LDLT_CHOL': 10.548, 'STRUMPACK_METIS_BICGSTAB': 0.192}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.129, 'STRUMPACK_METIS_PRECBICGSTAB': 7.149, 'EIGEN_LLT_CHOL': 10.715, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 3.404, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.228, 'STRUMPACK_METIS_AUTO': 7.188, 'EIGEN_LDLT_CHOL': 10.575, 'STRUMPACK_METIS_BICGSTAB': 0.175}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.212, 'STRUMPACK_METIS_PRECBICGSTAB': 13.016, 'EIGEN_LLT_CHOL': 37.51, 'EIGEN_CG': 0.087, 'STRUMPACK_SCOTCH_AUTO': 9.32, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.245, 'STRUMPACK_METIS_AUTO': 13.368, 'EIGEN_LDLT_CHOL': 37.296, 'STRUMPACK_METIS_BICGSTAB': 0.312}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.127, 'STRUMPACK_METIS_PRECBICGSTAB': 1.056, 'EIGEN_LLT_CHOL': 1.121, 'EIGEN_CG': 0.032, 'STRUMPACK_SCOTCH_AUTO': 3.2, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.095, 'STRUMPACK_METIS_AUTO': 1.161, 'EIGEN_LDLT_CHOL': 0.997, 'STRUMPACK_METIS_BICGSTAB': 0.189}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.206, 'STRUMPACK_METIS_PRECBICGSTAB': 8.627, 'EIGEN_LLT_CHOL': 92.035, 'EIGEN_CG': 0.092, 'STRUMPACK_SCOTCH_AUTO': 9.934, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.958, 'STRUMPACK_METIS_AUTO': 8.619, 'EIGEN_LDLT_CHOL': 91.862, 'STRUMPACK_METIS_BICGSTAB': 0.307}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.255, 'STRUMPACK_METIS_PRECBICGSTAB': 2.362, 'EIGEN_LLT_CHOL': 10.631, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 9.041, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.179, 'STRUMPACK_METIS_AUTO': 2.436, 'EIGEN_LDLT_CHOL': 10.662, 'STRUMPACK_METIS_BICGSTAB': 0.34}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.225, 'STRUMPACK_METIS_PRECBICGSTAB': 21.137, 'EIGEN_LLT_CHOL': 10.402, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 12.763, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 10.669, 'STRUMPACK_METIS_AUTO': 21.177, 'EIGEN_LDLT_CHOL': 10.333, 'STRUMPACK_METIS_BICGSTAB': 0.315}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.241, 'STRUMPACK_METIS_PRECBICGSTAB': 2.172, 'EIGEN_LLT_CHOL': 90.419, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.668, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.566, 'STRUMPACK_METIS_AUTO': 2.179, 'EIGEN_LDLT_CHOL': 89.63, 'STRUMPACK_METIS_BICGSTAB': 0.279}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.233, 'STRUMPACK_METIS_PRECBICGSTAB': 2.578, 'EIGEN_LLT_CHOL': 92.305, 'EIGEN_CG': 0.092, 'STRUMPACK_SCOTCH_AUTO': 11.19, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 10.01, 'STRUMPACK_METIS_AUTO': 3.229, 'EIGEN_LDLT_CHOL': 92.099, 'STRUMPACK_METIS_BICGSTAB': 0.335}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.144, 'STRUMPACK_METIS_PRECBICGSTAB': 3.185, 'EIGEN_LLT_CHOL': 38.486, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 2.302, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.247, 'STRUMPACK_METIS_AUTO': 3.253, 'EIGEN_LDLT_CHOL': 37.257, 'STRUMPACK_METIS_BICGSTAB': 0.19}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.143, 'STRUMPACK_METIS_PRECBICGSTAB': 9.015, 'EIGEN_LLT_CHOL': 39.376, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.863, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.508, 'STRUMPACK_METIS_AUTO': 9.12, 'EIGEN_LDLT_CHOL': 37.256, 'STRUMPACK_METIS_BICGSTAB': 0.186}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.205, 'STRUMPACK_METIS_PRECBICGSTAB': 3.112, 'EIGEN_LLT_CHOL': 38.814, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 9.385, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.666, 'STRUMPACK_METIS_AUTO': 3.138, 'EIGEN_LDLT_CHOL': 38.747, 'STRUMPACK_METIS_BICGSTAB': 0.305}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.147, 'STRUMPACK_METIS_PRECBICGSTAB': 5.217, 'EIGEN_LLT_CHOL': 38.33, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 2.107, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.214, 'STRUMPACK_METIS_AUTO': 5.163, 'EIGEN_LDLT_CHOL': 37.236, 'STRUMPACK_METIS_BICGSTAB': 0.192}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.201, 'STRUMPACK_METIS_PRECBICGSTAB': 1.496, 'EIGEN_LLT_CHOL': 12.298, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 3.027, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.398, 'STRUMPACK_METIS_AUTO': 1.559, 'EIGEN_LDLT_CHOL': 10.344, 'STRUMPACK_METIS_BICGSTAB': 0.252}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.152, 'STRUMPACK_METIS_PRECBICGSTAB': 16.789, 'EIGEN_LLT_CHOL': 39.483, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 4.052, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 2.382, 'STRUMPACK_METIS_AUTO': 16.9, 'EIGEN_LDLT_CHOL': 37.272, 'STRUMPACK_METIS_BICGSTAB': 0.204}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.138, 'STRUMPACK_METIS_PRECBICGSTAB': 1.801, 'EIGEN_LLT_CHOL': 10.68, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.362, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 0.978, 'STRUMPACK_METIS_AUTO': 1.8, 'EIGEN_LDLT_CHOL': 10.551, 'STRUMPACK_METIS_BICGSTAB': 0.191}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.198, 'STRUMPACK_METIS_PRECBICGSTAB': 1.616, 'EIGEN_LLT_CHOL': 39.581, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.224, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.23, 'STRUMPACK_METIS_AUTO': 1.914, 'EIGEN_LDLT_CHOL': 38.341, 'STRUMPACK_METIS_BICGSTAB': 0.222}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.225, 'STRUMPACK_METIS_PRECBICGSTAB': 15.657, 'EIGEN_LLT_CHOL': 88.909, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 19.75, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 11.06, 'STRUMPACK_METIS_AUTO': 15.706, 'EIGEN_LDLT_CHOL': 88.689, 'STRUMPACK_METIS_BICGSTAB': 0.326}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.261, 'STRUMPACK_METIS_PRECBICGSTAB': 29.572, 'EIGEN_LLT_CHOL': 89.648, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 18.482, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 17.707, 'STRUMPACK_METIS_AUTO': 29.669, 'EIGEN_LDLT_CHOL': 88.777, 'STRUMPACK_METIS_BICGSTAB': 0.361}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.194, 'STRUMPACK_METIS_PRECBICGSTAB': 4.539, 'EIGEN_LLT_CHOL': 38.655, 'EIGEN_CG': 0.089, 'STRUMPACK_SCOTCH_AUTO': 7.538, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.907, 'STRUMPACK_METIS_AUTO': 4.534, 'EIGEN_LDLT_CHOL': 38.433, 'STRUMPACK_METIS_BICGSTAB': 0.291}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.255, 'STRUMPACK_METIS_PRECBICGSTAB': 2.315, 'EIGEN_LLT_CHOL': 10.272, 'EIGEN_CG': 0.085, 'STRUMPACK_SCOTCH_AUTO': 9.238, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.204, 'STRUMPACK_METIS_AUTO': 2.359, 'EIGEN_LDLT_CHOL': 10.251, 'STRUMPACK_METIS_BICGSTAB': 0.332}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.212, 'STRUMPACK_METIS_PRECBICGSTAB': 8.535, 'EIGEN_LLT_CHOL': 92.073, 'EIGEN_CG': 0.092, 'STRUMPACK_SCOTCH_AUTO': 10.192, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.928, 'STRUMPACK_METIS_AUTO': 8.714, 'EIGEN_LDLT_CHOL': 91.913, 'STRUMPACK_METIS_BICGSTAB': 0.317}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.192, 'STRUMPACK_METIS_PRECBICGSTAB': 7.163, 'EIGEN_LLT_CHOL': 37.877, 'EIGEN_CG': 0.091, 'STRUMPACK_SCOTCH_AUTO': 7.639, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.752, 'STRUMPACK_METIS_AUTO': 7.368, 'EIGEN_LDLT_CHOL': 37.706, 'STRUMPACK_METIS_BICGSTAB': 0.281}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.21, 'STRUMPACK_METIS_PRECBICGSTAB': 5.363, 'EIGEN_LLT_CHOL': 92.384, 'EIGEN_CG': 0.093, 'STRUMPACK_SCOTCH_AUTO': 10.297, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.242, 'STRUMPACK_METIS_AUTO': 5.426, 'EIGEN_LDLT_CHOL': 92.081, 'STRUMPACK_METIS_BICGSTAB': 0.31}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.236, 'STRUMPACK_METIS_PRECBICGSTAB': 24.132, 'EIGEN_LLT_CHOL': 37.279, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 14.117, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 12.94, 'STRUMPACK_METIS_AUTO': 24.146, 'EIGEN_LDLT_CHOL': 37.265, 'STRUMPACK_METIS_BICGSTAB': 0.332}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.208, 'STRUMPACK_METIS_PRECBICGSTAB': 13.0, 'EIGEN_LLT_CHOL': 38.869, 'EIGEN_CG': 0.087, 'STRUMPACK_SCOTCH_AUTO': 9.016, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.064, 'STRUMPACK_METIS_AUTO': 13.105, 'EIGEN_LDLT_CHOL': 37.444, 'STRUMPACK_METIS_BICGSTAB': 0.308}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.189, 'STRUMPACK_METIS_PRECBICGSTAB': 11.536, 'EIGEN_LLT_CHOL': 12.296, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 8.899, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.744, 'STRUMPACK_METIS_AUTO': 11.493, 'EIGEN_LDLT_CHOL': 10.222, 'STRUMPACK_METIS_BICGSTAB': 0.28}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.166, 'STRUMPACK_METIS_PRECBICGSTAB': 2.214, 'EIGEN_LLT_CHOL': 37.327, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 2.194, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.257, 'STRUMPACK_METIS_AUTO': 2.254, 'EIGEN_LDLT_CHOL': 37.123, 'STRUMPACK_METIS_BICGSTAB': 0.219}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.217, 'STRUMPACK_METIS_PRECBICGSTAB': 2.25, 'EIGEN_LLT_CHOL': 37.384, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 8.608, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 7.791, 'STRUMPACK_METIS_AUTO': 2.881, 'EIGEN_LDLT_CHOL': 37.341, 'STRUMPACK_METIS_BICGSTAB': 0.316}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.169, 'STRUMPACK_METIS_PRECBICGSTAB': 2.709, 'EIGEN_LLT_CHOL': 89.645, 'EIGEN_CG': 0.034, 'STRUMPACK_SCOTCH_AUTO': 2.203, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.26, 'STRUMPACK_METIS_AUTO': 2.737, 'EIGEN_LDLT_CHOL': 89.453, 'STRUMPACK_METIS_BICGSTAB': 0.229}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.196, 'STRUMPACK_METIS_PRECBICGSTAB': 7.256, 'EIGEN_LLT_CHOL': 38.626, 'EIGEN_CG': 0.089, 'STRUMPACK_SCOTCH_AUTO': 9.048, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.642, 'STRUMPACK_METIS_AUTO': 7.452, 'EIGEN_LDLT_CHOL': 38.443, 'STRUMPACK_METIS_BICGSTAB': 0.282}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.176, 'STRUMPACK_METIS_PRECBICGSTAB': 1.932, 'EIGEN_LLT_CHOL': 92.358, 'EIGEN_CG': 0.036, 'STRUMPACK_SCOTCH_AUTO': 2.142, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.182, 'STRUMPACK_METIS_AUTO': 2.598, 'EIGEN_LDLT_CHOL': 92.121, 'STRUMPACK_METIS_BICGSTAB': 0.235}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.211, 'STRUMPACK_METIS_PRECBICGSTAB': 5.549, 'EIGEN_LLT_CHOL': 91.274, 'EIGEN_CG': 0.089, 'STRUMPACK_SCOTCH_AUTO': 10.383, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.514, 'STRUMPACK_METIS_AUTO': 5.427, 'EIGEN_LDLT_CHOL': 89.457, 'STRUMPACK_METIS_BICGSTAB': 0.315}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.161, 'STRUMPACK_METIS_PRECBICGSTAB': 13.28, 'EIGEN_LLT_CHOL': 10.731, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 4.034, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 2.097, 'STRUMPACK_METIS_AUTO': 13.295, 'EIGEN_LDLT_CHOL': 10.574, 'STRUMPACK_METIS_BICGSTAB': 0.192}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-1': {'STRUMPACK_SCOTCH_BICGSTAB': 0.164, 'STRUMPACK_METIS_PRECBICGSTAB': 16.91, 'EIGEN_LLT_CHOL': 39.507, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 4.097, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 2.33, 'STRUMPACK_METIS_AUTO': 16.984, 'EIGEN_LDLT_CHOL': 38.013, 'STRUMPACK_METIS_BICGSTAB': 0.204}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-2': {'STRUMPACK_SCOTCH_BICGSTAB': 0.131, 'STRUMPACK_METIS_PRECBICGSTAB': 7.196, 'EIGEN_LLT_CHOL': 10.903, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.668, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.295, 'STRUMPACK_METIS_AUTO': 17.028, 'EIGEN_LDLT_CHOL': 10.589, 'STRUMPACK_METIS_BICGSTAB': 0.181}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.297, 'STRUMPACK_METIS_PRECBICGSTAB': 2.952, 'EIGEN_LLT_CHOL': 90.403, 'EIGEN_CG': 0.088, 'STRUMPACK_SCOTCH_AUTO': 13.288, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 11.965, 'STRUMPACK_METIS_AUTO': 2.926, 'EIGEN_LDLT_CHOL': 88.515, 'STRUMPACK_METIS_BICGSTAB': 0.391}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.132, 'STRUMPACK_METIS_PRECBICGSTAB': 2.557, 'EIGEN_LLT_CHOL': 10.749, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.724, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 0.961, 'STRUMPACK_METIS_AUTO': 2.553, 'EIGEN_LDLT_CHOL': 10.586, 'STRUMPACK_METIS_BICGSTAB': 0.181}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.19, 'STRUMPACK_METIS_PRECBICGSTAB': 2.706, 'EIGEN_LLT_CHOL': 11.991, 'EIGEN_CG': 0.085, 'STRUMPACK_SCOTCH_AUTO': 7.529, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.71, 'STRUMPACK_METIS_AUTO': 2.812, 'EIGEN_LDLT_CHOL': 10.24, 'STRUMPACK_METIS_BICGSTAB': 0.286}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.28, 'STRUMPACK_METIS_PRECBICGSTAB': 2.601, 'EIGEN_LLT_CHOL': 37.425, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 10.201, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.244, 'STRUMPACK_METIS_AUTO': 2.714, 'EIGEN_LDLT_CHOL': 37.264, 'STRUMPACK_METIS_BICGSTAB': 0.361}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.248, 'STRUMPACK_METIS_PRECBICGSTAB': 2.544, 'EIGEN_LLT_CHOL': 92.703, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 11.652, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 10.217, 'STRUMPACK_METIS_AUTO': 3.155, 'EIGEN_LDLT_CHOL': 91.649, 'STRUMPACK_METIS_BICGSTAB': 0.35}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.178, 'STRUMPACK_METIS_PRECBICGSTAB': 1.239, 'EIGEN_LLT_CHOL': 1.138, 'EIGEN_CG': 0.032, 'STRUMPACK_SCOTCH_AUTO': 5.048, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.398, 'STRUMPACK_METIS_AUTO': 1.265, 'EIGEN_LDLT_CHOL': 1.121, 'STRUMPACK_METIS_BICGSTAB': 0.244}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.298, 'STRUMPACK_METIS_PRECBICGSTAB': 2.883, 'EIGEN_LLT_CHOL': 92.287, 'EIGEN_CG': 0.09, 'STRUMPACK_SCOTCH_AUTO': 14.098, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 11.828, 'STRUMPACK_METIS_AUTO': 2.854, 'EIGEN_LDLT_CHOL': 91.701, 'STRUMPACK_METIS_BICGSTAB': 0.386}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': {'STRUMPACK_SCOTCH_BICGSTAB': 0.18, 'STRUMPACK_METIS_PRECBICGSTAB': 3.895, 'EIGEN_LLT_CHOL': 10.469, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 7.002, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.212, 'STRUMPACK_METIS_AUTO': 3.997, 'EIGEN_LDLT_CHOL': 10.381, 'STRUMPACK_METIS_BICGSTAB': 0.268}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-16': {'STRUMPACK_SCOTCH_BICGSTAB': 0.146, 'STRUMPACK_METIS_PRECBICGSTAB': 1.817, 'EIGEN_LLT_CHOL': 10.297, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 2.922, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.094, 'STRUMPACK_METIS_AUTO': 1.831, 'EIGEN_LDLT_CHOL': 10.184, 'STRUMPACK_METIS_BICGSTAB': 0.202}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.178, 'STRUMPACK_METIS_PRECBICGSTAB': 6.294, 'EIGEN_LLT_CHOL': 10.704, 'EIGEN_CG': 0.088, 'STRUMPACK_SCOTCH_AUTO': 9.215, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 5.946, 'STRUMPACK_METIS_AUTO': 6.296, 'EIGEN_LDLT_CHOL': 10.593, 'STRUMPACK_METIS_BICGSTAB': 0.255}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-64': {'STRUMPACK_SCOTCH_BICGSTAB': 0.283, 'STRUMPACK_METIS_PRECBICGSTAB': 2.53, 'EIGEN_LLT_CHOL': 37.251, 'EIGEN_CG': 0.086, 'STRUMPACK_SCOTCH_AUTO': 10.385, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.159, 'STRUMPACK_METIS_AUTO': 2.682, 'EIGEN_LDLT_CHOL': 37.094, 'STRUMPACK_METIS_BICGSTAB': 0.367}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.196, 'STRUMPACK_METIS_PRECBICGSTAB': 2.071, 'EIGEN_LLT_CHOL': 10.665, 'EIGEN_CG': 0.089, 'STRUMPACK_SCOTCH_AUTO': 7.82, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 6.895, 'STRUMPACK_METIS_AUTO': 2.562, 'EIGEN_LDLT_CHOL': 10.601, 'STRUMPACK_METIS_BICGSTAB': 0.29}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.147, 'STRUMPACK_METIS_PRECBICGSTAB': 5.142, 'EIGEN_LLT_CHOL': 39.295, 'EIGEN_CG': 0.035, 'STRUMPACK_SCOTCH_AUTO': 2.22, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 1.339, 'STRUMPACK_METIS_AUTO': 5.3, 'EIGEN_LDLT_CHOL': 39.308, 'STRUMPACK_METIS_BICGSTAB': 0.189}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32': {'STRUMPACK_SCOTCH_BICGSTAB': 0.241, 'STRUMPACK_METIS_PRECBICGSTAB': 2.947, 'EIGEN_LLT_CHOL': 175.574, 'EIGEN_CG': 0.087, 'STRUMPACK_SCOTCH_AUTO': 10.526, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.153, 'STRUMPACK_METIS_AUTO': 3.209, 'EIGEN_LDLT_CHOL': 175.244, 'STRUMPACK_METIS_BICGSTAB': 0.336}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-4': {'STRUMPACK_SCOTCH_BICGSTAB': 0.125, 'STRUMPACK_METIS_PRECBICGSTAB': 4.05, 'EIGEN_LLT_CHOL': 11.315, 'EIGEN_CG': 0.033, 'STRUMPACK_SCOTCH_AUTO': 3.337, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 0.976, 'STRUMPACK_METIS_AUTO': 4.12, 'EIGEN_LDLT_CHOL': 10.26, 'STRUMPACK_METIS_BICGSTAB': 0.172}}

</details>

The goal is to now combine data into hierarchical sets for refined parameters, OpenMP threads, algorithms, and finally timings. We can do this by creating nested directionaries for each of the sets, allowing subselection of the specified data required for later plotting and analysis. Alternatively, we can create an in-memory SQLite3 database to more easily allowing subselection.


```python
#Dictionary method
param_omp_val_t = { }

keyVal = time_data_hwl.iteritems().next()[0].split('-') #Param set name, and subsequent threads used
for k,v in time_data_hwl.iteritems():
    k_par_t = k.split('-')
d = { k_par_t[0]: {int(k_par_t[1]) : v} }

param_omp_val_t.update(d)
```


```python
#DB method
import sqlite3
conn.close()
conn = sqlite3.connect(':memory:')
# Get a cursor object
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE hwl_omp(id INTEGER PRIMARY KEY, ref_param TEXT,
                       omp_threads INTEGER, algo TEXT, time REAL)
''')
for k,v in time_data_hwl.iteritems():
    k_par_t = k.split('-')
    for kk,vv in v.iteritems():
        cursor.execute('''INSERT INTO hwl_omp(ref_param, omp_threads, algo, time)
                  VALUES(?,?,?,?)''', ("-".join(k_par_t[:-1]), k_par_t[-1], kk, vv))
conn.commit()
```

We may not select from the database as follows:


```python
par='strum_5k_omp1_paramslevmar.parameter_flags=Rxy'
cursor.execute('''SELECT ref_param, omp_threads, algo, time FROM hwl_omp WHERE ref_param=? AND algo=?''',(par,'STRUMPACK_SCOTCH_BICGSTAB'))
all_rows = cursor.fetchall()
for row in all_rows:
    # row[0] returns the first column in the query (name), row[1] returns email column.
    print('{0} , {1}, {2} , {3}'.format(row[0], row[1], row[2], row[3]))
```

    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 64, STRUMPACK_SCOTCH_BICGSTAB , 0.229
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 32, STRUMPACK_SCOTCH_BICGSTAB , 0.175
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 8, STRUMPACK_SCOTCH_BICGSTAB , 0.144
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 2, STRUMPACK_SCOTCH_BICGSTAB , 0.143
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 4, STRUMPACK_SCOTCH_BICGSTAB , 0.147
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 1, STRUMPACK_SCOTCH_BICGSTAB , 0.152
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 16, STRUMPACK_SCOTCH_BICGSTAB , 0.166


We begin by making a list of the unique refinement parameter names listed in the DB, and also the algorithms used.


```python
cursor.execute('''SELECT DISTINCT ref_param FROM hwl_omp ''')
params = [s[0] for s in cursor.fetchall()]
cursor.execute('''SELECT DISTINCT algo FROM hwl_omp ''')
algos = [s[0] for s in cursor.fetchall()]
```

As we are visualising the performance changes for a variety of different numbers of OpenMP threads, $T$, it is worth generating a series of plots for the threads within the range $\{T:1≤T≤64,~\log_{2}{T}\in ℤ\_{\ge 0}\}$ for Haswell, and $\{T:1≤T≤256,~\log\_{2}{T}\in ℤ\_{\ge 0}\}$ for KNL (even if the upper limit allows 272). We now map the algorithm names to specific colours to allow us to more easily track them in subsequent plots.


```python
xlabels = algos
keys = algos
colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
key_col = {k:v for (k,v) in zip(keys,colours)}
```

By looking over the refinment parameters and plotting each subsequent algorithm for threads Vs. time on the same plots, we can determine the scalability and overall performance of each algorithm for the given data set being examined.


```python
marker_list=[l for l in matplotlib.markers.MarkerStyle.markers.keys() if not isinstance(l, (int))]
```


```python
# define the figure size and grid layout properties
figsize = (18, 15)
cols = int(np.ceil(np.sqrt(len(params))))
gs = gridspec.GridSpec(cols, cols)
gs.update(hspace=0.4)
fig_hwl_omp = plt.figure(num=1, figsize=figsize)
fig_hwl_omp.suptitle('5k images Haswell ', size=20)
ax = []
cbars= []
import numpy as np
i=0

from itertools import cycle
lineStyles = ["-","--","-.",":"]

for p in params:
    row = (i // cols)
    col = i % cols
    ax.append(fig_hwl_omp.add_subplot(gs[row, col]))
    i+=1
    lineStyles_cycle = cycle(lineStyles)
    markerStyles_cycle = cycle(marker_list)
    next(markerStyles_cycle)
    for al_idx,al in enumerate(algos):

        cursor.execute('''SELECT omp_threads, time FROM hwl_omp WHERE ref_param=? AND algo=?''',(p,al))
        np_omp_t = np.array(cursor.fetchall(),dtype=[('OMP', '<i8'), ('time', '<f8')])
        np_omp_t.sort()
        dat = zip(*np_omp_t)

        cset = ax[-1].plot(dat[0], dat[1], label=al, linewidth=2, 
                           linestyle=next(lineStyles_cycle), marker=next(markerStyles_cycle) )
        
    ax[-1].set_title(p.replace('levmar.parameter_flags=','').replace('strum_5k_omp1_params',''), size=16 )
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=10)


#gs.tight_layout(fig_hwl_omp,)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=12)
fig_hwl_omp.text(0.04, 0.5, 'time [s]', va='center', rotation='vertical')
fig_hwl_omp.text(0.5, 0.04, 'OMP_NUM_THREADS', va='center', rotation='horizontal')

plt.savefig('omp_5kframes_allsolvers_hwl_ompall.pdf',)
```


|   |
|---|
| ![png](/img/strumpacksolvermpi_dist_cori_files/strumpacksolvermpi_dist_cori_27_0.png) |


# OpenMP solver performance on KNL


```python
SBATCH_SCRIPT_KNL=\
"""#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -o <DATNAME>_knl_out_OMP.log
#SBATCH -e <DATNAME>_knl_out_OMP.err
#SBATCH -J CCTBX_STRUMPACK
#SBATCH --mail-user=loriordan@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00

#OpenMP settings:
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

#run the application:
cd /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST
source $PWD/miniconda/bin/activate myEnv
source $PWD/build/setpaths.sh
module load darshan
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
mkdir -p OMP_KNL
cd OMP_KNL

mkdir <DATNAME>_knl_out_OMP
cd <DATNAME>_knl_out_OMP

#Use FPE mask to avoid floating point exceptions. Need to further investigate reasons for these
for n in 256;do
    t_c=$((256/${n}))
    echo "OMP_NUM_THREADS=${n} BOOST_ADAPTBX_FPE_DEFAULT=1 time srun -n 1 -c 272 --cpu_bind=cores\
        libtbx.python /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/OMP_SOLVER.py <DATA_A> <DATA_B>"
    OMP_NUM_THREADS=${n} BOOST_ADAPTBX_FPE_DEFAULT=1 srun -n 1 -c 272 --cpu_bind=cores libtbx.python \
      /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/OMP_SOLVER.py <DATA_A> <DATA_B>\
      1> <DATNAME>_knl_out_OMP${n}.log 2> <DATNAME>_knl_out_OMP${n}.err
done
cat <DATNAME>_knl_out_OMP${n}.log >> <DATNAME>_knl_out_OMP.log
cat <DATNAME>_knl_out_OMP${n}.err >> <DATNAME>_knl_out_OMP.err
"""
```


```python
str_out={}
import os
threads_list = [1,2,4,8,16,32,64,128,256]
sub_scripts_knl = []
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        
        A_path = A_LIST[imgs_idx]; b_path = B_LIST[imgs_idx]
        dat_name = A_path.split('/')[-1][2:-4]

        print "Data Set Name:=%s"%(dat_name)

        #Ensure the A and b data are matched correctly
        assert(os.path.dirname(A_path) == os.path.dirname(b_path))
        SUBMIT = SBATCH_SCRIPT_KNL.replace('<DATA_A>',A_path)\
           .replace('<DATA_B>', b_path)\
           .replace('<DATNAME>', dat_name)
        SBATCH_SCRIPT_FILE = open("SBATCH_SCRIPT_knl_%s.sh"%(dat_name), "w")
        sub_scripts_knl.append("SBATCH_SCRIPT_knl_%s.sh"%(dat_name))
        SBATCH_SCRIPT_FILE.write(SUBMIT)
        SBATCH_SCRIPT_FILE.close()
        var = !sbatch {sub_scripts_knl[-1]}
        print var
```

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-
    ['Submitted batch job 11708194']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    ['Submitted batch job 11708195']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    ['Submitted batch job 11708198']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11708199']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-
    ['Submitted batch job 11708200']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-
    ['Submitted batch job 11708202']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    ['Submitted batch job 11708203']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Eta-
    ['Submitted batch job 11708206']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11708208']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11708209']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    ['Submitted batch job 11708211']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-
    ['Submitted batch job 11708213']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-
    ['Submitted batch job 11708215']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Deff-
    ['Submitted batch job 11708216']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11708217']
    Data Set Size:=32k
    Skipping 32k



```python
sub_scripts_knl
```




    ['SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Eta-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Deff-.sh',
     'SBATCH_SCRIPT_knl_strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-.sh']




```python
dat_list_knl_omp = {}
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        
        dat = A_LIST[imgs_idx].split('/')[-1][2:-4]
        var = !find ./OMP_KNL -iname '{dat}_knl_out_OMP256.log' #!ls OMP_KNL/{dat}_knl_out_OMP/ | grep 'log'
        dat_list_knl_omp.update({dat : var})
        print dat
print dat_list_knl_omp
```
<details>
<summary><p><b>---CLICK FOR OUTPUT---</b></p></summary>

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    strum_5k_omp1_paramslevmar.parameter_flags=Eta-
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-
    strum_5k_omp1_paramslevmar.parameter_flags=Deff-
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    Data Set Size:=32k
    Skipping 32k
    {'strum_5k_omp1_paramslevmar.parameter_flags=Deff-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_knl_out_OMP256.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-': ['./OMP_KNL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_knl_out_OMP/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_knl_out_OMP256.log']}

</details>


Create dictionary of timing values organised by keys of data set name, followed by algorithm name mapping to the timings


```python
time_data = {}
for key, value in dat_list_knl_omp.iteritems():
    with open(str(value[0])) as f:
        lines = f.read().splitlines()
        zlist = {}
        [zlist.update({l.split()[-7][:-1]:float(l.split()[-3][:-2])})\
          for l in lines if "#calls" in l]
        time_data.update({key : zlist})
print time_data
```

    {'strum_5k_omp1_paramslevmar.parameter_flags=Deff-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.679, 'STRUMPACK_METIS_PRECBICGSTAB': 5.431, 'EIGEN_LLT_CHOL': 54.04, 'EIGEN_CG': 0.101, 'STRUMPACK_SCOTCH_AUTO': 12.456, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.861, 'STRUMPACK_METIS_AUTO': 5.288, 'EIGEN_LDLT_CHOL': 53.954, 'STRUMPACK_METIS_BICGSTAB': 0.921}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-': {'STRUMPACK_SCOTCH_BICGSTAB': 1.054, 'STRUMPACK_METIS_PRECBICGSTAB': 10.319, 'EIGEN_LLT_CHOL': 417.372, 'EIGEN_CG': 0.331, 'STRUMPACK_SCOTCH_AUTO': 61.092, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 65.317, 'STRUMPACK_METIS_AUTO': 9.812, 'EIGEN_LDLT_CHOL': 417.778, 'STRUMPACK_METIS_BICGSTAB': 1.578}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 1.114, 'STRUMPACK_METIS_PRECBICGSTAB': 11.211, 'EIGEN_LLT_CHOL': 810.343, 'EIGEN_CG': 0.327, 'STRUMPACK_SCOTCH_AUTO': 60.934, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 58.865, 'STRUMPACK_METIS_AUTO': 11.428, 'EIGEN_LDLT_CHOL': 812.015, 'STRUMPACK_METIS_BICGSTAB': 1.572}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.968, 'STRUMPACK_METIS_PRECBICGSTAB': 9.781, 'EIGEN_LLT_CHOL': 178.381, 'EIGEN_CG': 0.331, 'STRUMPACK_SCOTCH_AUTO': 52.31, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 50.504, 'STRUMPACK_METIS_AUTO': 9.844, 'EIGEN_LDLT_CHOL': 178.571, 'STRUMPACK_METIS_BICGSTAB': 1.48}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.855, 'STRUMPACK_METIS_PRECBICGSTAB': 9.684, 'EIGEN_LLT_CHOL': 53.945, 'EIGEN_CG': 0.303, 'STRUMPACK_SCOTCH_AUTO': 46.912, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 41.549, 'STRUMPACK_METIS_AUTO': 9.912, 'EIGEN_LDLT_CHOL': 53.919, 'STRUMPACK_METIS_BICGSTAB': 1.361}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.819, 'STRUMPACK_METIS_PRECBICGSTAB': 7.24, 'EIGEN_LLT_CHOL': 417.288, 'EIGEN_CG': 0.111, 'STRUMPACK_SCOTCH_AUTO': 11.986, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 10.533, 'STRUMPACK_METIS_AUTO': 7.282, 'EIGEN_LDLT_CHOL': 417.468, 'STRUMPACK_METIS_BICGSTAB': 1.052}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 1.038, 'STRUMPACK_METIS_PRECBICGSTAB': 10.36, 'EIGEN_LLT_CHOL': 417.854, 'EIGEN_CG': 0.332, 'STRUMPACK_SCOTCH_AUTO': 68.231, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 65.317, 'STRUMPACK_METIS_AUTO': 10.417, 'EIGEN_LDLT_CHOL': 418.163, 'STRUMPACK_METIS_BICGSTAB': 1.541}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.692, 'STRUMPACK_METIS_PRECBICGSTAB': 5.489, 'EIGEN_LLT_CHOL': 54.031, 'EIGEN_CG': 0.101, 'STRUMPACK_SCOTCH_AUTO': 11.977, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 8.83, 'STRUMPACK_METIS_AUTO': 5.283, 'EIGEN_LDLT_CHOL': 53.959, 'STRUMPACK_METIS_BICGSTAB': 0.92}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-': {'STRUMPACK_SCOTCH_BICGSTAB': 1.089, 'STRUMPACK_METIS_PRECBICGSTAB': 10.986, 'EIGEN_LLT_CHOL': 811.03, 'EIGEN_CG': 0.327, 'STRUMPACK_SCOTCH_AUTO': 60.603, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 59.198, 'STRUMPACK_METIS_AUTO': 11.19, 'EIGEN_LDLT_CHOL': 811.903, 'STRUMPACK_METIS_BICGSTAB': 1.61}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.765, 'STRUMPACK_METIS_PRECBICGSTAB': 6.478, 'EIGEN_LLT_CHOL': 178.337, 'EIGEN_CG': 0.115, 'STRUMPACK_SCOTCH_AUTO': 13.928, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 11.754, 'STRUMPACK_METIS_AUTO': 6.223, 'EIGEN_LDLT_CHOL': 178.536, 'STRUMPACK_METIS_BICGSTAB': 0.994}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.611, 'STRUMPACK_METIS_PRECBICGSTAB': 4.491, 'EIGEN_LLT_CHOL': 7.389, 'EIGEN_CG': 0.109, 'STRUMPACK_SCOTCH_AUTO': 13.581, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 9.266, 'STRUMPACK_METIS_AUTO': 4.722, 'EIGEN_LDLT_CHOL': 7.316, 'STRUMPACK_METIS_BICGSTAB': 0.884}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.959, 'STRUMPACK_METIS_PRECBICGSTAB': 9.798, 'EIGEN_LLT_CHOL': 178.004, 'EIGEN_CG': 0.329, 'STRUMPACK_SCOTCH_AUTO': 53.007, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 50.558, 'STRUMPACK_METIS_AUTO': 10.168, 'EIGEN_LDLT_CHOL': 178.286, 'STRUMPACK_METIS_BICGSTAB': 1.483}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.815, 'STRUMPACK_METIS_PRECBICGSTAB': 7.52, 'EIGEN_LLT_CHOL': 417.622, 'EIGEN_CG': 0.115, 'STRUMPACK_SCOTCH_AUTO': 11.743, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 10.571, 'STRUMPACK_METIS_AUTO': 7.217, 'EIGEN_LDLT_CHOL': 418.061, 'STRUMPACK_METIS_BICGSTAB': 1.055}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.769, 'STRUMPACK_METIS_PRECBICGSTAB': 6.203, 'EIGEN_LLT_CHOL': 178.116, 'EIGEN_CG': 0.113, 'STRUMPACK_SCOTCH_AUTO': 14.376, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 11.694, 'STRUMPACK_METIS_AUTO': 6.518, 'EIGEN_LDLT_CHOL': 178.359, 'STRUMPACK_METIS_BICGSTAB': 0.996}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-': {'STRUMPACK_SCOTCH_BICGSTAB': 0.88, 'STRUMPACK_METIS_PRECBICGSTAB': 9.936, 'EIGEN_LLT_CHOL': 54.119, 'EIGEN_CG': 0.307, 'STRUMPACK_SCOTCH_AUTO': 46.507, 'STRUMPACK_SCOTCH_PRECBICGSTAB': 44.809, 'STRUMPACK_METIS_AUTO': 9.659, 'EIGEN_LDLT_CHOL': 54.078, 'STRUMPACK_METIS_BICGSTAB': 1.366}}



```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'serif',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (16,10)
matplotlib.rcParams['figure.dpi']= 150
import matplotlib.gridspec as gridspec
```


```python
# define the figure size and grid layout properties
figsize = (15, 12)
cols = 4
gs = gridspec.GridSpec(cols, cols)
gs.update(hspace=0.4)
fig1 = plt.figure(num=1, figsize=figsize)
fig1.text(0.04, 0.5, 'time [s]', va='center', rotation='vertical')

fig1.suptitle('5k images KNL OMP 256', size=20)
ax = []
cbars= []

xlabels = time_data.itervalues().next().keys()
keys = time_data.itervalues().next().keys()
colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
key_col = {k:v for (k,v) in zip(keys,colours)}

ax = []
i=0
matplotlib.rcParams.update({'font.size': 14})
for k,d in time_data.iteritems():
    
    row = (i // cols)
    col = i % cols
    ax.append(fig1.add_subplot(gs[row, col]))
    
    cset = ax[-1].bar(range(len(d)), d.values(), color=colours, align="center")
    ax[-1].set_title(k.replace('levmar.parameter_flags=','').replace('strum_5k_omp1_params',''), size=16 )
    ax[-1].set_xticklabels(['']*len(d.keys()))
    plt.yscale('log',basey=10)
    i+=1
    
# Ensure colours match the labels in the legend
leg_patch = []
for kk,vv in d.iteritems():
    leg_patch.append(mpatches.Patch(color=key_col[kk], label=kk))
plt.legend(handles=leg_patch,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=14)
plt.savefig('omp_1kframes_allsolvers_knl_omp256_grid.pdf', pad_inches=10)
```


|   |
|---|
| ![png](/img/strumpacksolvermpi_dist_cori_files/strumpacksolvermpi_dist_cori_36_0.png) |



```python
# define the figure size and grid layout properties
import matplotlib.patches as mpatches
matplotlib.rcParams.update({'font.size': 22})
figsize = (20, 16)
cols = 4
fig3 = plt.figure(num=3, figsize=figsize)
ax3 = plt.axes()
ii = []
uu=[]
xlabels = time_data.itervalues().next().keys()
keys = time_data.itervalues().next().keys()
colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
key_col = {k:v for (k,v) in zip(keys,colours)}
hatch_pattern = [ '/' , '\\' , '|' , '-' , '+' , 'x' ,'o' ,'O' , '.', '*']
legend_set = False

for k, d in time_data.iteritems():
    p_list=[]
    ax3.set_ylabel('time [s]')

    m = zip(d.keys(), d.values())
    m = sorted(m,key=lambda x: x[1])
    c_patches = []
    for p in reversed(xrange(len(m))):
        pbar = plt.bar(k, m[p][1], label=m[p][0], color=key_col[m[p][0]], edgecolor='k')
        #Add legend items until end of first iteration through loop, then stop adding
            
        if p==0 and not legend_set: 
            legend_set=True
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,)
ax3.set_yscale('log',basey=10)
plt.title('5k frames KNL OMP_NUM_THREADS=256', fontsize=24)
plt.setp(ax3.get_xticklabels(), rotation=90, ha='left')
xtl = [x.replace('levmar.parameter_flags=','').replace('strum_5k_omp1_params','') for x in time_data.keys()]
ax3.set_xticklabels(xtl);

plt.savefig('omp_5kframes_allsolvers_knl_omp256.pdf', pad_inches=10)
```


|   |
|---|
| ![png](/img/strumpacksolvermpi_dist_cori_files/strumpacksolvermpi_dist_cori_37_0.png) |


# MPI Performance on Haswell

We now use the same submission idea as the previous OpenMP examples, but using MPI instead. For a fully comprehensive exmaination of the various STRUMPACK methods used in the solution process, we examine almost all possible combinations using the parallel versions algorithms of SCOTCH and METIS, and all subsequent solver algorithms as listed above under the $\textrm{MPI STRUMPACK}$ set.


```python
MPI_SOLVER='''
from __future__ import division

import mpi4py
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI

assert MPI.Is_initialized()
assert MPI.Query_thread() == MPI.THREAD_FUNNELED

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scitbx.matrix import sqr,col
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.development.timers import Profiler

import boost.python
ext_omp = boost.python.import_ext("scitbx_examples_strumpack_solver_ext")
ext_mpi = boost.python.import_ext("scitbx_examples_strumpack_mpi_dist_solver_ext")
import sys
import numpy as np

import scipy.sparse as sps
len_x = None

if rank==0:
  A_mat = np.loadtxt(sys.argv[1],dtype={'names':('rows','cols','vals'),'formats':('i8','i8','f8')})
  b_vec = np.loadtxt(sys.argv[2])
  len_x = len(b_vec)
  n_rows = len(b_vec)
  n_cols = n_rows
  nnz = len(A_mat['vals'])
  print n_rows, n_rows/size
  #Convert the sparse CSR to flex doubles, then use them to solve using the implemented framework
  A_sp = sps.csr_matrix((A_mat['vals'],(A_mat['rows'],A_mat['cols']))) 

  #Check if upper/lower triangular only, and generate full if so
  tu=sps.triu(A_sp)
  tl=sps.tril(A_sp)
  sd=sps.diags(A_sp.diagonal())

  A_spS = A_sp
  if tu.nnz == sd.getnnz() or tl.nnz == sd.getnnz():
    A_spS = A_sp + A_sp.transpose() - sd

  import numpy as np
  row_idx_split = np.array_split(np.arange(n_rows),size)
  len_row_idx_split = flex.int( np.cumsum( np.append([0], [len(i) for i in row_idx_split]) ).tolist() )

  A_indptr = flex.int( A_spS.indptr )
  A_indices = flex.int( A_spS.indices )
  A_data = flex.double( A_spS.data )
  b = flex.double( b_vec )

  P = Profiler("EIGEN_CG")
  res_eig_cg = ext_omp.eigen_solver(3, n_rows, n_cols, A_indptr, A_indices, A_data, b)
  del P

else:
  A_spS=None
  row_idx_split=None
  len_row_idx_split=None
  b_vec=None
  n_cols=None

if size>1:
  #Broadcast data to each rank
  A_spS = comm.bcast(A_spS, root=0)
  row_idx_split = comm.bcast(row_idx_split, root=0)
  len_row_idx_split = comm.bcast(len_row_idx_split, root=0)
  b_vec = comm.bcast(b_vec, root=0)
  n_cols = comm.bcast(n_cols, root=0)

  #Take subset of data for each rank
  A_row_offset = flex.int(A_spS[row_idx_split[rank],:].indptr)
  A_col_offset = flex.int(A_spS[row_idx_split[rank],:].indices)
  A_values = flex.double(A_spS[row_idx_split[rank],:].data)
  b = flex.double(b_vec[row_idx_split[rank]])

  ################################################################################
  ################################################################################

  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.auto)
  strum_result_mpi_PTSCOTCH_AUTO = comm.gather(res_strum_mpi_local.x, root=0)
  del P

  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.auto)
  strum_result_mpi_PARMETIS_AUTO = comm.gather(res_strum_mpi_local.x, root=0)
  del P

  ################################################################################
  ################################################################################
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.direct)
  strum_result_mpi_PTSCOTCH_DIRECT = comm.gather(res_strum_mpi_local.x, root=0)
  del P
      
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.direct)
  strum_result_mpi_PARMETIS_DIRECT = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################  
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.refine)
  strum_result_mpi_PTSCOTCH_REFINE = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.refine)
  strum_result_mpi_PARMETIS_REFINE = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.prec_gmres)
  strum_result_mpi_PTSCOTCH_PRECGMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.prec_gmres)
  strum_result_mpi_PARMETIS_PRECGMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.gmres)
  strum_result_mpi_PTSCOTCH_GMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.gmres)
  strum_result_mpi_PARMETIS_GMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
    n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.prec_bicgstab)
  strum_result_mpi_PTSCOTCH_PRECBICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
    n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.prec_bicgstab)
  strum_result_mpi_PARMETIS_PRECBICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.bicgstab)
  strum_result_mpi_PTSCOTCH_BICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]),
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.bicgstab)
  strum_result_mpi_PARMETIS_BICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################
  if rank==0:
    PTSCOTCH_AUTO = flex.double()
    PARMETIS_AUTO = flex.double()
    PTSCOTCH_DIRECT = flex.double()
    PARMETIS_DIRECT = flex.double()
    PTSCOTCH_REFINE = flex.double()
    PARMETIS_REFINE = flex.double()
    PTSCOTCH_PRECGMRES = flex.double()
    PARMETIS_PRECGMRES = flex.double()
    PTSCOTCH_GMRES = flex.double()
    PARMETIS_GMRES = flex.double()
    PTSCOTCH_PRECBICGSTAB = flex.double()
    PARMETIS_PRECBICGSTAB = flex.double()
    PTSCOTCH_BICGSTAB = flex.double()
    PARMETIS_BICGSTAB = flex.double()
    for l in xrange(len(strum_result_mpi_PTSCOTCH_AUTO)):
      PTSCOTCH_AUTO.extend(strum_result_mpi_PTSCOTCH_AUTO[l])
      PARMETIS_AUTO.extend(strum_result_mpi_PARMETIS_AUTO[l])
      PTSCOTCH_DIRECT.extend(strum_result_mpi_PTSCOTCH_DIRECT[l])
      PARMETIS_DIRECT.extend(strum_result_mpi_PARMETIS_DIRECT[l])
      PTSCOTCH_REFINE.extend(strum_result_mpi_PTSCOTCH_REFINE[l])
      PARMETIS_REFINE.extend(strum_result_mpi_PARMETIS_REFINE[l])
      PTSCOTCH_PRECGMRES.extend(strum_result_mpi_PTSCOTCH_PRECGMRES[l])
      PARMETIS_PRECGMRES.extend(strum_result_mpi_PARMETIS_PRECGMRES[l])
      PTSCOTCH_GMRES.extend(strum_result_mpi_PTSCOTCH_GMRES[l])
      PARMETIS_GMRES.extend(strum_result_mpi_PARMETIS_GMRES[l])
      PTSCOTCH_PRECBICGSTAB.extend(strum_result_mpi_PTSCOTCH_PRECBICGSTAB[l])
      PARMETIS_PRECBICGSTAB.extend(strum_result_mpi_PARMETIS_PRECBICGSTAB[l])
      PTSCOTCH_BICGSTAB.extend(strum_result_mpi_PTSCOTCH_BICGSTAB[l])
      PARMETIS_BICGSTAB.extend(strum_result_mpi_PARMETIS_BICGSTAB[l])

if rank==0:  
  eps_tol=1e-3
  num_errors = 0
  for ii in xrange(len_x):
  ################################################################################
  ################################################################################
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_AUTO[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PARMETIS_AUTO:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_AUTO[ii], ii)
      
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_AUTO[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_DIRECT:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_AUTO[ii], ii)
  ################################################################################
  ################################################################################      
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_DIRECT[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_DIRECT:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_DIRECT[ii], ii)
      
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_DIRECT[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PARMETIS_DIRECT:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_DIRECT[ii], ii)
  ################################################################################
  ################################################################################      
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_REFINE[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_REFINE:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_REFINE[ii], ii)
      
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_REFINE[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PARMETIS_REFINE:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_REFINE[ii], ii)
      
  ################################################################################
  ################################################################################
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECGMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_PRECGMRES:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECGMRES[ii], ii)
      
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_PRECGMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PARMETIS_PRECGMRES:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_PRECGMRES[ii], ii)
  ################################################################################
  ################################################################################
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_GMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_GMRES:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_GMRES[ii], ii)
      
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_GMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PARMETIS_GMRES:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_GMRES[ii], ii)
  ################################################################################
  ################################################################################
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECBICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_PRECBICGSTAB:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECBICGSTAB[ii], ii)
      
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_PRECBICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PARMETIS_PRECBICGSTAB:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_PRECBICGSTAB[ii], ii)
  ################################################################################
  ################################################################################
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_BICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_BICGSTAB:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_BICGSTAB[ii], ii)
      
    if not approx_equal(PTSCOTCH_AUTO[ii], PARMETIS_BICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PARMETIS_BICGSTAB:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PARMETIS_BICGSTAB[ii], ii)
  ################################################################################
  ################################################################################

  assert (num_errors == 0)
'''
```


```python
MPI_SOLVER_FILE = open("MPI_SOLVER.py", "w")
MPI_SOLVER_FILE.write(MPI_SOLVER)
MPI_SOLVER_FILE.close()
```


```python
SBATCH_SCRIPT_MPI_HWL=\
"""#!/bin/bash
#SBATCH -N 1
#SBATCH -A m2859
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -o <DATNAME>_hwl_out_MPI.log
#SBATCH -e <DATNAME>_hwl_out_MPI.err
#SBATCH -J MPI_CCTBX_STRUMPACK
#SBATCH --mail-user=loriordan@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00

#run the application:
cd /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST
source $PWD/miniconda/bin/activate myEnv
source $PWD/build/setpaths.sh
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/4.9.3 
module load cray-mpich
module load darshan
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
mkdir -p MPI_HWL
cd MPI_HWL

mkdir -p <DATNAME>_hwl_out_MPI
cd <DATNAME>_hwl_out_MPI

#Use FPE mask to avoid floating point exceptions. Need to further investigate reasons for these
for n in 1 2 4 8 16 32;do
    t_c=$((256/${n}))
    echo "OMP_NUM_THREADS=1 BOOST_ADAPTBX_FPE_DEFAULT=1 srun -n ${n} -c 2 --cpu_bind=cores \
        libtbx.python /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/MPI_SOLVER.py <DATA_A> <DATA_B>"
    BOOST_ADAPTBX_FPE_DEFAULT=1 OMP_NUM_THREADS=1 srun -n ${n} -c 2 --cpu_bind=cores libtbx.python \
        /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/MPI_SOLVER.py <DATA_A> <DATA_B>\
        1> <DATNAME>_hwl_out_MPI${n}_OMP1.log \
        2> <DATNAME>_hwl_out_MPI${n}_OMP1.err
done
cat <DATNAME>_hwl_out_MPI${n}_OMP1.log >> <DATNAME>_hwl_out_MPI.log
cat <DATNAME>_hwl_out_MPI${n}_OMP1.err >> <DATNAME>_hwl_out_MPI.err
"""
```


```python
str_out={}
import os
proc_list = [1,2,4,8,16,32]
sub_scripts = []
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        
        A_path = A_LIST[imgs_idx]; b_path = B_LIST[imgs_idx]
        dat_name = A_path.split('/')[-1][2:-4]

        print "Data Set Name:=%s"%(dat_name)

        #Ensure the A and b data are matched correctly
        assert(os.path.dirname(A_path) == os.path.dirname(b_path))
        SUBMIT = SBATCH_SCRIPT_MPI_HWL.replace('<DATA_A>',A_path)\
           .replace('<DATA_B>', b_path).replace('<DATNAME>', dat_name)
        SBATCH_SCRIPT_FILE = open("SBATCH_SCRIPT_MPI_%s.sh"%(dat_name), "w")
        sub_scripts.append("SBATCH_SCRIPT_MPI_%s.sh"%(dat_name))
        SBATCH_SCRIPT_FILE.write(SUBMIT)
        SBATCH_SCRIPT_FILE.close()
        var = !sbatch {sub_scripts[-1]}
        print var

```

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-
    ['Submitted batch job 11742102']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    ['Submitted batch job 11742103']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-
    ['Submitted batch job 11742104']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11742105']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-
    ['Submitted batch job 11742106']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-
    ['Submitted batch job 11742107']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    ['Submitted batch job 11742108']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Eta-
    ['Submitted batch job 11742109']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11742110']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11742111']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-
    ['Submitted batch job 11742112']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-
    ['Submitted batch job 11742113']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-
    ['Submitted batch job 11742114']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Deff-
    ['Submitted batch job 11742115']
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-
    ['Submitted batch job 11742116']
    Data Set Size:=32k
    Skipping 32k



```python
%squeue -u mlxd
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>JOBID</th>
      <th>PARTITION</th>
      <th>NAME</th>
      <th>USER</th>
      <th>ST</th>
      <th>TIME</th>
      <th>NODES</th>
      <th>NODELIST(REASON)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



We now extract the timing data from the output files.


```python
dat_list_hwl_mpi = {}
procs_list = [1,2,4,8,16,32]
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        for MPI in procs_list:
            dat = A_LIST[imgs_idx].split('/')[-1][2:-4]
            var = !find ./MPI_HWL -iname '{dat}_hwl_out_MPI{MPI}_OMP1.log' #!ls OMP_KNL/{dat}_knl_out_OMP/ | grep 'log'
            dat_list_hwl_mpi.update({dat + str(MPI) : var})
print dat_list_hwl_mpi
```
<details>
<summary><p><b><span style="color:#ff3333; border:2px white solid; font-size:20px">---CLICK FOR OUTPUT---</span></b></p></summary>


    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Data Set Size:=32k
    Skipping 32k
    {'strum_5k_omp1_paramslevmar.parameter_flags=Eta-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16': [], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': [], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16': [], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': [], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32': [], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-_hwl_out_MPI32_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI8_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-16': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Deff-_hwl_out_MPI16_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI4_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI1_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': ['./MPI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI2_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32': []}

</details>

```python
time_data_hwl_mpi = {}
for key, value in dat_list_hwl_mpi.iteritems():
    try:
        with open(str(value[0])) as f:
            lines = f.read().splitlines()
            zlist = {}

            [zlist.update({l.split()[-7][:-1]:float(l.split()[-3][:-2])})\
              for l in lines if "#calls" in l if "RANK=0" in l]
            time_data_hwl_mpi.update({key : zlist})
    except Exception as e:
        #pass
        print "Could not find key:=%s; val:=%s"%(key,value)
print time_data_hwl_mpi

#Dictionary method
param_omp_val_t = { }

keyVal = time_data_hwl_mpi.iteritems().next()[0].split('-') #Param set name, and subsequent threads used
for k,v in time_data_hwl_mpi.iteritems():
    k_par_t = k.split('-')
d = { k_par_t[0]: {int(k_par_t[1]) : v} }

param_omp_val_t.update(d)
```

<details>
<summary><p><b><span style="color:#ff3333; border:2px white solid; font-size:20px">---CLICK FOR OUTPUT---</span></b></p></summary>

    Could not find key:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16; val:=[]
    Could not find key:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16; val:=[]
    Could not find key:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16; val:=[]
    Could not find key:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32; val:=[]
    Could not find key:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32; val:=[]
    Could not find key:=strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32; val:=[]
    {'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.377, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 263.06, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 274.221, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 267.332, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 263.905, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 271.979, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 275.638, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.374, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.38, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 274.612, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 263.232, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 274.499, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.379, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 274.55}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.073, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 46.725, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 108.069, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 47.029, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 47.467, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 47.46, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 109.744, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.07, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.297, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 112.564, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 46.754, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 108.255, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.302, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 108.355}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.076, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 7.171, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 8.61, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 7.195, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 7.131, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 8.743, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 8.612, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.074, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.098, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 8.628, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 7.171, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 10.491, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.099, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 8.641}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.545, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 127.133, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 188.357, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 129.103, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 124.669, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 130.811, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 188.413, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.562, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.292, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 188.353, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 129.758, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 189.037, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.29, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 188.607}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.054, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 13.258, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 14.0, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 13.484, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 14.585, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 16.393, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 14.021, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.059, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.122, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 13.983, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 13.18, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 15.935, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.121, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 14.017}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.363, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 229.896, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 263.268, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 230.369, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 230.151, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 234.615, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 263.059, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.425, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.365, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 263.321, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 231.2, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 262.557, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.352, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 263.249}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.068, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 245.259, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 159.444, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 245.016, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 245.106, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 344.778, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 159.465, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.08, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.332, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 159.223, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 245.284, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 159.386, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.337, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 159.529}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.545, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 175.759, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 201.781, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 177.869, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 172.002, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 168.02, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 201.674, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.557, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.31, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 202.481, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 176.156, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 202.062, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.329, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 202.468}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.336, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 318.469, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 576.819, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 313.926, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 333.898, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 318.047, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 576.087, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.357, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.46, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 576.475, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 321.394, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 575.932, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.47, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 575.605}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.641, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 69.17, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 120.168, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 72.781, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 70.909, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 71.393, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 120.106, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.654, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.282, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 120.172, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 70.624, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 120.061, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.266, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 120.123}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.521, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 21.541, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 39.594, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 21.91, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 21.532, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 22.713, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 39.426, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.522, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.176, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 39.354, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 22.69, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 39.128, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.19, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 39.75}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-32': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.599, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 56.953, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 109.418, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 56.738, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 59.359, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 57.286, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 107.481, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.594, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.251, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 108.388, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 57.096, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 107.493, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.222, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 108.546}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.083, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 389.64, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 804.783, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 384.048, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 380.559, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 272.169, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 803.955, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.09, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.995, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 808.838, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 385.991, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 805.282, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.417, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 812.196}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.175, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 23.928, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 33.556, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 23.698, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 23.657, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 29.766, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 33.875, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.169, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.227, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 33.961, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 27.559, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 35.336, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.239, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 33.958}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.628, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 22.493, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 50.251, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 22.472, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 23.404, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 27.338, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 50.006, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.631, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.179, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 50.204, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 21.641, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 50.444, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.19, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 50.901}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.265, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 15.07, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 17.349, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 15.085, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 13.233, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 16.943, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 17.365, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.262, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.137, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 17.339, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 15.09, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 21.916, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.142, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 17.381}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.131, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 8.481, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 10.012, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 7.16, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 7.078, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 9.926, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 10.013, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.129, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.099, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 10.005, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 7.787, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 12.256, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.099, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 10.027}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.118, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 385.122, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 222.356, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 382.105, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 382.329, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 377.351, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 221.778, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.111, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.399, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 221.681, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 381.623, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 222.113, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.403, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 222.216}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.092, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 41.273, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 55.047, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 41.085, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 41.325, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 42.537, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 55.03, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.045, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.228, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 54.794, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 41.128, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 55.374, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.227, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 55.156}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.922, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 43.946, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 78.292, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 44.25, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 43.982, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 46.695, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 78.74, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.926, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.23, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 78.585, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 43.691, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 79.065, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.234, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 79.08}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.546, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 177.996, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 202.696, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 180.55, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 173.05, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 170.661, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 202.04, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.563, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.32, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 202.263, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 179.348, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 202.466, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.337, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 202.565}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.887, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 30.796, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 45.404, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 31.075, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 30.933, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 33.812, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 45.822, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.882, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.177, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 45.887, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 30.818, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 50.028, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.184, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 45.504}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.337, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 279.249, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 352.229, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 282.009, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 282.522, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 279.656, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 352.299, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.314, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.392, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 351.88, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 283.544, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 351.731, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.407, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 354.661}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.245, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 9.246, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 10.252, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 7.928, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 8.49, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 10.284, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 10.296, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.255, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.109, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 10.264, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 10.56, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 12.404, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.112, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 10.292}, 'strum_5k_omp1_paramslevmar.parameter_flags=Eta-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.073, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 17.46, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 22.367, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 17.554, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 18.062, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 24.564, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 22.392, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.062, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.182, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 22.356, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 20.5, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 24.387, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.172, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 22.417}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.749, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 158.852, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 220.457, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 160.051, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 159.813, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 161.727, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 220.356, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.752, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.363, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 220.89, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 161.456, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 220.139, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.361, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 220.378}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.377, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 285.454, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 350.845, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 287.878, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 285.744, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 283.293, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 351.373, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.344, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.4, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 352.611, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 288.8, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 352.502, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.418, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 350.992}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.058, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 322.072, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 159.57, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 236.406, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 236.404, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 344.464, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 159.655, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.065, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.335, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 159.627, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 235.631, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 159.38, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.338, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 159.234}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-8': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.267, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 15.215, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 17.352, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 14.679, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 14.705, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 17.716, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 17.393, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.264, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.15, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 17.36, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 16.732, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 21.875, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.137, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 17.4}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.154, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 374.719, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 467.482, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 376.725, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 375.094, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 367.214, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 471.374, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.144, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.415, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 470.531, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 372.734, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 473.236, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.985, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 471.742}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.074, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 388.168, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 793.975, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 384.979, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 383.48, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 276.486, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 794.002, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.079, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 1.275, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 809.669, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 386.351, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 809.856, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.415, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 805.284}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.303, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 19.466, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 24.415, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 18.033, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 20.882, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 20.517, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 24.471, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.315, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.204, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 24.466, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 18.739, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 27.637, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.191, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 24.54}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.342, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 316.531, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 570.196, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 312.023, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 321.268, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 318.028, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 568.415, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.352, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.451, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 570.843, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 318.213, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 570.759, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 1.104, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 571.073}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.296, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 10.424, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 12.081, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 10.696, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 11.564, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 10.393, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 12.095, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.298, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.136, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 12.085, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 9.824, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 15.018, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.135, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 12.119}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.242, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 9.415, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 10.252, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 8.405, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 9.085, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 10.88, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 10.244, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.252, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.108, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 10.301, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 8.544, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 12.302, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.108, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 10.339}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.924, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 46.732, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 89.622, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 46.289, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 46.354, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 49.876, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 89.434, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.925, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.243, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 89.257, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 45.722, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 92.893, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.238, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 89.622}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.894, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 34.231, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 46.911, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 34.458, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 33.737, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 35.213, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 46.634, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.863, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.2, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 46.728, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 33.824, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 46.706, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.207, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 46.828}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.311, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 13.913, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 11.962, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 9.068, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 9.456, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 12.289, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 11.958, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.286, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.149, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 11.959, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 10.052, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 13.964, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.151, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 11.997}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.307, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 16.315, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 24.51, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 15.973, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 15.547, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 20.192, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 24.514, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.306, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.194, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 24.489, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 17.722, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 27.655, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.192, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 24.574}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.593, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 57.691, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 108.085, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 56.996, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 59.571, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 58.148, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 108.132, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.597, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.241, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 108.188, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 57.218, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 108.166, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.239, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 107.38}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.081, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 22.212, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 22.792, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 22.004, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 21.815, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 27.346, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 22.774, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.085, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.182, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 22.816, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 21.802, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 29.048, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.176, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 22.856}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.171, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 375.794, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 460.855, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 376.431, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 375.278, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 367.417, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 460.885, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.142, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.412, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 463.505, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 373.408, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 461.448, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.548, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 461.259}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.95, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 37.377, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 52.03, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 37.409, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 37.254, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 37.96, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 51.95, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.967, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.226, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 52.171, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 37.027, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 52.202, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.225, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 52.147}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.519, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 62.999, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 94.903, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 61.7, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 62.59, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 74.788, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 94.998, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.49, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.231, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 94.941, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 63.899, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 94.8, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.245, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 95.026}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.072, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 46.084, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 106.879, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 46.374, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 46.652, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 46.967, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 107.848, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.059, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.289, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 109.065, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 46.043, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 106.807, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.3, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 107.365}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.058, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 40.878, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 54.114, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 40.748, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 40.956, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 41.923, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 53.734, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.04, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.226, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 54.066, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 40.747, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 55.019, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.221, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 54.287}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.563, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 128.55, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 192.176, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 129.752, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 126.635, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 132.613, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 191.394, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.557, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.28, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 192.117, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 130.742, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 193.063, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.284, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 190.255}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.139, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 13.726, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 16.067, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 15.493, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 13.661, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 16.743, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 16.045, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.136, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.135, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 16.05, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 13.546, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 20.855, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.13, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 16.076}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.154, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 23.607, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 33.746, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 23.742, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 26.937, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 30.325, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 33.985, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.166, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.255, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 33.815, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 27.1, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 34.447, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.237, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 33.899}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.133, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 7.709, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 10.016, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 7.708, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 7.558, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 8.971, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 10.024, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.135, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.102, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 10.03, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 7.465, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 12.31, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.102, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 10.026}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.987, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 47.16, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 101.952, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 47.626, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 47.501, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 50.639, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 106.858, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.025, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.273, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 105.014, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 47.418, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 104.596, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.237, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 102.824}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.362, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 198.873, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 274.559, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 201.519, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 199.496, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 270.707, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 273.407, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.361, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.348, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 273.71, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 201.242, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 273.583, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.364, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 273.76}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.506, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 21.197, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 39.129, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 21.57, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 21.49, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 22.15, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 39.11, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.497, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.177, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 39.166, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 22.621, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 38.817, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.186, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 39.162}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.059, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 23.664, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 24.656, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 26.142, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 26.228, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 28.716, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 24.614, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.056, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.167, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 24.606, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 23.319, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 30.023, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.166, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 24.7}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.955, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 37.362, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 51.966, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 37.413, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 37.229, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 37.73, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 52.301, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.974, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.227, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 52.043, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 37.029, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 51.982, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.223, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 52.536}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.042, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 23.628, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 24.62, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 26.063, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 26.217, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 28.8, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 24.597, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.038, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.175, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 24.593, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 23.263, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 30.835, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.161, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 24.682}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.479, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 62.617, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 94.536, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 61.691, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 61.739, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 64.444, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 94.422, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.493, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.241, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 94.481, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 63.647, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 94.763, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.235, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 94.62}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-16': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.14, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 13.711, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 15.909, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 15.477, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 13.488, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 19.589, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 15.894, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.145, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.135, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 15.911, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 13.543, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 20.246, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.134, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 15.908}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-8': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.087, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 22.242, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 22.915, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 21.917, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 21.746, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 27.372, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 22.883, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.087, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.183, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 22.926, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 21.73, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 28.07, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.178, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 22.956}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.112, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 384.465, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 222.323, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 383.156, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 382.761, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 378.274, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 220.729, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.111, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.391, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 221.624, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 384.927, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 221.07, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.395, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 221.299}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Deff-32': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.636, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 25.086, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 49.519, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 24.682, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 23.279, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 27.925, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 49.506, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.64, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.189, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 49.538, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 22.549, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 49.824, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.192, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 49.807}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.358, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 234.939, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 268.953, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 234.727, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 234.694, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 239.878, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 268.671, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.355, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.354, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 269.204, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 235.822, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 266.997, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.355, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 270.42}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-4': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.863, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 34.355, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 47.033, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 34.873, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 33.847, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 35.015, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 46.816, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.866, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.195, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 46.817, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 33.942, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 47.342, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.195, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 46.983}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.987, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 47.641, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 102.111, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 47.923, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 47.412, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 51.119, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 102.103, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.001, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.278, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 101.853, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 47.594, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 102.011, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.236, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 102.627}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-1': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Deff-levmar.parameter_flags=Eta-2': {'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 0.943, 'STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=0': 46.484, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 89.474, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=0': 46.099, 'STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=0': 45.682, 'STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=0': 48.944, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 89.432, 'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 0.926, 'STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=0': 0.243, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 89.394, 'STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=0': 45.893, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 91.979, 'STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=0': 0.242, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 89.737}}

</details>

    

    ValueErrorTraceback (most recent call last)

    <ipython-input-76-14ce243fdcad> in <module>()
         19 for k,v in time_data_hwl_mpi.iteritems():
         20     k_par_t = k.split('-')
    ---> 21 d = { k_par_t[0]: {int(k_par_t[1]) : v} }
         22 
         23 param_omp_val_t.update(d)


    ValueError: invalid literal for int() with base 10: 'levmar.parameter_flags=Eta'



```python
#DB method
import sqlite3
#conn.close()
conn = sqlite3.connect(':memory:')
# Get a cursor object
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE hwl_mpi(id INTEGER PRIMARY KEY, ref_param TEXT,
                       mpi_proc INTEGER, algo TEXT, time REAL)
''')
for k,v in time_data_hwl_mpi.iteritems():
    k_par_t = k.split('-')
    for kk,vv in v.iteritems():
        cursor.execute('''INSERT INTO hwl_mpi(ref_param, mpi_proc, algo, time)
                  VALUES(?,?,?,?)''', ("-".join(k_par_t[:-1]), k_par_t[-1], kk, vv))
conn.commit()
```


```python
cursor.execute('''SELECT DISTINCT ref_param FROM hwl_mpi ''')
params = [s[0] for s in cursor.fetchall()]
cursor.execute('''SELECT DISTINCT algo FROM hwl_mpi ''')
algos = [s[0] for s in cursor.fetchall()]

par='strum_5k_omp1_paramslevmar.parameter_flags=Rxy'
cursor.execute('''SELECT ref_param, mpi_proc, algo, time FROM hwl_mpi WHERE ref_param=? AND algo=?''',(par,'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0'))
all_rows = cursor.fetchall()
for row in all_rows:
    # row[0] returns the first column in the query (name), row[1] returns email column.
    print('{0} , {1}, {2} , {3}'.format(row[0], row[1], row[2], row[3]))
```

    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 32, STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 , 1.245
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 16, STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 , 1.267
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 4, STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 , 0.95
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 2, STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 , 0.987
    strum_5k_omp1_paramslevmar.parameter_flags=Rxy , 8, STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 , 1.059



```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
xlabels = algos
keys = algos
colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
key_col = {k:v for (k,v) in zip(keys,colours)}
marker_list=[l for l in matplotlib.markers.MarkerStyle.markers.keys() if not isinstance(l, (int))]
```


```python
# define the figure size and grid layout properties
figsize = (18, 15)
cols = int(np.ceil(np.sqrt(len(params))))
gs = gridspec.GridSpec(cols, cols)
gs.update(hspace=0.4)
fig_hwl_mpi = plt.figure(num=1, figsize=figsize)
fig_hwl_mpi.suptitle('5k images Haswell MPI single node', size=20)
ax = []
cbars= []
import numpy as np
i=0

from itertools import cycle
lineStyles = ["-","--","-.",":"]

for p in params:
    row = (i // cols)
    col = i % cols
    ax.append(fig_hwl_mpi.add_subplot(gs[row, col]))
    i+=1
    lineStyles_cycle = cycle(lineStyles)
    markerStyles_cycle = cycle(marker_list)
    next(markerStyles_cycle)
    for al_idx,al in enumerate(algos):

        cursor.execute('''SELECT mpi_proc, time FROM hwl_mpi WHERE ref_param=? AND algo=?''',(p,al))
        np_omp_t = np.array(cursor.fetchall(),dtype=[('MPI', '<i8'), ('time', '<f8')])
        np_omp_t.sort()
        dat = zip(*np_omp_t)

        cset = ax[-1].plot(dat[0], dat[1], label=al.split("_RANK=0")[0], linewidth=2, 
                           linestyle=next(lineStyles_cycle), marker=next(markerStyles_cycle) )
        
    ax[-1].set_title(p.replace('levmar.parameter_flags=','').replace('strum_5k_omp1_params',''), size=16 )
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=10)
    plt.xticks([2**n for n in xrange(1,6)])

#gs.tight_layout(fig_hwl_omp,)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=12)
fig_hwl_mpi.text(0.04, 0.5, 'time [s]', va='center', rotation='vertical')
fig_hwl_mpi.text(0.5, 0.04, 'MPI_PROC', va='center', rotation='horizontal')

plt.savefig('mpi_5kframes_allsolvers_hwl_mpiall.pdf',)
```


|   |
|---|
| ![png](/img/strumpacksolvermpi_dist_cori_files/strumpacksolvermpi_dist_cori_51_0.png) |


# Multiple nodes with MPI on Haswell

Now, we can examine the scalability of the above solvers across multiple nodes using MPI.


```python
MPI_MULTI_SOLVER_PARMETIS='''
from __future__ import division

import mpi4py
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI

assert MPI.Is_initialized()
assert MPI.Query_thread() == MPI.THREAD_FUNNELED

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scitbx.matrix import sqr,col
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.development.timers import Profiler

import boost.python
ext_omp = boost.python.import_ext("scitbx_examples_strumpack_solver_ext")
ext_mpi = boost.python.import_ext("scitbx_examples_strumpack_mpi_dist_solver_ext")
import sys
import numpy as np

import scipy.sparse as sps
len_x = None

if rank==0:
  A_mat = np.loadtxt(sys.argv[1],dtype={'names':('rows','cols','vals'),'formats':('i8','i8','f8')})
  b_vec = np.loadtxt(sys.argv[2])
  len_x = len(b_vec)
  n_rows = len(b_vec)
  n_cols = n_rows
  nnz = len(A_mat['vals'])
  print n_rows, n_rows/size
  #Convert the sparse CSR to flex doubles, then use them to solve using the implemented framework
  A_sp = sps.csr_matrix((A_mat['vals'],(A_mat['rows'],A_mat['cols']))) 

  #Check if upper/lower triangular only, and generate full if so
  tu=sps.triu(A_sp)
  tl=sps.tril(A_sp)
  sd=sps.diags(A_sp.diagonal())

  A_spS = A_sp
  if tu.nnz == sd.getnnz() or tl.nnz == sd.getnnz():
    A_spS = A_sp + A_sp.transpose() - sd

  import numpy as np
  row_idx_split = np.array_split(np.arange(n_rows),size)
  len_row_idx_split = flex.int( np.cumsum( np.append([0], [len(i) for i in row_idx_split]) ).tolist() )

  A_indptr = flex.int( A_spS.indptr )
  A_indices = flex.int( A_spS.indices )
  A_data = flex.double( A_spS.data )
  b = flex.double( b_vec )

  P = Profiler("EIGEN_CG")
  res_eig_cg = ext_omp.eigen_solver(3, n_rows, n_cols, A_indptr, A_indices, A_data, b)
  del P

else:
  A_spS=None
  row_idx_split=None
  len_row_idx_split=None
  b_vec=None
  n_cols=None

if size>1:
  #Broadcast data to each rank
  A_spS = comm.bcast(A_spS, root=0)
  row_idx_split = comm.bcast(row_idx_split, root=0)
  len_row_idx_split = comm.bcast(len_row_idx_split, root=0)
  b_vec = comm.bcast(b_vec, root=0)
  n_cols = comm.bcast(n_cols, root=0)

  #Take subset of data for each rank
  A_row_offset = flex.int(A_spS[row_idx_split[rank],:].indptr)
  A_col_offset = flex.int(A_spS[row_idx_split[rank],:].indices)
  A_values = flex.double(A_spS[row_idx_split[rank],:].data)
  b = flex.double(b_vec[row_idx_split[rank]])

  ################################################################################
  ################################################################################

  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.auto)
  strum_result_mpi_PARMETIS_AUTO = comm.gather(res_strum_mpi_local.x, root=0)
  del P

  ################################################################################
  ################################################################################
      
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.direct)
  strum_result_mpi_PARMETIS_DIRECT = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################  
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.refine)
  strum_result_mpi_PARMETIS_REFINE = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.prec_gmres)
  strum_result_mpi_PARMETIS_PRECGMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.gmres)
  strum_result_mpi_PARMETIS_GMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################

  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
    n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.prec_bicgstab)
  strum_result_mpi_PARMETIS_PRECBICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  
  P = Profiler("STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]),
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.parmetis, ext_mpi.bicgstab)
  strum_result_mpi_PARMETIS_BICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  if rank==0:
    PARMETIS_AUTO = flex.double()
    PARMETIS_DIRECT = flex.double()
    PARMETIS_REFINE = flex.double()
    PARMETIS_PRECGMRES = flex.double()
    PARMETIS_GMRES = flex.double()
    PARMETIS_PRECBICGSTAB = flex.double()
    PARMETIS_BICGSTAB = flex.double()
    for l in xrange(len(strum_result_mpi_PARMETIS_BICGSTAB)):
      PARMETIS_AUTO.extend(strum_result_mpi_PARMETIS_AUTO[l])
      PARMETIS_DIRECT.extend(strum_result_mpi_PARMETIS_DIRECT[l])
      PARMETIS_REFINE.extend(strum_result_mpi_PARMETIS_REFINE[l])
      PARMETIS_PRECGMRES.extend(strum_result_mpi_PARMETIS_PRECGMRES[l])
      PARMETIS_GMRES.extend(strum_result_mpi_PARMETIS_GMRES[l])
      PARMETIS_PRECBICGSTAB.extend(strum_result_mpi_PARMETIS_PRECBICGSTAB[l])
      PARMETIS_BICGSTAB.extend(strum_result_mpi_PARMETIS_BICGSTAB[l])

if rank==0:  
  eps_tol=1e-3
  num_errors = 0
  for ii in xrange(len_x):
  ################################################################################
  ################################################################################
  
    if not approx_equal(PARMETIS_AUTO[ii], PARMETIS_DIRECT[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PARMETIS_AUTO:=%f != PARMETIS_DIRECT:=%f @ [%d] "%(PARMETIS_AUTO[ii], PARMETIS_DIRECT[ii], ii)
      
  ################################################################################
  ################################################################################ 
  
    if not approx_equal(PARMETIS_AUTO[ii], PARMETIS_REFINE[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PARMETIS_AUTO:=%f != PARMETIS_REFINE:=%f @ [%d] "%(PARMETIS_AUTO[ii], PARMETIS_REFINE[ii], ii)
      
  ################################################################################
  ################################################################################      
    if not approx_equal(PARMETIS_AUTO[ii], PARMETIS_PRECGMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PARMETIS_AUTO:=%f != PARMETIS_PRECGMRES:=%f @ [%d] "%(PARMETIS_AUTO[ii], PARMETIS_PRECGMRES[ii], ii)
      
  ################################################################################
  ################################################################################
  
    if not approx_equal(PARMETIS_AUTO[ii], PARMETIS_GMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PARMETIS_AUTO:=%f != PARMETIS_GMRES:=%f @ [%d] "%(PARMETIS_AUTO[ii], PARMETIS_GMRES[ii], ii)
      
  ################################################################################
  ################################################################################
  
    if not approx_equal(PARMETIS_AUTO[ii], PARMETIS_PRECBICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PARMETIS_AUTO:=%f != PARMETIS_PRECBICGSTAB:=%f @ [%d] "%(PARMETIS_AUTO[ii], PARMETIS_PRECBICGSTAB[ii], ii)
      
  ################################################################################
  ################################################################################
  
    if not approx_equal(PARMETIS_AUTO[ii], PARMETIS_BICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PARMETIS_AUTO:=%f != PARMETIS_BICGSTAB:=%f @ [%d] "%(PARMETIS_AUTO[ii], PARMETIS_BICGSTAB[ii], ii)
      
   ################################################################################
  ################################################################################
 
  assert (num_errors == 0)
'''
MPI_MULTI_SOLVER_FILE = open("MPI_MULTI_SOLVER_PARMETIS.py", "w")
MPI_MULTI_SOLVER_FILE.write(MPI_MULTI_SOLVER_PARMETIS)
MPI_MULTI_SOLVER_FILE.close()
```


```python
MPI_MULTI_SOLVER_PTSCOTCH='''
from __future__ import division

import mpi4py
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
from mpi4py import MPI

assert MPI.Is_initialized()
assert MPI.Query_thread() == MPI.THREAD_MULTIPLE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scitbx.matrix import sqr,col
from cctbx.array_family import flex
from libtbx.test_utils import approx_equal
from libtbx.development.timers import Profiler

import boost.python
ext_omp = boost.python.import_ext("scitbx_examples_strumpack_solver_ext")
ext_mpi = boost.python.import_ext("scitbx_examples_strumpack_mpi_dist_solver_ext")
import sys
import numpy as np

import scipy.sparse as sps
len_x = None

if rank==0:
  A_mat = np.loadtxt(sys.argv[1],dtype={'names':('rows','cols','vals'),'formats':('i8','i8','f8')})
  b_vec = np.loadtxt(sys.argv[2])
  len_x = len(b_vec)
  n_rows = len(b_vec)
  n_cols = n_rows
  nnz = len(A_mat['vals'])
  print n_rows, n_rows/size
  #Convert the sparse CSR to flex doubles, then use them to solve using the implemented framework
  A_sp = sps.csr_matrix((A_mat['vals'],(A_mat['rows'],A_mat['cols']))) 

  #Check if upper/lower triangular only, and generate full if so
  tu=sps.triu(A_sp)
  tl=sps.tril(A_sp)
  sd=sps.diags(A_sp.diagonal())

  A_spS = A_sp
  if tu.nnz == sd.getnnz() or tl.nnz == sd.getnnz():
    A_spS = A_sp + A_sp.transpose() - sd

  import numpy as np
  row_idx_split = np.array_split(np.arange(n_rows),size)
  len_row_idx_split = flex.int( np.cumsum( np.append([0], [len(i) for i in row_idx_split]) ).tolist() )

  A_indptr = flex.int( A_spS.indptr )
  A_indices = flex.int( A_spS.indices )
  A_data = flex.double( A_spS.data )
  b = flex.double( b_vec )

  P = Profiler("EIGEN_CG")
  res_eig_cg = ext_omp.eigen_solver(3, n_rows, n_cols, A_indptr, A_indices, A_data, b)
  del P

else:
  A_spS=None
  row_idx_split=None
  len_row_idx_split=None
  b_vec=None
  n_cols=None

if size>1:
  #Broadcast data to each rank
  A_spS = comm.bcast(A_spS, root=0)
  row_idx_split = comm.bcast(row_idx_split, root=0)
  len_row_idx_split = comm.bcast(len_row_idx_split, root=0)
  b_vec = comm.bcast(b_vec, root=0)
  n_cols = comm.bcast(n_cols, root=0)

  #Take subset of data for each rank
  A_row_offset = flex.int(A_spS[row_idx_split[rank],:].indptr)
  A_col_offset = flex.int(A_spS[row_idx_split[rank],:].indices)
  A_values = flex.double(A_spS[row_idx_split[rank],:].data)
  b = flex.double(b_vec[row_idx_split[rank]])

  ################################################################################
  ################################################################################

  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_AUTO_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.auto)
  strum_result_mpi_PTSCOTCH_AUTO = comm.gather(res_strum_mpi_local.x, root=0)
  del P

  ################################################################################
  ################################################################################
      
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_DIRECT_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.direct)
  strum_result_mpi_PTSCOTCH_DIRECT = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  ################################################################################
  ################################################################################  
  
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_REFINE_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.refine)
  strum_result_mpi_PTSCOTCH_REFINE = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_PRECGMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.prec_gmres)
  strum_result_mpi_PTSCOTCH_PRECGMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_GMRES_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.gmres)
  strum_result_mpi_PTSCOTCH_GMRES = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################

  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_PRECBICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]), 
    n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.prec_bicgstab)
  strum_result_mpi_PTSCOTCH_PRECBICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  
  P = Profiler("STRUMPACK_MPI_DIST_PTSCOTCH_BICGSTAB_RANK=%d"%rank)
  res_strum_mpi_local = ext_mpi.strumpack_mpi_dist_solver(len(row_idx_split[rank]),
      n_cols, comm, A_row_offset, A_col_offset, A_values, b, len_row_idx_split, ext_mpi.ptscotch, ext_mpi.bicgstab)
  strum_result_mpi_PTSCOTCH_BICGSTAB = comm.gather(res_strum_mpi_local.x, root=0)
  del P
  
  ################################################################################
  ################################################################################
  if rank==0:
    PTSCOTCH_AUTO = flex.double()
    PTSCOTCH_DIRECT = flex.double()
    PTSCOTCH_REFINE = flex.double()
    PTSCOTCH_PRECGMRES = flex.double()
    PTSCOTCH_GMRES = flex.double()
    PTSCOTCH_PRECBICGSTAB = flex.double()
    PTSCOTCH_BICGSTAB = flex.double()
    for l in xrange(len(strum_result_mpi_PTSCOTCH_BICGSTAB)):
      PTSCOTCH_AUTO.extend(strum_result_mpi_PTSCOTCH_AUTO[l])
      PTSCOTCH_DIRECT.extend(strum_result_mpi_PTSCOTCH_DIRECT[l])
      PTSCOTCH_REFINE.extend(strum_result_mpi_PTSCOTCH_REFINE[l])
      PTSCOTCH_PRECGMRES.extend(strum_result_mpi_PTSCOTCH_PRECGMRES[l])
      PTSCOTCH_GMRES.extend(strum_result_mpi_PTSCOTCH_GMRES[l])
      PTSCOTCH_PRECBICGSTAB.extend(strum_result_mpi_PTSCOTCH_PRECBICGSTAB[l])
      PTSCOTCH_BICGSTAB.extend(strum_result_mpi_PTSCOTCH_BICGSTAB[l])

if rank==0:  
  eps_tol=1e-3
  num_errors = 0
  for ii in xrange(len_x):
  ################################################################################
  ################################################################################
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_DIRECT[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_DIRECT:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_DIRECT[ii], ii)
      
  ################################################################################
  ################################################################################ 
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_REFINE[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_REFINE:=%f @ [%d] "%(PARMETIS_AUTO[ii], PTSCOTCH_REFINE[ii], ii)
      
  ################################################################################
  ################################################################################      
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECGMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_PRECGMRES:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECGMRES[ii], ii)
      
  ################################################################################
  ################################################################################
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_GMRES[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_GMRES:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_GMRES[ii], ii)
      
  ################################################################################
  ################################################################################
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECBICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_PRECBICGSTAB:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_PRECBICGSTAB[ii], ii)
      
  ################################################################################
  ################################################################################
  
    if not approx_equal(PTSCOTCH_AUTO[ii], PTSCOTCH_BICGSTAB[ii], eps=eps_tol):
      num_errors += 1
      print "Error in PTSCOTCH_AUTO:=%f != PTSCOTCH_BICGSTAB:=%f @ [%d] "%(PTSCOTCH_AUTO[ii], PTSCOTCH_BICGSTAB[ii], ii)
      
   ################################################################################
  ################################################################################
 
  assert (num_errors == 0)
'''
MPI_MULTI_SOLVER_FILE = open("MPI_MULTI_SOLVER_PTSCOTCH.py", "w")
MPI_MULTI_SOLVER_FILE.write(MPI_MULTI_SOLVER_PTSCOTCH)
MPI_MULTI_SOLVER_FILE.close()
```


```python
SBATCH_SCRIPT_MPI_MULTI_HWL=\
"""#!/bin/bash
#SBATCH -N <NODES>
#SBATCH -A m2859
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -o <DATNAME>_hwl_out_MPI_MULTI.log
#SBATCH -e <DATNAME>_hwl_out_MPI_MULTI.err
#SBATCH -J MPI_MULTI_CCTBX_STRUMPACK
#SBATCH --mail-user=loriordan@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 01:00:00

#run the application:
cd /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST
source $PWD/miniconda/bin/activate myEnv
source $PWD/build/setpaths.sh
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/4.9.3 
module load cray-mpich
module load darshan
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
mkdir -p MPI_MULTI_HWL
cd MPI_MULTI_HWL

mkdir -p <DATNAME>_hwl_out_MPI_MULTI
cd <DATNAME>_hwl_out_MPI_MULTI

#Use FPE mask to avoid floating point exceptions. Need to further investigate reasons for these
PROC=$((32*<NODES>))
echo "OMP_NUM_THREADS=1 BOOST_ADAPTBX_FPE_DEFAULT=1 srun -n ${PROC} -c 2 --cpu_bind=cores \
        libtbx.python /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/MPI_MULTI_SOLVER_PARMETIS.py <DATA_A> <DATA_B>"
BOOST_ADAPTBX_FPE_DEFAULT=1 OMP_NUM_THREADS=1 srun -n ${PROC} -c 2 --cpu_bind=cores libtbx.python \
        /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/MPI_MULTI_SOLVER_PARMETIS.py <DATA_A> <DATA_B>\
        1> <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.log \
        2> <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.err

cat <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.log >> <DATNAME>_hwl_out_MPI_MULTI.log
cat <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.err >> <DATNAME>_hwl_out_MPI_MULTI.err
"""
```


```python
SBATCH_SCRIPT_MPI_MULTI_HWL_PTSCOTCH=\
"""#!/bin/bash
#SBATCH -N <NODES>
#SBATCH -A m2859
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -o <DATNAME>_hwl_out_MPI_MULTI.log
#SBATCH -e <DATNAME>_hwl_out_MPI_MULTI.err
#SBATCH -J MPI_MULTI_CCTBX_STRUMPACK
#SBATCH --mail-user=loriordan@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00

#run the application:
cd /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST
source $PWD/miniconda/bin/activate myEnv
source $PWD/build/setpaths.sh
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/4.9.3 
module load cray-mpich
module load darshan
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
mkdir -p MPI_MULTI_HWL
cd MPI_MULTI_HWL

mkdir -p <DATNAME>_hwl_out_MPI_MULTI_PTSCOTCH
cd <DATNAME>_hwl_out_MPI_MULTI_PTSCOTCH

#Use FPE mask to avoid floating point exceptions. Need to further investigate reasons for these
#Note: cray-mpich requires environment variables to set the MPI threading level when built with -craympich-mt option 
#See https://www.hpc.kaust.edu.sa/tips/multi-threaded-mpi for examples

PROC=$((32*<NODES>))
echo "OMP_NUM_THREADS=1 BOOST_ADAPTBX_FPE_DEFAULT=1 MPICH_MAX_THREAD_SAFETY=multiple srun -n ${PROC} -c 2 --cpu_bind=cores \
        libtbx.python /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/MPI_MULTI_SOLVER_PTSCOTCH.py <DATA_A> <DATA_B>"
BOOST_ADAPTBX_FPE_DEFAULT=1 OMP_NUM_THREADS=1 MPICH_MAX_THREAD_SAFETY=multiple srun -n ${PROC} -c 2 --cpu_bind=cores libtbx.python \
        /global/cscratch1/sd/mlxd/STRUMPACK_MPI_DIST/MPI_MULTI_SOLVER_PTSCOTCH.py <DATA_A> <DATA_B>\
        1> <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.log \
        2> <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.err

cat <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.log >> <DATNAME>_hwl_out_MPI_MULTI.log
cat <DATNAME>_hwl_out_MPI_MULTI${PROC}_OMP1.err >> <DATNAME>_hwl_out_MPI_MULTI.err
"""
```

PTSCOTCH FAILURES CAUSING JOBS TO FAIL: ERROR: hdgraphFold2: communication error; Issue is likely due to the requirement of MPI_THREAD_MULTIPLE support in the MPI_Init_thread() initialisation routine. Building this support in with Cori requires using the `-craympich-mt` flag during compilation, and `MPICH_MAX_THREAD_SAFETY=multiple` environment variable at runtime. A rebuild on mpi4py with this option enabled, as well as the PTSCOTCH and STRUMPACK libraries was attempted to rectify this issue.

A test of this support in mpi4py after rebuilding was performed with:
```
#Fails
srun -n 2 libtbx.python -c "import mpi4py; mpi4py.rc.threads = True; mpi4py.rc.thread_level = 'multiple'; from mpi4py import MPI; assert MPI.Query_thread()==MPI.THREAD_MULTIPLE"

#Succeeds
MPICH_MAX_THREAD_SAFETY=multiple srun -n 2 libtbx.python -c "import mpi4py; mpi4py.rc.threads = True; mpi4py.rc.thread_level = 'multiple'; from mpi4py import MPI; assert MPI.Query_thread()==MPI.THREAD_MULTIPLE"
```

Though, the PTSCOTCH enabled routines still fail on more than a single node. This requires a further examination to determine the reason for failure.

PARMETIS NOT FAILING; USING FOR MULTINODE TESTING.


```python
str_out={}
import os
node_list = [2]
sub_scripts = []

for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        if len(sub_scripts)>0:
            continue
        A_path = A_LIST[imgs_idx]; b_path = B_LIST[imgs_idx]
        dat_name = A_path.split('/')[-1][2:-4]

        print "Data Set Name:=%s"%(dat_name)

        #Ensure the A and b data are matched correctly
        assert(os.path.dirname(A_path) == os.path.dirname(b_path))
        for NODES in node_list:
            SUBMIT = SBATCH_SCRIPT_MPI_MULTI_HWL.replace('<DATA_A>',A_path)\
               .replace('<DATA_B>', b_path).replace('<DATNAME>', dat_name).replace('<NODES>', str(NODES))
            SBATCH_SCRIPT_FILE = open("SBATCH_SCRIPT_MPI_MULTI_%s.sh"%(dat_name), "w")
            sub_scripts.append("SBATCH_SCRIPT_MPI_MULTI_%s.sh"%(dat_name))
            SBATCH_SCRIPT_FILE.write(SUBMIT)
            SBATCH_SCRIPT_FILE.close()
            var = !sbatch {sub_scripts[-1]}
            print var
```

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Data Set Name:=strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-
    ['Submitted batch job 11822447']
    Data Set Size:=32k
    Skipping 32k



```python
dat_list_hwl_mpi_multi_parmetis = {}
nodes_list = [1,2,4,8,16,32]
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "5k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        for NODES in nodes_list:
            MPI=32*NODES
            dat = A_LIST[imgs_idx].split('/')[-1][2:-4]
            var = !find ./MPI_MULTI_HWL -iname '{dat}_hwl_out_MPI_MULTI{MPI}_OMP1.log' #!ls OMP_KNL/{dat}_knl_out_OMP/ | grep 'log'
            if len(var)!=0:
                dat_list_hwl_mpi_multi_parmetis.update({dat + str(MPI) : var})
print dat_list_hwl_mpi_multi_parmetis
```

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Data Set Size:=32k
    Skipping 32k
    {'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-512': ['./MPI_MULTI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI512_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-1024': ['./MPI_MULTI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI1024_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-64': ['./MPI_MULTI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI_MULTI/strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-_hwl_out_MPI_MULTI64_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-64': ['./MPI_MULTI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI64_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-256': ['./MPI_MULTI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI256_OMP1.log'], 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-128': ['./MPI_MULTI_HWL/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI/strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-_hwl_out_MPI_MULTI128_OMP1.log']}



```python
time_data_hwl_mpi_multi_parmetis = {}
for key, value in dat_list_hwl_mpi_multi_parmetis.iteritems():
    try:
        with open(str(value[0])) as f:
            lines = f.read().splitlines()
            zlist = {}

            [zlist.update({l.split()[-7][:-1]:float(l.split()[-3][:-2])})\
              for l in lines if "#calls" in l if any( x in l for x in ["EIGEN_CG","RANK=0"]) ]
            time_data_hwl_mpi_multi_parmetis.update({key : zlist})
    except Exception as e:
        #pass
        print "Could not find key:=%s; val:=%s %s"%(key,value,e)
print time_data_hwl_mpi_multi_parmetis
```

    {'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-512': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.978, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 2.472, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 2.468, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.936, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 2.543, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 2.42, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 13.575, 'EIGEN_CG': 0.11}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-1024': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 2.134, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 2.404, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 2.433, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 2.182, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 2.483, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 2.447, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 8.012, 'EIGEN_CG': 0.117}, 'strum_5k_omp1_paramslevmar.parameter_flags=Rxy-levmar.parameter_flags=Bfactor-levmar.parameter_flags=Eta-64': {}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-64': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.561, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 16.318, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 16.658, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.547, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 16.749, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 16.843, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 19.145, 'EIGEN_CG': 0.217}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-256': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.636, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 2.513, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 2.526, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.678, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 2.468, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 2.367, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 5.943, 'EIGEN_CG': 0.111}, 'strum_5k_omp1_paramslevmar.parameter_flags=Bfactor-128': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 1.584, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 4.239, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 4.245, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 1.615, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 4.237, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 4.173, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 6.396, 'EIGEN_CG': 0.14}}



```python
#DB method
import sqlite3
conn.close()
conn = sqlite3.connect(':memory:')
# Get a cursor object
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE hwl_mpi_multi(id INTEGER PRIMARY KEY, ref_param TEXT,
                       procs INTEGER, algo TEXT, time REAL)
''')
for k,v in time_data_hwl_mpi_multi_parmetis.iteritems():
    k_par_t = k.split('-')
    for kk,vv in v.iteritems():
        cursor.execute('''INSERT INTO hwl_mpi_multi(ref_param, procs, algo, time)
                  VALUES(?,?,?,?)''', ("-".join(k_par_t[:-1]), k_par_t[-1], kk, vv))
        print "-".join(k_par_t[:-1]), k_par_t[-1], kk, vv
conn.commit()
```

    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 1.978
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 2.472
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 2.468
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 1.936
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 2.543
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 2.42
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 13.575
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 512 EIGEN_CG 0.11
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 2.134
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 2.404
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 2.433
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 2.182
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 2.483
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 2.447
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 8.012
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 1024 EIGEN_CG 0.117
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 1.561
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 16.318
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 16.658
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 1.547
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 16.749
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 16.843
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 19.145
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 64 EIGEN_CG 0.217
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 1.636
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 2.513
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 2.526
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 1.678
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 2.468
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 2.367
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 5.943
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 256 EIGEN_CG 0.111
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 1.584
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 4.239
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 4.245
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 1.615
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 4.237
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 4.173
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 6.396
    strum_5k_omp1_paramslevmar.parameter_flags=Bfactor 128 EIGEN_CG 0.14



```python
cursor.execute('''SELECT DISTINCT ref_param FROM hwl_mpi_multi ''')
params = [s[0] for s in cursor.fetchall()]
cursor.execute('''SELECT DISTINCT algo FROM hwl_mpi_multi ''')
algos = [s[0] for s in cursor.fetchall()]
```


```python
# define the figure size and grid layout properties
figsize = (10, 6)
cols = int(np.ceil(np.sqrt(len(params))))
gs = gridspec.GridSpec(cols, cols)
gs.update(hspace=0.4)
fig_hwl_mpi = plt.figure(num=1, figsize=figsize)
fig_hwl_mpi.suptitle('5k images Haswell MPI multi node', size=20)
ax = []
cbars= []
import numpy as np
i=0

from itertools import cycle
lineStyles = ["-","--","-.",":"]

for p in params:
    row = (i // cols)
    col = i % cols
    ax.append(fig_hwl_mpi.add_subplot(gs[row, col]))
    i+=1
    lineStyles_cycle = cycle(lineStyles)
    markerStyles_cycle = cycle(marker_list)
    next(markerStyles_cycle)
    for al_idx,al in enumerate(algos):

        cursor.execute('''SELECT procs, time FROM hwl_mpi_multi WHERE ref_param=? AND algo=?''',(p,al))
        np_omp_t = np.array(cursor.fetchall(),dtype=[('MPI', '<i8'), ('time', '<f8')])
        np_omp_t.sort()
        dat = zip(*np_omp_t)

        cset = ax[-1].plot(dat[0], dat[1], label=al.split("_RANK=0")[0], linewidth=2, 
                           linestyle=next(lineStyles_cycle), marker=next(markerStyles_cycle) )
        
    ax[-1].set_title(p.replace('levmar.parameter_flags=','').replace('strum_5k_omp1_params',''), size=16 )
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=10)
    plt.xticks([2**n for n in xrange(6,11)],[str(2**ii) for ii in xrange(1,6)])

#gs.tight_layout(fig_hwl_omp,)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=12)
fig_hwl_mpi.text(0.04, 0.5, 'time [s]', va='center', rotation='vertical')
fig_hwl_mpi.text(0.5, 0.04, 'Haswell Nodes', va='center', rotation='horizontal')

plt.savefig('mpi_5kframes_allsolvers_hwl_mpi_multi_bfactor.pdf',)
```


|   |
|---|
| ![png](/img/strumpacksolvermpi_dist_cori_files/strumpacksolvermpi_dist_cori_64_0.png) |


For the examined data parameter refinement data set (`Bfactor`) the MPI distributed routines using the working ParMETIS library showed some scaling, but saturated at approximated 8 Haswell nodes. EIGEN's conjugate gradient solver still dominates all of the above in terms of performance, with runtimes on the order of 100 ms. 

We may next increase the data size and see if the solvers scale better.

## 32k images in data set


```python
str_out={}
import os
node_list = [1,2,4,8]
sub_scripts = []

for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "32k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        if len(sub_scripts)>0:
            continue
        A_path = A_LIST[imgs_idx]; b_path = B_LIST[imgs_idx]
        dat_name = A_path.split('/')[-1][2:-4]

        print "Data Set Name:=%s"%(dat_name)

        #Ensure the A and b data are matched correctly
        assert(os.path.dirname(A_path) == os.path.dirname(b_path))
        for NODES in node_list:
            SUBMIT = SBATCH_SCRIPT_MPI_MULTI_HWL.replace('<DATA_A>',A_path)\
               .replace('<DATA_B>', b_path).replace('<DATNAME>', dat_name).replace('<NODES>', str(NODES))
            SBATCH_SCRIPT_FILE = open("SBATCH_SCRIPT_MPI_MULTI_%s.sh"%(dat_name), "w")
            sub_scripts.append("SBATCH_SCRIPT_MPI_MULTI_%s.sh"%(dat_name))
            SBATCH_SCRIPT_FILE.write(SUBMIT)
            SBATCH_SCRIPT_FILE.close()
            var = !sbatch {sub_scripts[-1]}
            print var
```

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Skipping 5k
    Data Set Size:=32k
    Data Set Name:=strum_32k_omp1_paramslevmar.parameter_flags=Eta-
    ['Submitted batch job 11797392']
    ['Submitted batch job 11797395']
    ['Submitted batch job 11797398']
    ['Submitted batch job 11797401']



```python
dat_list_hwl_mpi_multi_parmetis_32k = {}
nodes_list = [1,2,4,8,16,32]
for imgs_size in list_idx:
    print "Data Set Size:=%s"%imgs_size
    #Subselect the smallest data size for now
    if imgs_size != "32k":
        print "Skipping %s"%imgs_size
        continue
        
    for imgs_idx in list_idx[imgs_size]:
        for NODES in nodes_list:
            MPI=32*NODES
            dat = A_LIST[imgs_idx].split('/')[-1][2:-4]
            var = !find ./MPI_MULTI_HWL -iname '{dat}_hwl_out_MPI_MULTI{MPI}_OMP1.log' #!ls OMP_KNL/{dat}_knl_out_OMP/ | grep 'log'
            if len(var)!=0:
                dat_list_hwl_mpi_multi_parmetis_32k.update({dat + str(MPI) : var})
print dat_list_hwl_mpi_multi_parmetis_32k
```

    Data Set Size:=10k
    Skipping 10k
    Data Set Size:=1k
    Skipping 1k
    Data Set Size:=5k
    Skipping 5k
    Data Set Size:=32k
    {'strum_32k_omp1_paramslevmar.parameter_flags=Eta-32': ['./MPI_MULTI_HWL/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI32_OMP1.log'], 'strum_32k_omp1_paramslevmar.parameter_flags=Eta-64': ['./MPI_MULTI_HWL/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI64_OMP1.log'], 'strum_32k_omp1_paramslevmar.parameter_flags=Eta-256': ['./MPI_MULTI_HWL/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI256_OMP1.log'], 'strum_32k_omp1_paramslevmar.parameter_flags=Eta-128': ['./MPI_MULTI_HWL/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI/strum_32k_omp1_paramslevmar.parameter_flags=Eta-_hwl_out_MPI_MULTI128_OMP1.log']}



```python
time_data_hwl_mpi_multi_parmetis_32k = {}
for key, value in dat_list_hwl_mpi_multi_parmetis_32k.iteritems():
    try:
        with open(str(value[0])) as f:
            lines = f.read().splitlines()
            zlist = {}

            [zlist.update({l.split()[-7][:-1]:float(l.split()[-3][:-2])})\
              for l in lines if "#calls" in l if any( x in l for x in ["EIGEN_CG","RANK=0"]) ]
            time_data_hwl_mpi_multi_parmetis_32k.update({key : zlist})
    except Exception as e:
        #pass
        print "Could not find key:=%s; val:=%s %s"%(key,value,e)
print time_data_hwl_mpi_multi_parmetis_32k
```

    {'strum_32k_omp1_paramslevmar.parameter_flags=Eta-32': {}, 'strum_32k_omp1_paramslevmar.parameter_flags=Eta-64': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 7.985, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 529.119, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 527.819, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 7.539, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 528.022, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 524.183, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 529.64, 'EIGEN_CG': 0.349}, 'strum_32k_omp1_paramslevmar.parameter_flags=Eta-128': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 7.858, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 163.248, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 162.918, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 7.69, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 163.758, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 162.985, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 162.717, 'EIGEN_CG': 0.336}, 'strum_32k_omp1_paramslevmar.parameter_flags=Eta-256': {'STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0': 7.744, 'STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0': 51.414, 'STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0': 51.385, 'STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0': 7.439, 'STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0': 51.436, 'STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0': 51.387, 'STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0': 53.823, 'EIGEN_CG': 0.437}}



```python
#DB method
import sqlite3
conn.close()
conn_parmetis_32k = sqlite3.connect(':memory:')
# Get a cursor object
cursor = conn_parmetis_32k.cursor()
cursor.execute('''
    CREATE TABLE hwl_mpi_multi_32k(id INTEGER PRIMARY KEY, ref_param TEXT,
                       procs INTEGER, algo TEXT, time REAL)
''')
for k,v in time_data_hwl_mpi_multi_parmetis_32k.iteritems():
    k_par_t = k.split('-')
    for kk,vv in v.iteritems():
        cursor.execute('''INSERT INTO hwl_mpi_multi_32k(ref_param, procs, algo, time)
                  VALUES(?,?,?,?)''', ("-".join(k_par_t[:-1]), k_par_t[-1], kk, vv))
        print "-".join(k_par_t[:-1]), k_par_t[-1], kk, vv
conn_parmetis_32k.commit()
```

    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 7.985
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 529.119
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 527.819
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 7.539
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 528.022
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 524.183
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 529.64
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 64 EIGEN_CG 0.349
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 7.858
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 163.248
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 162.918
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 7.69
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 163.758
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 162.985
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 162.717
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 128 EIGEN_CG 0.336
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 STRUMPACK_MPI_DIST_PARMETIS_GMRES_RANK=0 7.744
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 STRUMPACK_MPI_DIST_PARMETIS_DIRECT_RANK=0 51.414
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 STRUMPACK_MPI_DIST_PARMETIS_REFINE_RANK=0 51.385
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 STRUMPACK_MPI_DIST_PARMETIS_BICGSTAB_RANK=0 7.439
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 STRUMPACK_MPI_DIST_PARMETIS_PRECGMRES_RANK=0 51.436
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 STRUMPACK_MPI_DIST_PARMETIS_PRECBICGSTAB_RANK=0 51.387
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 STRUMPACK_MPI_DIST_PARMETIS_AUTO_RANK=0 53.823
    strum_32k_omp1_paramslevmar.parameter_flags=Eta 256 EIGEN_CG 0.437



```python
cursor.execute('''SELECT DISTINCT ref_param FROM hwl_mpi_multi_32k ''')
params = [s[0] for s in cursor.fetchall()]
cursor.execute('''SELECT DISTINCT algo FROM hwl_mpi_multi_32k ''')
algos = [s[0] for s in cursor.fetchall()]
```


```python
# define the figure size and grid layout properties
figsize = (10, 6)
cols = int(np.ceil(np.sqrt(len(params))))
gs = gridspec.GridSpec(cols, cols)
gs.update(hspace=0.4)
fig_hwl_mpi_32k = plt.figure(num=1, figsize=figsize)
fig_hwl_mpi_32k.suptitle('32k images Haswell MPI multi node', size=20)
ax = []
cbars= []
import numpy as np
i=0

from itertools import cycle
lineStyles = ["-","--","-.",":"]

for p in params:
    row = (i // cols)
    col = i % cols
    ax.append(fig_hwl_mpi_32k.add_subplot(gs[row, col]))
    i+=1
    lineStyles_cycle = cycle(lineStyles)
    markerStyles_cycle = cycle(marker_list)
    next(markerStyles_cycle)
    for al_idx,al in enumerate(algos):

        cursor.execute('''SELECT procs, time FROM hwl_mpi_multi_32k WHERE ref_param=? AND algo=?''',(p,al))
        np_omp_t = np.array(cursor.fetchall(),dtype=[('MPI', '<i8'), ('time', '<f8')])
        np_omp_t.sort()
        dat = zip(*np_omp_t)
        
        ax[-1].plot(dat[0], dat[1], label=al.split("_RANK=0")[0], linewidth=2, 
                           linestyle=next(lineStyles_cycle), marker=next(markerStyles_cycle) )
        
    ax[-1].set_title(p.replace('levmar.parameter_flags=','').replace('strum_32k_omp1_params',''), size=16 )
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=10)
    plt.xticks([2**n for n in xrange(6,9)],[str(2**ii) for ii in xrange(1,4)])

#gs.tight_layout(fig_hwl_omp,)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=12)
fig_hwl_mpi_32k.text(0.04, 0.5, 'time [s]', va='center', rotation='vertical')
fig_hwl_mpi_32k.text(0.5, 0.04, 'Haswell Nodes', va='center', rotation='horizontal')

plt.savefig('mpi_32kframes_allsolvers_hwl_mpi_multi_bfactor.pdf',)
```


|   |
|---|
| ![png](/img/strumpacksolvermpi_dist_cori_files/strumpacksolvermpi_dist_cori_72_0.png) |


While the EIGEN solver is a multiple or so slover than the previous example, it still remains in the 100 ms range, with the next best peformance given by the BICGSTAB or GMRES routines (though no scaling observed). The other routines show scaling across the examined nodes, bu were not examined beyond 8.
