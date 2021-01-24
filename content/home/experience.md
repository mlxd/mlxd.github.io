+++
# Experience widget.
widget = "experience"  # Do not modify this line!
active = true  # Activate this widget? true/false

title = "Experience"
subtitle = ""

# Order that this section will appear in.
weight = 8

# Date format for experience
#   Refer to https://sourcethemes.com/academic/docs/customization/#date-format
#date_format = "January 2006"
date_format = "2006"

# Experiences.
#   Add/remove as many `[[experience]]` blocks below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin/end multi-line descriptions with 3 quotes `"""`.
#[[experience]]
#  title = "CEO"
#  company = "GenCoin"
#  company_url = ""
#  location = "California"
#  date_start = "2017-01-01"
#  date_end = ""
#  description = """
#  Responsibilities include:
#  
#  * Analysing
#  * Modelling
#  * Deploying
#  """
#
#[[experience]]
#  title = "Professor"
#  company = "University X"
#  company_url = ""
#  location = "California"
#  date_start = "2016-01-01"
#  date_end = "2016-12-31"
#  description = """Taught electronic engineering and researched semiconductor physics."""

[[experience]]
  title = "Research Computational Scientist [Postdoctoral researcher], Senior Computational Scientist [Research Fellow]"
  company = "Irish Centre for High-End Computing (ICHEC) [NUI Galway]"
  date_start = "2019-01-01"
  #date_end = "2020-12-11"
  location = "Dublin, Ireland"
  description = """
  *	Led research and development of a hybrid classical HPC-quantum algorithm for NLP tasks. Using Python, C++, PyBind11, Catch2, Docker, OpenMP, MPI and backed by Intel’s Quantum Simulator (Intel-QS), succeeded in demonstrating the encoding of corpus data and performing similarity comparisons between different sentences. Available at [Code](https://github.com/ICHEC/QNLP) and [Publication](https://doi.org/10.1088/2632-2153/abbd2e)
  * Point-of-contact and technical lead for several national and international collaborations between both industry and academia on quantum software projects [URL](https://www.ichec.ie/about/activities/novel-technologies/ichecs-quantum-programming-ireland-qpi-initiative).
  * Represnt the quantum research team at ICHEC for several international conferences, including talks at ISC 2019, Frankfurt, Germany, Intel DevCon 2019, Denver Colorado, and an invited talk at QNLP 2019, Oxford.  [Talk (Youtube)](https://www.youtube.com/watch?v=rG0_SKCx09A)
  """

[[experience]]
  title = "Visiting researcher"
  company = "Okinawa Institute of Science and Technology Graduate University (OIST)"
  date_start = "2018-07-01"
  date_end = "2018-08-29"
  location = "Okinawa, Japan"
  description = """
  * Collaborated on a study of chaotic dynamics in quantum superfluid systems. Work available at [https://arxiv.org/abs/1812.04759](https://arxiv.org/abs/1812.04759). 
  * Further developed and generalised the Bose--Einstein simulation suite [GPUE](https://github.com/GPUE-group/GPUE). Published in the Journal of Open Source Software as [GPUE: Graphics Processing Unit Gross--Pitaevskii Equation solver](http://joss.theoj.org/papers/10.21105/joss.01037). """

[[experience]]
  title = "Postdoctoral fellow"
  company = "Lawrence Berkeley National Laboratory"
  date_start = "2017-04-28"
  date_end = "2018-05-31"
  location = "Berkeley, CA, USA"
  description = """Researcher involved the ExaFEL project.
  Responsibilities included:  

  * Building a Docker-enabled pipeline for deployment at NERSC.
  * Integration of STRUMPACK distributed sparse linear-algebra package for crystal parameter fitting.
  * Scalability and bottle-neck investigations of the software at scale.
  * Real-time data analysis and feedback during protein crystallography experiments.
  * Implementing OpenMP and MPI parallelised algorithms.

  Code contributions available at [CCTBX](https://github.com/cctbx/cctbx_project), [DIALS](https://github.com/dials/dials), [ExaFEL Project](https://github.com/exafel/exafel_project).  
  Results leading to publications: [2018](publication/dials2018/), [2019](publication/pnas2019)
  """

[[experience]]
  title = "PhD student"
  company = "Okinawa Institute of Science and Technology Graduate University (OIST)"
  date_start = "2012-09-30"
  date_end = "2017-03-23"
  location = "Okinawa, Japan"
  description = """Researched cold atomic systems, which specific emphasis on the non-equilibrium dynamics of vortex lattice carrying Bose–Einstein condensates. Works and responsibilities included:

  * Thesis: [Non-equilibrium vortex dynamics in rapidly rotating Bose-Einstein condensates](https://oist.repo.nii.ac.jp/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=182&item_no=1&page_id=15&block_id=79)
  * Developed GPUE ([modern](https://github.com/GPUE-group/GPUE),[original](https://github.com/mlxd/GPUE)), a software suite for simulating linear and nonlinear Schrödinger equation dynamics. The work was used largely to investigate vortex behaviour in Bose--Einstein condensates.
  * Responsible for HPC-related interactions between the lab and university.
  * Chair of the OIST Student Assembly (2015-2016).

  See [publications](/publication) for a list of published works.

  """

[[experience]]
  title = "Research assistant (PhD student)"
  company = "University College Cork (UCC)"
  date_start = "2011-09-01"
  date_end = "2012-09-30"
  location = "Cork, Ireland"
  description = """Transfered to OIST with [Prof. Busch](https://groups.oist.jp/qsu) after completing first year of PhD programme."""

[[experience]]
  title = "Software developer"
  company = "IBM, Dublin Software Labs"
  date_start = "2010-08-01"
  date_end = "2011-07-01"
  location = "Dublin, Ireland"
  description = """
  * Developer on [IBM Solution Installer](https://www.ibm.com/support/knowledgecenter/en/SSHRKX_8.0.0/config/int_sol_installer.html), used for installing server-side applications with WebSphere Portal platform.
  * Experience using Java, J2EE, XML, XSLT, Ant, Shell scripting, JDBC, Python, C.
  * Deployed and maintained WebSphere Portal, Lotus Sametime, and Lotus Connections software stacks for development and testing.
  """

[[experience]]
  title = "Product engineer (intern)"
  company = "Analog Devices"
  date_start = "2009-01-05"
  date_end = "2009-08-31"
  location = "Limerick, Ireland"
  description = """
  * Analysed and assisted in trouble-shooting of die failures.
  * Trained in operation of light-emission microscopy (LEM), optical beam-induced resistance change (OBIRCH) methods, focussed ion beam (FIB) and scanning electron microscope (SEM) equipment.
  * Programmed wafer data analysis and input automation routines in BASH, C, and VB.
  """

+++
