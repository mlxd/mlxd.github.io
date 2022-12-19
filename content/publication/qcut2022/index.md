+++
title = "Fast quantum circuit cutting with randomized measurements"
date = "2022-07-29"
draft = false

authors = ["Angus Lowe", "Matija Medvidović", "Anthony Hayes", "Lee J. O'Riordan", "Thomas R Bromley", "Juan Miguel Arrazola", "Nathan Killoran"]
publication_types = ["2"]

publication = "arXiv:2207.14734"
publication_short = "arXiV"

abstract = "We propose a new method to extend the size of a quantum computation beyond the number of physical qubits available on a single device. This is accomplished by randomly inserting measure-and-prepare channels to express the output state of a large circuit as a separable state across distinct devices. Our method employs randomized measurements, resulting in a sample overhead that is O(4^k/epsilon^2), where epsilon is the accuracy of the computation and k the number of parallel wires that are "cut" to obtain smaller sub-circuits. We also show an information-theoretic lower bound of Omega(2^k/epsilon^2) for any comparable procedure. We use our techniques to show that circuits in the Quantum Approximate Optimization Algorithm (QAOA) with $p$ entangling layers can be simulated by circuits on a fraction of the original number of qubits with an overhead that is roughly 2^(O(pk)), where $k$ is the size of a known balanced vertex separator of the graph which encodes the optimization problem. We obtain numerical evidence of practical speedups using our method applied to the QAOA, compared to prior work. Finally, we investigate the practical feasibility of applying the circuit cutting procedure to large-scale QAOA problems on clustered graphs by using a 30-qubit simulator to evaluate the variational energy of a 129-qubit problem as well as carry out a 62-qubit optimization."
abstract_short = ""

image_preview = ""

math = true
selected = false

projects = []
slides = ""

tags = []

url_code = ""
url_dataset = ""
url_pdf = ""
url_project = "" #"project/uqs"
url_slides = ""
url_video = ""

url_custom = [
    { name = "arXiV", url = "https://arxiv.org/abs/2207.14734"}
]
+++
