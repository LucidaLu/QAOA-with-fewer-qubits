  Maximum cut (Max-Cut) problem is one of the most important combinatorial optimization problems because of its various applications in real life, and recently Quantum Approximate Optimization Algorithm (QAOA) has been widely employed to solve it. However, as the size of the problem increases, the number of qubits required will become larger. With the aim of saving qubits, we propose a coupling framework for designing QAOA circuits to solve larger-scale Max-Cut problem. This framework relies on a classical algorithm that approximately solves a certain variant of Max-Cut, and we derive an approximation guarantee theoretically, assuming the approximation ratio of the classical algorithm and QAOA. Furthermore we design a heuristic approach that fits in our framework and perform sufficient numerical experiments, where we solve Max-Cut on various $24$-vertex Erdős-Rényi graphs. Our framework only consumes $18$ qubits and achieves $0.9950$ approximation ratio on average, which outperforms the previous methods showing $0.9778$ (quantum algorithm using the same number of qubits) and $0.9643$ (classical algorithm). The experimental results indicate our well-designed quantum-classical coupling framework gives satisfactory approximation ratio while reduces the qubit cost, which sheds light on more potential computing power of NISQ devices.
