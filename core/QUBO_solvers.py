# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/3/20 10:29
# @Author  : Wen Xixiang
# @Version : python3.9

import numpy as np
import warnings
from scipy import sparse
from collections import defaultdict
import dimod
import itertools
from minorminer import find_embedding
import os
import time
from greedy import SteepestDescentSolver
import matplotlib.pyplot as plt
from typing import Any, Literal, Tuple, Union, List, Iterable, Dict
from dimod.binary import BinaryQuadraticModel as BQM
from collections import ChainMap
import networkx as nx
from functools import reduce
from .utils import to_fixed_point, b2f, show_obj_curve
from dwave.system import DWaveSampler, FixedEmbeddingComposite
import dwave_networkx as dnx
import dimod


def check_dimension(func):
    """
    This is a decorator detecting the dimensions of input data.
    For a linear system equation, coefficient matrix A should have
    2 dimension, right hand side of the linear system should be a
    column vector. And the number of rows of A is equal to the size of b.

    Parameters
    ----------
    func:
        The function to decorate. It should be a function used to solving
        linear system equation Ax=b. A should be an instance of numpy
        ndarray or scipy sparse matrices and has 2 dimension. b should be
        an instance of numpy ndarray which has 1 dimension.
    """

    def wrapper(*args, **kwargs):
        A = args[1]
        b = args[2]
        if isinstance(A, np.ndarray) or sparse.isspmatrix(A):
            if A.ndim != 2:
                raise ValueError("The dimension of A should be 2.")
        else:
            raise TypeError(
                "Matrix 'A' should be an instance of numpy ndarray or scipy sparse matrices!"
            )
        if isinstance(b, np.ndarray):
            if b.ndim != 1 and np.all(np.array(b.shape) > 1):
                raise ValueError("The dimension of b should be 1.")
            else:
                b = b.reshape(-1, 1)  # Change vector b to column vector.
        else:
            raise TypeError("Vector 'b' should be an instance of np.ndarray!")
        if A.shape[0] != len(b):
            raise ValueError(
                "Dimensions of input matrix and vector do not match and cannot be multiplied!"
            )

        new_args = (args[0], A, b)
        return func(*new_args, **kwargs)

    return wrapper


class BinarySampleSolver(object):
    """A solver that uses sampling methods to solve systems of linear equations.
    There are only two cases for all the elements of the solution, high or low.

    Parameters
    ----------
    sampler : dimod.Sampler
        A tool to sample from a given problem to obtain the system's minimum
        energy.
    num_sample : int
        The number of samples taken by the sampler.
    high_value : int or float
        The higher value of the solution to a system of linear equations.
    low_value : int or float
        The lower value of the solution to a system of linear equations.
    postprocess : bool
        Whether to use post-processing after sampling.
    resultMod : {'lowest_energy', 'lowest_obj'}, optional
        The way to process the sampling result (default='lowest_energy').
    num_spin_reversal_transforms : int, optional
        Number of spin reversal transforms used in D-Wave quantum annealer
        (default=0), which reduce the impact of possible analog and systematic
        errors). For detailed instructions, please refer to D-Wave
        documentation: https://docs.dwavesys.com/docs/latest/doc_handbook.html.
    embedding_dict : dict
        The embedding method requires a dictionary format where the key
        represents a node in the target structure, and the corresponding value
        is a chain of nodes in the quantum devices' structure.
        In essence, embedding associates a sequence of nodes (a chain) on
        the D-Wave annealer with a problem node.

    """

    def __init__(
        self, sampler, num_sample=1000, high_value=2, low_value=1, *args, **kwargs
    ):

        self.sampler = sampler
        self.num_sample = num_sample
        self.h_value = high_value
        self.l_value = low_value
        self.postprocess = kwargs.pop("postprocess", None)
        self.resultMod = kwargs.pop("resultMod", None)
        self.num_spin_reversal_transforms = kwargs.pop(
            "num_spin_reversal_transforms", None
        )  # This parameter only used for QPU!
        self.embedding = kwargs.pop("embedding_dict", None)

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, new_sampler):
        if not isinstance(new_sampler, dimod.Sampler):
            raise TypeError(
                "Sampler should be an instance of sampler from dwave ocean software."
            )

        if isinstance(new_sampler, DWaveSampler):
            self._mode = "Quantum_Annealing"
            self._sampler_target_graph = new_sampler.to_networkx_graph()
            warnings.warn(
                "'embedding' indicates the way mapping our problem "
                " If not specified, the default settings('minor_embedding') "
                " will be used for DWaveSampler."
            )
            self.embedding = None
            # Embedding result will be flushed after update 'sampler' property.
            # You can respecify the result by loading embedding file and assign embedding property.
        else:
            self._mode = "Simulated_Annealing"
            self._sampler_target_graph = None
            self.embedding = None

        self._sampler = new_sampler

    @property
    def num_sample(self):
        return self._ns

    @num_sample.setter
    def num_sample(self, new_ns):
        if not isinstance(new_ns, int):
            raise TypeError("The number of samples should be an integer!")
        else:
            self._ns = new_ns

    @property
    def h_value(self):
        return self._hv

    @h_value.setter
    def h_value(self, new_hv):
        if not isinstance(new_hv, (int, float)):
            raise TypeError(
                "The higher value of permeability should be an integer or float!"
            )
        else:
            self._hv = new_hv

    @property
    def l_value(self):
        return self._lv

    @l_value.setter
    def l_value(self, new_lv):
        if not isinstance(new_lv, (int, float)):
            raise TypeError(
                "The lower value of permeability should be an integer or float!"
            )
        else:
            self._lv = new_lv

    @property
    def postprocess(self):
        return self._p_p

    @postprocess.setter
    def postprocess(self, new_command):
        if new_command is None:
            warnings.warn(
                "'postprocess' indicates whether to optimize the result"
                " after sampler finish its work. Default method is"
                " 'SteepestDescentSolver'."
            )
            self._p_p = False
        else:
            self._p_p = new_command

    @property
    def resultMod(self):
        return self._rm

    @resultMod.setter
    def resultMod(self, new_rm):
        if new_rm is None:
            warnings.warn(
                "'resultMod' indicates the way to process the sampling result."
                " If not specified, the default Settings will be used"
            )
            self._rm = "lowest_energy"
        elif new_rm not in ["lowest_energy", "lowest_obj"]:
            warnings.warn(
                "Incorrect use of 'resultMod' " "Use default `codingMod` setting."
            )
            self._rm = "lowest_energy"
        else:
            self._rm = new_rm

    @property
    def sampling_method(self):
        return self._mode

    @property
    def num_spin_reversal_transforms(self):
        if self._mode == "Simulated_Annealing":
            warnings.warn(
                "Spin reversal transforms only used in Quantum Sampler! "
                "Setting this value does not affect the SA sampling result."
            )
        return self._nsrt

    @num_spin_reversal_transforms.setter
    def num_spin_reversal_transforms(self, new_nsrt):
        if new_nsrt is None:
            new_nsrt = 0
        if not isinstance(new_nsrt, int):
            raise TypeError(
                "The number of spin-reversal transforms should be an integer!"
            )
        else:
            self._nsrt = new_nsrt

    @property
    def embedding(self):
        if self._mode == "Simulated_Annealing":
            warnings.warn(
                "Embedding only used in Quantum Sampler."
                "Setting embedding method does not affect the SA sampling result."
            )
        return self._embedding

    @embedding.setter
    def embedding(self, new_embedding):
        if new_embedding is None:
            new_embedding = dict()

        if not isinstance(new_embedding, dict):
            raise TypeError("The embedding should be an instance of dict!")
        else:
            self._embedding = new_embedding

    @property
    def sampler_target_graph(self):
        return self._sampler_target_graph

    @property
    def num_emb_qubits(self):
        if len(self._embedding) > 0:
            num_qb_each_nodes = map(
                lambda x: len(x), self._embedding.values()
            )  # number of qubits corresponding to each nodes
            return sum(list(num_qb_each_nodes))
        else:
            warnings.warn("No embedding method is assigned or calculated.")

    @property
    def chain_num_qubits(self):
        if len(self._embedding) > 0:
            temp = list(map(lambda x: len(x), self._embedding.values()))
            chain_qubits = sorted(
                temp, reverse=True
            )  # number of qubits corresponding to each nodes
            return chain_qubits
        else:
            warnings.warn("No embedding method is assigned or calculated.")

    def _cal_Q_const(self, A, b):
        """Calculating the Q matrix and const item. Q matrix is used to
        construct QUBO problem, Q and const can be used to construct BQM
        problem.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.

        Returns
        -------
        Q : {(M*R, N*R), (M*R, N*R)}, array_like
            Q matrix in dict form. R is the number of coding bit.
        const : float
            Constant term of object function.
        """
        I = np.ones([A.shape[1], 1])
        Q = (self._hv - self._lv) ** 2 * A.T @ A - 2 * (self._hv - self._lv) * np.diag(
            ((b - self._lv * A @ I).reshape(1, -1) @ A).squeeze()
        )
        const = (b - self._lv * A @ I).reshape(1, -1) @ (b - self._lv * A @ I)
        return Q, const.squeeze()

    def _cal_QUBO(self, A, b):
        Q, _ = self._cal_Q_const(A, b)
        return {"Q": Q}

    def _cal_BQM(self, A, b):
        Q, const = self._cal_Q_const(A, b)
        return {"bqm": BQM.from_qubo(Q, offset=const)}

    def _cal_ISING(self, A, b):
        Q, const = self._cal_Q_const(A, b)
        bqm = BQM.from_qubo(Q, offset=const)
        ising_problem = BQM.to_ising(bqm)
        return {"h": ising_problem[0], "J": ising_problem[1]}  # h, J

    def _opt_obj(self, A, b, sampleset):
        """Find the optimal sample result which has the lowest object function.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        sampleset : dict or list
            The result sampled from D-wave sampler.

        Returns
        -------
        x : array_like
            The solution with the minimum objective function.
        """
        lstspst = sampleset.lowest()  # lowest_sampleset
        if len(lstspst) >= 20:
            lstspst = lstspst.truncate(20)
        obj = np.inf
        x = None
        for sample in lstspst.samples():
            tmpres = np.array((list(sample.values())))  # temper result
            tmpx = b2f(tmpres, self._lv, self._hv)
            tmpobj = np.linalg.norm(A @ tmpx - b)
            if tmpobj <= obj:
                obj = tmpobj
                x = tmpx
        return x

    def _problem2qsampler(self, A, b, label="temp"):
        """
        Mapping the structure of problem to the structure of Quantum annealer.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        label : str, optional
            The identifier for the current problem,
            used to distinguish the embedding file.
        """
        if len(self._embedding) == 0 and self._mode == "Quantum_Annealing":
            filename = f"{label}_minior_embedding_{A.shape[1]}bits.npy"
            if not os.path.exists(os.path.join(os.getcwd(), filename)):
                warnings.warn(
                    "The embedding file does not exist in the current path and will be calculated and saved "
                    "automatically ..."
                )
                self._embedding = self._problem_mapping(A, b)
                np.save(filename, self._embedding)
            else:
                self._embedding = np.load(filename, allow_pickle=True).tolist()
            self._sampler = FixedEmbeddingComposite(self._sampler, self._embedding)

    def _problem_mapping(self, A, b, **kwargs):
        """Find the minor embedding for our QUBO problem.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.

        Returns
        -------
        embedding : dict
            Minor Embedding result found by the D-wave ocean software.
        """
        target_structure = dimod.child_structure_dfs(
            self._sampler
        )  # 获取目标采样器的结构

        Q_dick = self._cal_Q_const(A, b, **kwargs)
        bqm = dimod.as_bqm(Q_dick, dimod.BINARY)  # 将qubo问题转化为bqm问题

        source_edgelist = list(
            itertools.chain(bqm.quadratic, ((v, v) for v in bqm.linear))
        )  # 当前问题的edge列表

        target_edgelist = target_structure.edgelist  # 目标采样器的edge列表

        # 寻找映射， 这只是一个映射关系，其实与Q具体的值没有关系, 重点是Q矩阵的形式，
        # 因此我才可以把它提前计算保存好，以减少花费在寻找嵌入方案上的时间
        embedding = find_embedding(source_edgelist, target_edgelist)
        return embedding

    @check_dimension
    def solve(self, A, b, problem_type="BQM", label="temp"):
        """
        Solving the linear system by direct sampling method using SA sampler.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        problem_type : {'BQM', 'QUBO', 'ISING'}, optional
            The type of problem being solved by D-Wave Sampler. Default is
            'BQM', which has the positive objective function. This is because
            it provides the sampler about the constant part of the objective
            function, thereby resulting in better performance.The remaining
            two methods have similar effects.
        label : str, optional
            The identifier for the current problem,
            used to distinguish the embedding file.

        Returns
        -------
        x : np.ndarray
            The solution of linear system which only has two value.

        """
        # Mapping the problem on Quantum devices
        self._problem2qsampler(A, b, label=label)

        if problem_type == "BQM":
            construct_method = self._cal_BQM
            sample_method = self._sampler.sample
            pp_method = SteepestDescentSolver().sample

        elif problem_type == "QUBO":
            construct_method = self._cal_QUBO
            sample_method = self._sampler.sample_qubo
            pp_method = SteepestDescentSolver().sample_qubo

        elif problem_type == "ISING":
            construct_method = self._cal_ISING
            sample_method = self._sampler.sample_ising
            pp_method = SteepestDescentSolver().sample_ising

        else:
            NotImplementedError

        print("============================================")
        # Calculate the sampling problem ('QUBO', 'BQM' or 'Ising')
        sampling_problem = construct_method(A, b)

        # Sampling
        T0 = time.time()
        if self._mode == "Simulated_Annealing":
            sampleset = sample_method(**sampling_problem, num_reads=self._ns)
        else:
            sampleset = sample_method(
                **sampling_problem,
                num_reads=self._ns,
                num_spin_reversal_transforms=self._nsrt,
            )
            qpu_acct = sampleset.info["timing"]["qpu_access_time"]
            print("* QPU access time: {} us".format(qpu_acct))
        print(f"* The sampling process took: {(time.time() - T0):.2f} s")

        if self._p_p == True:
            """
            Steepest greedy descent over the subspace is run
            sequentially on samples returned by the QPU.
            """
            T1 = time.time()
            sampleset = pp_method(**sampling_problem, initial_states=sampleset)
            print(f"** The postprocess time is: {(time.time() - T1):.3f} s")

        # Convert the sampling result to variable we supposed to solving.
        if self._rm == "lowest_obj":
            x = self._opt_obj(
                A, b, sampleset
            )  # Select best sampling result from the top 20 lowest energy state.
        elif self._rm == "lowest_energy":
            # Select the lowest energy state as sampling result
            result = np.array(list(sampleset.first.sample.values()))
            x = b2f(result, self._lv, self._hv).reshape(-1, 1)  # transfer q to x
        else:
            raise NotImplementedError

        print("============================================")

        return x

    def show_sampler_structure(self, **kwargs):
        """
        Show the structure of quantum annealer we are using.

        Parameters
        ----------
        kwargs : dict
            Additional options used in dnx.draw_chimera. Such as
            'with_label', 'node_color', 'node_size' and so on.
            More detailed discription can be found in
            https://docs.ocean.dwavesys.com/en/stable/docs_dnx/reference/generated/dwave_networkx.drawing.chimera_layout.draw_chimera.html

        """
        if self.sampler_target_graph is not None:

            options = {
                "with_labels": False,
                "node_color": "blue",
                "node_size": 10,
                "edge_color": (0.1, 0.1, 0.6, 0.3),
                "width": 1,
                "linewidths": 0.01,
            }

            kkwargs = dict(ChainMap(kwargs, options))

            graph_family = self.sampler_target_graph.graph["family"]
            if graph_family == "pegasus":
                draw_method = dnx.draw_pegasus
            elif graph_family == "chimera":
                draw_method = dnx.draw_chimera
            draw_method(self.sampler_target_graph, **kkwargs)

    def show_embedding_result(self, embedding, **kwargs):
        """
        Plot the embedding result.
        """
        if self.sampler_target_graph is not None:
            options = {
                "show_labels": False,
                "unused_color": (0.9, 0.9, 0.9, 1.0),  # None 表示不显示未嵌入的部分
                # 'interaction_edges' : [(1691, 1695)], # only show the edge we defined!
                "node_size": 10,
                "overlapped_embedding": False,
                # Default false, If True, chains in the embedding parameter may overlap (contain the same vertices in G)
                "width": 1,
            }

            node_list = [list(embedding.values())[i][0] for i in range(len(embedding))]
            corr_edges = list(nx.utils.pairwise(node_list))

            graph_family = self.sampler_target_graph.graph["family"]
            if graph_family == "pegasus":
                draw_method_embedding = dnx.draw_pegasus_embedding
            elif graph_family == "chimera":
                draw_method_embedding = dnx.draw_chimera_embedding
            draw_method_embedding(
                self.sampler_target_graph,
                embedding,
                interaction_edges=corr_edges,
                **options,
            )


class DirectSampleSolver(BinarySampleSolver):
    """
    A solver that uses sampling methods to solve systems of linear equations.
    The solution of the system of linear equations is represented by an encoding
     consisting of multiple bits.

    Parameters
    ----------
    R : int
        The length of the encoded bit string.
    fixed_point : int
        The parameters used for encoding a bit string, which determine
        the upper limit of encodable numbers.
    regularization : bool
        Whether to employ regularization techniques to alleviate the
        ill-conditioned nature of the system of linear equations.
    codingMod : {'pn', 'p', 'n'}, Optional
        The parameter determines the type of encoded numbers, which corresponds
        to the type of solutions. If all solutions are positive, the 'p' mode
        is used. If all solutions are negative, the 'n' mode is used. If there
        are both positive and negative solutions, the 'pn' mode is used. The
        default mode is 'pn'."

    """

    def __init__(self, R=5, fixed_point=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_encoding_bit = R
        self.fixed_point = fixed_point
        self.codingMod = kwargs.pop("codingMod", None)
        self.bit_value = self._cal_bit_value()
        self.IPI = None  # save the Intermediate process of iteration

    @property
    def num_encoding_bit(self):
        return self._R

    @num_encoding_bit.setter
    def num_encoding_bit(self, new_R):
        if not isinstance(new_R, int):
            raise TypeError("The number of bits used to encoding should be an integer!")
        else:
            self._R = new_R

    @property
    def fixed_point(self):
        return self._fp

    @fixed_point.setter
    def fixed_point(self, new_fp):
        if not isinstance(new_fp, int):
            raise TypeError("Fixed point should be an integer!")
        else:
            self._fp = new_fp

    @property
    def bit_value(self):
        return self._bv

    @bit_value.setter
    def bit_value(self, new_bv):
        if not isinstance(new_bv, np.ndarray):
            raise TypeError(
                f"The dtype of bit value should be np.ndarray! \n"
                f"Not the {type(new_bv)}!"
            )
        else:
            self._bv = new_bv

    @property
    def codingMod(self):
        return self._cm

    @codingMod.setter
    def codingMod(self, new_cm):
        if new_cm is None:
            warnings.warn(
                "'codingMod' indicates the way bit string to represent a number!"
                " It contain three mod, 'p', 'n', 'pn'"
                " If not specified, the default Settings will be used"
            )
            self._cm = "pn"

        elif new_cm not in ["p", "n", "pn"]:
            warnings.warn(
                "Incorrect use of 'codingMod' " "Use default `codingMod` setting."
            )
            self._cm = "pn"
        else:
            self._cm = new_cm

    def _problem2qsampler(self, A, b, label="temp", **kwargs):
        """
        Mapping the structure of problem to the structure of Quantum annealer.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        label : str, optional
            The identifier for the current problem,
            used to distinguish the embedding file.
        """
        if len(self._embedding) == 0 and self._mode == "Quantum_Annealing":
            filename = f"{label}_minior_embedding_{self._R * A.shape[1]}bits.npy"
            if not os.path.exists(os.path.join(os.getcwd(), filename)):
                warnings.warn(
                    "The embedding file does not exist in the current path and will be calculated and saved automatically ..."
                )
                self._embedding = self._problem_mapping(A, b, **kwargs)
                np.save(filename, self._embedding)
            else:
                self._embedding = np.load(filename, allow_pickle=True).tolist()
            self._sampler = FixedEmbeddingComposite(self._sampler, self._embedding)

    def _cal_bit_value(self):
        """
        Calculating the value of each bit in coding string.
        This method has three mode, 'pn' indicates the encoded values are both
        positive and negative. 'n' indicates negative and 'p' indicates positive.

        Notes
        -----
        The formula is as following (j0 = fixed_point)

        .. math:: x_{i}=\sum_{j=1}^{R} 2^{j_{0}-j} q_{(i-1) R+j}

        Examples
        --------
        >>> R = 3
        >>> fp = 1
        >>> cal_bit_value(num_bits,fixed_point,'p')
        array([2.  , 1. , 0.5])
        """
        if self._cm == "pn":
            return np.array(
                [
                    -(2**self._fp) if i == 0 else 2.0 ** (self._fp - i)
                    for i in range(0, self._R)
                ]
            )
        elif self._cm == "p":
            return np.array([2.0 ** (self._fp - i) for i in range(self._R)])
        elif self._cm == "n":
            return np.array([-(2.0 ** (self._fp - i)) for i in range(self._R)])
        else:
            raise ValueError

    def _cal_B_matrix(self, N):
        """
        Calculating the matrix used to discrete the object matrix.

        Parameters
        ----------
        N : int
            The dimension of the solution vector.

        Returns
        -------
        B : np.ndarray
            The matrix used to discrete the object matrix.
        """
        B = np.zeros([N, N * self._R])
        for i in range(N):
            B[i, i * self._R : (i + 1) * self._R] = self._bv
        return B

    def _cal_QUBO(self, A, b, **kwargs):
        # Q, _ = self._cal_Q_const(A, b, **kwargs)
        Q = self._cal_Q_const(A, b, **kwargs)
        return {"Q": Q}

    def _cal_BQM(self, A, b, **kwargs):
        # Q, const = self._cal_Q_const(A, b, **kwargs)
        Q = self._cal_Q_const(A, b, **kwargs)
        return {"bqm": BQM.from_qubo(Q)}  # , offset=const)}

    def _cal_ISING(self, A, b, **kwargs):
        # Q, const = self._cal_Q_const(A, b, **kwargs)
        # bqm = BQM.from_qubo(Q, offset=const)
        Q = self._cal_Q_const(A, b, **kwargs)
        bqm = BQM.from_qubo(Q)
        ising_problem = BQM.to_ising(bqm)
        return {"h": ising_problem[0], "J": ising_problem[1]}  # h, J

    def _cal_Q_const(self, A, b, **kwargs):
        """Calculating the Q matrix and const item. Q matrix is used to construct QUBO problem,
        Q and const can be used to construct BQM problem.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        B : {N, N*R}, array_like
            The encoding matrix, which transfer sampling result to real/complex vector.
        alpha : float
            The factor for Tikhonov regularization. If set to zero, no regularization
            method will be applied.
        D : array_like
            Finite difference operators for implementing higher-order Tikhonov
            regularization.
        Returns
        -------
        Q : {(M*R, N*R), (M*R, N*R)}, array_like
            Q matrix in dict form. R is the number of coding bit.
        const : float
            Constant term of object function.
        """
        B = kwargs["B"]
        alpha = kwargs["alpha"]
        D = kwargs["D"]

        A_disc = A @ B
        AA_disc = A_disc.T @ A_disc

        if alpha:
            Q = (AA_disc + alpha * (D @ B).T @ D @ B) - 2 * np.diag(
                (b.reshape(1, -1) @ A_disc).squeeze()
            )
        else:
            Q = A_disc.T @ A_disc - 2 * np.diag(
                (b.reshape(1, -1) @ A_disc).squeeze()
            )  # no regulization

        Q = np.tril(Q) + np.tril(Q, -1)
        const = b.reshape(1, -1) @ b.reshape(-1, 1)

        return Q, const

    def _opt_obj(self, A, b, sampleset, B, non_linear_obj=None):
        """Find the optimal sample result which has the lowest object function.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        sampleset : dict or list
            The result sampled from D-wave sampler.
        B : {N, N*R}, array_like
            The encoding matrix, which transfer sampling result to
            real/complex vector.
        non_linear_obj : func, optional
            These methods can only solve linear systems of equations,
            so for nonlinear problems, they can only approximate them as
            linear problems and then solve them. This can potentially result
            in suboptimal solutions for the nonlinear problem, where the
            optimal solution corresponds to a suboptimal solution for the
            linear problem. Providing a nonlinear objective function here
            may improve the solution outcome.

        Returns
        -------
        x : array_like
            The solution with the minimum objective function.
        """

        if non_linear_obj is None:

            def default_obj(x):
                return np.linalg.norm(A @ x - b) ** 2

        else:
            default_obj = non_linear_obj

        lstspst = sampleset.lowest()  # lowest_sampleset
        if len(lstspst) >= 20:
            lstspst = lstspst.truncate(20)
        obj = np.inf
        x = None
        for sample in lstspst.samples():
            tmpres = np.array((list(sample.values())))  # temper result
            tmpx = (B @ tmpres).reshape(-1, 1)
            tmpobj = default_obj(tmpx)
            if tmpobj <= obj:
                obj = tmpobj
                x = tmpx
        return x

    @check_dimension
    def solve(
        self,
        A,
        b,
        problem_type="BQM",
        label="temp",
        alpha=None,
        D=None,
        non_linear_obj=None,
        sp=True,
    ):
        """
        Solving the linear system iteratively.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        problem_type : string
            The problem type used during sampling which has three options:
            "BQM", "QUBO" and "Ising".
        label : string
            Making a description of current problem which is used to distinguish
            file name of embedding result.
        alpha : float
            The factor of Tikhonov regularization. If it equals to zero, no regularization
            method will be used.
        D : array_like
            Finite difference operators for implementing higher-order Tikhonov
            regularization. Default is an identity operator.
        non_linear_obj : Callable[..., Any]
            The outer function used to decide the best sampling result. This method will
            take advantage of the diversity of sampling results so that the solution can
            satisfy the objective function that cannot be encoded as a quadratic form.
        sp : bool
            Whether to print the processes of this program. Default is True.

        Returns
        -------
        x : {(M, 1), (N, 1)}, array_like
            Solution of the linear system.

        """
        self._problem2qsampler(A, b, label=label)
        B = self._cal_B_matrix(A.shape[1])

        if D is None:
            D = np.eye(A.shape[1])

        if alpha is None:
            alpha = 0

        if problem_type == "BQM":
            construct_method = self._cal_BQM
            sample_method = self._sampler.sample
            pp_method = SteepestDescentSolver().sample

        elif problem_type == "QUBO":
            construct_method = self._cal_QUBO
            sample_method = self._sampler.sample_qubo
            pp_method = SteepestDescentSolver().sample_qubo

        elif problem_type == "ISING":
            construct_method = self._cal_ISING
            sample_method = self._sampler.sample_ising
            pp_method = SteepestDescentSolver().sample_ising

        else:
            NotImplementedError

        if sp:
            print("============================================")

        # Calculate the sampling problem ('QUBO', 'BQM' or 'Ising')
        sampling_problem = construct_method(A, b, B=B, alpha=alpha, D=D)

        # Sampling
        T0 = time.time()
        if self._mode == "Simulated_Annealing":
            sampleset = sample_method(**sampling_problem, num_reads=self._ns)
        else:
            sampleset = sample_method(
                **sampling_problem,
                num_reads=self._ns,
                num_spin_reversal_transforms=self._nsrt,
            )
            qpu_acct = sampleset.info["timing"]["qpu_access_time"]
            if sp:
                print("* QPU access time: {} us".format(qpu_acct))
        if sp:
            print(f"* The sampling process took: {(time.time() - T0):.2f} s")

        if self._p_p:
            """
            Steepest greedy descent over the subspace is run
            sequentially on samples returned by the QPU.
            """
            T1 = time.time()
            sampleset = pp_method(**sampling_problem, initial_states=sampleset)
            if sp:
                print(f"** The postprocess time is: {(time.time() - T1):.3f} s")

        # Convert the sampling result to variable we supposed to solving.
        if self._rm == "lowest_obj":
            x = self._opt_obj(A, b, sampleset, B, non_linear_obj=non_linear_obj)
            # Select best sampling result from the top 20 lowest energy state.
        elif self._rm == "lowest_energy":
            # Select the lowest energy state as sampling result
            result = np.array(list(sampleset.first.sample.values()))
            x = (B @ result).reshape(-1, 1)  # transfer q to x
        else:
            raise NotImplementedError

        if sp:
            print("============================================")

        return x

    @check_dimension
    def it_solve(
        self,
        A,
        b,
        L=1.0,
        iter_num=10,
        problem_type="BQM",
        label="temp",
        alpha=None,
        drate=1.0,
        D=None,
        non_linear_obj=None,
        sp=True,
        save_results=False,
        show_obj=True,
        eps=1e-8,
    ):
        """
        Solving the linear system iteratively.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        L : float
            Scale factor, complement coding range.
        iter_num : int
            Maximum number of iteration.
        problem_type : string
            The problem type used during sampling which has three options:
            "BQM", "QUBO" and "Ising".
        label : string
            Making a description of current problem which is used to distinguish
            file name of embedding result.
        alpha : float
            The factor of Tikhonov regularization. If it equals to zero, no regularization
            method will be used.
        drate : float
            The proportion of alpha changes per iteration.
        D : array_like
            Finite difference operators for implementing higher-order Tikhonov
            regularization. Default is an identity operator.
        non_linear_obj : Callable[..., Any]
            The outer function used to decide the best sampling result. This method will
            take advantage of the diversity of sampling results so that the solution can
            satisfy the objective function that cannot be encoded as a quadratic form.
        sp : bool
            Whether to print the processes of this program. Default is True.
        save_results : bool (False)
            Save the middle results during the iteration.
        show_obj : bool
            Whether to show the object function curve.
        eps :  float
            Control the precision of the iteration process.

        Returns
        -------
        x : {(M, 1), (N, 1)}, array_like
            Solution of the linear system.

        """
        tmp_set = {}
        self._problem2qsampler(A, b, label=label)

        B0 = self._cal_B_matrix(A.shape[1])

        if D is None:
            D = np.eye(A.shape[1])

        if alpha is None:
            alpha = 0

        obj_set = []

        if problem_type == "BQM":
            construct_method = self._cal_BQM
            sample_method = self._sampler.sample
            pp_method = SteepestDescentSolver().sample

        elif problem_type == "QUBO":
            construct_method = self._cal_QUBO
            sample_method = self._sampler.sample_qubo
            pp_method = SteepestDescentSolver().sample_qubo

        elif problem_type == "ISING":
            construct_method = self._cal_ISING
            sample_method = self._sampler.sample_ising
            pp_method = SteepestDescentSolver().sample_ising

        else:
            NotImplementedError

        for it in range(iter_num):

            if sp:
                print("\n ============================================")
            B = L * B0
            # Calculate the sampling problem ('QUBO', 'BQM' or ''Ising)
            sampling_problem = construct_method(A, b, B=B, alpha=alpha, D=D)

            # Sampling
            T0 = time.time()
            if self._mode == "Simulated_Annealing":
                sampleset = sample_method(
                    **sampling_problem, num_reads=self._ns, large_sparse_opt=True
                )
            else:
                sampleset = sample_method(
                    **sampling_problem,
                    num_reads=self._ns,
                    num_spin_reversal_transforms=self._nsrt,
                )
                qpu_acct = sampleset.info["timing"]["qpu_access_time"]
                if sp:
                    print("* QPU access time: {} us".format(qpu_acct))
            if sp:
                print(
                    f"* The sampling time in {it + 1} iteration was: {(time.time() - T0):.2f} s"
                )

            if self._p_p == True:
                """
                Steepest greedy descent over the subspace is run
                sequentially on samples returned by the QPU.
                """
                T1 = time.time()
                sampleset = pp_method(**sampling_problem, initial_states=sampleset)
                if sp:
                    print(f"** The postprocess time is: {(time.time() - T1):.3f} s")

            # Convert the sampling result to variable we supposed to solving.
            if self._rm == "lowest_obj":
                _x = self._opt_obj(
                    A, b, sampleset, B0, non_linear_obj=non_linear_obj
                )  # Select best sampling result from the top 20 lowest energy state.
            elif self._rm == "lowest_energy":
                # Select the lowest energy state as sampling result
                result = np.array(list(sampleset.first.sample.values()))
                _x = (B0 @ result).reshape(-1, 1)  # transfer q to x
            else:
                raise NotImplementedError

            # using the idea of the steepest descent method to calculating the local optimal "L".
            A_x = A @ _x
            if alpha:
                D_x = D @ _x
                L = float(
                    (b.reshape(1, -1) @ A_x)
                    / (A_x.T @ A_x + alpha * D_x.reshape(1, -1) @ D_x)
                )

            else:
                L = float((b.reshape(1, -1) @ A_x) / (A_x.T @ A_x))

            if save_results:
                tmp_set[f"iter_{it + 1}"] = _x

            obj = np.linalg.norm(A @ _x - b)
            # Append latest object function value
            obj_set.append(obj)

            if sp:
                print(f"* Current Object Function is {obj}")

            if obj <= eps or it == iter_num - 1:
                x = np.copy(_x)
                if sp:
                    print("Satisfying the termination condition, stop the iteration.")
                if sp:
                    print("\n ============================================")
                # plot the objection curve after iteration.
                if show_obj:
                    show_obj_curve(obj_set)
                if save_results:
                    self.IPI = tmp_set  # Save the Intermediate process of iteration
                break
            else:
                x = np.copy(_x)
                if alpha:
                    alpha = alpha * drate
                if sp:
                    print(f"* The step size of current iteration:{L}")
        return x


class IterSampleSolver(DirectSampleSolver):
    """
    A solver that uses sampling methods to solve systems of linear equations.
    The solver uses an iterative method for solving, which is more accurate
    than direct solver.

    Parameters
    ----------
    x0 : array_like
        The initial guess for the solution of the system of linear equations.
    L : float
        The guess for the range of solution.
    maxiter : int, optional
        The maximum number of iterations(default=100).
    epsilon : float, optional
        The upper limit of solution accuracy(default=1e-5).
    show_obj_curve : bool, optional
        Whether to display the curve of the objective function's variation
        after iteration (default=True).
    """

    def __init__(self, x0, L=1, maxiter=100, epsilon=1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._RM = None
        self._L = L
        self._bak_L = L
        self._x0 = x0.reshape(-1, 1)
        self._bak_x0 = x0.reshape(-1, 1)
        self.maxiter = maxiter
        self.epsilon = epsilon
        self._sobj = kwargs.pop(
            "show_obj_curve", False
        )  # whether to show the curve of object function after iteration.
        self._best_x = None

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, new_x0):
        if not isinstance(new_x0, np.ndarray):
            raise TypeError(
                f"The dtype of bit value should be np.ndarray! \n"
                f"Not the {type(new_x0)}!"
            )
        else:
            self._x0 = new_x0

    @property
    def best_x(self):
        return self._best_x

    def reset(self):
        self._L = self._bak_L
        self._x0 = self._bak_x0
        self._best_x = None
        self.IPI = None

    def _cal_Q_const(self, A, b, **kwargs):
        """Calculating the Q matrix and const item. Q matrix is used to construct QUBO problem,
        Q and const can be used to construct BQM problem.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        B : {N, N*R}, array_like
            The encoding matrix, which transfer sampling result to real/complex vector.
        alpha : float
            The factor for Tikhonov regularization. If set to zero, no regularization
            method will be applied.
        D : array_like
            Finite difference operators for implementing higher-order Tikhonov
            regularization.
        Returns
        -------
        Q : {(M*R, N*R), (M*R, N*R)}, array_like
            Q matrix in dict form. R is the number of coding bit.
        const : float
            Constant term of object function.
        """
        alpha = kwargs.get("alpha", 0)

        ATA = A.T @ A
        if alpha:
            B = kwargs["B"]
            D = kwargs["D"]

            D_B = D @ B
            D_x0 = D @ self._x0
            Q = (ATA + alpha * D_B.T @ D_B) - 2 * np.diag(
                (b.reshape(1, -1) @ A - alpha * D_x0.reshape(1, -1) @ D_B).squeeze()
            )
            # const = b.reshape(1, -1) @ b.reshape(-1, 1) + alpha * D_x0.reshape(1, -1) @ D_x0
            Q = np.tril(Q) + np.tril(Q, -1)

        else:
            Q = ATA - 2 * np.diag((b.reshape(1, -1) @ A).squeeze())
            Q = np.tril(Q) + np.tril(Q, -1)
            # const = b.reshape(1, -1) @ b.reshape(-1, 1)

        return Q  # , const.squeeze()

    def _opt_obj(self, A, b, sampleset, B, non_linear_obj=None):
        """Find the optimal sample result which has the lowest object function.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        sampleset : dict or list
            The result sampled from D-wave sampler.
        non_linear_obj : func, optional
            These methods can only solve linear systems of equations,
            so for nonlinear problems, they can only approximate them as
            linear problems and then solve them. This can potentially result
            in suboptimal solutions for the nonlinear problem, where the
            optimal solution corresponds to a suboptimal solution for the
            linear problem. Providing a nonlinear objective function here
            may improve the solution outcome.

        Returns
        -------
        w : array_like
            The solution with the minimum objective function.
        """
        if non_linear_obj is None:

            def default_obj(x):
                return np.linalg.norm(A @ x - b) ** 2

        else:
            default_obj = non_linear_obj

        lstspst = sampleset.lowest(
            atol=10
        )  # lowest_sampleset atol --> absolute tolerance
        if len(lstspst) >= 20:
            lstspst = lstspst.truncate(20)
        obj = np.inf
        w = None
        for sample in lstspst.samples():
            samp_values = np.array((list(sample.values())))  # temper result
            tmp_w = (B @ samp_values).reshape(-1, 1)
            tmp_x = self._x0 + self._L * tmp_w
            tmp_obj = default_obj(tmp_x)
            if tmp_obj <= obj:
                obj = tmp_obj
                w = tmp_w
        return w

    @check_dimension
    def solve(
        self,
        A,
        b,
        label="temp",
        alpha=None,
        drate=1.0,
        D=None,
        non_linear_obj=None,
        sp=False,
        save_results=False,
    ):
        """
        Solving the linear system iteratively.

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        problem_type : string
            The problem type used during sampling which has three options:
            "BQM", "QUBO" and "Ising".
        label : string
            Making a description of current problem which is used to distinguish
            file name of embedding result.
        alpha : float
            The factor of Tikhonov regularization. If it equals to zero, no regularization
            method will be used.
        drate : float
            The proportion of alpha changes per iteration.
        D : array_like
            Finite difference operators for implementing higher-order Tikhonov
            regularization. Default is an identity operator.
        non_linear_obj : Callable, optional
            These methods can only solve linear systems of equations,
            so for nonlinear problems, they can only approximate them as
            linear problems and then solve them. This can potentially result
            in suboptimal solutions for the nonlinear problem, where the
            optimal solution corresponds to a suboptimal solution for the
            linear problem. Providing a nonlinear objective function here
            may improve the solution outcome.
        sp : bool
            Whether to print the processes of this program. Default is True.
        save_results : bool (False)
            Save the middle results during the iteration.

        Returns
        -------
        x : {(M, 1), (N, 1)}, array_like
            Solution of the linear system.

        """
        tmp_set = {}
        B = self._cal_B_matrix(A.shape[1])
        M = A @ B
        self._problem2qsampler(M, b, label=label, alpha=alpha)

        if D is None:
            D = np.eye(A.shape[1])

        if alpha is None:
            alpha = 0.0

        obj_set = []
        obj = np.linalg.norm(A @ self._x0 - b)

        construct_method = self._cal_QUBO
        sample_method = self._sampler.sample_qubo
        pp_method = SteepestDescentSolver().sample_qubo

        tmp_set["iter_0"] = self._x0.squeeze()
        best_obj = np.inf
        for it in range(self.maxiter):
            if sp:
                print("\n ============================================")

            d = b - A @ self._x0

            # Append latest object function value
            obj_set.append(obj)
            # Calculate the sampling problem ('QUBO', 'BQM' or ''Ising)
            sampling_problem = construct_method(
                A=self._L * M, b=d, B=self._L * B, alpha=alpha, D=D
            )

            # Sampling
            T0 = time.time()
            if self._mode == "Simulated_Annealing":
                sampleset = sample_method(
                    **sampling_problem, num_reads=self._ns, large_sparse_opt=True
                )
            else:
                sampleset = sample_method(
                    **sampling_problem,
                    num_reads=self._ns,
                    # num_spin_reversal_transforms=self._nsrt,
                )
                qpu_acct = sampleset.info["timing"]["qpu_access_time"]
                if sp:
                    print("* QPU access time: {} us".format(qpu_acct))
            if sp:
                print(
                    f"* The sampling time in {it + 1} iteration was: {(time.time() - T0):.2f} s"
                )

            if self._p_p == True:
                """
                Steepest greedy descent over the subspace is run
                sequentially on samples returned by the QPU.
                """
                T1 = time.time()
                sampleset = pp_method(**sampling_problem, initial_states=sampleset)
                if sp:
                    print(f"** The postprocess time is: {(time.time() - T1):.3f} s")

            # Convert the sampling result to variable we supposed to solving.
            if self._rm == "lowest_obj":
                _x = self._opt_obj(
                    A, b, sampleset, B, non_linear_obj=non_linear_obj
                )  # Select best sampling result from the top 20 lowest energy state.

            elif self._rm == "lowest_energy":
                # Select the lowest energy state as sampling result
                q_star = np.array(list(sampleset.first.sample.values()))
                _x = (B @ q_star).reshape(-1, 1)  # transfer q to x

            else:
                raise NotImplementedError

            if np.all(
                _x == 0
            ):  # if all the sampling result is equals to zero, stop the iteration.
                if sp:
                    print(
                        "All the sampling result is equals to zero, stop the iteration."
                    )
                if sp:
                    print("\n ============================================")
                if self._sobj:
                    show_obj_curve(obj_set)
                if save_results:
                    self.IPI = tmp_set  # Save the Intermediate process of iteration

                break

            # using the idea of the steepest descent method to calculating the local optimal "L".
            A_x = A @ _x
            if not alpha:
                L1 = float((d.reshape(1, -1) @ A_x) / (A_x.T @ A_x))
            else:
                D_x = D @ _x
                L1 = float(
                    (
                        d.reshape(1, -1) @ A_x
                        - alpha * (D @ self._x0).reshape(1, -1) @ D_x
                    )
                    / (A_x.T @ A_x + alpha * D_x.reshape(1, -1) @ D_x)
                )

            # updating the solution.
            x = self._x0 + L1 * _x

            if save_results:
                tmp_set[f"iter_{it + 1}"] = x.squeeze()

            obj = np.linalg.norm(A @ x - b)
            if sp:
                print(f"* Current Object Function is {obj}")

            if (
                obj <= self.epsilon
                or abs(obj_set[-1] - obj) / obj_set[-1] <= self.epsilon
                or it == self.maxiter - 1
            ):
                self._x0 = x

                if obj <= best_obj:
                    best_obj = obj
                    self._best_x = self._x0

                if sp:
                    print("Satisfying the termination condition, stop the iteration.")
                if sp:
                    print(f"The lowest obj funciton is {best_obj}")
                if sp:
                    print("\n ============================================")
                # plot the objection curve after iteration.
                if self._sobj:
                    show_obj_curve(obj_set)
                if save_results:
                    self.IPI = tmp_set  # Save the Intermediate process of iteration
                break
            else:
                self._x0 = x

                if it > 0:
                    if obj <= best_obj:
                        best_obj = obj
                        self._best_x = self._x0
                else:
                    self._best_x = self._x0

                self._L = L1
                alpha = alpha * drate
                if sp:
                    print(f"* The step size of current iteration:{self._L}")

        return self._x0

    @check_dimension
    def original_QUBO_solve(self, A, b, steplen=0.75, non_linear_obj=None):
        """
        Original iterative method introduced in (Souza et al., 2022).

        Parameters
        ----------
        A : {(M, M), (M, N)}, array_like
            Coefficient matrix of linear system.
        b : (M, 1), array_like
            Right hand side of linear system.
        steplen : float
            A variable that regulates the fluctuation of L.
        non_linear_obj : func, optional
            These methods can only solve linear systems of equations,
            so for nonlinear problems, they can only approximate them as
            linear problems and then solve them. This can potentially result
            in suboptimal solutions for the nonlinear problem, where the
            optimal solution corresponds to a suboptimal solution for the
            linear problem. Providing a nonlinear objective function here
            may improve the solution outcome.

        Returns
        -------
        x : {(M, 1), (N, 1)}, array_like
            Solution of the linear system.
        """
        I = np.ones(A.shape[1]).reshape(-1, 1)
        obj_set = []
        obj = np.linalg.norm(A @ self._x0 - b)
        for it in range(self.maxiter):
            print("============================================")
            # Append latest object function value
            obj_set.append(obj)
            # Calculate the Q Matrix for QUBO problem
            m = A.shape[1]
            B = self._cal_B_matrix(m)
            bb = (b + self._L * A @ I - A @ self._x0) / self._L  # 右边项
            A_disc = A @ B  # 相当于M与bit_value做Kronecker积
            AA_disc = A_disc.T @ A_disc
            Q = AA_disc - 2 * np.diag((bb.reshape(1, -1) @ A_disc).squeeze())
            Q = np.tril(Q) + np.tril(Q, -1)
            # const = (bb.reshape(1, -1) @ bb.reshape(-1, 1)).squeeze()
            # Sampling
            T0 = time.time()
            sampleset = self._sampler.sample_qubo(Q, num_reads=self._ns)
            print(f"* The {it + 1} sampling run time is: {(time.time() - T0):.2f} s")

            if self._rm == "lowest_obj":
                _x = self._opt_obj(
                    A, b, sampleset, non_linear_obj
                )  # Select best sampling result from the top 20 lowest energy state.
            elif self._rm == "lowest_energy":
                result = np.array(list(sampleset.first.sample.values()))
                _x = to_fixed_point(result, self._bv).reshape(-1, 1)  # transfer q to x
            else:
                raise NotImplementedError

            # updating the solution.
            x = self._x0 + self._L * (_x - I)
            obj = np.linalg.norm(A @ x - b)

            print(f"* Current Object Function is {obj}")

            if (
                obj <= self.epsilon
                or np.linalg.norm((x - self._x0) / x) <= self.epsilon
                or it == self.maxiter - 1
            ):
                self._x0 = x
                print("* Satisfying the termination condition, stop the iteration.")
                print("============================================")
                # plot the objection curve after iteration.
                if self._sobj:
                    show_obj_curve(obj_set)
                break
            else:
                self._x0 = x
                self._L = self._L * steplen
                print(f"* The step size of current iteration:{self._L}")
        return self._x0
