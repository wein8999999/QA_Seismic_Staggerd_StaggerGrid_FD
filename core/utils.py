# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/3/13 17:38
# @Author  : Wen Xixiang
# @Version : python3.9
import numpy as np
import scipy as sp
import networkx as nx
import sys
from typing import Any, Literal, Tuple, Union, List, Iterable
from scipy.signal import fftconvolve
import warnings

from matplotlib import pyplot as plt


def cholesky_solver(A, b):
    cho, low = sp.linalg.cho_factor(A)
    solution = sp.linalg.cho_solve((cho, low), b)
    return solution


# 使用SVD分解法求解（奇异矩阵将去掉其零空间）：
def SVD_solver(Matrix, b):
    U, A, Vh = np.linalg.svd(Matrix, hermitian=True)
    if A.shape[0] != np.linalg.matrix_rank(A):  # 若不满秩，则需要去除零空间
        index = A != 0
        # 对角阵要删除所有零行和零列
        A = A[index]

        # U要去除对应的列
        U = U[:, index]

        # Vh要去除对应的行
        Vh = Vh[index, :]
    Matrix_inv = Vh.T @ np.linalg.inv(np.diag(A)) @ U.T  # 求广义逆
    solution = Matrix_inv @ b
    return solution


def steepest_decent_solver(Q, b, x0=None, miter_num=1000):
    def obj_func(coeff_delta_vec):
        # 二次型目标函数
        return (coeff_delta_vec.T @ Q @ coeff_delta_vec - 2 * b.T @ coeff_delta_vec).squeeze()

    def obj_func_diff(coeff_delta_vec):
        # 目标函数对于x的导数
        return 2 * Q @ coeff_delta_vec - 2 * b

    M = Q.shape[0]
    if x0 is None:
        x0 = np.ones(M).reshape(-1, 1) / 100

    for iter_num in range(miter_num):
        obj_diff_k = obj_func_diff(x0)
        # print(f"当前小循环中目标函数值(与delta_a相关)为：{obj_func(x0)}")
        step = (obj_diff_k.T @ obj_diff_k).squeeze() / (obj_diff_k.T @ Q @ obj_diff_k).squeeze()
        # step = 8.8 # 步长因在0到2/(\lambda_max(Q))之间 2/(np.linalg.eigvals(Q).max())
        x1 = x0 - 0.3 * step * obj_diff_k  #
        x0 = x1

    print(f"当前小循环中目标函数值(与delta_a相关)为：{obj_func(x0)}")
    return x0


def to_fixed_point(binary: np.ndarray, bit_value: np.ndarray) -> np.ndarray:
    """ Convert a binary array to a floating-point array represented by bit values.

    在得到采样后的二进制比特串时, 通过使用此函数将二进制比特串转化为十进制数,
    前提是需要知道每一位比特对应的值，即函数中的bit_value. 需要注意的是bit_value的
    长度等于编码每一个十进制数所使用比特数量(也就是R). binary的长度为R*m, m为待求参
    数的个数.

    Parameters
    ----------
    binary : np.ndarray
        The bit string obtained after sampling.
    bit_value : np.ndarray
        An array consisted by the value of each bit.

    Examples
    --------
    >>> binary = np.array([1,0,1,0,0,1,0,1,0,1,1,0,1,1,1])
    >>> bit_value = np.array([2,1,0.5,0.25,0.125])
    >>> to_fixed_point(binary,bit_value)
    >>> array([2.5  , 2.625, 2.875])
    """

    if not isinstance(binary, np.ndarray):
        raise TypeError(f"`binary` must be the instance of np.ndarray, not {type(binary)}")
    if not isinstance(bit_value, np.ndarray):
        raise TypeError(f"`bit_value` must be the instance of np.ndarray, not {type(bit_value)}")

    binary, bit_value = binary.flatten(), bit_value.flatten()

    # If sampling model is ISING, change the {-1,1} to {0,1}
    # binary = np.array(binary + np.abs(binary)) / 2

    num_binary_entry = len(binary)
    num_bits = len(bit_value)
    num_x_entry = num_binary_entry // num_bits
    if num_x_entry * num_bits != num_binary_entry:
        raise ValueError("The length of q or bit_value is incorrect.")
    float_array = np.array([bit_value @ binary[i * num_bits: (i + 1) * num_bits] for i in range(num_x_entry)])
    return float_array.reshape(-1, 1)


def b2f(binary: np.ndarray,
        low_value: Union[int, float],
        high_value: Union[int, float]) -> np.ndarray:
    """Convert a binary array to a floating-point array represented by two values.

    Parameters
    ----------
    binary : np.ndarray
        Binary array.
    low_value : Union[int, float]
        Low value.
    high_value : Union[int, float]
        High value.

    Returns
    -------
    float_array : np.ndarray
        Floating-point array.
    """

    if low_value > high_value:
        low_value, high_value = high_value, low_value
    elif low_value == high_value:
        warnings.warn("`low_value` and `high_value` are the same.")

    # If sampling model is ISING, change the {-1,1} to {0,1}
    binary = np.array(binary + np.abs(binary)) / 2
    float_array = np.array(binary.flatten() * high_value - (binary - 1) * low_value)

    return float_array


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Flush both streams to ensure all content is written
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # Close the log file when done
        self.log.close()


def find_long_simple_path(chimera_graph):
    """
    Finding a long simple path in a chimera_graph. Detail discription can be found
    in O'Malley's article (2018, doi: 10.1038/s41598-018-25206-0).
    """

    rows = chimera_graph.graph['rows']
    columns = chimera_graph.graph['columns']
    nodes_list = sorted(list(chimera_graph.nodes))
    unit_cells = {}  # save each unit cell and its containing nodes

    for i in range(rows * columns):
        nl_array = np.array(nodes_list, dtype=int)
        temp = nl_array[np.nonzero(nl_array > i * 8 - 1)]
        unit_nodes = temp[np.nonzero(temp <= (i + 1) * 8 - 1)].tolist()
        unit_cells[i] = unit_nodes

    # The process of finding the longest path is in the "S" order.
    right_order = np.array(list(unit_cells.keys())).reshape(rows, columns)
    right_order[1::2] = np.flip(right_order[1::2], axis=1)
    right_order = right_order.flatten()

    def delete_repeat_nodes(old_section, pre_longest_path):
        new_section = [pre_longest_path[-1]]
        for item in old_section:
            if item not in pre_longest_path:
                new_section.append(item)
        return new_section

    # 将上边的图分为两个unit为一组的节
    total_longest_path = []
    for i in range(rows * columns):

        # Finding the longest path in sub-graph.
        if i == 0:
            section = unit_cells[right_order[i]] + unit_cells[right_order[i + 1]]
            sub_chimera_graph = nx.subgraph(chimera_graph, section)
            source = section[0]
            target = section[-1]
        elif i == rows * columns - 1:
            section = unit_cells[right_order[i]]
            source = section[0]
            target = section[-1]
        else:
            section = unit_cells[right_order[i]] + unit_cells[right_order[i + 1]]
            section = delete_repeat_nodes(section, section_longest_path)
            sub_chimera_graph = nx.subgraph(chimera_graph, section)
            source = target
            # 如果寻找路径过程失败了， 回来改改这里的"-3"
            target = unit_cells[right_order[i + 1]][-3]  # If failed to finding the path, come here and change the "-3"
        print(f"source={source}, target={target}")
        section_longest_path = []
        for path in nx.all_simple_paths(sub_chimera_graph, source=source, target=target):
            if len(section_longest_path) < len(path):
                section_longest_path = path

        print(f'i = {i}, the longest path of this section is:', section_longest_path)
        total_longest_path += section_longest_path[1:]

    # print(f"The total number of nodes is {len(chimera_graph.nodes)}, our longest path contain {len(total_longest_path)} nodes")
    # To get each path as the corresponding list of edges, you can use the networkx.utils.pairwise() helper function:
    # corr_edges = list(nx.utils.pairwise(total_longest_path))
    # print(corr_edges)

    return total_longest_path


def show_obj_curve(obj_set):
    fig, ax = plt.subplots(1, 1, figsize=(5, 7))
    ax.plot([i for i in obj_set[1:]], 'o-', color='crimson')
    ax.set_ylabel('Misfit Function')
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
    ax.tick_params(direction='in', length=3, width=1)
    # ax.legend(loc=1, fontsize='small')
    plt.pause(3.)
    plt.close(fig)


def Taylor_Coef_forward(M=4):
    """
    使用泰勒展开法求解有限差分的系数, 使用向前差商, 即dx = (x_i+1 - x_i) / delta

    Parameters
    ----------
    M : int
        The length of stagger grid finite difference operator.

    Returns
    -------
    coeff_vec : array_like
        The coefficient vector of staggered grid finite difference.
    """

    # Order = M*2 # 阶数为M的二倍
    A = np.zeros([M, M])
    b = np.zeros([M, 1])
    for i in range(M):
        b[i] = 1 if i == 0 else 0
        for j in range(M):
            A[i, j] = (j + 1) ** (i + 1)

    lu, piv = sp.linalg.lu_factor(A)
    x = sp.linalg.lu_solve((lu, piv), b)
    return x


def Taylor_Coef_center(M=4):
    """
    使用泰勒展开法求解有限差分的系数, 使用中心差商

    Parameters
    ----------
    M : int
        The length of stagger grid finite difference operator.

    Returns
    -------
    coeff_vec : array_like
        The coefficient vector of staggered grid finite difference.
    """

    # Order = M*2 # 阶数为M的二倍
    A = np.zeros([M, M])
    b = np.zeros([M, 1])
    for i in range(M):
        b[i] = 1 if i == 0 else 0
        for j in range(M):
            A[i, j] = (j + 1) ** (2 * (i + 1))

    lu, piv = sp.linalg.lu_factor(A)
    x = sp.linalg.lu_solve((lu, piv), b)
    return x

def construct_LD_1D_forward(N, M, D):
    """
    N : number of model
    M : 近似一阶微分的差分模板的长度, 使用前向差分，有限差分的精度为M
    构建1维模型的D阶吉洪诺夫正则化算子
    D : 微分算子的阶数
    """
    if D == 0:
        LD = np.eye(N)
    else:
        coeff = Taylor_Coef_forward(M)  # 一阶有限差分格式的差分系数
        element = np.vstack([-np.sum(coeff), coeff])  # 一阶有限差分矩阵中每行的系数
        D_element = np.array([[1]])
        for i in range(D):
            D_element = fftconvolve(D_element, element)
        D_element = D_element[1:]
        LD = np.zeros([N, N])
        for i in range(N):
            if i < N - (D * M):
                LD[i, i+1:i + D * M + 1] = D_element.squeeze()
            else:
                tmp = i - N + (D * M) + 1
                LD[i, i+1:i + D * M + 1 - tmp] = D_element[:-tmp].squeeze()
        for i in range(N):
            LD[i, i] = -np.sum(LD[i, :])

    return LD

def construct_LD_1D_forward_ex(N, M, D):
    """
    测试下来我发现
    N : number of model
    M : 近似一阶微分的差分模板的长度, 有限差分的精度为M, 删除差分模板超模的部分
    D : 微分算子的阶数
    """
    if D == 0:
        LD = np.eye(N)
    else:
        coeff = Taylor_Coef_forward(M)  # 一阶有限差分格式的差分系数
        element = np.vstack([-np.sum(coeff), coeff])  # 一阶有限差分矩阵中每行的系数
        D_element = np.array([[1]])
        for i in range(D):
            D_element = fftconvolve(D_element, element)
        LD = np.zeros([N - (D * M), N])
        for i in range(N - (D * M)):
            LD[i, i:i + D * M + 1] = D_element.squeeze()
    return LD

def construct_LD_1D_center(N, M, D):
    """
    适合于正则化的差分矩阵的求取方法. 使用中心差分

    N1,2 : 模型的x方向, 模型的y方向
    M : 近似一阶微分的差分模板的长度, 有限差分的精度为2*M, 具体内容在胡光辉老师的全波形反演书上
    构建1维模型的D阶吉洪诺夫正则化算子
    D : 微分算子的阶数, 注意这种情况下的微分阶数为2*D
    """
    if D == 0:
        LD = np.eye(N)
    else:
        coeff = Taylor_Coef_center(M)  # 一阶有限差分格式的差分系数
        element = np.vstack([coeff[::-1], -2 * np.sum(coeff), coeff])  # 一阶有限差分矩阵中每行的系数
        D_element = np.array([[1]])
        for i in range(D):
            D_element = fftconvolve(D_element, element)
        D_element[D * M] = 0.  # 将中间的点置为0, 方便之后求取合适的中心点值

        # 卷积之后的差分算子长度应该是 2*DM+1
        LD = np.zeros([N, N])
        # 构建粗糙化矩阵
        for i in range(N):
            # 如果可能，添加非中心元素
            if i > 0:
                if i <= D * M:
                    tmp1 = D * M - 1 - i
                    tmp1 = None if tmp1 < 0 else tmp1
                    LD[i, i - 1::-1] = D_element[D * M - 1:tmp1:-1].squeeze()
                else:
                    LD[i, i - 1:i - D * M - 1:-1] = D_element[D * M - 1::-1].squeeze()
            if i < N:
                if i >= N - D * M:
                    tmp11 = i - (N- D*M) + 1
                    LD[i, i + 1:i + 1 + D * M] = D_element[D * M + 1:-tmp11].squeeze()
                else:
                    LD[i, i + 1:i + 1 + D * M] = D_element[D * M + 1:].squeeze()

        for i in range(N):
            LD[i, i] = -np.sum(LD[i, :])
    return LD

def construct_LD_2D_center(N1, N2, M, D):
    """
    适合于正则化的差分矩阵的求取方法. 使用中心差分, 原始版本见useful_tookits.py

    N1,2 : 模型的x方向, 模型的y方向
    M : 近似一阶微分的差分模板的长度, 有限差分的精度为2*M, 具体内容在胡光辉老师的全波形反演书上
    构建1维模型的D阶吉洪诺夫正则化算子
    D : 微分算子的阶数, 注意这种情况下的微分阶数为2*D
    """
    if D == 0:
        LD = np.eye(N1 * N2)
    else:
        coeff = Taylor_Coef_center(M)  # 一阶有限差分格式的差分系数
        element = np.vstack([coeff[::-1], -2 * np.sum(coeff), coeff])  # 一阶有限差分矩阵中每行的系数
        D_element = np.array([[1]])
        for i in range(D):
            D_element = fftconvolve(D_element, element)
        D_element[D * M] = 0.  # 将中间的点置为0, 方便之后求取合适的中心点值

        # 卷积之后的差分算子长度应该是 2*DM+1
        LD = np.zeros([N1 * N2, N1 * N2])
        k = 0
        # 构建粗糙化矩阵
        for i in range(N1):
            for j in range(N2):
                Matrix = np.zeros([N1, N2])

                # 如果可能，添加非中心元素
                if i > 0:
                    if i <= D * M:
                        tmp1 = D * M - 1 - i
                        tmp1 = None if tmp1 < 0 else tmp1
                        Matrix[i - 1::-1, j] = D_element[D * M - 1:tmp1:-1].squeeze()
                    else:
                        Matrix[i - 1:i - D * M - 1:-1, j] = D_element[D * M - 1::-1].squeeze()
                if j > 0:
                    if j <= D * M:
                        tmp2 = D * M - 1 - j
                        tmp2 = None if tmp2 < 0 else tmp2
                        Matrix[i, j - 1::-1] = D_element[D * M - 1:tmp2:-1].squeeze()
                    else:
                        Matrix[i, j - 1:j - D * M - 1:-1] = D_element[D * M - 1::-1].squeeze()
                if i < N1:
                    if i >= N1 - D * M:
                        tmp11 = i - (N1- D*M) + 1
                        Matrix[i + 1:i + 1 + D * M, j] = D_element[D * M + 1:-tmp11].squeeze()
                    else:
                        Matrix[i + 1:i + 1 + D * M, j] = D_element[D * M + 1:].squeeze()
                if j < N2:
                    if j >= N2 - D * M:
                        tmp22 = j - (N2- D*M) + 1
                        Matrix[i, j + 1:j + 1 + D * M] = D_element[D * M + 1:-tmp22].squeeze()
                    else:
                        Matrix[i, j + 1:j + 1 + D * M] = D_element[D * M + 1:].squeeze()

                # 将 mtmp 整形并放入 L 的相应行

                LD[k, :] = Matrix.reshape([N1 * N2, 1]).squeeze()
                k = k + 1

        for i in range(N1 * N2):
            LD[i, i] = -np.sum(LD[i, :])
    return LD


def construct_LD_2D_forward(N1, N2, M, D):
    """
    适合于正则化的差分矩阵的求取方法. 使用前向差分, 原始版本见useful_tookits.py

    N1,2 : 模型的x方向, 模型的y方向
    M : 近似一阶微分的差分模板的长度, 有限差分的精度为M, 具体内容在胡光辉老师的全波形反演书上
    构建1维模型的D阶吉洪诺夫正则化算子
    D : 微分算子的阶数
    """
    if D == 0:
        LD = np.eye(N1 * N2)
    else:
        coeff = Taylor_Coef_forward(M)  # 一阶有限差分格式的差分系数
        element = np.vstack([-np.sum(coeff), coeff])  # 一阶有限差分矩阵中每行的系数
        D_element = np.array([[1]])
        for i in range(D):
            D_element = fftconvolve(D_element, element)
        D_element = D_element[1:]
        LD = np.zeros([N1 * N2, N1 * N2])
        k = 0
        for i in range(0, N1):
            for j in range(0, N2):

                Matrix = np.zeros([N1, N2])

                if i + D * M <= N1 - 1:
                    Matrix[i + 1:i + D * M + 1, j] = D_element.squeeze()
                else:
                    tmp1 = i + D * M - N1 + 1
                    Matrix[i + 1:i + D * M + 1 - tmp1, j] = D_element[:-tmp1].squeeze()
                if j + D * M <= N2 - 1:
                    Matrix[i, j + 1:j + D * M + 1] = D_element.squeeze()
                else:
                    tmp2 = j + D * M - N2 + 1
                    Matrix[i, j + 1:j + D * M + 1 - tmp2] = D_element[:-tmp2].squeeze()

                LD[k, :] = Matrix.reshape([N1 * N2, 1]).squeeze()
                k = k + 1

        for i in range(N1 * N2):
            LD[i, i] = -np.sum(LD[i, :])
    return LD


def construct_LD_2D_ex(N1, N2, M, D):
    """
    求取一个2D模型函数的微分, 尺寸为N1*N2, 使用中心点周围M个点来近似该点的一阶微分, 更高阶的微分可以通过卷积来计算.
    此函数会去除超出模型边界的部分, 求得的有限差分算子的L^T@L可能会是奇异的. 若要设计为吉洪诺夫正则化的粗化矩阵, 还需要进行一些改进
    简单来说, 就是保留边在边界以内的不完全差分算子部分, 并且让中心点为其余点之和的复数.

    N1,2 : 模型的x方向, 模型的y方向
    M : 近似一阶微分的差分模板的长度, 由于使用前向差分, 有限差分的精度为M
    D : 微分算子的阶数
    """
    if D == 0:
        LD = np.eye(N1 * N2)
    else:
        coeff = Taylor_Coef_forward(M)  # 一阶有限差分格式的差分系数
        element = np.vstack([-np.sum(coeff), coeff])  # 一阶有限差分矩阵中每行的系数
        D_element = np.array([[1]])
        for i in range(D):
            D_element = fftconvolve(D_element, element)
        row_num_LD = (N1 - D * M) * (N2 - D * M)
        LD = np.zeros([row_num_LD, N1 * N2])
        k = 0
        for i in range(0, N1 - D * M):
            for j in range(0, N2 - D * M):
                Matrix = np.zeros([N1, N2])
                Matrix[i, j:j + D * M + 1] = D_element.squeeze()
                Matrix[i:i + D * M + 1, j] += D_element.squeeze()
                LD[k, :] = Matrix.reshape([N1 * N2, 1]).squeeze()
                k = k + 1
    return LD


def Laplace_operator(nx, nz):
    """计算一个二阶精度的基于中心差分模板的有限差分粗化矩阵,'ex'版本更适合对数据做微分"""
    # 假设 n 和 PSCALE 已经定义
    n = nx * nz
    # PSCALE = ...

    # 初始化粗糙化矩阵 L
    L = np.zeros([n, n])
    k = 0  # Python 中索引从 0 开始

    # 构建粗糙化矩阵
    for i in range(nx):
        for j in range(nz):
            mtmp = np.zeros([nx, nz])

            # 如果可能，添加非中心元素
            if i > 0:
                mtmp[i - 1, j] = 1
            if j > 0:
                mtmp[i, j - 1] = 1
            if i < nx - 1:
                mtmp[i + 1, j] = 1
            if j < nz - 1:
                mtmp[i, j + 1] = 1
            # 将 mtmp 整形并放入 L 的相应行
            L[k, :] = mtmp.flatten()
            # L[k, :] = mtmp.flatten()
            k += 1

    # 设置 L 的对角元素值，考虑边缘和角落
    for i in range(n):
        L[i, i] = -np.sum(L[i, :])

    return L


def Laplace_operator_ex(nx, nz):
    """与PEIP第四章的差分算子对应, 采用M=1, 精度为2阶的中心差分模板,
    截去超模的部分
    """

    # 假设 n 和 PSCALE 已经定义
    n = nx * nz

    # 初始化粗糙化矩阵 L
    L = np.zeros([(nx - 2) * (nz - 2), n])
    k = 0  # Python 中索引从 0 开始

    # 构建粗糙化矩阵
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            mtmp = np.zeros([nx, nz])

            # 如果可能，添加非中心元素
            mtmp[i - 1, j] = 1
            mtmp[i, j - 1] = 1
            mtmp[i + 1, j] = 1
            mtmp[i, j + 1] = 1

            mtmp[i, j] = -4
            # 将 mtmp 整形并放入 L 的相应行
            L[k, :] = mtmp.flatten()
            # L[k, :] = mtmp.flatten()
            k += 1

    return L

def find_the_lowest_point(data_res_norm, model_norm, show_L_curve=True):
    """
    即使得norm+data_res最小的点
    """
    data_res = data_res_norm/data_res_norm.max()
    model = model_norm/model_norm.max()
    mesh = data_res + model
    index = np.argmin(np.abs(mesh))

    if show_L_curve is True:
        plt.figure(figsize=(5,5))
        plt.plot(data_res_norm, model_norm, color='black')
        plt.plot(data_res_norm[index], model_norm[index], 'o')
        plt.xlabel(r'$\left \| Gm-d \right \|_{2}$', math_fontfamily='stix')
        plt.ylabel(r"$\left \| m \right \|_{2}$", math_fontfamily='stix')
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('L-curve_for_test.png', dpi=600, format='png')
        plt.show()

    return index

def get_optimal_reg_factor(G, d, npoints=100, L=None, alpha_range=None, show_L_curve=False):
    """
    给出的正则化参数范围能更好地确定L曲线的拐点
    当程序出错的时候可以试试把注释部分去掉, 因为我认为它们可能不会用到
    L : 是有限差分算子, 用于高阶Tikhonov正则化
    """
    # calculate the norms eta=||m|| and rho=||Gm-d||, 借助svd的版本化简了一下参考PEIP书96页
    eta = np.zeros([npoints, 1])
    rho = np.copy(eta)
    _, n = G.shape
    if L is None:
        L = np.eye(n)
        LTL = L
    else:
        LTL = L.T @ L
    GTG = G.T @ G
    # 注意只有0阶Tikhonov正则化才能用这个方法

    if alpha_range is None:
        # 如果不提供 alpha的范围, 默认总1e-5到1e5取值
        reg_param = np.logspace(-5, 5, npoints)
    else:
        reg_param = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), npoints)

    for i in range(npoints):
        m_alpha = np.linalg.inv(GTG + reg_param[i] * LTL) @ G.T @ d
        eta[i] = np.linalg.norm(m_alpha)
        rho[i] = np.linalg.norm(G @ m_alpha - d)

    lowest_point = find_the_lowest_point(rho, eta, show_L_curve=show_L_curve)
    best_alpha = reg_param[lowest_point]

    return best_alpha
