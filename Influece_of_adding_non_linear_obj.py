# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/6/20 16:58
# @Author  : Wen Xixiang
# @Version : python3.9

import sys
import numpy as np
from pyQS import stagger_FD_coeff as sfci
from pyQS.utils import Logger
from neal import SimulatedAnnealingSampler
# from dwave.system import DWaveSampler
from pyQS.QUBO_solvers_bak_newest import IterSampleSolver
import time
# 一样的参数， 分别跑10次，看看有没有区别！
# 将控制台的结果输出到a.log文件，可以改成a.txt, 注意要在终端运行，否则可能不会保存计算结果
# sys.stdout = Logger(f"Inf_add_non_linear.txt", sys.stdout)
sys.stdout = Logger(f"Inf_add_non_linear_{time.strftime('%Y%m%d%H%M', time.localtime())}.txt", sys.stdout)


pi = np.pi

v = 4500
h = 10
tao = 0.001
freq_max_set = [174]  # [40, 52, 58] 174 # 地震波频率中的最大频率成分 M=4 : 40, M=6 : 52, M=8 : 58
M_set = [8]  # [4,6,8]
sample_num = 100
max_iter_num = 20
max_iter4_solver = 10
epsilon = 1e-5

print(f'v={v}')
print(f'freq={174}')
print(f'M={8}')
print(f'sample time = {sample_num}')
print(f'max_iter_num = {max_iter_num}')
print(f'max_iter_num_in_iterative_solver = {max_iter4_solver}')
print('b_alpha = 1*0.1*it change in big loop')  # alpha = 1 * 0.8 ** it
print('s_alpha = b_alpha change in small loop')

sampler = SimulatedAnnealingSampler()
for it_num in range(50):
    print(f" >>> Start the {it_num} experiment.")
    qa_a0_nonreg_dict = {}
    for M, freq_max in zip(M_set, freq_max_set):
        linear_sys = sfci.Stagger_FD_coeff_1D(v, h, tao, freq_max)
        sis = IterSampleSolver(sampler=sampler,
                               x0=np.ones(M) / 100,  # initial vector of Delta a
                               num_sample=sample_num,
                               R=10,
                               fixed_point=1,
                               epsilon=1e-5,
                               maxiter=max_iter4_solver,  # max iteration in small iteration
                               resultMod='lowest_energy',
                               regularization=False,
                               # num_spin_reversal_transforms=10,
                               # postprocess=True
                               )
        qa_a0 = linear_sys.solve(M=M,
                                 beta=1,
                                 max_iter_num=max_iter_num,
                                 mode_of_construct='direct',
                                 solver=sis.solve,
                                 epsilon=epsilon,
                                 show_process=False)
        qa_a0_nonreg_dict[f'original'] = qa_a0

    qa_a0_reg_dict = {}
    for M, freq_max in zip(M_set, freq_max_set):
        linear_sys = sfci.Stagger_FD_coeff_1D(v, h, tao, freq_max)
        sis = IterSampleSolver(sampler=sampler,
                               x0=np.ones(M) / 100,  # initial vector of Delta a
                               num_sample=sample_num,
                               R=10,
                               fixed_point=1,
                               epsilon=1e-5,
                               maxiter=max_iter4_solver,  # max iteration in small iteration
                               resultMod='lowest_obj',
                               regularization=False,  # alpha=1*0.1**it Which is same as reg-lstsq method
                               # num_spin_reversal_transforms=10,
                               # postprocess=True
                               )
        qa_a0 = linear_sys.solve(M=M,
                                 beta=1,
                                 max_iter_num=max_iter_num,
                                 mode_of_construct='direct',
                                 epsilon=epsilon,
                                 solver=sis.solve,
                                 show_process=False,
                                 add_non_linear_obj=True
                                 )
        qa_a0_reg_dict[f'add_non_linear_obj'] = qa_a0

    print(f" <<< End the {it_num} experiment.")
