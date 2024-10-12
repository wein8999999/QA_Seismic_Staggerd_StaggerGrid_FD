# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/6/20 16:58
# @Author  : Wen Xixiang
# @Version : python3.9

# 讨论正则化对于问题求解的影响
# 通过王老师的文章我了解到，加入正则化技术的方法比最小二乘法求解过程更加稳定，那正则化技术用在
# 量子退火过程中会不会有同样的效果呢？

import sys
import numpy as np
from pyQS import stagger_FD_coeff as sfci
from pyQS.utils import Logger
from neal import SimulatedAnnealingSampler
# from dwave.system import DWaveSampler
import time
from pyQS.QUBO_solvers_bak_newest import IterSampleSolver

# 一样的参数， 分别跑10次，看看有没有区别！
# 将控制台的结果输出到a.log文件，可以改成a.txt, 注意要在终端运行，否则可能不会保存计算结果
# sys.stdout = Logger(f"Inf_reg_new.txt", sys.stdout)
sys.stdout = Logger(f"Inf_reg_{time.strftime('%Y%m%d%H%M', time.localtime())}.txt", sys.stdout)

# 注意在这一部分实验的时候, 将迭代的约束修改得更加严格, 保证每次求解都可以达到20次迭代

pi = np.pi

v = 4500
h = 10
tao = 0.001
freq_max_set = [174]  # [40, 52, 58] 174 # 地震波频率中的最大频率成分 M=4 : 40, M=6 : 52, M=8 : 58
M_set = [8]  # [4,6,8]
sample_num = 100
max_iter_num = 20
max_iter4_solver = 5
# epsilon = 1e-5
print(f'v={v}')
print(f'freq={174}')
print(f'M={8}')
print(f'sample time = {sample_num}')
print(f'max_iter_num = {max_iter_num}')
print(f'max_iter_num_in_iterative_solver = {max_iter4_solver}')
print('b_alpha = 1*0.1*it change in big loop')  # alpha = 1 * 0.8 ** it
print('s_alpha = b_alpha change in small loop')

# 定义量子退火器
# sampler = DWaveSampler(token='DEV-c0791d9680a5ed42fbf4cb52d391e2f1045ee6bc',
#                        solver='Advantage_system6.2',
#                        region='na-west-1')
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
                                 epsilon=1e-15,
                                 show_process=False)
        qa_a0_nonreg_dict[f'Non_reg'] = qa_a0

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
                               resultMod='lowest_energy',
                               regularization=True,  # alpha=1*0.1**it Which is same as reg-lstsq method
                               # num_spin_reversal_transforms=10,
                               # postprocess=True
                               )
        qa_a0 = linear_sys.solve(M=M,
                                 alpha=1,
                                 beta=1,
                                 max_iter_num=max_iter_num,
                                 mode_of_construct='direct',
                                 epsilon=1e-15,
                                 solver=sis.solve,
                                 show_process=False)
        qa_a0_reg_dict[f'Reg'] = qa_a0

    print(f" <<< End the {it_num} experiment.")
