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
from pyQS.QUBO_solvers_bak_newest import IterSampleSolver

# 一样的参数， 分别跑10次，看看有没有区别！
# 将控制台的结果输出到a.log文件，可以改成a.txt, 注意要在终端运行，否则可能不会保存计算结果
sys.stdout = Logger(f"Inf_new_Old.txt", sys.stdout)
# sys.stdout = Logger(f"New_and_Old_{time.strftime('%Y%m%d%H%M', time.localtime())}.txt", sys.stdout)

pi = np.pi

v = 4500
h = 10
tao = 0.001
freq_max_set = [174]  # [40, 52, 58] 174 # 地震波频率中的最大频率成分 M=4 : 40, M=6 : 52, M=8 : 58
M_set = [8]  # [4,6,8]
sample_num = 100  # 采样次数
max_iter_num = 20
max_iter4_solver = 5
epsilon = 1e-5
print(f'v={v}')
print(f'freq={174}')
print(f'M={8}')
print(f'sample time = {sample_num}')
print(f'max_iter_num = {max_iter_num}')
print(f'max_iter_num_in_iterative_solver = {max_iter4_solver}')
print(f'epsilon = {epsilon}')
print('non-regularization!')
print('L varies between 0.0001 and 100')
step_L_list = np.linspace(0.0001, 100, 51)
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
                               epsilon=epsilon,
                               maxiter=max_iter4_solver,  # max iteration in small iteration
                               resultMod='lowest_energy',
                               regularization=False,
                               # num_spin_reversal_transforms=10,
                               # postprocess=True
                               )
        qa_a0 = linear_sys.solve(M=M,
                                 alpha=0,
                                 beta=1,
                                 epsilon=epsilon,
                                 max_iter_num=max_iter_num,
                                 mode_of_construct='direct',
                                 solver=sis.solve,
                                 show_process=False)
        qa_a0_nonreg_dict[f'New method'] = qa_a0

    qa_a0_reg_dict = {}
    step_L = step_L_list[it_num]
    for M, freq_max in zip(M_set, freq_max_set):
        print(f'!!!L={step_L}!!!')
        linear_sys = sfci.Stagger_FD_coeff_1D(v, h, tao, freq_max)
        sis = IterSampleSolver(sampler=sampler,
                               x0=np.ones(M) / 100,  # initial vector of Delta a
                               num_sample=sample_num,
                               L=step_L,
                               # L and steplen are hard to estimate, but can make significant influence to result!
                               steplen=0.75,
                               R=10,
                               fixed_point=1,  # in original algorithm fp must equal to 1, because x\in [0,2]
                               epsilon=epsilon,
                               maxiter=max_iter4_solver,  # max iteration in small iteration
                               resultMod='lowest_energy',
                               regularization=False,  # alpha=1*0.1**it Which is same as reg-lstsq method
                               codingMod='p'
                               # num_spin_reversal_transforms=10,
                               # postprocess=True
                               )
        qa_a0 = linear_sys.solve(M=M,
                                 alpha=0,
                                 beta=1,
                                 epsilon=epsilon,
                                 max_iter_num=max_iter_num,
                                 mode_of_construct='direct',
                                 solver=sis.original_QUBO_solve,
                                 show_process=False)
        qa_a0_reg_dict["Souza's Method"] = qa_a0

    print(f" <<< End the {it_num} experiment.")
