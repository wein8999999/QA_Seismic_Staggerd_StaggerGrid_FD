# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/6/29 17:48
# @Author  : Wen Xixiang
# @Version : python3.9

import matplotlib.pyplot as plt
import numpy as np

config = {
    # "font.family":'Calibri',
    "mathtext.fontset": 'stix',
}
plt.rcParams.update(config)

label_set = ['Steepest Descent', 'Fixed Descent'] # ['No-added', 'Added']  #  ['Non-reg', 'Reg'] #  ['non_reg', 'reg'] ['old', 'new']
max_iter = 20
epsilon = 1e-5

log_file = open('Inf_reg_20230817.txt', 'r')
lf = log_file.readlines()
log_file.close()

data_list = []
counter = 0
for line in lf:
    if 'Current object function' in line and '*' not in line:
        content = line.strip().split()
        data_list.append(float(content[-1]))
        if counter == 0:
            original_obj = float(content[-1])
        counter += 1

indx = np.argwhere((original_obj == np.array(data_list))).squeeze()  # index of original object function
indx_non_reg = indx[0::2]
indx_reg = indx[1::2]

nreg_obj_data = {}
reg_obj_data = {}
for ind in range(len(indx_non_reg)):
    # print(indx_non_reg[ind] + 1, indx_reg[ind])
    nreg_obj_data[ind] = np.array([data for data in data_list[indx_non_reg[ind] + 1:indx_reg[ind]]])
    if ind < len(indx_non_reg) - 1:
        # print(indx_reg[ind]+1, indx_non_reg[ind+1])
        reg_obj_data[ind] = np.array([data for data in data_list[indx_reg[ind] + 1:indx_non_reg[ind + 1]]])
    else:
        # print(indx_reg[ind]+1, 'end')
        reg_obj_data[ind] = np.array([data for data in data_list[indx_reg[ind] + 1:]])


# 计算中位数曲线
# def median_curve_and_error(obj_data, max_iter_num):
#     obj_in_each_iter = [[]] * max_iter_num  # 统计每轮迭代的目标函数数组
#     for j, obj_line in enumerate(list(obj_data.values())):
#         if len(obj_line)!= max_iter_num: # 如果迭代不满最大迭代次数，就重复最后一次迭代目标函数，一面影响中位数曲线
#             obj_line = np.hstack([obj_line, [obj_line[-1]]*(max_iter_num-len(obj_line))])
#         for i in range(len(obj_line)):
#             obj_in_each_iter[i] = obj_in_each_iter[i] + [obj_line[i]]
#
#     median_line = np.zeros(max_iter_num)
#     # RMSE = np.zeros(20) # 均方根误差太大
#     error_lines = np.zeros([2, max_iter_num])  # 记录每一个位置的最大最小值
#
#     for i, obj_list in enumerate(obj_in_each_iter):
#         if len(obj_list) != 0:
#             median_line[i] = np.median(obj_list)  # np.array(obj_list).mean()  #
#             # RMSE[i] = np.sum(np.abs(mean_line_nreg[i] - np.array(obj_list)))/len(obj_list)
#             error_lines[0, i] = np.abs(np.min(median_line[i] - np.array(obj_list)))  # 上界
#             error_lines[1, i] = -np.abs(np.max(median_line[i] - np.array(obj_list)))  # 下界
#     return median_line, error_lines

# 计算平均值曲线, 并计算标准差
def mean_curve_and_standard_deviation(obj_data, max_iter_num):
    obj_in_each_iter = [[]] * max_iter_num  # 统计每轮迭代的目标函数数组
    for j, obj_line in enumerate(list(obj_data.values())):
        if len(obj_line)!= max_iter_num: # 如果迭代不满最大迭代次数，就重复最后一次迭代目标函数，一面影响中位数曲线
            obj_line = np.hstack([obj_line, [obj_line[-1]]*(max_iter_num-len(obj_line))])
        for i in range(len(obj_line)):
            obj_in_each_iter[i] = obj_in_each_iter[i] + [obj_line[i]]

    mean_line = np.zeros(max_iter_num)
    SD = np.zeros(20)  # 记录每一次迭代的标准差

    for i, obj_list in enumerate(obj_in_each_iter):
        if len(obj_list) != 0:
            mean_line[i] = np.mean(obj_list)   #
            SD[i] = np.sqrt(np.sum((mean_line[i] - np.array(obj_list))**2)/len(obj_list))
    return mean_line, SD

mean_line_nreg, SD_nreg = mean_curve_and_standard_deviation(nreg_obj_data, max_iter)
mean_line_reg, SD_reg = mean_curve_and_standard_deviation(reg_obj_data, max_iter)

lw = 1.1
fig, ax = plt.subplots(1, 1, figsize=(5, 7))
ax.spines['left'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['right'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)

# 将所有目标函数曲线也绘制出来
# for ind in range(len(indx_non_reg)):
#     ax.plot([i for i in range(1, len(nreg_obj_data[ind]) + 1)],
#             nreg_obj_data[ind], 'o-',markersize=5,
#             color='crimson',alpha=0.3, linewidth=0.5)
#     ax.plot([i for i in range(1, len(reg_obj_data[ind]) + 1)],
#             reg_obj_data[ind], 'o-', markersize=5,
#             color='skyblue', alpha=0.3, linewidth=0.5)

# ax.plot([i for i in range(1, len(mean_line_nreg) + 1)],
#         mean_line_nreg, 'o-',
#         color='firebrick',
#         markeredgecolor='maroon',
#         markerfacecolor='crimson',
#         label=label_set[0])
#
# ax.fill_between([i for i in range(1, len(mean_line_nreg) + 1)],
#                 mean_line_nreg - SD_nreg,
#                 mean_line_nreg + SD_nreg,
#                 color='crimson',
#                 alpha=0.2)

# ax.errorbar([i for i in range(1, len(mean_line_nreg)+1)],mean_line_nreg,yerr=SD_nreg,fmt='-o',color = 'black',label="0.5-2ndHeating",elinewidth=1,capsize=3)
ax.plot([i for i in range(1, len(mean_line_reg) + 1)],
        mean_line_reg, 'o-',
        color='darkblue',
        markeredgecolor='darkblue',
        markerfacecolor='royalblue',
        label=label_set[1])

ax.fill_between([i for i in range(1, len(mean_line_reg) + 1)],
                mean_line_reg - SD_reg,
                mean_line_reg + SD_reg,
                color='skyblue',
                alpha=0.2)

ax.set_ylabel('Object Function', fontfamily='Times New Roman',
              fontsize=13, fontweight='bold',
              labelpad=-5)
ax.set_xlabel('Iterations', fontfamily='Times New Roman',fontsize=13, fontweight='bold')
ax.set_yscale('log')
# ax.set_ylim([1e-10, 1e15])
ax.set_xlim([1, 20])
ax.set_xticks([1,5,10,15,20])
# ax.set_yticks([1e-10, 1e-5, 1, 1e5, 1e10, 1e15])
ax.tick_params(top=True, right=True, which='major',
               direction='in', length=4, width=1.0,
               colors='black', labelsize=10)

ax.tick_params(top=True, right=True, direction='in',
               which="minor", length=2, width=0.8, colors='black')

ax.grid(visible=True, axis='both', which='major', linestyle='--', linewidth=0.5)

ax.axhline(y=epsilon, linestyle='--', linewidth=1.2,
           color='black', label='$\epsilon=10^{-5}$',
           )
lw = 1.5
ax.spines['left'].set_linewidth(lw)
ax.spines['top'].set_linewidth(lw)
ax.spines['right'].set_linewidth(lw)
ax.spines['bottom'].set_linewidth(lw)

ax.legend(title='',
          fontsize='small',
          loc='lower left',
          borderpad=1.,  # 留白
          borderaxespad=2.,  # 图框边界距离坐标轴距离
          labelspacing=0.5,  # 图例条目间距
          edgecolor='black',
          handlelength=2.)
# ax.legend(loc=1,fontsize='small')
# plt.tight_layout()
plt.show(block=True)
# fig.savefig('New_old_reuslt.pdf', dpi=600, format='pdf')
