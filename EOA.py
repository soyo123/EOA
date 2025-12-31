# import packages
import os
from opfunu.cec_based.cec2022 import *
import numpy as np
from copy import deepcopy
from scipy.special import gamma  # 使用 scipy.special.gamma 替代 np.math.gamma

PopSize = 400
DimSize = 100
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1
curIter = 0
MaxIter = int(MaxFEs / PopSize * 2)

gamma_val = 1.0  # 趋光行为强度
beta = 0.1  # 固定随机搜索强度
alpha_max = 0.5  # 随机扰动强度

# initialize the population randomly
def Initialization(func):
    global Pop, FitPop
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])

# boundary check
def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi

# levy flight
def LevyFlight():
    beta = 1.5
    sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, DimSize)
    v = np.random.normal(0, 1, DimSize)
    step = u / (np.abs(v) ** (1 / beta))
    return step

# compute diversity
def ComputeDiversity():
    global Pop
    g_best = np.mean(Pop, axis=0)  # Use mean as global best approximation
    diversity = np.mean(np.linalg.norm(Pop - g_best, axis=1))
    return diversity

# EOA core function
def EOA(func):
    global Pop, FitPop, gamma_val, alpha_max, curIter, MaxIter

    new_Pop = np.zeros_like(Pop)
    g_best_idx = np.argmin(FitPop)
    g_best = Pop[g_best_idx]

    alpha = alpha_max * (1 - curIter / MaxIter)  # 动态调整噪声强度

    # 根据适应值排名选择行为
    ranks = np.argsort(FitPop)  # 按适应值升序排序
    threshold = int(0.7 * PopSize)  # 前 70% 使用趋光行为

    for i in range(PopSize):
        c2 = 1 / (1 + np.exp(-gamma_val * (FitPop[i] - FitPop[g_best_idx])))
        r2 = np.random.uniform(0, 1, DimSize)
        noise = alpha * np.random.uniform(-1, 1, DimSize)

        if ranks[i] < threshold:
            # Phototaxis (趋光行为)
            candidate = Pop[i] + c2 * r2 * (g_best - Pop[i]) + noise
        else:
            # Random search (异养行为) with Levy flight
            levy = LevyFlight()
            candidate = Pop[i] + (g_best - Pop[i]) * levy

        candidate = Check(candidate)  # Ensure boundary handling
        candidate_fitness = func(candidate)

        # 仅当新个体优于当前个体时更新
        if candidate_fitness < FitPop[i]:
            new_Pop[i] = candidate
            FitPop[i] = candidate_fitness
        else:
            new_Pop[i] = Pop[i]  # 保留原个体

    # 更新种群
    Pop[:] = new_Pop

# run EOA
def RunEOA(func):
    global FitPop, curIter, TrialRuns, DimSize
    All_Trial_Best = []
    for trial in range(TrialRuns):
        BestList = []
        curIter = 0
        np.random.seed(1998 + 18 * trial)
        Initialization(func)
        BestList.append(min(FitPop))
        while curIter < MaxIter:
            EOA(func)  # 更新一代
            curIter += 1
            BestList.append(min(FitPop))
        All_Trial_Best.append(BestList)
    np.savetxt(f"./EOA/EOA_Data/CEC2022/F{FuncNum+1}_{DimSize}D.csv", All_Trial_Best, delimiter=",")

# main function
def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2022 = [F12022(Dim), F22022(Dim), F32022(Dim), F42022(Dim), F52022(Dim), F62022(Dim),
               F72022(Dim), F82022(Dim), F92022(Dim), F102022(Dim), F112022(Dim), F122022(Dim)]

    for i in range(len(CEC2022)):
        FuncNum = i
        RunEOA(CEC2022[i].evaluate)

if __name__ == "__main__":
    if not os.path.exists('./EOA/EOA_Data/CEC2022'):
        os.makedirs('./EOA/EOA_Data/CEC2022')
    Dims = [10, 20]
    for Dim in Dims:
        main(Dim)
