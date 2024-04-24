from BenchmarkFunctions import BenchmarkFunctions
import numpy as np

def initialization(nP, dim, ub, lb):
    # 算 ub 长度，找变量边界。
    X = np.zeros((nP, dim))
    for i in range(nP):
        # 得到的 X 每个元素都在 i 维的指定边界范围内随机分布。
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)

    return X


# 龙格库塔，算就完了
def RungeKutta(XB, XW, DelX):
    dim = XB.shape[0]
    C = np.random.randint(1, 3) * (1 - np.random.rand())
    r1 = np.random.rand(1, dim)
    r2 = np.random.rand(1, dim)

    K1 = 0.5 * (np.random.rand() * XW - C * XB)
    K2 = 0.5 * (np.random.rand() * (XW + r2 * K1 * DelX / 2) - (C * XB + r1 * K1 * DelX / 2))
    K3 = 0.5 * (np.random.rand() * (XW + r2 * K2 * DelX / 2) - (C * XB + r1 * K2 * DelX / 2))
    K4 = 0.5 * (np.random.rand() * (XW + r2 * K3 * DelX) - (C * XB + r1 * K3 * DelX))

    XRK = (K1 + 2 * K2 + 2 * K3 + K4)
    SM = 1 / 6 * XRK

    return SM

# 搞个服从均匀分布的随机数数组
def Unifrnd(a, b, c, dim):
    a2 = a / 2
    b2 = b / 2
    mu = a2 + b2
    sig = b2 - a2
    z = mu + sig * (2 * np.random.rand(c, dim) - 1)
    return z

# 取解优化
def Rndx(nP, i):
    qi = np.random.permutation(nP-1)   # 总的解数
    qi = qi[qi != i]  # 剔除i
    A, B, C = qi[:3]  # 和原算法一样取第1，2，3个
    return A, B, C

def RUN(nP, MaxIt, lb, ub, dim, fobj):
    # 记录所有方案的适配程度
    Cost = np.zeros(nP)

    # 调用 Initialize 初始化解集
    X = initialization(nP, dim, ub, lb)
    Xnew2 = np.zeros((1, dim))

    # 原来算法里记录收敛曲线来评估算法的性能和收敛速度的。
    Convergence_curve = np.zeros(MaxIt)

    # 计算适合度
    for i in range(nP):
        Cost[i] = fobj(X[i, :])

    # 找个最好的
    ind = np.argmin(Cost)
    Best_Cost = Cost[ind]
    Best_X = X[ind, :]
    Convergence_curve[0] = Best_Cost

    # 主要的的部分----------------------------
    it = 1
    while it < MaxIt:
        it += 1

        # 更新 f
        f = 20 * np.exp(-12 * (it / MaxIt))

        # 算个平均值
        Xavg = np.mean(X, axis=0)

        # SF调整搜索步长的大小，越后面越小
        SF = 2 * (0.5 - np.random.rand(1, nP)) * f

        for i in range(nP):
            ind_l = np.argmin(Cost)
            _ = Cost[ind_l]

            lBest = X[ind_l, :]

            # 确定 ABC
            A, B, C = Rndx(nP, i)
            ind1 = np.argmin(Cost[[A, B, C]])
            _ = Cost[ind1]

            # 是时候求 X 了
            # 调整当前解的位置
            gama = np.random.rand() * (np.array(X[i, :]) - np.random.rand(dim) * (np.array(ub) - np.array(lb))) * np.exp(-4 * it / MaxIt)
            # 用全局最佳解 Best_X 与当前解 X 之间的差加上前面的 gama。表示在当前位置上搜索的方向和步长。
            Stp = np.random.rand(1, dim) * ((Best_X - np.random.rand() * Xavg) + gama)
            # 计算搜索步长的大小
            DelX = 2 * np.random.rand(1, dim) * np.abs(Stp)

            # 算 RungeKutta 的 Xb 与 Xw
            if Cost[i] < Cost[ind1]:
                Xb = X[i, :]
                Xw = X[ind1, :]
            else:
                Xb = X[ind1, :]
                Xw = X[i, :]

            # 搜索机制
            SM = RungeKutta(Xb, Xw, DelX)

            L = np.random.rand(1, dim) < 0.5
            Xc = L * X[i, :] + (1 - L) * X[A, :]
            Xm = L * Best_X + (1 - L) * lBest

            vec = [1, -1]
            r = np.random.choice(vec)

            g = 2 * np.random.rand()
            mu = 0.5 + 0.1 * np.random.randn(1, dim)

            # 确定新 X
            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[0, i] * g * Xc) + SF[0, i] * (SM) + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[0, i] * g * Xm) + SF[0, i] * (SM) + mu * (X[A, :] - X[B, :])

            # 超出范围？
            FU = Xnew > ub
            FL = Xnew < lb
            Xnew = Xnew * (~(FU + FL)) + ub * FU + lb * FL
            CostNew = fobj(np.transpose(Xnew))

            if CostNew < Cost[i]:
                X[i, :] = Xnew
                Cost[i] = CostNew

            # ESQ 算法，50% 概率执行
            if np.random.rand() < 0.5:
                # 指数衰减因子
                EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
                r = np.floor(Unifrnd(-1, 2, 1, 1))

                u = 2 * np.random.rand(1, dim)
                w = Unifrnd(0, 2, 1, dim) * EXP

                # 调用 Rndx 获取 ABC
                A, B, C = Rndx(nP, i)
                Xavg = (X[A, :] + X[B, :] + X[C, :]) / 3

                beta = np.random.rand(1, dim)
                Xnew1 = beta * (Best_X) + (1 - beta) * (Xavg)
                Xnew2 = np.zeros((1, dim))

                # 根据 X 算 X2
                for j in range(dim):
                    if w[0, j] < 1:
                        Xnew2[0, j] =  Xnew1[0, j] + r[0, 0] * w[0, j] * ((Xnew1[0, j] - Xavg[j]) + np.random.randn())

                    else:
                        Xnew2[0, j] = (Xnew1[0, j] - Xavg[j]) + r[0, 0] * w[0, j] * ((u[0, j] * Xnew1[0, j] - Xavg[j]) + np.random.randn())

                FU = Xnew2 > ub
                FL = Xnew2 < lb
                Xnew2 = Xnew2 * (~(FU + FL)) + ub * FU + lb * FL

                # 算新解的适应度，如果更好，则更新当前解
                CostNew = fobj(np.transpose(Xnew2))
                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                # 如果不如，也有一定概率接受它
                else:
                    if np.random.rand() < w[0, np.random.randint(dim)]:
                        SM = RungeKutta(X[i, :], Xnew2, DelX)
                        Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[0, i] * (
                                    SM + (2 * np.random.rand(1, dim) * Best_X - Xnew2))
                        FU = Xnew > ub
                        FL = Xnew < lb
                        Xnew = Xnew * (~(FU + FL)) + ub * FU + lb * FL
                        CostNew = fobj(np.transpose(Xnew))

                        if CostNew < Cost[i]:
                            X[i, :] = Xnew
                            Cost[i] = CostNew

            # 确定最好的方案
            if Cost[i] < Best_Cost:
                Best_X = X[i, :]
                Best_Cost = Cost[i]

        # 保存最佳方案
        Convergence_curve[it-1] = Best_Cost
        print('it : {}, Best Cost = {}'.format(it, Convergence_curve[it-1]))

    return Best_Cost, Best_X, Convergence_curve

# test
nP = 50
MaxIt = 500
function_name = 'F2'
dimd = 5
lb, ub, dim, fobj = BenchmarkFunctions(function_name, dimd)

Best_Cost, Best_X, Convergence_curve = RUN(nP, MaxIt, lb, ub, dim, fobj)

print("最优解:", Best_X)
print("最优值:", Best_Cost)