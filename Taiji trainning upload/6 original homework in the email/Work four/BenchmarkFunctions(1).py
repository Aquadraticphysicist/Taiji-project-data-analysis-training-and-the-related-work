import numpy as np

#定义BenchmarkFunctions函数
"""根据函数名称返回函数的上界、下界、维度和目标函数。

    参数:
    F (str): 函数的名称，例如 'F1', 'F2' 等。
    dimd 用户定义的维度

    返回:
    lb (float): 下界。
    ub (float): 上界。
    dim (int): 维度。
    fobj (function): 目标函数。"""

def BenchmarkFunctions(F, dimd):
    global fobj, dim, ub, lb
    D = dimd #设定维度为30
    if F == 'F1':
        fobj = F1()
        lb = -100 #定义下界
        ub = 100  #定义上界
        dim = D  #给予维度
    elif F == 'F2':
        fobj = F2()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F3':
        fobj = F3()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F4':
        fobj = F4()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F5':
        fobj = F5()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F6':
        fobj = F6()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F7':
        fobj = F7()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F8':
        fobj = F8()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F9':
        fobj = F9()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F10':
        fobj = F10()
        lb = -32.768
        ub = 32.768
        dim = D
    elif F == 'F11':
        fobj = F11()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F12':
        fobj = F12()
        lb = -100
        ub = 100
        dim = D
    elif F == 'F13':
        fobj = F13()
        lb = -600
        ub = 600
        dim = D
    elif F == 'F14':
        fobj = F14()
        lb = -50
        ub = 50
        dim = D
    else:
        raise ValueError("Invalid function choice")

    return lb, ub, dim, fobj  #返回需要的下界，上界，维度，函数表达式

#定义每个计算函数
#F1到F6为单峰函数，F7到F14为多峰函数
def F1():
    def inner(x):
        D = len(x)
        z = x[0] ** 2 + 10 ** 6 * sum(x[1:] ** 2) #计算F1的每一维度的数值
        return z
    return inner  #返回F1函数表达式


def F2():
    def inner(x):
        D = len(x)
        f = np.zeros(D)
        for i in range(D):
           f[i] = np.abs(x[i]) ** (i + 1) #计算F2的每一维度的数值
           z = sum(f)
        return z
    return inner  #返回F2函数权柄


def F3():
    def inner(x):
        z = sum(x ** 2) + (sum(np.multiply(0.5, x))) ** 2 + (sum(np.multiply(0.5, x))) ** 4
        return z
    return inner #返回F3函数权柄


def F4():
    def inner(x):
        D = len(x)
        ff = np.zeros(D)
        for i in range(D - 1):
            ff[i] = 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2

        z = sum(ff)
        return z
    return inner  # 返回内部函数 inner 的权柄



def F5():
    def inner(x):
        D = len(x)
        z = 10 ** 6 * x[0] ** 2 + sum(x[1:] ** 2)
        return z
    return inner  # 返回内部函数 inner 的权柄



def F6():
    def inner(x):
        D = len(x)
        f = np.zeros(D)
        for i in range(D):
            f[i] = (((10 ** 6) ** ((i - 1) / (D - 1))) * x[i] ** 2)

        z = sum(f)
        return z
    return inner  # 返回内部函数 inner 的权柄



def F7():
    def inner(x):
        D = len(x)
        f = np.zeros(D)
        for i in range(D):
            if i == D - 1:
                f[i] = 0.5 + (np.sin(np.sqrt(x[i] ** 2 + x[0] ** 2)) ** 2 - 0.5) / (
                        1 + 0.001 * (x[i] ** 2 + x[0] ** 2)) ** 2
            else:
                f[i] = 0.5 + (np.sin(np.sqrt(x[i] ** 2 + x[i + 1] ** 2)) ** 2 - 0.5) / (
                        1 + 0.001 * (x[i] ** 2 + x[i + 1] ** 2)) ** 2

        z = sum(f)
        return z
    return inner  # 返回内部函数 inner 的权柄



def F8():
    def inner(x):
        D = len(x)
        f = np.zeros(D)
        w = np.zeros(D)
        for i in range(D - 1):
            w[i] = 1 + (x[i] - 1) / 4
            f[i] = (w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2)

        w[D - 1] = 1 + (x[D - 1] - 1) / 4
        z = np.sin(np.pi * w[0]) ** 2 + sum(f) + (w[D - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[D - 1]) ** 2)
        return z
    return inner  # 返回内部函数 inner 的权柄



def F9():
    def inner(x):
        D = len(x)
        f = np.zeros(D)
        for i in range(D):
            y = x[i] + 420.9687462275036
            if np.abs(y) < 500:
                f[i] = y * np.sin(np.abs(y) ** 0.5)
            else:
                if y > 500:
                    f[i] = (500 - np.mod(y, 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(y, 500)))) - (y - 500) ** 2 / (
                            10000 * D)
                else:
                    if y < -500:
                        f[i] = (np.mod(np.abs(y), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(y), 500) - 500))) - (
                                y + 500) ** 2 / (10000 * D)

        z = 418.9829 * D - sum(f)
        return z
    return inner  # 返回内部函数 inner 的权柄



def F10():
    def inner(x):
        D = len(x)
        z = -20 * np.exp(-0.2 * ((1 / D) * (sum(x ** 2))) ** 0.5) - np.exp(
            1 / D * sum(np.cos(np.multiply(2 * np.pi, x)))) + 20 + np.exp(1)
        return z
    return inner  # 返回内部函数 inner 的权柄



def F11():
    def inner(x):
        D = len(x)
        x = x + 0.5
        a = 0.5
        b = 3
        kmax = 20
        c1 = np.zeros(kmax + 2)
        c2 = np.zeros(kmax + 2)
        c1[:kmax + 2] = a ** np.arange(0, kmax + 2)
        c2[:kmax + 2] = 2 * np.pi * b ** np.arange(0, kmax + 2)
        f = 0
        c = -w(0.5, c1, c2)

        for i in range(D):  # Iterate over indices from 1 to D
             f = f + w(np.transpose(x[:, i]), c1, c2)
             z = f + c * D
        return z
    return inner #返回F11函数表达式



#定义w函数
def w(x, c1, c2):
    """
    计算加权余弦和。

    参数:
    x (numpy.ndarray): 输入向量。
    c1 (numpy.ndarray): 权重向量1。
    c2 (numpy.ndarray): 权重向量2。

    返回:
    y (numpy.ndarray): 加权余弦和。
    """
    y = np.zeros((len(x), 1)) #遍历维度
    for k in range(len(x)):  #对每个维度进行余弦函数变换，加权计算
        y[k] = np.sum(np.multiply(c1, np.cos(np.multiply(c2, x[:, k]))))

    return y


def F12():
    def inner(x):
        D = len(x)
        z = (np.abs(sum(x ** 2) - D)) ** (1 / 4) + (0.5 * sum(x ** 2) + sum(x)) / D + 0.5
        return z
    return inner #返回F12函数表达式


def F13():
    def inner(x):
        dim = len(x)
        z = sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.array([np.arange(1, dim + 1)])))) + 1
        return z
    return inner


def F14():
    def inner(x):
        dim = len(x)
        z = (np.pi / dim) * (10 * ((np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2) + sum(np.multiply((((x[1:] + 1) / 4) ** 2),
        (1 + 10.0 * ((np.sin(np.multiply(np.pi, (1 + (x[1:] + 1) / 4)))) ** 2)))) + (((x[-1] + 1) / 4) ** 2)
                         +sum(Ufun(x[1:], 10, 100, 4)))
        return z
    return inner #返回F14函数表达式

"""
x: 输入向量。
a: 控制函数的位置。
k: 控制函数的尖峰程度。
m: 控制函数的形状。
"""
def Ufun(x, a, k, m):
    o = np.multiply(np.multiply(k, ((x - a) ** m)), (x > a)) + np.multiply(np.multiply(k, ((-x - a) ** m)), (x < (-a)))
    return o

#调用的例子
if __name__ == "__main__":
    function_name = 'F12'
    lb, ub, dim, fobj = BenchmarkFunctions(function_name)
    print("Function Name:", function_name)
    print("Lower Bound:", lb)
    print("Upper Bound:", ub)
    print("Dimension:", dim)
    x = np.random.uniform(lb, ub, dim)
    print("Objective Value:", fobj)
    result = fobj(x)
    print(result)

