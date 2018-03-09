import numpy as np

# 区分的に一様な確率密度関数
def piecewise_uniform(x):
    sections = [[i, i+5] for i in range(0, 80, 5)]    # [0,80]を16個の区間に分割
    nums = [ 41.,  22.,  16.,  86., 114., 106.,  95.,  72.,  48.,  41.,  32., 16.,  15.,   4.,   6.,   1.]    # この区間における実データ(train.csv)の度数分布
    for i in range(len(sections)):
        if sections[i][0] <= x and x <= sections[i][1]:
            return (nums[i] / (5 * np.sum(nums)))

# 乱数生成器を作成
def genrandom(func, domain, value_range):
    while(1):
        # 乱数ペア(x_candidate, y_candidate) を生成
        x_candidate = domain[-1] * np.random.uniform(0,1)
        y_candidate = value_range[-1] * np.random.uniform(0,1)
        
        # y_candidate が上の確率密度関数の値より小さかったら採用
        if y_candidate <= func(x_candidate):
            return x_candidate
