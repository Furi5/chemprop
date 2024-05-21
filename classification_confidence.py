import pandas as pd
import numpy as np


# 校准数据集
calibration_data = [...]

# 计算校准数据集的非一致性评分
calibration_scores = []
for data in calibration_data:
    _, std = toxpredict(data)
    score = calculate_nonconformity_score(std)
    calibration_scores.append(score)

# 新数据
new_data = ...

# 计算新数据的非一致性评分
_, new_std = toxpredict(new_data)
new_score = calculate_nonconformity_score(new_std)

# 计算置信度
confidence = calculate_confidence(calibration_scores, new_score)

print(f"The confidence for the new data is: {confidence:.2f}")
