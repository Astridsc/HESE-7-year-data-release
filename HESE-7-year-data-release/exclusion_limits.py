   
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

mean = 0
SD = 1

one_sd = norm.cdf(SD, mean, SD) - norm.cdf(-SD, mean, SD)
two_sd = norm.cdf(2 * SD, mean, SD) - norm.cdf(-2 * SD, mean, SD)
three_sd = norm.cdf(3 * SD, mean, SD) - norm.cdf(-3 * SD, mean, SD)

alpha =   1 - two_sd  # 2sigma confidence level



