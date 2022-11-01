import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# read in data, adjust to right file and separation
data = pd.read_csv('Final_DOE.txt', sep = "/", header=None, names=('VM_type','Arrival_type','Model','Response_time'))

# print(data)

#Make ANOVA model
model = ols("""Response_time ~ C(VM_type) + C(Arrival_type) + C(Model) + 
	C(VM_type):C(Arrival_type) + C(VM_type):C(Model) + C(Arrival_type):C(Model) +
	C(Arrival_type):C(Model):C(VM_type)""", data=data).fit()


#perform 2-level-three-way ANOVA
# print("Type 2:")
# print(sm.stats.anova_lm(model, typ=2))
print("Type 3:")
print(sm.stats.anova_lm(model, typ=3))