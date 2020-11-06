import pandas as pd
import numpy as np


#Discretizacion en pandas
print(pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3,
             labels = ['good', 'medium', 'bad'],retbins = True))

