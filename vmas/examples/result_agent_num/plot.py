import numpy as np
data = np.loadtxt(open("progress_2_agents.csv","rb"),delimiter=",",skiprows=1,usecols=[1,2,3])
print(data) 