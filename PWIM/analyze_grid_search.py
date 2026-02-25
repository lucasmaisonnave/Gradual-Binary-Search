import matplotlib.pyplot as plt
import json
import os
import numpy as np

filename = './results/grid_search_WA_4bits_dichotomie_per_proj.json'
folder = 'histograms/grid_search/dicho/'
try:
    # Read existing data from the file
    with open(filename, 'r') as file:
        existing_data = json.load(file)
except FileNotFoundError:
    # If the file doesn't exist, start with an empty list
    existing_data = []
except json.JSONDecodeError:
    # If the file is empty or not valid JSON, start with an empty list
    existing_data = []

PPL = []
MCS = []
n = []
bit = []
layer = []
proj = []
for data in existing_data:
    if data['config']['bit'] == 4:
        PPL.append(data['ppl'])
        proj.append(len(data['config']['n'][-1]))
        n.append(list(data['config']['n'][-1].values())[-1])
        bit.append(data['config']['bit'])
        layer.append(data['config']['layer'])



plt.scatter(np.array(layer) * 7 + np.array(proj), PPL, c=bit, alpha= (np.array(n) + 1)/10)
plt.ylabel('PPL')
plt.xlabel('layer')
plt.legend()
plt.grid()
plt.savefig(folder + 'PPLvsLayer_4bits_dichotomie_per_proj.png')
plt.close()

plt.scatter(n, PPL, c = layer)
plt.ylabel('PPL')
plt.xlabel('n')
plt.legend()
plt.grid()
plt.savefig(folder + 'PPLvsN_4bits.png')
plt.close()

