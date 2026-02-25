import matplotlib.pyplot as plt
import json
import os
import numpy as np

filename = 'grid_search_WA_6bits_dichotomie_per_proj.json'
folder = 'results/grid_search/meta-llama/Meta-Llama-3-8B/'
try:
    # Read existing data from the file
    with open(folder + filename, 'r') as file:
        existing_data = json.load(file)
except FileNotFoundError:
    # If the file doesn't exist, start with an empty list
    existing_data = []
except json.JSONDecodeError:
    # If the file is empty or not valid JSON, start with an empty list
    existing_data = []

DATA = []
for data in existing_data:
    ppl = data["ppl"]
    proj = data["config"]["proj"]
    layer = data["config"]["layer"]
    bit = data["config"]["bit"]

    config = {"max_bit": [], 'layer': layer, 'proj': proj}

    for n in data["config"]["n"]:
        d = {}
        keys = n.keys()
        for k in keys:
            d[k] = {"max": n[k], "bit": bit}
        config["max_bit"].append(d)

    results = {"config": config, "ppl": ppl}
    DATA.append(results)
        

with open(folder + 'test.json', 'w') as file:
    json.dump(DATA, file, indent=4)