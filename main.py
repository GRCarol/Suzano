import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hibrido import Optimizer
from setup_params import Params

current_dir = os.getcwd()
path = os.path.join(current_dir, 'input_case.xlsx')
hybrid = True

optimizer = Optimizer(path, max_days=30, max_iter=5)
if hybrid:
    all_simulations, all_times = optimizer.heuristics(iterations=5, verbose=True, generate_plot=True, simple_info=True)
else:
    all_simulations = []
    all_times = []
    data = Params(path, dias=30, time_slow=0)
    optimizer.run(data, [], [], verbose=True, use_max_time=False)

dados = {
    'all_simulations': all_simulations,
    'all_times': all_times
}

path = os.path.join(current_dir, 'resultado_final.json')
with open(path, 'w') as f:
    json.dump(dados, f)
