import json
import numpy as np
import math

classes = {0: 'Small', 1: 'Medium', 2: 'Large'}
stats = {0: {'hd95': [], 'undefined': 0, 'total': 0},
         1: {'hd95': [], 'undefined': 0, 'total': 0},
         2: {'hd95': [], 'undefined': 0, 'total': 0}}

with open('results/evaluation_20260926T155034543326Z/slices.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        cls = data['routing_class']
        methods = data.get('methods', {})
        if 'heuristic' in methods:
            final = methods['heuristic']
            hd95 = final.get('hd95_surface_px')
            stats[cls]['total'] += 1
            if hd95 is None or math.isnan(hd95):
                stats[cls]['undefined'] += 1
            else:
                stats[cls]['hd95'].append(hd95)

for cls, name in classes.items():
    valid = stats[cls]['hd95']
    mean_hd95 = np.mean(valid) if len(valid) > 0 else float('nan')
    print(f"{name} | Valid Avg HD95: {mean_hd95:.4f} | Undefined: {stats[cls]['undefined']} / {stats[cls]['total']}")
