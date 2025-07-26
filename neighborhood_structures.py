import random
import numpy as np

random.seed(97)
np.random.seed(97)
def structures():
    return [
        {'same_farm': True, 'nearby': True},
        {'same_farm': True, 'nearby': False},
        {'same_farm': False, 'nearby': True},
        {'same_farm': False, 'nearby': False},

    ]

def get_farm(params, up):
    return next(f for f in params.F if params.P[up][f] == 1)


def shake(x, params, same_farm=True, nearby=True):
    def farm_constraint(up, current_farm):
        if same_farm:
            return get_farm(params, up) == current_farm
        return get_farm(params, up) != current_farm

    x1 = np.array([v for v in x])
    chosen_up = random.choice(x)
    chosen_index = x.index(chosen_up)
    farm = get_farm(params, chosen_up)
    farm_ups = [i for i, _ in enumerate(params.ups) if farm_constraint(i, farm)]
    if len(farm_ups) == 1:
        return x

    if same_farm:
        relative_idx = farm_ups.index(chosen_up)
    else:
        relative_up = sorted(farm_ups, key=lambda z: abs(params.VI[z] - params.VI[chosen_up]))[0]
        relative_idx = farm_ups.index(relative_up)
    if relative_idx == len(farm_ups):
        alternatives = [relative_idx - 1]
    elif relative_idx == 0:
        alternatives = [1]
    else:
        if nearby:
            alternatives = [relative_idx - 1, relative_idx + 1]
        else:
            left = farm_ups[:relative_idx]
            if len(left) > 1:
                left = farm_ups[:relative_idx - 1]
            right = farm_ups[relative_idx + 1:]
            if len(right) > 1:
                right = farm_ups[relative_idx + 2:]
            alternatives = left + right

    if nearby and not same_farm:
        alternatives.append(farm_ups[relative_idx])
    alternatives = [a for a in alternatives if a not in x]
    if len(alternatives) == 0:
        return x

    new_up = random.choice(alternatives)
    x1[chosen_index] = new_up
    return list(x1)


def neighborhood_change(c1, c2, k, new_eval, best_eval, valid_ups, best_valid_ups, x_values, best_values):
    if new_eval < best_eval:
        print(f'Nova melhor solução: {new_eval}')
        c = c2
        best_eval = new_eval
        b_output = valid_ups
        b_values = x_values
    else:
        k += 1
        c = c1
        b_output = best_valid_ups
        b_values = best_values
    return c, k, best_eval, b_output, b_values

