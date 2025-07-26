from time import time
from gurobipy import Model, GRB, quicksum
import numpy as np
import random

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from neighborhood_structures import structures, shake, neighborhood_change
from plot_result import build_html
from setup_params import Params

random.seed(97)
np.random.seed(97)

# =============================
# MODELO
# =============================

STATUS_NAMES = {
    GRB.LOADED: "Loaded",
    GRB.OPTIMAL: "Optimal",
    GRB.INFEASIBLE: "Infeasible",
    GRB.INF_OR_UNBD: "Infeasible or Unbounded",
    GRB.UNBOUNDED: "Unbounded",
    GRB.CUTOFF: "Cutoff",
    GRB.ITERATION_LIMIT: "Iteration Limit",
    GRB.NODE_LIMIT: "Node Limit",
    GRB.TIME_LIMIT: "Time Limit",
    GRB.SOLUTION_LIMIT: "Solution Limit",
    GRB.INTERRUPTED: "Interrupted",
    GRB.NUMERIC: "Numeric Error",
    GRB.SUBOPTIMAL: "Suboptimal",
    GRB.USER_OBJ_LIMIT: "User Objective Limit",
}

class Optimizer:
    def __init__(self, path, max_days=30, gurobi_time_limit=20, max_iter=100):
        self.path = path
        self.gurobi_time_limit = gurobi_time_limit
        self.max_days = max_days
        self.max_iter = max_iter

    def run(self, data, constraints, forced_ups, verbose=False, use_max_time=None, return_first=False, start=None):

        m = Model("Planejamento Transporte Celulose")
        if not verbose:
            m.setParam("OutputFlag", 0)
        m.setParam("Heuristics", 0.7)
        m.setParam("MIPFocus", 1)
        if use_max_time is not None:
            m.setParam('TimeLimit', use_max_time)
        if return_first:
            m.setParam('SolutionLimit', 1)

        diff_DB = (max(data.DB) - min(data.DB))

        db_min = m.addVars(data.D, data.T, name="db_min", lb=0, ub=max(data.DB))
        db_max = m.addVars(data.D, data.T, name="db_max", lb=0, ub=max(data.DB))
        x = m.addVars(data.I, data.D, data.K, data.T, vtype=GRB.INTEGER, name="x")
        y = m.addVars(data.I, data.K, data.T, vtype=GRB.BINARY, name="y")
        z = m.addVars(data.K, data.T, data.F, vtype=GRB.BINARY, name="z")
        u = m.addVars(data.I, data.D, data.T, vtype=GRB.BINARY, name="u")
        v = m.addVars(data.I, data.T, name="v")
        delta = m.addVars(data.I, data.T, vtype=GRB.BINARY, name="delta")
        theta = m.addVars(data.K, data.T, vtype=GRB.BINARY, name="theta")
        r = m.addVars(data.I, data.K, data.T, vtype=GRB.BINARY, name="r")
        # e = m.addVars(data.I, vtype=GRB.INTEGER, lb=0, ub=data.dias + 1, name="e")

        m.update()
        if start is not None:
            for var in m.getVars():
                if var.VarName in start and start[var.VarName] is not None:
                    try:
                        var.start = start[var.VarName]
                    except:
                        pass  # ignora erro se valor for inválido para a variável


        # Função Objetivo
        m.setObjective(quicksum(db_max[d, t] - db_min[d, t] for d in data.D for t in data.T), GRB.MINIMIZE)

        # =============================
        # RESTRIÇÕES 5.1 a 5.5
        # =============================

        for d in data.D:
            for t in data.T:
                m.addConstr(db_max[d, t] >= db_min[d, t], name=f'5.0-{(d, t)}')

        for up in constraints:
            m.addConstr(quicksum(y[up, k, t] for k in data.K for t in data.T) == 0, name=f'Set constraint {up}')

        # 5.1) Cálculo de DBmin e DBmax por dia/fábrica
        for i in data.I:
            for d in data.D:
                for t in data.T:
                    m.addConstr(db_max[d, t] >= data.DB[i] * u[i, d, t], name=f'5.1.1-{(i, d, t)}')
                    m.addConstr(db_min[d, t] <= data.DB[i] + diff_DB * (1 - u[i, d, t]), name=f'5.1.2-{(i, d, t)}')
                    m.addConstr(db_min[d, t] >= min(data.DB) * u[i, d, t], name=f'5.1.3-{(i, d, t)}')

        # for i in data.I:
        #     for k in data.K:
        #         for t in data.T:
        #             m.addConstr(e[i] <= t + (1 - r[i, k, t]) * (data.dias + 1), name=f'Set e {(i, k, t)}')
        #             m.addConstr(e[i] >= t * r[i, k, t], name=f"LowerBound_e_{i}_{k}_{t}")
        #
        # for up_1, up_2 in constraints:
        #     m.addConstr(e[up_1] <= e[up_2], name=f'Set constraint {(up_1, up_2)}')
        #
        for up in forced_ups:
            m.addConstr(quicksum(y[up, k, t] for k in data.K for t in data.T) >= 1, name=f'Set constraint {up}')

        # 5.2) Ativação de u[i,d,t] se x[i,d,k,t] > 0
        for i in data.I:
            for d in data.D:
                for t in data.T:
                    m.addConstr(u[i, d, t] >= quicksum(x[i, d, k, t] for k in data.K) / data.M, name=f'5.2.1-{(i, d, t)}')
                    m.addConstr(u[i, d, t] <= quicksum(x[i, d, k, t] for k in data.K), name=f'5.2.2-{(i, d, t)}')

        # 5.3) Ativação de z[k,t,f] se x[i,d,k,t] > 0 em P[i,f] = 1
        for k in data.K:
            for t in data.T:
                for f in data.F:
                    soma = quicksum(x[i, d, k, t] * data.P[i][f] for i in data.I for d in data.D)
                    m.addConstr(z[k, t, f] >= soma / data.M, name=f'5.3.1-{(k, t, f)}')
                    m.addConstr(z[k, t, f] <= soma, name=f'5.3.2-{(k, t, f)}')

        # 5.4) Ativação de y[i,k,t] se x[i,d,k,t] > 0
        for i in data.I:
            for k in data.K:
                for t in data.T:
                    soma = quicksum(x[i, d, k, t] for d in data.D)
                    m.addConstr(y[i, k, t] >= soma / data.M, name=f'5.4.1-{(i, k, t)}')
                    m.addConstr(y[i, k, t] <= soma, name=f'5.4.2-{(i, k, t)}')

        # 5.5) Atendimento à demanda de fábrica
        for d in data.D:
            for t in data.T:
                soma_volume = quicksum(x[i, d, k, t] * data.CAPACIDADE[i][d][t] for i in data.I for k in data.K)
                m.addConstr(soma_volume <= data.DEM_max[d][t], name=f'5.5.1-{(d, t)}')
                m.addConstr(soma_volume >= data.DEM_min[d][t], name=f'5.5.2-{(d, t)}')

        # 5.6) RSP médio ponderado (alternativo)
        for d in data.D:
            num = quicksum(data.RSP[i] * x[i, d, k, t] * data.CAPACIDADE[i][d][t] for i in data.I for k in data.K for t in data.T)
            den = quicksum(x[i, d, k, t] * data.CAPACIDADE[i][d][t] for i in data.I for k in data.K for t in data.T)
            m.addConstr(num <= data.RSP_max[d][0] * den, name=f'5.6.1-{d}')
            m.addConstr(num >= data.RSP_min[d][0] * den, name=f'5.6.2-{d}')

        # 5.7) Limite de gruas
        for k in data.K:
            for t in data.T:
                m.addConstr(quicksum(y[i, k, t] for i in data.I) <= data.GRUA[k], name=f'5.7-{(k, t)}')

        # 5.8) Exclusividade por fazenda
        for k in data.K:
            for t in data.T:
                m.addConstr(quicksum(z[k, t, f] for f in data.F) <= 1, name=f'5.8-{(k, t)}')

        # 5.9) Frota por transportador (adaptado)
        for k in data.K:
            for t in data.T:
                total_volume = quicksum(v[i, t] for i in data.I)
                total_viagens = quicksum(x[i, d, k, t] for i in data.I for d in data.D)

                # Se volume for suficiente, theta[k,t] deve ser 1
                m.addConstr(total_volume >= data.V_min[k, t] * theta[k, t], name=f"theta_volume_lb_{k}_{t}")
                m.addConstr(total_volume <= data.V_min[k, t] - 1e-5 + data.MV * theta[k, t], name=f"theta_volume_ub_{k}_{t}")

                # Frota mínima apenas se theta == 1
                m.addConstr(total_viagens >= data.FROTA_min[k] * theta[k, t], name=f"frota_min_{k}_{t}")

                # Frota máxima sempre
                m.addConstr(total_viagens <= data.FROTA_max[k], name=f"frota_max_{k}_{t}")

        # 5.10) Percentual mínimo de veículos
        for i in data.I:
            for k in data.K:
                for t in data.T:
                    num = quicksum(x[i, d, k, t] for d in data.D)
                    den = quicksum(x[h, d, k, t] for h in data.I for d in data.D)
                    m.addConstr(num + (1 - y[i, k, t]) * data.M >= data.TETA[k] * den, name=f'5.10-{(i, k, t)}')

        # 5.11) Balanço de massa
        for i in data.I:
            m.addConstr(v[i, 0] == data.VI[i] - quicksum(x[i, d, k, 1] * data.CAPACIDADE[i][d][0] for d in data.D for k in data.K), name=f'5.11.1-{i}')
            for t in data.T:
                if t > 0:
                    m.addConstr(v[i, t] == v[i, t - 1] - quicksum(x[i, d, k, t] * data.CAPACIDADE[i][d][t] for d in data.D for k in data.K), name=f'5.11.2-{(i, t)}')

        # 5.12) Veículo não pode atender UP sem volume
        for i in data.I:
            for t in data.T:
                m.addConstr(v[i, t] <= data.MV * delta[i, t], name=f'5.12.1-{(i, t)}')
                m.addConstr(v[i, t] >= 1e-5 - data.MV * (1 - delta[i, t]), name=f'5.12.2-{(i, t)}')
                if t > 0:
                    m.addConstr(quicksum(x[i, d, k, t] for d in data.D for k in data.K) <= data.M * delta[i, t - 1], name=f'5.12.3-{(i, t)}')

        # 5.13) Troca de fazenda após volume 0
        for k in data.K:
            for t in data.T:
                if t > 0:
                    for f in data.F:
                        lhs = z[k, t, f] * data.qt_i
                        rhs = z[k, t - 1, f] * quicksum(delta[i, t - 1] * data.P[i][f] for i in data.I)
                        m.addConstr(lhs >= rhs, name=f'5.13-{(k, t)}')

        # 5.14) UPs pequenas com transporte contínuo
        for i in data.I:
            for k in data.K:
                for t in data.T:
                    if t > 0:
                        m.addConstr(y[i, k, t] >= y[i, k, t - 1] * delta[i, t - 1] * data.L7[i], name=f'5.14-{(i, k, t)}')

        # 5.15) Ativação de variável de entrada na UP
        for i in data.I:
            for k in data.K:
                m.addConstr(r[i, k, 0] == y[i, k, 0], name=f'5.15.1-{(i, k)}')
                for t in data.T:
                    if t > 0:
                        m.addConstr(r[i, k, t] <= y[i, k, t], name=f'5.15.2-{(i, k, t)}')
                        m.addConstr(r[i, k, t] <= 1 - y[i, k, t - 1], name=f'5.15.3-{(i, k, t)}')
                        m.addConstr(r[i, k, t] >= y[i, k, t] - y[i, k, t - 1], name=f'5.15.4-{(i, k, t)}')

        # 5.16) UPs grandes: até 2 janelas de transporte
        for i in data.I:
            for k in data.K:
                m.addConstr(quicksum(r[i, k, t] for t in data.T) * data.U7[i] <= 2, name=f'5.16-{(i, k)}')

        m.optimize()
        if m.status == GRB.INFEASIBLE:
            return None, m, None

        # try:
        #     new_x = {(i, d, k, t): x[(i, d, k, t)].X for i in data.I for k in data.K for d in data.D for t in data.T}
        # except:
        #     return None, m, None
        return (x, v), m, m.ObjVal

    @staticmethod
    def evaluate(data, x):
        return sum(
            max([data.DB[i] for i in data.I if sum(x[i, d, k, t].X for k in data.K) > 0])
            - min([data.DB[i] for i in data.I if sum(x[i, d, k, t].X for k in data.K) > 0])
            for d in data.D for t in data.T
        )

    def optimize(
            self, data, verbose=False, generate_plot=False, simple_info=True, constraints=None,
            forced_ups=None, use_max_time=None, return_first=False, start=None
        ):
        if constraints is None:
            constraints = list()

        if forced_ups is None:
            forced_ups = list()

        if simple_info:
            print(f'Rodando para {self.max_days} dias')
        output, m, obj = self.run(data, constraints, forced_ups, verbose, use_max_time, return_first, start)
        if simple_info:
            print(f'Status: {STATUS_NAMES[m.status]}')
            if m.status != GRB.INFEASIBLE:
                print(f'Valor da função objetivo: {m.ObjVal}')
            if output is None:
                return None, None, None

        # if verbose:
        #     self.generate_output(m)

        return m, obj, output

    def heuristics(self, verbose=False, generate_plot=False, simple_info=True, iterations=5):
        data = Params(self.path, dias=self.max_days, time_slow=0)

        all_simulations = []
        all_times = []
        for s in range(iterations):
            start_time = time()
            m, obj, output, obj_values = self.vns(data, verbose, generate_plot, simple_info)
            end_time = time()
            print(f'Resultado em {end_time - start_time} segundos: {obj}')
            print(f'Final: {obj}')
            x, v = output
            build_html(data, x, v)
            all_simulations.append(obj_values)
            all_times.append(end_time - start_time)

        return all_simulations, all_times

    def vns(self, params, verbose, generate_plot, simple_info, k_max=None):
        constraints = random.sample(params.I, 5)
        neighborhoods = structures()
        if k_max is None:
            k_max = len(neighborhoods)
        k_max = max(len(neighborhoods), k_max)
        best_eval = np.inf
        already_visited = []
        k = 0
        best_valid_ups = None
        best_values = None
        counter = 0
        obj_values = []
        while k < k_max and counter < self.max_iter:
            c = neighborhoods[k]
            constraints2 = shake(constraints, params, same_farm=c['same_farm'], nearby=c['nearby'])
            if tuple(constraints2) in already_visited:
                k += 1
                continue
            already_visited.append(tuple(constraints2))

            m, current_obj, output = self.optimize(
                data=params, verbose=verbose, generate_plot=generate_plot,
                simple_info=simple_info, constraints=constraints2, use_max_time=self.gurobi_time_limit
            )
            counter += 1
            if current_obj is None or np.isinf(current_obj):
                continue

            x_values = {v.VarName: v.X for v in m.getVars()}
            x, v = output
            valid_ups = [i for i in params.I if any((x[i, d, k, t].X > 0 for d in params.D for k in params.K for t in params.T))]
            m, current_obj, output, valid_ups = self.local_search(valid_ups, params, verbose, generate_plot, simple_info, start=x_values)
            x_values = {v.VarName: v.X for v in m.getVars()}
            # if current_obj < best_eval:
            #     if generate_plot:
            #         x, v = output
            #         build_html(params, x, v)
            constraints, k, best_eval, best_valid_ups, best_values = neighborhood_change(
                constraints, constraints2, k, current_obj, best_eval, valid_ups, best_valid_ups, x_values, best_values
            )
            obj_values.append(best_eval)

        print('Última busca local...')
        m, current_obj, output, valid_ups = self.local_search(best_valid_ups, params, verbose, generate_plot, simple_info, use_max_time=180, start=best_values)
        # x, v = output
        # build_html(params, x, v)
        return m, current_obj, output, obj_values

    def local_search(self, valid_ups, params, verbose, generate_plot, simple_info, use_max_time=60, start=None):

        used_farms = {next(f for f, p in enumerate(params.P[up]) if p == 1) for up in valid_ups}
        options = [i for i in params.I if i not in valid_ups and next(f for f, p in enumerate(params.P[i]) if p == 1) in used_farms]
        constraints = [i for i in params.I if i not in valid_ups and i not in options]
        m, current_obj, output = self.optimize(
            data=params, verbose=verbose, generate_plot=generate_plot,
            simple_info=simple_info, constraints=constraints, forced_ups=valid_ups, use_max_time=use_max_time, start=start
        )
        x, v = output
        valid_ups = [i for i in params.I if any((x[i, d, k, t].X > 0 for d in params.D for k in params.K for t in params.T))]

        return m, current_obj, output, valid_ups

    @staticmethod
    def generate_output(m):
        if m.status == GRB.INF_OR_UNBD:
            print("Modelo infactível ou ilimitado. Forçando verificação de infactibilidade...")

            m.setParam(GRB.Param.Presolve, 0)
            m.setParam(GRB.Param.DualReductions, 0)

            m.optimize()

            if m.status == GRB.INFEASIBLE:
                print("Modelo confirmado como infactível. Calculando IIS...")
                m.computeIIS()
                m.write("modelo.ilp")
                for c in m.getConstrs():
                    if c.IISConstr:
                        print(f"Restrição no IIS: {c.ConstrName}")

                for v in m.getVars():
                    if v.IISLB or v.IISUB:
                        print(f"Variável no IIS (limite): {v.VarName} | LB: {v.IISLB}, UB: {v.IISUB}")
        elif m.status == GRB.INFEASIBLE:
            print("Modelo inviável! Gerando IIS...")
            m.computeIIS()
            m.write("infeasible.ilp")
            print("IIS gravado em 'infeasible.ilp'.")
            print("\n=== IIS ===")
            for c in m.getConstrs():
                if c.IISConstr:
                    print(f"Restrição: {c.ConstrName}")

            for v in m.getVars():
                if v.IISLB:
                    print(f"Lower Bound ativo: {v.VarName}")
                if v.IISUB:
                    print(f"Upper Bound ativo: {v.VarName}")

