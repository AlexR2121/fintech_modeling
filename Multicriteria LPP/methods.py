import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable

class MultiObjectiveProblem:
    def __init__(self, objs_coefs, types_of_optimization,
                 free_coefs = None,
                 A_uneq=None, b_uneq=None,
                 uneq_types=None, A_eq=None,
                 b_eq=None, vars_constraints=None):

        self.objs_coefs = objs_coefs
        self.free_coefs = free_coefs
        self.types_of_optimization = types_of_optimization

        self.A_uneq = A_uneq
        self.b_uneq = b_uneq
        self.uneq_types = uneq_types

        self.A_eq = A_eq
        self.b_eq = b_eq

        self.vars_constraints = vars_constraints

        for i in range(objs_coefs.shape[0]):
            if self.types_of_optimization[i] == 'min':
                self.objs_coefs[i] *= -1
                self.free_coefs[i] *= -1

    def add_constraints(self, model, var_list):
        if self.A_uneq is not None:
            for i, uneq_left_part in enumerate(self.A_uneq):
                constr_left_part = 0
                for coef, x in zip(uneq_left_part, var_list):
                    constr_left_part += coef * x
                if self.uneq_types[i] == '>=':
                    model += (constr_left_part >= self.b_uneq[i], f'uneq_constr_{i}')
                else:
                    model += (constr_left_part <= self.b_uneq[i], f'uneq_constr_{i}')
        if self.A_eq is not None:
            for i, eq_left_part in enumerate(self.A_eq):
                constr_left_part = 0
                for coef, x in zip(eq_left_part, var_list):
                    constr_left_part += coef * x
                model += (constr_left_part == self.b_eq[i], f'eq_constr_{i}')

        return model

    def _set_vars_and_objs(self):
        n_vars = self.objs_coefs.shape[1]
        low_bounds = [0] * n_vars
        high_bounds = [None] * n_vars

        if self.vars_constraints is not None:
            for i, var_constraint in enumerate(self.vars_constraints):
                if var_constraint == '<=':
                    low_bounds[i] = None
                    high_bounds[i] = 0
                elif var_constraint == 'arb':
                    low_bounds[i] = None

        var_list = []
        for i in range(n_vars):
            x = LpVariable(f'x{i}', low_bounds[i], high_bounds[i])
            var_list.append(x)

        obj_funs = [0] * self.objs_coefs.shape[0]

        for i, obj_coef in enumerate(self.objs_coefs):
            for coef, x in zip(obj_coef, var_list):
                obj_funs[i] += coef * x
            obj_funs[i] += self.free_coefs[i]
        return var_list, obj_funs

    def solve_successive(self, concession):

        var_list, obj_funs = self._set_vars_and_objs()

        q = []
        x_opt = []
        for i in range(len(obj_funs)):
            model = LpProblem("problem", LpMaximize)

            model = self.add_constraints(model, var_list)

            if isinstance(concession, float):
                m = 1-concession
                for j, k in enumerate(q):
                    if self.types_of_optimization[j] == 'min':
                        model += (obj_funs[j] >= k * (1 + m), f"concession_{j}")
                    else:
                        model += (obj_funs[j] >= k * m, f"concession_{j}")
            else:
                for j, k in enumerate(q):
                    model += (obj_funs[j] >= k - concession[j], f"concession_{j}")

            model += obj_funs[i]

            model.solve()

            q.append(model.objective.value())
            x_loc_opt = dict()
            for var in model.variables():
                x_loc_opt[var.name] = var.value()
            x_opt.append(x_loc_opt)

        sol = pd.DataFrame(x_opt)
        objs_vals = np.dot(sol.values, self.objs_coefs.T)

        for i in range(objs_vals.shape[1]):
            if self.types_of_optimization[i] == 'min':
                objs_vals[:, i] *= -1
                objs_vals[:, i] -= self.free_coefs[i]
            else:
                objs_vals[:, i] += self.free_coefs[i]


        concessions_names = [f'Concession {i}' for i in range(len(obj_funs))]
        obj_names = [f'Objective {i}' for i in range(len(obj_funs))]

        sol[obj_names] = objs_vals
        sol.index = concessions_names
        sol = sol.round(2)

        return sol

    def solve_eps_constraints(self, epsilons, index_of_main_obj):
        var_list, obj_funs = self._set_vars_and_objs()

        model = LpProblem("problem", LpMaximize)

        model = self.add_constraints(model, var_list)
        for j, k in enumerate(epsilons):
            if j != index_of_main_obj:
                model += (obj_funs[j] >= k, f"constr_obj_{j}")
            else:
                model += obj_funs[j]

        model.solve()

        x_opt = dict()
        for var in model.variables():
            x_opt[var.name] = var.value()

        sol = pd.DataFrame(x_opt, index = ['Optimal values'])

        objs_vals = np.dot(sol.values, self.objs_coefs.T)

        obj_names = [f'Objective {i}' for i in range(len(obj_funs))]
        obj_names[index_of_main_obj] = 'Main objective'
        sol[obj_names] = objs_vals

        sol = sol.round(2)

        return sol

