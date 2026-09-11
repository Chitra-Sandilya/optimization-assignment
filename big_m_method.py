from fractions import Fraction

M_VALUE = 10**6


class BigMSimplex:
    def __init__(self, num_vars, var_names):
        self.num_vars = num_vars
        self.var_names = list(var_names)
        self.rows = []
        self.rhs = []
        self.basis = []
        self.obj = []
        self.obj_M = []

    def add_row(self, coeffs, rhs, basic_var_index):
        self.rows.append([Fraction(c) for c in coeffs])
        self.rhs.append(Fraction(rhs))
        self.basis.append(basic_var_index)

    def set_objective(self, real_coeffs, M_coeffs):
        self.obj = [Fraction(c) for c in real_coeffs]
        self.obj_M = [Fraction(c) for c in M_coeffs]

    def _reduced_cost(self, j):
        real = self.obj[j]
        Mpart = self.obj_M[j]
        for i, bvar in enumerate(self.basis):
            factor = self.rows[i][j]
            real -= factor * self._basis_real_cost(i)
            Mpart -= factor * self._basis_M_cost(i)
        return real, Mpart

    def _basis_real_cost(self, row_i):
        return self._col_real_cost(self.basis[row_i])

    def _basis_M_cost(self, row_i):
        return self._col_M_cost(self.basis[row_i])

    def _col_real_cost(self, j):
        return self.obj[j]

    def _col_M_cost(self, j):
        return self.obj_M[j]

    def solve(self, max_iter=50, verbose=True):
        iteration = 0
        while True:
            iteration += 1
            n_cols = len(self.obj)

            reduced = []
            for j in range(n_cols):
                real, Mp = self._reduced_cost(j)
                reduced.append((real, Mp))

            if verbose:
                self._print_tableau(iteration, reduced)

            entering = None
            most_negative = (Fraction(0), Fraction(0))
            for j, (real, Mp) in enumerate(reduced):
                val = (Mp, real)
                if val < most_negative:
                    most_negative = val
                    entering = j

            if entering is None:
                if verbose:
                    print(f"\n>>> Optimality reached after {iteration - 1} pivot(s).\n")
                break

            ratios = []
            for i in range(len(self.rows)):
                a_ij = self.rows[i][entering]
                if a_ij > 0:
                    ratios.append((self.rhs[i] / a_ij, i))
            if not ratios:
                raise Exception("Problem is unbounded.")

            _, leaving_row = min(ratios, key=lambda t: t[0])

            if verbose:
                print(f"Entering variable: {self.var_names[entering]}  |  "
                      f"Leaving variable: {self.var_names[self.basis[leaving_row]]}\n")

            self._pivot(leaving_row, entering)

            if iteration > max_iter:
                raise Exception("Max iterations exceeded.")

        return self._extract_solution()

    def _pivot(self, pivot_row, pivot_col):
        pivot_val = self.rows[pivot_row][pivot_col]
        self.rows[pivot_row] = [v / pivot_val for v in self.rows[pivot_row]]
        self.rhs[pivot_row] = self.rhs[pivot_row] / pivot_val

        for i in range(len(self.rows)):
            if i == pivot_row:
                continue
            factor = self.rows[i][pivot_col]
            if factor != 0:
                self.rows[i] = [self.rows[i][k] - factor * self.rows[pivot_row][k]
                                 for k in range(len(self.rows[i]))]
                self.rhs[i] = self.rhs[i] - factor * self.rhs[pivot_row]

        self.basis[pivot_row] = pivot_col

    def _extract_solution(self):
        n_cols = len(self.obj)
        solution = [Fraction(0)] * n_cols
        for i, bvar in enumerate(self.basis):
            solution[bvar] = self.rhs[i]

        z_real = Fraction(0)
        z_M = Fraction(0)
        for j in range(n_cols):
            z_real += self.obj[j] * solution[j]
            z_M += self.obj_M[j] * solution[j]

        return solution, z_real, z_M

    def _print_tableau(self, iteration, reduced):
        print(f"--- Iteration {iteration} ---")
        header = "Basis".ljust(8) + "".join(name.rjust(10) for name in self.var_names) + "RHS".rjust(10)
        print(header)
        for i in range(len(self.rows)):
            row_str = self.var_names[self.basis[i]].ljust(8)
            row_str += "".join(str(self._fmt(v)).rjust(10) for v in self.rows[i])
            row_str += str(self._fmt(self.rhs[i])).rjust(10)
            print(row_str)
        cj_row = "Cj-Zj".ljust(8)
        for real, Mp in reduced:
            if Mp != 0:
                cj_row += f"{self._fmt(Mp)}M".rjust(10)
            else:
                cj_row += str(self._fmt(real)).rjust(10)
        print(cj_row)
        print()

    @staticmethod
    def _fmt(x):
        if isinstance(x, Fraction):
            if x.denominator == 1:
                return str(x.numerator)
            return f"{float(x):.4f}"
        return str(x)


def solve_example_problem():
    var_names = ["x1", "x2", "S1", "S2", "A1", "A2"]

    model = BigMSimplex(num_vars=2, var_names=var_names)

    model.add_row([3, 1, 0, 0, 1, 0], 3, basic_var_index=4)
    model.add_row([4, 3, -1, 0, 0, 1], 6, basic_var_index=5)
    model.add_row([1, 2, 0, 1, 0, 0], 4, basic_var_index=3)

    real_coeffs = [4, 1, 0, 0, 0, 0]
    M_coeffs =    [0, 0, 0, 0, 1, 1]
    model.set_objective(real_coeffs, M_coeffs)

    solution, z_real, z_M = model.solve(verbose=True)

    print("=" * 50)
    print("OPTIMAL SOLUTION")
    print("=" * 50)
    for name, val in zip(var_names, solution):
        print(f"  {name} = {BigMSimplex._fmt(val)}")
    print(f"\n  Optimal Z = {BigMSimplex._fmt(z_real)}"
          + (f"  (+ {BigMSimplex._fmt(z_M)}*M residual -> INFEASIBLE if nonzero)" if z_M != 0 else ""))
    print("=" * 50)


if __name__ == "__main__":
    solve_example_problem()