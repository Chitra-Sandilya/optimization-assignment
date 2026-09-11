from fractions import Fraction
import copy


def vogel_approximation_method(supply, demand, cost, verbose=True):
    m, n = len(supply), len(demand)
    supply = supply[:]
    demand = demand[:]
    allocation = [[0] * n for _ in range(m)]
    row_active = [True] * m
    col_active = [True] * n

    step = 0
    while any(row_active) and any(col_active):
        step += 1

        row_penalty = {}
        for i in range(m):
            if not row_active[i]:
                continue
            costs_in_row = sorted(cost[i][j] for j in range(n) if col_active[j])
            if len(costs_in_row) >= 2:
                row_penalty[i] = costs_in_row[1] - costs_in_row[0]
            elif len(costs_in_row) == 1:
                row_penalty[i] = costs_in_row[0]

        col_penalty = {}
        for j in range(n):
            if not col_active[j]:
                continue
            costs_in_col = sorted(cost[i][j] for i in range(m) if row_active[i])
            if len(costs_in_col) >= 2:
                col_penalty[j] = costs_in_col[1] - costs_in_col[0]
            elif len(costs_in_col) == 1:
                col_penalty[j] = costs_in_col[0]

        best_row = max(row_penalty, key=lambda k: row_penalty[k]) if row_penalty else None
        best_col = max(col_penalty, key=lambda k: col_penalty[k]) if col_penalty else None
        row_pen_val = row_penalty.get(best_row, -1)
        col_pen_val = col_penalty.get(best_col, -1)

        if row_pen_val >= col_pen_val:
            i = best_row
            j = min((jj for jj in range(n) if col_active[jj]), key=lambda jj: cost[i][jj])
        else:
            j = best_col
            i = min((ii for ii in range(m) if row_active[ii]), key=lambda ii: cost[ii][j])

        qty = min(supply[i], demand[j])
        allocation[i][j] = qty
        supply[i] -= qty
        demand[j] -= qty

        if verbose:
            print(f"Step {step}: allocate x{i+1}{j+1} = {qty}  "
                  f"(cost {cost[i][j]}, row-penalty={row_pen_val}, col-penalty={col_pen_val})")

        if supply[i] == 0:
            row_active[i] = False
        if demand[j] == 0:
            col_active[j] = False

    return allocation


def modi_method(allocation, cost, supply, demand, verbose=True):
    m, n = len(supply), len(demand)
    allocation = copy.deepcopy(allocation)
    iteration = 0

    while True:
        iteration += 1
        basic_cells = [(i, j) for i in range(m) for j in range(n) if allocation[i][j] > 0]

        needed = m + n - 1
        if len(basic_cells) < needed:
            for i in range(m):
                for j in range(n):
                    if allocation[i][j] == 0 and (i, j) not in basic_cells:
                        basic_cells.append((i, j))
                        allocation[i][j] = 0
                        if len(basic_cells) == needed:
                            break
                if len(basic_cells) == needed:
                    break

        adj = {f"r{i}": [] for i in range(m)}
        adj.update({f"c{j}": [] for j in range(n)})
        for (i, j) in basic_cells:
            adj[f"r{i}"].append((f"c{j}", (i, j)))
            adj[f"c{j}"].append((f"r{i}", (i, j)))

        u = {i: None for i in range(m)}
        v = {j: None for j in range(n)}
        u[0] = Fraction(0)
        visited = {"r0"}
        frontier = ["r0"]
        while frontier:
            node = frontier.pop()
            for neighbor, (ci, cj) in adj[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                if node.startswith("r") and neighbor.startswith("c"):
                    i, j = int(node[1:]), int(neighbor[1:])
                    v[j] = Fraction(cost[i][j]) - u[i]
                else:
                    j, i = int(node[1:]), int(neighbor[1:])
                    u[i] = Fraction(cost[i][j]) - v[j]
                frontier.append(neighbor)

        opp_cost = {}
        for i in range(m):
            for j in range(n):
                if allocation[i][j] == 0:
                    opp_cost[(i, j)] = Fraction(cost[i][j]) - (u[i] + v[j])

        if verbose:
            print(f"\n--- MODI Iteration {iteration} ---")
            print("u:", {f"u{i+1}": str(u[i]) for i in range(m)})
            print("v:", {f"v{j+1}": str(v[j]) for j in range(n)})
            print("Opportunity costs (non-basic cells):")
            for (i, j), d in opp_cost.items():
                print(f"  d{i+1}{j+1} = {d}")

        negative_cells = {k: val for k, val in opp_cost.items() if val < 0}
        if not negative_cells:
            if verbose:
                print("\n>>> All opportunity costs >= 0  => solution is OPTIMAL.\n")
            break

        entering = min(negative_cells, key=lambda k: negative_cells[k])
        if verbose:
            print(f"\nMost negative opportunity cost at x{entering[0]+1}{entering[1]+1} "
                  f"= {negative_cells[entering]}  -> this cell enters the basis.")

        start, target = f"r{entering[0]}", f"c{entering[1]}"
        path = _find_path(adj, start, target)
        loop_cells = [entering]
        for k in range(len(path) - 1):
            node_a, node_b = path[k], path[k + 1]
            for neighbor, cell in adj[node_a]:
                if neighbor == node_b:
                    loop_cells.append(cell)
                    break

        signs = [1 if idx % 2 == 0 else -1 for idx in range(len(loop_cells))]

        minus_cells = [loop_cells[idx] for idx, s in enumerate(signs) if s == -1]
        theta = min(allocation[i][j] for (i, j) in minus_cells)

        if verbose:
            print(f"Closed loop: {[(f'x{i+1}{j+1}', '+' if s==1 else '-') for (i, j), s in zip(loop_cells, signs)]}")
            print(f"theta (shift amount) = {theta}")

        for (i, j), s in zip(loop_cells, signs):
            allocation[i][j] += s * theta

        if verbose:
            _print_allocation(allocation)

    return allocation


def _find_path(adj, start, target):
    visited = {start}
    queue = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        if node == target:
            return path
        for neighbor, _ in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    raise Exception("No path found - allocation is not a valid spanning tree (check degeneracy).")


def _print_allocation(allocation):
    print("Updated allocation:")
    for row in allocation:
        print("  ", row)


def main():
    supply = [250, 300, 400]
    demand = [200, 225, 275, 250]
    cost = [
        [11, 13, 17, 14],
        [16, 18, 14, 10],
        [21, 24, 13, 10],
    ]

    assert sum(supply) == sum(demand), "Problem must be balanced (total supply = total demand)."

    print("=" * 60)
    print("STAGE 1: Vogel's Approximation Method (initial solution)")
    print("=" * 60)
    vam_allocation = vogel_approximation_method(supply, demand, cost)
    print("\nVAM initial allocation:")
    _print_allocation(vam_allocation)
    vam_cost = sum(vam_allocation[i][j] * cost[i][j] for i in range(3) for j in range(4))
    print(f"VAM initial total cost = {vam_cost}")

    print("\n" + "=" * 60)
    print("STAGE 2: MODI Method (optimality test & improvement)")
    print("=" * 60)
    optimal_allocation = modi_method(vam_allocation, cost, supply, demand)

    total_cost = sum(optimal_allocation[i][j] * cost[i][j] for i in range(3) for j in range(4))

    print("=" * 60)
    print("FINAL OPTIMAL SHIPMENT PLAN")
    print("=" * 60)
    for i, row in enumerate(optimal_allocation):
        for j, qty in enumerate(row):
            if qty > 0:
                print(f"  Ship {qty} units from O{i+1} -> D{j+1}  (unit cost {cost[i][j]})")
    print(f"\n  MINIMUM TOTAL TRANSPORTATION COST = {total_cost}")
    print("=" * 60)


if __name__ == "__main__":
    main()