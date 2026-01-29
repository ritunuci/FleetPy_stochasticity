import gurobipy as gp
from gurobipy import GRB

try:
    print("Starting Gurobi WLS test...")
    model = gp.Model("wls_test")
    x = model.addVar(name="x")
    y = model.addVar(name="y")
    model.setObjective(x + y, GRB.MAXIMIZE)
    model.addConstr(x + 2 * y <= 4, "c0")
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print("Gurobi WLS test succeeded!")
        print(f"x = {x.X}, y = {y.X}")
    else:
        print(f"Gurobi failed to solve the model. Status: {model.Status}")
except Exception as e:
    print(f"Error: {e}")

