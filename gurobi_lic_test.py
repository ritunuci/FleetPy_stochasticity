import gurobipy as gp
from gurobipy import GRB

print("Starting license check...")

with gp.Env() as env:

    try:
        print("LICENSEID:", env.getParam("LICENSEID"))
    except:
        pass

    try:
        print("WLSACCESSID:", env.getParam("WLSACCESSID"))
    except:
        pass

    try:
        print("Token server:", env.getParam("TOKENSERVER"))
    except:
        pass

    try:
        print("Compute server:", env.getParam("COMPUTESERVER"))
    except:
        pass

    print("\nEnvironment initialized successfully.")

    with gp.Model("test", env=env) as model:
        x = model.addVar()
        y = model.addVar(name="y")
        model.setObjective(x + y, GRB.MAXIMIZE)
        model.addConstr(x + 2 * y <= 4, "c0")
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            print("Gurobi Solved Successfully!")
            print(f"x = {x.X}, y = {y.X}")
        else:
        	print(f"Gurobi failed to solve the model. Status: {model.Status}")
print("\nDone.")