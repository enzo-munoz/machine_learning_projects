#!/usr/bin/env python
# coding: utf-8

# ## Problem Description: Optimizing Energy Production and Distribution in a Renewable Energy Grid
# 
# A utility company operates a renewable energy grid with energy sources such as solar, wind, and hydro, with the option of backup from conventional sources. The objective is to optimize the amount of energy produced by each source to meet demand across various locations while minimizing costs and adhering to environmental and operational constraints.
# 
# ### Energy Sources and Production Plants
# 
# Each energy source has three plants, with specific capacities and costs:
# 
# | Energy Source | Plant | Production Capacity (MW) | Cost of Production ($/MWh) |
# | ------------- | ----- | ------------------------ | -------------------------- |
# | Solar         | S1    | 0 - 45                  | 20                         |
# |               | S2    | 0 - 80                  | 18                         |
# |               | S3    | 0 - 90                  | 19                         |
# | Wind          | W1    | 0 - 100                 | 15                         |
# |               | W2    | 0 - 110                 | 14                         |
# |               | W3    | 0 - 140                 | 16                         |
# | Hydro         | H1    | 0 - 155                 | 12                         |
# |               | H2    | 0 - 220                 | 11                         |
# |               | H3    | 0 - 320                 | 12                         |
# | Conventional  | B1    | 0 - 400                 | 40                         |
# 
# ### Demand Centers and Transmission Limits
# 
# The company supplies energy to four demand centers with specific demand requirements, transmission losses, and maximum transmission capacities:
# 
# | Demand Center | Demand (MW) | Transmission Loss (%) | Max Transmission (MW) |
# | ------------- | ----------- | --------------------- | ---------------------- |
# | D1            | 400         | 5                     | 100                    |
# | D2            | 375         | 7                     | 130                    |
# | D3            | 300         | 6                     | 80                     |
# | D4            | 280         | 10                    | 120                    |
# 
# ### Objective
# 
# The goal is to minimize the total cost of energy production and distribution while meeting demand at each demand center. Costs include:
# - **Production Costs**: The cost to produce energy at each plant.
# - **Transmission Losses**: The cost of energy lost in transmission from each plant to each demand center.
# 
# ### Constraints
# 
# 1. **Demand Satisfaction**: Each demand center must receive enough energy to meet its demand.
# 2. **Production Capacity**: Energy produced by each plant must not exceed its maximum capacity.
# 3. **Transmission Limits**: Energy transmitted from each plant to each demand center must respect infrastructure limits.
# 4. **Environmental Constraint**: Use of conventional energy must be limited to no more than 20% of total energy produced.
# 
# ---
# 
# 
# 

# ### Formulation with pyomo

# In[ ]:


from pyomo.environ import *

# Define the Model
model = ConcreteModel('Energy Production and Distribution Optimization')

# Sets for Plants and Demand Centers
model.sources = Set(initialize=['S1', 'S2', 'S3', 'W1', 'W2', 'W3', 'H1', 'H2', 'H3', 'B1'], doc='Energy production plants')
model.demand_centers = Set(initialize=['D1', 'D2', 'D3', 'D4'], doc='Demand centers')

# Parameters for Production Capacities and Costs
production_capacity = {
    'S1': 45, 'S2': 80, 'S3': 90,  # Solar
    'W1': 100, 'W2': 110, 'W3': 140, # Wind
    'H1': 155, 'H2': 220, 'H3': 320, # Hydro
    'B1': 400  # Conventional backup
}

production_cost = {
    'S1': 20, 'S2': 18, 'S3': 19,
    'W1': 15, 'W2': 14, 'W3': 16,
    'H1': 12, 'H2': 11, 'H3': 12,
    'B1': 40
}

model.capacity = Param(model.sources, initialize=production_capacity, doc='Max production capacity for each plant')
model.cost = Param(model.sources, initialize=production_cost, doc='Cost of production for each plant')

# Parameters for Demand and Transmission Losses
demand = {'D1': 400, 'D2': 375, 'D3': 300, 'D4': 280}
# Updated transmission_loss parameter with all combinations defined
transmission_loss = {
    ('S1', 'D1'): 5, ('S1', 'D2'): 5, ('S1', 'D3'): 5, ('S1', 'D4'): 5,
    ('S2', 'D1'): 5, ('S2', 'D2'): 5, ('S2', 'D3'): 5, ('S2', 'D4'): 5,
    ('S3', 'D1'): 5, ('S3', 'D2'): 5, ('S3', 'D3'): 5, ('S3', 'D4'): 5,

    ('W1', 'D1'): 7, ('W1', 'D2'): 7, ('W1', 'D3'): 7, ('W1', 'D4'): 7,
    ('W2', 'D1'): 7, ('W2', 'D2'): 7, ('W2', 'D3'): 7, ('W2', 'D4'): 7,
    ('W3', 'D1'): 7, ('W3', 'D2'): 7, ('W3', 'D3'): 7, ('W3', 'D4'): 7,

    ('H1', 'D1'): 6, ('H1', 'D2'): 6, ('H1', 'D3'): 6, ('H1', 'D4'): 6,
    ('H2', 'D1'): 6, ('H2', 'D2'): 6, ('H2', 'D3'): 6, ('H2', 'D4'): 6,
    ('H3', 'D1'): 6, ('H3', 'D2'): 6, ('H3', 'D3'): 6, ('H3', 'D4'): 6,

    ('B1', 'D1'): 10, ('B1', 'D2'): 10, ('B1', 'D3'): 10, ('B1', 'D4'): 10
}


model.demand = Param(model.demand_centers, initialize=demand, doc='Demand at each center')
model.transmission_loss = Param(model.sources , model.demand_centers, initialize=transmission_loss, doc='Transmission loss percentage')

# Decision Variables
model.production = Var(model.sources, domain=NonNegativeReals, doc='Energy production at each plant')
model.distribution = Var(model.sources, model.demand_centers, domain=PositiveReals, doc='Energy distributed from each plant to each demand center')

# Objective Function: Minimize Total Cost (Production + Transmission Losses)
def objective_rule(model):
    production_costs = sum(model.cost[p] * model.production[p] for p in model.sources)
    transmission_costs = sum(model.distribution[p, d] * model.transmission_loss[p, d] / 100 for p in model.sources for d in model.demand_centers)
    return production_costs + transmission_costs

model.objective = Objective(rule=objective_rule, sense=minimize)

# Constraints

# 1. Demand Satisfaction Constraint: Demand at each center must be met
def demand_satisfaction_rule(model, d):
    return sum(model.distribution[p, d] for p in model.sources) >= model.demand[d]

model.demand_satisfaction = Constraint(model.demand_centers, rule=demand_satisfaction_rule)

# 2. Production Capacity Constraint: Each plant's production should not exceed its capacity
def production_capacity_rule(model, p):
    return model.production[p] <= model.capacity[p]

model.production_capacity = Constraint(model.sources, rule=production_capacity_rule)

# 3. Transmission Limit Constraint: Distributed energy cannot exceed production at each plant
def transmission_limit_rule(model, p):
    return sum(model.distribution[p, d] for d in model.demand_centers) <= model.production[p]

model.transmission_limit = Constraint(model.sources, rule=transmission_limit_rule)

# 4. Environmental Constraint: Conventional energy (B1) limited to 20% of total production
def environmental_constraint_rule(model):
    return model.production['B1'] <= 0.2 * sum(model.production[p] for p in model.sources)

model.environmental_constraint = Constraint(rule=environmental_constraint_rule)


# In[ ]:


# Solve the model
Solver = SolverFactory('gurobi')
Results = Solver.solve(model)

# Display solution in a clear and detailed way
print("### Energy Production and Distribution Optimization Results ###\n")

# Objective value (Total Cost)
print(f"Total Cost (Objective Value): ${model.objective():,.2f}\n")

# Production Levels at Each Plant
print("Energy Production Levels (MW):")
for plant in model.sources:
    print(f"  Plant {plant}: {model.production[plant]():.2f} MW")

print("\nEnergy Distribution to Demand Centers (MW):")
for plant in model.sources:
    for center in model.demand_centers:
        print(f"  Energy from {plant} to {center}: {model.distribution[plant, center]():.2f} MW")

# Check Demand Satisfaction
print("\nDemand Satisfaction at Each Center (MW):")
for center in model.demand_centers:
    total_received = sum(model.distribution[plant, center]() for plant in model.sources)
    print(f"  Demand Center {center}: {total_received:.2f} MW received (Demand: {model.demand[center]} MW)")

# Display if Environmental Constraint is Met
total_production = sum(model.production[plant]() for plant in model.sources)
conventional_production = model.production['B1']()
if conventional_production <= 0.2 * total_production:
    print("\nEnvironmental Constraint: Conventional energy usage is within allowed limit.")
else:
    print("\nEnvironmental Constraint: Warning! Conventional energy usage exceeds allowed limit.")

# Print summary of results
print("\nSummary:")
print(f"  Total Renewable Production: {total_production - conventional_production:.2f} MW")
print(f"  Total Conventional Production: {conventional_production:.2f} MW")
print(f"  Total Production: {total_production:.2f} MW")
print(f"  Transmission Loss Cost: ${sum(model.distribution[p, d]() * model.transmission_loss[p, d] / 100 for p in model.sources for d in model.demand_centers):,.2f}")
print("-------------------------------------------------------------------")


# In[ ]:


# Enable dual variables collection
model.dual = Suffix(direction=Suffix.IMPORT)

# Solve the model
solver = SolverFactory('gurobi')
results = solver.solve(model)

# Check if the solver found an optimal solution
if (results.solver.status != 'ok') or (results.solver.termination_condition != 'optimal'):
    print("Solver did not find an optimal solution.")
else:
    # Sensitivity Analysis: Access dual values for constraints
    print("\nSensitivity Analysis (Constraint Duals):")

    # Duals for demand satisfaction constraints
    for d in model.demand_centers:
        if model.demand_satisfaction[d] in model.dual:
            print(f"Dual for demand_satisfaction[{d}]: {model.dual[model.demand_satisfaction[d]]:.2f}")
        else:
            print(f"Dual for demand_satisfaction[{d}]: Not available")

    # Duals for production capacity constraints
    for p in model.sources:
        if model.production_capacity[p] in model.dual:
            print(f"Dual for production_capacity[{p}]: {model.dual[model.production_capacity[p]]:.2f}")
        else:
            print(f"Dual for production_capacity[{p}]: Not available")

    # Duals for transmission limit constraints
    for p in model.sources:
        if model.transmission_limit[p] in model.dual:
            print(f"Dual for transmission_limit[{p}]: {model.dual[model.transmission_limit[p]]:.2f}")
        else:
            print(f"Dual for transmission_limit[{p}]: Not available")

    # Dual for environmental constraint
    if model.environmental_constraint in model.dual:
        print(f"Dual for environmental_constraint: {model.dual[model.environmental_constraint]:.2f}")
    else:
        print("Dual for environmental_constraint: Not available")


# In[ ]:


from pyomo.environ import *

# Create a model instance
model_1 = ConcreteModel()

# Sets for plants, demand centers, and months
plants = ['S1', 'S2', 'S3', 'W1', 'W2', 'W3', 'H1', 'H2', 'H3', 'B1']
demand_centers = ['D1', 'D2', 'D3', 'D4']
months = range(1, 13)  # 12-month planning period
energy_sources = {'Solar': ['S1', 'S2', 'S3'],
                  'Wind': ['W1', 'W2', 'W3'],
                  'Hydro': ['H1', 'H2', 'H3'],
                  'Conventional': ['B1']}

# Parameters for production cost, capacity, demand, and maintenance
production_cost = {'S1': 20, 'S2': 18, 'S3': 19,
                   'W1': 15, 'W2': 14, 'W3': 16,
                   'H1': 12, 'H2': 11, 'H3': 12,
                   'B1': 40}
capacity = {'S1': 60, 'S2': 70, 'S3': 65,
            'W1': 110, 'W2': 100, 'W3': 130,
            'H1': 160, 'H2': 210, 'H3': 185,
            'B1': 220}
fixed_activation_cost = 1000  # Fixed cost for each active plant
min_production_threshold = 10  # Minimum production for active plants
maintenance_threshold = 50     # Lower threshold to ensure maintenance is triggered
maintenance_cost = 500         # Fixed maintenance cost for each maintenance month
monthly_demand = {'D1': [350, 260, 270, 160, 180, 220, 290, 310, 240, 200, 220, 200],
                  'D2': [200, 210, 220, 230, 240, 250, 260, 270, 250, 240, 230, 220],
                  'D3': [100, 105, 110, 115, 120, 125, 130, 135, 125, 120, 115, 110],
                  'D4': [180, 190, 200, 210, 220, 230, 240, 250, 230, 220, 210, 200]}
transmission_loss = {'D1': 0.05, 'D2': 0.07, 'D3': 0.06, 'D4': 0.10}
max_transmission = {'D1': 100, 'D2': 130, 'D3': 80, 'D4': 120}

# Decision Variables
model_1.x = Var(plants, demand_centers, months, within=NonNegativeReals)  # Energy produced and transmitted monthly
model_1.y = Var(plants, months, within=Binary)  # Binary variable for plant activation (1 if active, 0 otherwise)
model_1.maintenance = Var(plants, months, within=Binary)  # Binary variable for maintenance (1 if maintenance, 0 otherwise)

# Objective: Minimize total cost (production + activation + maintenance)
model_1.obj = Objective(expr=sum(production_cost[i] * sum(model_1.x[i, j, m] for j in demand_centers for m in months)
                                for i in plants) +
                        sum(fixed_activation_cost * model_1.y[i, m] for i in plants for m in months) +
                        sum(maintenance_cost * model_1.maintenance[i, m] for i in plants for m in months),
                        sense=minimize)

# Constraints List
model_1.constraints = ConstraintList()

# Demand satisfaction with transmission losses
for j in demand_centers:
    for m in months:
        model_1.constraints.add(
            sum(model_1.x[i, j, m] * (1 - transmission_loss[j]) for i in plants) >= monthly_demand[j][m]
        )

# Production capacity for each plant per month
for i in plants:
    for m in months:
        model_1.constraints.add(
            sum(model_1.x[i, j, m] for j in demand_centers) <= capacity[i] * (1 - model_1.maintenance[i, m])
        )

# Transmission limit for each plant-demand center-month
for i in plants:
    for j in demand_centers:
        for m in months:
            model_1.constraints.add(
                model_1.x[i, j, m] <= max_transmission[j]
            )

# Minimum production threshold for active plants each month
for i in plants:
    for m in months:
        model_1.constraints.add(
            sum(model_1.x[i, j, m] for j in demand_centers) >= min_production_threshold * model_1.y[i, m]
        )

# Maintenance enforcement if a plant operates above threshold for 3 consecutive months
for i in plants:
    for m in range(1, 10):  # Only consider up to month 10 for 3-month consecutive check
        model_1.constraints.add(
            model_1.y[i, m] + model_1.y[i, m+1] + model_1.y[i, m+2] - 3 * model_1.maintenance[i, m+3] <= 2
        )

# Limit on the number of active plants per energy source
max_active_plants = {'Solar': 2, 'Wind': 2, 'Hydro': 2, 'Conventional': 1}
for source, max_plants in max_active_plants.items():
    for m in months:
        model_1.constraints.add(
            sum(model_1.y[i, m] for i in energy_sources[source]) <= max_plants
        )

# Conditional constraint: If `W1` is active, then `H1` must also be active in the same month
for m in months:
    model_1.constraints.add(
        model_1.y['W1', m] <= model_1.y['H1', m]
    )


# In[ ]:


# Create a solver and solve the model
solver = SolverFactory('gurobi')
result = solver.solve(model_1, tee=True)

# Check solver status
if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
    # Display results
    print("\n=== Optimal Solution Found ===\n")

    # Total cost breakdown
    total_production_cost = sum(production_cost[i] * sum(model_1.x[i, j, m].value for j in demand_centers for m in months) for i in plants)
    total_activation_cost = sum(fixed_activation_cost * model_1.y[i, m].value for i in plants for m in months)
    total_maintenance_cost = sum(maintenance_cost * model_1.maintenance[i, m].value for i in plants for m in months)
    total_cost = model_1.obj()

    print(f"Total Production Cost: ${total_production_cost:.2f}")
    print(f"Total Activation Cost: ${total_activation_cost:.2f}")
    print(f"Total Maintenance Cost: ${total_maintenance_cost:.2f}")
    print(f"Overall Total Cost: ${total_cost:.2f}\n")

    # Display monthly production and distribution plan
    print("=== Monthly Production and Distribution Plan ===")
    for m in months:
        print(f"\nMonth {m}:")
        for i in plants:
            production = sum(model_1.x[i, j, m].value or 0 for j in demand_centers)  # Ensure value is non-null
            print(f"  Plant {i} - Production: {production:.2f} MW")
            for j in demand_centers:
                distribution = model_1.x[i, j, m].value or 0
                print(f"    Distribution to {j}: {distribution:.2f} MW")

    # Display activation and maintenance status per plant per month
    print("\n=== Plant Activation and Maintenance Schedule ===")
    for i in plants:
        print(f"\nPlant {i}:")
        maintenance_months = []
        for m in months:
            is_active = model_1.y[i, m].value > 0.5 if model_1.y[i, m].value is not None else False
            is_maintenance = model_1.maintenance[i, m].value > 0.5 if model_1.maintenance[i, m].value is not None else False
            status = "Active" if is_active else "Inactive"
            maintenance_status = "Maintenance" if is_maintenance else "No Maintenance"
            print(f"  Month {m} - Status: {status}, Maintenance: {maintenance_status}")
            if is_maintenance:
                maintenance_months.append(m)

        # Summary for each plant's maintenance schedule
        if maintenance_months:
            print(f"  Maintenance Months: {', '.join(map(str, maintenance_months))}")
        else:
            print("  No Maintenance Required")

else:
    print("Solver did not find an optimal solution. Check model constraints and solver settings.")

