# LP: scalar version
# This script solves the LP on page 18 of Vanderbei:
# maximize      −2x − y
# subject to 
#               −x + y ≤ −1
#               −x − 2y ≤ −2
#               y ≤ 1
#               x, y ≥ 0.

# load necessary packages
# make sure you have installed these
using JuMP, GLPK

# using GLPK solver
model = Model(GLPK.Optimizer) 

# define variables
@variable(model, x >= 0)
# equivalent way:
# @variable(model, x)
# @constraint(model, x >= 0)
@variable(model, 0 <= y <= 1)

# @variable(model, lb <= x <= ub, start=x0) # define varaibles with lower/upper bounds and initial value

# add constraints
@constraint(model, -x + y <= -1)
@constraint(model, -x - 2y <= -2)


# define objective function
@objective(model, Max, -2x - y)

# print model
print(model)

# To solve the optimization problem, we call the optimize function.
optimize!(model)

# show status after optimizing -- is the problem infeasible or unbounded?
@show termination_status(model)
@show primal_status(model)
@show dual_status(model)

# show the optimal values (multiple ways to do this)
println("optimal x = ", value(x))   # method 1: your own print statement with value() command
@show value(x);                     # method 2: using @show with built-in printing
y_opt = JuMP.value(y);
println("optimal y = ", y_opt);     # method 3: save optimal value in a new variable
# @show value(y);
@show objective_value(model);