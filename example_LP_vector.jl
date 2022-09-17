# LP: vector version
# This script solves 2 LPs.

##############################################
# First, Exercise 2.1 of Vanderbei:
#=
maximize     6x1 + 8x2 + 5x3 + 9x4
subject to   2x1 +  x2 +  x3 + 3x4 ≤ 5
              x1 + 3x2 +  x3 + 2x4 ≤ 3
              x1,   x2,   x3,   x4 ≥ 0.
=#
##############################################

using JuMP, GLPK, LinearAlgebra

vector_model = Model(GLPK.Optimizer)

# constraint matrix coefficients
A = [2 1 1 3;
     1 3 1 2]

# constraint right-hand side
b = [5; 3]

# objective function coefficients
c = [6; 8; 5; 9]

# vectorized variable: @variable(vector_model, x[1:n])
@variable(vector_model, x[1:4] >= 0)
# equivalent way:
# @variable(vector_model, x[1:4])
# @constraint(vector_model, x .>= 0)

# vectorized constraints
@constraint(vector_model, A * x .<= b) # define through Linear algebra
# Or elementwise:
# n = length(b);
# @constraint(vector_model, [i=1:n],sum(A[i,j]*x[j] for j=1:4) <= b[i])
# The following (without the ".") causes an error:
# @constraint(vector_model, A * x <= b)


@objective(vector_model, Max, c' * x)
# equivalent way if you are "using LinearAlgebra"
# @objective(vector_model, Max, dot(c, x))

print(vector_model)

optimize!(vector_model)

@show value.(x);
# @show value(x); # This is the wrong format; it will cause an error
# x_opt = JuMP.value.(x);
# println("value(x)= ", x_opt);
@show objective_value(vector_model);

###########################################################
#=
Second, Exercise 2.11 of Vanderbei:
minimize      x12 + 8x13 + 9x14 + 2x23 + 7x24 + 3x34
subject to    x12 +  x13 +  x14                      ≥ 1
             −x12               +  x23 +  x24        = 0
                  −  x13        −  x23        + x34  = 0
                            x14        +  x24 + x34  ≤ 1
              x12,   x13, . . . . . . . . . . , x34  ≥ 0.
=#
###########################################################

vector_model = Model(GLPK.Optimizer)

# decision variables, stored as an upper triangular matrix
@variable(vector_model, x[i=1:3,j=i+1:4] >= 0)

# constraints --- set up matrices A so that a single constraint sums the elementwise product A[i,j]*x[i,j]
A1 = zeros(3,4)
A1[1, [2,3,4]] = [1,1,1]
@constraint(vector_model, sum( A1[i,j]*x[i,j] for i=1:3, j=i+1:4 ) >= 1)
# remark: A.*x will cause an error, since x is stored sparsely
# i.e., only the upper triangular elements of x exist as a matrix
A2 = zeros(3,4)
A2[1, 2]       = -1
A2[2, [3,4]]   = [1,1]
@constraint(vector_model, sum( A2[i,j]*x[i,j] for i=1:3, j=i+1:4 ) == 0)
A3 = zeros(3,4)
A3[[1,2],3]    = [-1, -1]
A3[3, 4]       = 1
@constraint(vector_model, sum( A3[i,j]*x[i,j] for i=1:3, j=i+1:4 ) == 0)
A4 = zeros(3,4)
A4[[1,2,3], 4] = [1,1,1]
@constraint(vector_model, sum( A4[i,j]*x[i,j] for i=1:3, j=i+1:4 ) <= 1)

# objective function
c = [0 1 8 9; # coefficients
     0 0 2 7;
     0 0 0 3]
@objective(vector_model, Min, sum( c[i,j]*x[i,j] for i=1:3, j=i+1:4) )

print(vector_model)

optimize!(vector_model)

@show value.(x);
@show objective_value(vector_model);