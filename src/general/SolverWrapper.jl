export MockSolverResult
export params

# Wrap Optim
params(x::Optim.UnivariateOptimizationResults) = [x.minimum]
params(x::Optim.MultivariateOptimizationResults) = x.minimum

# Wrap NLsolve
params(x::NLsolve.SolverResults) = x.zero

# Wrap mockup solution
immutable MockSolverResult{T <: Number}
    params::Vector{T}
end

params(x::MockSolverResult) = x.params