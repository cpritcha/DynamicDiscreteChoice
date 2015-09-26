export NFXPSolver
export stage1_solve, stage2_solve

immutable NFXPSolver{T} end

"""
The stage 1 solver solves the first stage of the NFXP problem. 

The stage 1 problem could be an constrained or uncontrained problem
"""
function stage1_solve(method::NFXPSolver{Any}, problem)
    error("not implemented")
end

function stage2_solve(method::NFXPSolver{Any}, problem)
    error("not implemented")
end

stage1_n_params(method::NFXPSolver{Any}) = error("not implemented")
stage2_n_params(method::NFXPSolver{Any}) = error("not implemented")