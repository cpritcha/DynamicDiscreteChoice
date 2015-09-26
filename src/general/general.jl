export General

module General
import Distributions: Categorical, params
import Optim
import NLsolve

sample(model, start_state; n_reps=500, n_runs=10) = error("not implemented")

include("ConfidenceInterval.jl")
include("SolverWrapper.jl")
include("Transition.jl")
include("Utility.jl")
include("VariableSpace.jl")

end