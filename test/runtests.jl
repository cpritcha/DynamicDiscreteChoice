module TestRunner
using DynamicDiscreteChoice
using FactCheck
import Distributions

const D = Distributions
const G = General
include("general/VariableSpace.jl")
include("general/Transition.jl")

end