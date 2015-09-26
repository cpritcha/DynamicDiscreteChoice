export ConfInt, OPG, vcov

abstract ConfInt

vcov(result) = error("not implemented")

immutable OPG <: ConfInt end

immutable ParametricBootstrap <: ConfInt
    reps::Int
end

immutable ResampleBootstrap <: ConfInt 
    reps::Int
end