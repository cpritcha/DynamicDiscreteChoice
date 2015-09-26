export Utility, UtilityFamily

immutable Utility{T}
    θ::Vector{Float64}
end
Base.call(u::Utility, state, action) = error("not implemented")

immutable UtilityFamily{T} end
Base.call{T}(u::UtilityFamily{T}, θ) = Utility{T}(θ)