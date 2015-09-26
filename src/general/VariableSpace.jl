export VariableSpace, DiscreteVariableSpace

abstract VariableSpace

type DiscreteVariableSpace{V,T,N} <: VariableSpace
    names::Vector{Symbol}
    levels::T
    size::Tuple{Vararg{Int}}
end

function DiscreteVariableSpace{T}(spacename::Symbol, names::Vector{Symbol}, levels::T)
    size = tuple([length(level) for level in levels]...)
    N = length(levels)
    return DiscreteVariableSpace{Val{spacename}, T, N}(names, levels, size)
end

Base.sub2ind(x::DiscreteVariableSpace, inds::Tuple{Vararg{Int}}) = sub2ind(x.size, inds...)
Base.ind2sub(x::DiscreteVariableSpace, i::Int) = ind2sub(x.size, i)
Base.size(x::DiscreteVariableSpace) = x.size
Base.length(x::DiscreteVariableSpace) = prod(x.size)

@generated function Base.getindex{V,T,N}(x::DiscreteVariableSpace{V,T,N}, inds::Tuple{Vararg{Int}})
	expr = :(())
	for i=1:N
		push!(expr.args, :(x.levels[$i][inds[$i]]))
	end
	expr
end
Base.getindex{T <: VariableSpace}(x::T, i::Int) = getindex(x, ind2sub(x, i))