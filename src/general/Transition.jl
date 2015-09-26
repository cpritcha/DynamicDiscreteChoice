export Transition, DiscreteTransition, SparseCategoricalTransition, SparseVec
export condprob, expectation, prob, sample, set_params!

abstract Transition

condprob(t::Transition, s, a) = error("not implemented")
prob(t::Transition, s, a, s_next) = error("not implemented")
sample(t::Transition, s, a) = error("not implemented")
set_params!(t::Transition, θ::AbstractVector) = error("not implemented")
# -------------------------

abstract DiscreteTransition <: Transition

immutable SparseVec
	inds::Vector{Int64}
	vals::Vector{Float64}
end

Base.length(x::SparseVec) = length(x.inds)

"""
Computes the expected value of value function at a particular state
"""
function expectation(cp::SparseVec, V::Vector{Float64})
	v = 0.0
	for i=1:length(cp)
		v += cp.vals[i]*V[cp.inds[i]]
	end
	return v
end

immutable SparseCategoricalTransition <: DiscreteTransition
    transition::Matrix{Vector{Int64}}
    categorical::Categorical
    statespace
    actionspace
end
function Base.copy(t::SparseCategoricalTransition)
    SparseCategoricalTransition(
        t.transition,
        Categorical(copy(t.categorical.p)),
        t.statespace,
        t.actionspace)
end

function condprob(t::SparseCategoricalTransition, sind, aind)
    linear_state_ind = sub2ind(t.statespace, sind)
    linear_action_ind = sub2ind(t.actionspace, aind)
    
    linear_state_inds = t.transition[linear_state_ind, linear_action_ind]
    return SparseVec(linear_state_inds, t.categorical.p)
end

function _is_edge(sind, sind_max)
    n = length(sind_max)
    for i=1:n
        if (sind[i] < sind_max[i]) && (sind[i] > 1)
            return false
        end
    end
    return true
end

function prob(t::SparseCategoricalTransition, sind, aind, sind_next) # FIXME: bad edges
    linear_sind = sub2ind(t.statespace, sind)
    linear_aind = sub2ind(t.actionspace, aind)
    linear_sind_next = sub2ind(t.statespace, sind_next)
    
    feasible_linear_sinds = t.transition[linear_sind, linear_aind]
    
    ind = findfirst(feasible_linear_sinds, linear_sind_next)

    if ind > 0    # if the transition is feasible
        
        if _is_edge(sind_next, size(t.statespace))
            probability = t.categorical.p[ind]
            n_feasible_transitions = length(feasible_linear_sinds)
            
            i = ind + 1
            current_linear_sind = linear_sind_next
            while i <= n_feasible_transitions
                current_linear_sind = feasible_linear_sinds[i]
                
                if current_linear_sind != linear_sind_next
                    return probability
                end
                probability += t.categorical.p[i]
                i += 1
            end
            return probability
        else
            return t.categorical.p[ind]
        end
    else
        error("P(s' = $sind_next | s = $sind, a = $aind) = 0")
    end 
end

function sample(t::SparseCategoricalTransition, sind, aind)
    ind = rand(t.categorical)
    return ind2sub(t.statespace, condprob(t, sind, aind).inds[ind])
end

function set_params!(t::SparseCategoricalTransition, θ::AbstractVector)
    θ_old = t.categorical.p
    n = length(θ_old)
    n == length(θ) ? nothing : error("length mismatch!\nexpected: $n, got: $(length(θ))")
    for i=1:n
        θ_old[i] = θ[i]
    end
end
# ============================

abstract TransitionFamily{T <: Transition}
Base.call(tf::TransitionFamily, θ) = error("not implemented") 

# -----------------------------