using Optim
import Distributions: Gumbel
import Calculus: gradient
abstract VariableSpace
abstract Transition
abstract RandTransition

type DiscreteVariableSpace{T,N} <: VariableSpace
	names::Vector{Symbol}
	levels::T
	size::Tuple{Vararg{Int}}
end

type ValueIterationState
	V::Vector{Float64}
	Vnew::Vector{Float64}
	k::Int
end
function ValueIterationState{T <: VariableSpace}(statespace::T)
	n = length(statespace)
	V = zeros(Float64, n)
	Vnew = copy(V)
	return ValueIterationState(V, Vnew, 0)
end

immutable Params
	transition::Vector{Float64}
	utility::Vector{Float64}
end

type NFXPModel
	statespace
	actionspace
	transition
	utility
	beta
end

type FittedNFXPModel
	vi_state::ValueIterationState
	model::NFXPModel
	s1_opt
	s2_opt
	data
end

doc"""
Create a variable space
"""
function DiscreteVariableSpace{T}(names::Vector{Symbol}, levels::T)
	size = tuple([length(level) for level in levels]...)
	N = length(levels)
	return DiscreteVariableSpace{T, N}(names, levels, size)
end

Base.sub2ind(x::DiscreteVariableSpace, inds::Tuple{Vararg{Int}}) = sub2ind(x.size, inds...)
Base.ind2sub(x::DiscreteVariableSpace, i::Int) = ind2sub(x.size, i)
Base.size(x::DiscreteVariableSpace) = x.size
Base.length(x::DiscreteVariableSpace) = prod(x.size)

function findval(arr, x)
	ind = 0
	for i=1:length(arr)
		if arr[i]==x
			ind = i
			break
		end
	end
	return ind
end
@generated function Base.getindex{T,N}(x::DiscreteVariableSpace{T,N}, inds::Tuple{Vararg{Int}})
	expr = :(())
	for i=1:N
		push!(expr.args, :(x.levels[$i][inds[$i]]))
	end
	expr
end
Base.getindex{T <: VariableSpace}(x::T, i::Int) = getindex(x, ind2sub(x, i))

# ----- SparseVec
immutable SparseVec
	inds::Vector{Int64}
	vals::Vector{Float64}
end

Base.length(x::SparseVec) = length(x.inds)

doc"""
Computes the expected value of value function at a particular state
"""
function expectation(cp::SparseVec, V::Vector{Float64})
	v = 0.0
	for i=1:length(cp)
		v += cp.vals[i]*V[cp.inds[i]]
	end
	return v
end

# ----- Transition
abstract Transition

function valueiterate(model::NFXPModel, sind, V, params_u)
	utility = model.utility
	transition = model.transition
	beta = model.beta
	actionspace = model.actionspace
	statespace = model.statespace
	state = statespace[sind]
	tmp = 0.0
	for i = 1:length(actionspace)
		aind = ind2sub(actionspace, i)
		action = actionspace[aind]
		cp = condprob(transition, sind, aind)
		tmp += exp(utility(state, action, params_u) + beta*expectation(cp, V))
	end
	return log(tmp)
end

function maxabsdiff(x, y)
	n = length(x)
	res = 0.0
	for i=1:n
		cand_res = abs(x[i]-y[i])
		if cand_res > res
			res = cand_res
		end
	end 
	res
end

function valueiteration!(vi_state::ValueIterationState, model::NFXPModel, params_u; ϵ=1e-6)
	transition = model.transition
	beta = model.beta
	statespace = model.statespace
	n = length(statespace)
	V = vi_state.V
	Vnew = vi_state.Vnew
	
	k = 1
	while true
		for i=1:n
			sind = ind2sub(statespace, i)
			Vnew[i] = valueiterate(model, sind, V, params_u)
		end
		Vnew, V = V, Vnew
		if maxabsdiff(V, Vnew) < ϵ
			vi_state.k = k
			break
		end
		k += 1
	end
end

# ----- Loglikelihood
function s1_fullloglikelihood_i(transition, data_i)
	sinds = data_i[1]
	ainds = data_i[2]
	M = length(ainds) - 1
	loglik = 0.0
	lik = 1.0
	for i=1:M
		p = prob(transition, sinds[i], ainds[i], sinds[i+1])
		loglik += log(p)
		lik *= p
	end
	return loglik
end

function s1_fullloglikelihood(transition, data)
	N = length(data)
	loglik = 0.0
	for i=1:N
		data_i = data[i]
		loglik += s1_fullloglikelihood_i(transition, data_i)
	end
	return loglik
end

function s2_loglikelihood_i(model::NFXPModel, sind, aind, V, params_u)
	utility = model.utility
	transition = model.transition
	actionspace= model.actionspace
	statespace = model.statespace
	beta = model.beta

	cp = condprob(transition, sind, aind)

	state = statespace[sind]
	action = actionspace[aind]

	numer = exp(utility(state, action, params_u) + beta*expectation(cp, V))
	denom = 0.0
	for i = 1:length(actionspace)
		aind = ind2sub(actionspace, i)
		action = actionspace[aind]
		cp = condprob(transition, sind, aind)
		v = utility(state, action, params_u) + beta*expectation(cp, V)
		if abs(v) > 700
			println("i: $i  action: $action  state: $state  params_u: $params_u")
			error("OVERFLOW!")
		end
		denom += exp(v)
	end
	return log(numer/denom)
end

function s2_fullloglikelihood_i(model::NFXPModel, data_i, V, params_u)
	transition = model.transition
	actionspace = model.actionspace
	statespace = model.statespace
	state_data = data_i[1]
	action_data = data_i[2]
	T = length(state_data)

	loglik = 0.0
	for t=1:T
		aind = action_data[t]
		sind  = state_data[t]
		loglik += s2_loglikelihood_i(model, sind, aind, V, params_u)
	end
	return loglik
end

function s2_fullloglikelihood!(vi_state::ValueIterationState, model::NFXPModel, data, params_u)
	valueiteration!(vi_state, model, params_u)
	V = vi_state.V
	N = length(data)
	loglik = 0.0
	for i=1:N
		loglik += s2_fullloglikelihood_i(model, data[i], V, params_u)
	end
	return loglik
end
		
immutable MockOpt
	minimum::Vector{Float64}
end

doc"""
Sample from a fitted NFXP model
"""
function predict(result, n::Int)
	model = result.model
	utility = model.utility
	transition = model.transition
	actionspace = model.actionspace
	statespace = model.statespace
	beta = model.beta

	V = result.vi_state.V
	params_t = result.s1_opt.minimum
	params_u = result.s2_opt.minimum

	n_actions = length(actionspace)

	g = Gumbel()

	start_state = 1

	aind = ind2sub(actionspace, 1)
	sind = ind2sub(statespace, start_state)

	action_inds = Array(typeof(aind), n)
	state_inds = Array(typeof(sind), n)
	util_shocks = zeros(Float64, n, n_actions)

	V_action = zeros(Float64, n_actions)

	for i=1:n
		state_inds[i] = sind
		
		util_shock = rand(g, n_actions)
		util_shocks[i,:] = util_shock

		state = statespace[sind]
		for j=1:n_actions
			aind = ind2sub(actionspace, j)
			action = actionspace[j]
			cp = condprob(transition, sind, aind)
			V_action[j] = utility(state, action, params_u) + 
				util_shock[j] + beta*expectation(cp, V)
		end
		best_action = ind2sub(actionspace, indmax(V_action))
		action_inds[i] = best_action
		sind = rand(transition, sind, best_action)
	end
	return (state_inds, action_inds, util_shocks)
end

function predict(model::FittedNFXPModel, n::Int, n_agents::Int)
	data = Array(Any, n_agents)
	for i=1:length(data)
		data[i] = predict(model::FittedNFXPModel, n)
	end
	return data
end

function forwarddifference(f, params, eps=1e-4)
	f_0 = f(params)
	n = length(params) 
	J = Array(Float64, n)
	
	new_params = copy(params)
	for i=1:n
		param_i = params[i]
		new_params[i] += eps
		f_cand = f(new_params)
		J[i] = (f_cand - f_0)/eps
		println("a: ", new_params)
		new_params[i] = param_i
		println("b: ", new_params)
	end
	return J
end

immutable NFXPClosure
    V::Vector{Float64}
    model::NFXPModel
    params_u::Vector{Float64}
    params_t::Vector{Float64}
end
function NFXPClosure(model::NFXPModel, transition_maker::Function, params_u, params_t)
    vi_state = ValueIterationState(model.statespace)
    transition = transition_maker(model.statespace, model.actionspace, params_t)
    newmodel = NFXPModel(model.statespace, model.actionspace, 
                         transition, model.utility, model.beta)
    valueiteration!(vi_state, newmodel, params_u)
    return NFXPClosure(vi_state.V, newmodel, params_u, params_t)
end
function fullloglikelihood_i(fc::NFXPClosure, data_i)
    s1_ll = -s1_fullloglikelihood_i(fc.model.transition, 
                                    data_i)
    s2_ll = -s2_fullloglikelihood_i(fc.model, 
                                    data_i, fc.V, 
                                    fc.params_u)
    return s1_ll + s2_ll
end
function ∇fullloglikelihood_i(vfc::Vector{NFXPClosure}, data_i; ϵ=1e-4)
    fll = [fullloglikelihood_i(fc, data_i) for fc in vfc]
    #println("fll: ", fll)
    # at the estimated parameter value
    f_0 = fll[1]
    n = length(fll) - 1
    J = Array(Float64, n)
    for i=1:n
        J[i] = (fll[i+1]-f_0)/ϵ
    end
    return J
end
function forwarddifference(model::NFXPModel, transition_maker::Function, params_u, params_t; ϵ=1e-5)
    n_u = length(params_u)
    n_t = length(params_t)
    n = n_u + n_t
    vfc = Array(NFXPClosure, n + 1)
    vfc[1] = NFXPClosure(model, transition_maker, params_u, params_t)
    
    for i=1:n_u
        new_params_u = copy(params_u)
        new_params_u[i] += ϵ
        println("θ_utility_new: ", new_params_u)
        vfc[i + 1] = NFXPClosure(model, transition_maker, new_params_u, params_t)
    end
    
    for i=1:n_t
        new_params_t = copy(params_t)
        new_params_t[i] += ϵ
        println("before θ_transition_new: ", new_params_t)
        new_params_t /= sum(new_params_t)
        println("after θ_transition_new: ", new_params_t)
        vfc[i + n_u + 1] = NFXPClosure(model, transition_maker, params_u, new_params_t)
    end
    return vfc
end

abstract CI
immutable BHHH_CI <: CI end
doc"""
Resampling bootstrap for balanced panels

Each agent in a panel is sampled with replacement.

`n` samples are taken (typically the same as the number
of agents in the dataset).
"""
immutable ResamplingBootstrapCI <: CI
    n::Int
end
doc"""
Parametric bootstrap for panels
"""
immutable ParametricBootstrapCI <: CI
    n::Int
end

doc"""
Computes an asymptotic BHHH confidence interval
"""
function vcov(result::FittedNFXPModel, transition_maker::Function)
    params_t = result.s1_opt.minimum
    params_u = result.s2_opt.minimum
    m = length(params_u) + length(params_t)
    model = result.model
    data = result.data
    n = length(data)
    if n < 10
        warn("n=$n is too small")
    end
    
    fd = forwarddifference(model, transition_maker, params_u, params_t)
    vcov_matrix = zeros(Float64, m, m)
    for i=1:n
        J = ∇fullloglikelihood_i(fd, data[i])
        #println("i: ", i, " J: ", J)
        vcov_matrix += J*J'
    end
    return vcov_matrix/n
end

doc"""
Computes a resampling bootstrap confidence interval
"""
vcov(result::FittedNFXPModel, rbs::ResamplingBootstrapCI) = error("not implemented")

doc"""
Computes a parametric bootstrap confidence interval
"""
function vcov(result::FittedNFXPModel, pbs::ParametricBootstrapCI)
    params_t = result.s1_opt.minimum
    params_u = result.s2_opt.minimum
    m = length(params_u) + length(params_t)
    n = pbs.n
    params = zeros(Float64, n, m)
    
    # fit the 
    
    for i=1:n
    end
end