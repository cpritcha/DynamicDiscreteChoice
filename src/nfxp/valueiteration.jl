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

function valueiterate(model::NFXP, sind, V)
    utility = model.utility
	transition = model.transition
    discount = model.discount
	actionspace = model.actionspace
	statespace = model.statespace
	state = statespace[sind]
	tmp = 0.0
	for i = 1:length(actionspace)
		aind = ind2sub(actionspace, i)
		action = actionspace[aind]
		cp = condprob(transition, sind, aind)
        tmp += exp(utility(state, action) + discount*expectation(cp, V))
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

function valueiteration!(vi_state::ValueIterationState, model::NFXP; ϵ=1e-7)
	transition = model.transition
    discount   = model.discount
	statespace = model.statespace
	n = length(statespace)
	V = vi_state.V
	Vnew = vi_state.Vnew
	
	k = 1
	while true
		for i=1:n
			sind = ind2sub(statespace, i)
			Vnew[i] = valueiterate(model, sind, V)
		end
		Vnew, V = V, Vnew
		if maxabsdiff(V, Vnew) < ϵ
			vi_state.k = k
			break
		end
		k += 1
	end
end