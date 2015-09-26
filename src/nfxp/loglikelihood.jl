function s1_loglikelihood_it(transition, sind, aind, sind_next)
    p = prob(transition, sind, aind, sind_next)
    return log(p)
end

function s1_fullloglikelihood_i(transition, data_i)
	sinds = data_i[1]
	ainds = data_i[2]
	T = length(ainds) - 1
	loglik = 0.0
	for t=1:T
        logp = s1_loglikelihood_it(transition, sinds[t], ainds[t], sinds[t+1])
		loglik += logp
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

function s2_loglikelihood_it(model::NFXP, sind, aind, V)
	utility = model.utility
	transition = model.transition
	actionspace= model.actionspace
	statespace = model.statespace
    discount = model.discount

	cp = condprob(transition, sind, aind)

	state = statespace[sind]
	action = actionspace[aind]

    numer = exp(utility(state, action) + discount*expectation(cp, V))
	denom = 0.0
	for i = 1:length(actionspace)
		aind = ind2sub(actionspace, i)
		action = actionspace[aind]
		cp = condprob(transition, sind, aind)
        v = utility(state, action) + discount*expectation(cp, V)
		if abs(v) > 700
			println("i: $i  action: $action  state: $state  params_u: $params_u")
			error("OVERFLOW!")
		end
		denom += exp(v)
	end
	return log(numer/denom)
end

function s2_fullloglikelihood_i(model::NFXP, data_i, V)
	state_data = data_i[1]
	action_data = data_i[2]
	T = length(state_data)

	loglik = 0.0
	for t=1:T
		aind = action_data[t]
		sind  = state_data[t]
		loglik += s2_loglikelihood_it(model, sind, aind, V)
	end
	return loglik
end

function s2_fullloglikelihood!(vi_state::ValueIterationState, model::NFXP, data)
    valueiteration!(vi_state, model)
	V = vi_state.V
	N = length(data)
	loglik = 0.0
	for i=1:N
        loglik += s2_fullloglikelihood_i(model, data[i], V)
	end
	return loglik
end