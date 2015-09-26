export mockup, sample

function mockup(modelfamily, θ_transition, θ_utility)
    vi_state = ValueIterationState(modelfamily.statespace)
    stage1_result = MockSolverResult(θ_transition)
    stage2_result = MockSolverResult(θ_utility)
    
    utility = modelfamily.utilityfamily(θ_utility)
    transition = modelfamily.transitionfamily(Nullable(θ_transition))
    model = NFXP(modelfamily.statespace,
                 modelfamily.actionspace,
                 transition,
                 utility,
                 modelfamily.discount)
    valueiteration!(vi_state, model)
    
    NFXPResult(vi_state, stage1_result, stage2_result, model, [], [], modelfamily)
end

function policy(V::Vector{Float64}, V_action::Vector{Float64}, model::NFXP, sind, util_shock)
    actionspace = model.actionspace
    statespace = model.statespace
    transition = model.transition
    utility = model.utility
    discount = model.discount
    
    n_actions = length(actionspace)
    
    state = statespace[sind]
    
    for j=1:n_actions
        aind = ind2sub(actionspace, j)
        action = actionspace[j]
        cp = condprob(transition, sind, aind)
        V_action[j] = utility(state, action) + 
            util_shock[j] + discount*expectation(cp, V)
    end
    best_action = ind2sub(actionspace, indmax(V_action))
    return best_action
end

function _sample(result::NFXPResult, start_state, n_reps)
    model = result.model
    transition = model.transition
    actionspace = model.actionspace
    statespace = model.statespace

    V = result.vi_state.V
    params_t = params(result.stage1_result)
    params_u = params(result.stage2_result)

    n_actions = length(actionspace)

    g = Gumbel()

    aind = ind2sub(actionspace, 1)
    sind = ind2sub(statespace, start_state)

    action_inds = Array(typeof(aind), n_reps)
    state_inds = Array(typeof(sind), n_reps)
    util_shocks = zeros(Float64, n_reps, n_actions)

    V_action = zeros(Float64, n_actions)

    for i=1:n_reps
        state_inds[i] = sind

        util_shock = rand(g, n_actions)
        util_shocks[i,:] = util_shock

        best_action = policy(V, V_action, model, sind, util_shock)
        action_inds[i] = best_action
        sind = sample(transition, sind, best_action)
    end
    return (state_inds, action_inds, util_shocks, sind)
end

function sample(result::NFXPResult, start_state; episode_length::Int=100, n_agents::Int=100)
    samples = []
    for i=1:n_agents
        tmp = _sample(result, start_state, episode_length)
        start_state = sub2ind(result.model.statespace, tmp[4])
        push!(samples, tmp)
    end
    return samples
end