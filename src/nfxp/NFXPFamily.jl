export NFXPFamily
export stage1_problem, stage2_problem, fit

type NFXPFamily
    statespace
    actionspace
    transitionfamily
    utilityfamily
    discount
end

immutable NFXP
    statespace::DiscreteVariableSpace
    actionspace::DiscreteVariableSpace
    transition
    utility::Utility
    discount::Float64
end

params_utility(model::NFXP) = model.utility.θ
params_transition(model::NFXP) = model.transition.categorical.p

function stage1_problem(transition, data)
    function obj(θ_transition)
        set_params!(transition, θ_transition)
        return -s1_fullloglikelihood(transition, data)
    end
    return obj
end

function stage2_problem(model_family::NFXPFamily, data, transition, vi_state)
    function obj(θ_utility)
        utility = model_family.utilityfamily(θ_utility)
        model = NFXP(model_family.statespace, 
            model_family.actionspace,
            transition,
            utility,
            model_family.discount)
        return -s2_fullloglikelihood!(vi_state, model, data)
    end
    return obj
end

function fit{T}(model_family::NFXPFamily, data, method::NFXPSolver{T})
    transition = model_family.transitionfamily()
    
    s1_problem = stage1_problem(transition, data)
    s1_result = stage1_solve(method, s1_problem)
    s1_n_params = stage1_n_params(method)
    θ_transition = params(s1_result)[1:s1_n_params]
    
    vi_state = ValueIterationState(model_family.statespace)
    s2_problem = stage2_problem(model_family, data, transition, vi_state)
    s2_result = stage2_solve(method, s2_problem)
    θ_utility = params(s2_result) 
    
    model = NFXP(
        model_family.statespace,
        model_family.actionspace,
        transition,
        model_family.utilityfamily(θ_utility),
        model_family.discount)
    
    return NFXPResult(
        vi_state,
        s1_result,
        s2_result,
        model,
        method,
        data,
        model_family)
end