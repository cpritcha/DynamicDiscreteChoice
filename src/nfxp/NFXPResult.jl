type NFXPResult
    vi_state
    stage1_result
    stage2_result
    model::NFXP
    solver
    data
    model_family::NFXPFamily
end

function Base.show(io::IO, x::NFXPResult)
    println(io, "Stage 1: ")
    println(io, x.stage1_result)
    println(io, "\nStage 2: ")
    println(io, x.stage2_result)
end

immutable NFXPClosure
    vi_state::ValueIterationState
    model::NFXP
end
function NFXPClosure(model::NFXP)
    vi_state = ValueIterationState(model.statespace)
    valueiteration!(vi_state, model)
    return NFXPClosure(vi_state, model)
end
function fullloglikelihood_i(fc::NFXPClosure, data_i)
    s1_ll = -s1_fullloglikelihood_i(fc.model.transition, 
                                    data_i)
    s2_ll = -s2_fullloglikelihood_i(fc.model, 
                                    data_i, fc.vi_state.V)
    return s1_ll + s2_ll
end
function ∇fullloglikelihood_i(vfc::Vector{NFXPClosure}, data_i; ϵ=1e-5)
    fll = [fullloglikelihood_i(fc, data_i) for fc in vfc]
    # at the estimated parameter value
    #println("fll: ", fll)
    f_0 = fll[1]
    n = length(fll) - 1
    J = Array(Float64, n)
    for i=1:n
        J[i] = (fll[i+1]-f_0)/ϵ
    end
    return J
end

function loglikelihood_it(model::NFXP, sind, aind, sind_next, aind_next, V)
    transition = model.transition
    s1_ll = s1_loglikelihood_it(transition, sind, aind, sind_next)
    s2_ll = s2_loglikelihood_it(model, sind_next, aind_next, V)
    return s1_ll + s2_ll
end

function ∇loglikelihood_it(nfxp_closures::Vector{NFXPClosure}, sind, aind, sind_next, aind_next; ϵ=1e-4)
    ll = [loglikelihood_it(nfxp_closure.model, sind, aind, sind_next, aind_next, nfxp_closure.vi_state.V)
        for nfxp_closure in nfxp_closures]
    f_0 = ll[1]
    n = length(ll) - 1
    J = Array(Float64, n)
    for i=1:n
        J[i] = (ll[i+1]-f_0)/ϵ
    end
    return J
end

function forwarddifference(model::NFXP, model_family::NFXPFamily; ϵ=1e-6)
    θ_utility = params_utility(model)
    θ_transition = params_transition(model)
    n_u = length(θ_utility)
    n_t = length(θ_transition)
    n = n_u + n_t
    vfc = Array(NFXPClosure, n + 1)
    vfc[1] = NFXPClosure(model)
    
    for i=1:n_u
        θ_utility_new = copy(θ_utility)
        θ_utility_new[i] += ϵ
        newutility = model_family.utilityfamily(θ_utility_new)
        newmodel = NFXP(model.statespace,
            model.actionspace,
            model.transition,
            newutility,
            model.discount)
        vfc[i + 1] = NFXPClosure(newmodel)
    end
    
    for i=1:n_t
        θ_transition_new = copy(θ_transition)
        θ_transition_new[i] += ϵ
        θ_transition_new /= sum(θ_transition_new)

        newtransition = copy(model.transition)
        set_params!(newtransition, θ_transition_new)

        newmodel = NFXP(model.statespace,
            model.actionspace,
            newtransition,
            model.utility,
            model.discount)
        vfc[i + n_u + 1] = NFXPClosure(newmodel)
    end
    return vfc
end

doc"""
Computes an asymptotic OPG confidence interval
"""
function vcov(result::NFXPResult, method::OPG; invert=true)
    model_family = result.model_family
    model = result.model
    data = result.data
    
    m_transition = stage1_n_params(result.solver)
    m_utility = length(params(result.stage2_result))
    m = m_transition + m_utility
    
    N = length(data)
    if N < 10
        warn("n=$n is too small")
    end
    
    fd = forwarddifference(model, model_family)
    vcov_matrix = zeros(Float64, m, m)
    Ts = [length(data[i])-1 for i=1:N]
    for i=1:N
        J = ∇fullloglikelihood_i(fd, data[i])[1:m]
        vcov_matrix += J*J'
    end
    return invert ? inv(vcov_matrix) : vcov_matrix 
end