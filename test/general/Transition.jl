facts("SparseCategoricalTransition") do
    statespace = G.DiscreteVariableSpace(
        :states,
        [:peanut_butter_inventory],
        ([0.0; 1.0],))
    actionspace = G.DiscreteVariableSpace(
        :actions,
        [:peanut_butter_purchase],
        ([:dont_purchase, :purchase],))
    transit_matrix = Array(Vector{Int}, 3, 2)
    
    transit_matrix[1, 1] = [1; 2]
    transit_matrix[2, 1] = [1; 1]
    transit_matrix[3, 1] = [3; 3]
    
    transit_matrix[1, 2] = [2; 2]
    transit_matrix[2, 2] = [2; 2]
    transit_matrix[3, 2] = [1; 1]
    categorical = D.Categorical([0.4; 0.6])
    
    transition = G.SparseCategoricalTransition(
        transit_matrix,
        categorical,
        statespace,
        actionspace)
    
    @fact        G.prob(transition, (1,), (1,), (1,)) --> 0.4
    @fact        G.prob(transition, (1,), (1,), (2,)) --> 0.6
    @fact_throws G.prob(transition, (1,), (1,), (3,))   
 
    @fact_throws G.prob(transition, (1,), (2,), (1,))
    @fact        G.prob(transition, (1,), (2,), (2,)) --> roughly(1.0)
    @fact_throws G.prob(transition, (1,), (2,), (3,))
    
    @fact        G.prob(transition, (3,), (1,), (3,)) --> roughly(1.0)
    @fact        G.prob(transition, (3,), (2,), (1,)) --> roughly(1.0)
end 