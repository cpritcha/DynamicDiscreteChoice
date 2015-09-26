facts("DiscreteVariableSpace") do
    purchase = G.DiscreteVariableSpace(
        :PurchaseAmount,
        [:Purchase],
        ([0.5; 1.0; 1.5],))
    @fact typeof(purchase) --> G.DiscreteVariableSpace{Val{:PurchaseAmount}, Tuple{Vector{Float64}}, 1}
end