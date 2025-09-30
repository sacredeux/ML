using Random

patterns = [-1 -1 -1; -1 1 1; 1 -1 1; 1 1 -1]

println(patterns)
println("The dimensions of patterns is: ", size(patterns))

M = 4
n_XOR, N = size(patterns)
weights = randn(Float32, (M, N)) / 20
threshold_hidden = zeros(Float32, M)
threshold_visible = zeros(Float32, N)

println("threshold_hidden: ", threshold_hidden)
println("threshold_visible: ", threshold_visible)

println(weights)
println("The dimensions of weights is: ", size(weights))

vmax = 10_000
k = 10
eta = 0.01
p0 = 1

function p(x)
    (1 .+ tanh.(x)) / 2
end

for nu ∈ 1:vmax
    sample_indices = rand(1:n_XOR, p0)

    delta_weights = zeros(Float32, (M, N))
    delta_threshold_hidden = zeros(Float32, M)
    delta_threshold_visible = zeros(Float32, N)

    # println(sample_indices)
    for idx ∈ sample_indices
        visible_0 = patterns[idx, :]
        localfield_hidden_0 = (weights * visible_0) - threshold_hidden 
        hidden = [rand() < x ? -1 : 1 for x in p(localfield_hidden_0)]
        
        visible = nothing
        localfield_hidden = nothing
        for t in 1:k
            localfield_visible = (transpose(weights) * hidden) - threshold_visible
            visible = [rand() < x ? -1 : 1 for x in p(localfield_visible)]
            
            localfield_hidden = (weights * visible) - threshold_hidden 
            hidden = [rand() < x ? -1 : 1 for x in p(localfield_hidden)]
        end

        delta_weights += tanh.(localfield_hidden_0) * transpose(visible_0) - tanh.(localfield_hidden) * transpose(visible)
        delta_threshold_hidden -= (tanh.(localfield_hidden_0) -  tanh.(localfield_hidden))
        delta_threshold_visible -= (visible_0 - visible)
    end

    global weights += (eta / p0).*delta_weights
    global threshold_hidden += (eta / p0).*delta_threshold_hidden
    global threshold_visible += (eta / p0).*delta_threshold_visible

end

begin
    local visible = rand((-1,1), 3)
    for t in 1:10_000
        localfield_hidden = (weights * visible) - threshold_hidden 
        hidden = [rand() < x ? -1 : 1 for x in p(localfield_hidden)]

        localfield_visible = (transpose(weights) * hidden) - threshold_visible
        visible = [rand() < x ? -1 : 1 for x in p(localfield_visible)]
    end

    Q = zeros(n_XOR)
    for t in 1:1_000_000
        localfield_hidden = (weights * visible) - threshold_hidden 
        hidden = [rand() < x ? -1 : 1 for x in p(localfield_hidden)]

        localfield_visible = (transpose(weights) * hidden) - threshold_visible
        visible = [rand() < x ? -1 : 1 for x in p(localfield_visible)]

        match_idx = findfirst(row -> row == visible, eachrow(patterns))
        if match_idx !== nothing
            Q[match_idx] += 1
        end
    end

    dkl = -1/4*sum(log.(4*Q))
    println("The DKL is = ", dkl)

end





println(weights)
