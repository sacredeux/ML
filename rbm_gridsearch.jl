using Random
using Distributions
using StatsBase
using Plots
using Printf
using DelimitedFiles

function p(x)
    (1 .+ exp.(-2x)).^-1
    # (1 .+ tanh.(x)) / 2
end

function constrastive_divergence(M, vmax, eta_0, eta_end, p0, k, sigma)
    patterns = [-1 -1 -1; -1 1 1; 1 -1 1; 1 1 -1]

    # println(patterns)
    # println("The dimensions of patterns is: ", size(patterns))
    n_XOR, N = size(patterns)
    mu = 0              # Mean
    # sigma = 1         # STD
    normal_distribution = Normal(mu, sigma)
    weights = rand(normal_distribution, (M, N))
    threshold_hidden = zeros(Float32, M)
    threshold_visible = zeros(Float32, N)

    # Best settings so far: 
    # vmax = 50_000
    # k = 20
    # eta = 0.005
    # p0 = 4

    # vmax = 10_000
    # k = 17
    # eta_end = 0.005
    # eta_0 = 0.5
    # p0 = 1

    for v ∈ 1:vmax
        eta = eta_end + eta_0 * (1 - v/vmax)

        sample_indices = sample(1:n_XOR, p0, replace=false)
        
        delta_weights = zeros(Float32, (M, N))
        delta_threshold_hidden = zeros(Float32, M)
        delta_threshold_visible = zeros(Float32, N)

        # println(sample_indices)
        for idx ∈ sample_indices
            visible_0 = patterns[idx, :]
            localfield_hidden_0 = (weights * visible_0) - threshold_hidden 
            hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden_0)]
            
            visible = nothing
            localfield_hidden = nothing
            for t in 1:k
                localfield_visible = (weights' * hidden) - threshold_visible
                visible = [rand() < x ? 1 : -1 for x in p(localfield_visible)]
                
                localfield_hidden = (weights * visible) - threshold_hidden 
                hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden)]
            end

            delta_weights .+= tanh.(localfield_hidden_0) * visible_0' - tanh.(localfield_hidden) * visible'
            delta_threshold_hidden .-= (tanh.(localfield_hidden_0) -  tanh.(localfield_hidden))
            delta_threshold_visible .-= (visible_0 - visible)
        end

        weights .+= (eta / p0).*delta_weights
        threshold_hidden .+= (eta / p0).*delta_threshold_hidden
        threshold_visible .+= (eta / p0).*delta_threshold_visible
    end
    # println(weights)
    return (weights, threshold_hidden, threshold_visible)
end



function discover_dynamics(weights, threshold_hidden, threshold_visible, n_gibbs)
    patterns = [-1 -1 -1; -1 1 1; 1 -1 1; 1 1 -1]
    n_XOR, N = size(patterns)
    visible = rand((-1,1), N)
    for t in 1:200_000
        # println(visible)
        localfield_hidden = (weights * visible) - threshold_hidden 
        hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden)]
        
        localfield_visible = (weights' * hidden) - threshold_visible
        visible = [rand() < x ? 1 : -1 for x in p(localfield_visible)]
    end

    Q = zeros(n_XOR)
    # n_gibbs = 10_000_000
    # n_gibbs = 1_000_000
    for t in 1:n_gibbs
        localfield_hidden = (weights * visible) - threshold_hidden 
        hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden)]

        localfield_visible = (weights' * hidden) - threshold_visible
        visible = [rand() < x ? 1 : -1 for x in p(localfield_visible)]

        match_idx = findfirst(row -> row == visible, eachrow(patterns))
        if match_idx !== nothing
            Q[match_idx] += 1
        end
    end

    Q /= n_gibbs
    dkl = -1/4*sum(log.(4*Q))
    # println("The DKL is = ", dkl)
    return dkl
end


function run()

    eta0s = [0.05]
    etaends = [0.001]
    vmaxs = [25_000]
    # sigmas = [1, 1.5, 2, 3, 4]
    sigmas = [1]
    ks = [10]
    ngibbs = [1_000_000]
    # eta0s = [0.075]
    # etaends = [0.001]
    # vmaxs = [100]
    # sigmas = [1]
    # ks = [2]
    # ngibbs = [100_000, 1_000_000]

    # eta0s = [0.05]
    # etaends = [0.001]

    results = zeros(length(eta0s)*length(etaends)*length(vmaxs)*length(sigmas)*length(ks)*length(ngibbs), 9)
    i = 0
    
    open("delim_file_zoom5.txt", "w") do io
        for eta_0 ∈ eta0s
        for eta_end ∈ etaends
        for vmax ∈ vmaxs
        for sigma ∈ sigmas
        for k ∈ ks
        for ngibb ∈ ngibbs
            i += 1
            # if eta_end > eta_0
            #     continue
            # end
            dkls = zeros(20)

            for r in 1:20
                weights, threshold_hidden, threshold_visible = constrastive_divergence(2, vmax, eta_0, eta_end, 4, k, sigma)
                dkl = discover_dynamics(weights, threshold_hidden, threshold_visible, ngibb)
                dkls[r] = dkl
            end
            # println("For eta_0, eta_end = ", eta_0,"" ,eta_end)
            @printf("For eta_0 = %.1e \teta_end = %.1e\tvmax = %.1e\tsigma = %.1e\tk = %.1e\tngibb = %.1e", eta_0, eta_end, vmax, sigma, k, ngibb)
            @printf("\navg(dkl) = %.3f\nstd(dkl)=%.3f\nmax(dkl)=%.3f\n", mean(dkls), std(dkls), maximum(dkls))
            # println(dkls)
            res = [eta_0, eta_end, vmax, sigma, k, ngibb, mean(dkls), std(dkls), maximum(dkls)]
            results[i,:] = res
            writedlm(io, res')
        end # ngibb
        end # k
        end # sigma
        end # vmax
        end # eta end
        end # eta 0

        
    end # Write to file

end

run()

