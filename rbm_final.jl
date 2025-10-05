using Random
using Distributions
using StatsBase
using Plots
using Printf
using DelimitedFiles
using LaTeXStrings
gr()

function p(x, beta=1)
    (1 .+ exp.(-2*beta*x)).^-1
end

function constrastive_divergence(M, vmax, eta_0, eta_end, p0, k, sigma)
    patterns = [-1 -1 -1; -1 1 1; 1 -1 1; 1 1 -1]
    n_XOR, N = size(patterns)
    mu = 0              # Mean
    normal_distribution = Normal(mu, sigma)
    weights = rand(normal_distribution, (M, N))
    threshold_hidden = zeros(Float32, M)
    threshold_visible = zeros(Float32, N)

    for v ∈ 1:vmax
        eta = eta_end + eta_0 * (1 - v/vmax)
        beta = 1/(M+1)*(M + v/vmax)

        sample_indices = sample(1:n_XOR, p0, replace=false)
        
        delta_weights = zeros(Float32, (M, N))
        delta_threshold_hidden = zeros(Float32, M)
        delta_threshold_visible = zeros(Float32, N)

        # println(sample_indices)
        for idx ∈ sample_indices
            visible_0 = patterns[idx, :]
            localfield_hidden_0 = (weights * visible_0) - threshold_hidden 
            hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden_0, beta)]
            # hidden_0 = copy(hidden)
            
            visible = nothing
            localfield_hidden = nothing
            for t in 1:k
                localfield_visible = (weights' * hidden) - threshold_visible
                visible = [rand() < x ? 1 : -1 for x in p(localfield_visible, beta)]
                
                localfield_hidden = (weights * visible) - threshold_hidden 
                hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden, beta)]
            end

            delta_weights .+= tanh.(localfield_hidden_0) * visible_0' - tanh.(localfield_hidden) * visible'
            # delta_weights .+= hidden_0 * visible_0' - hidden * visible'
            delta_threshold_hidden .-= (tanh.(localfield_hidden_0) -  tanh.(localfield_hidden))
            # delta_threshold_hidden .-= (hidden_0 -  hidden)
            delta_threshold_visible .-= (visible_0 - visible)
        end

        weights .+= (eta / p0).*delta_weights
        threshold_hidden .+= (eta / p0).*delta_threshold_hidden
        threshold_visible .+= (eta / p0).*delta_threshold_visible
    end
    # println(weights)
    return (weights, threshold_hidden, threshold_visible)
end


"""
Finds the DKL by counting the frequencies of the XOR states and estimating their probability distribution.  
"""
function discover_dynamics(weights, threshold_hidden, threshold_visible, n_gibbs)
    patterns = [-1 -1 -1; -1 1 1; 1 -1 1; 1 1 -1]
    n_XOR, N = size(patterns)
    visible = rand((-1,1), N)
    # Perform a number of Gibbs sampling steps to "forget" the initial visible state.
    for t in 1:200_000
        # println(visible)
        localfield_hidden = (weights * visible) - threshold_hidden 
        hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden)]
        
        localfield_visible = (weights' * hidden) - threshold_visible
        visible = [rand() < x ? 1 : -1 for x in p(localfield_visible)]
    end

    # After forgetting the initial state, start storing statistics for DKL calculation
    Q = zeros(n_XOR)
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


"""
This function is used to explore what number of Gibbs sampling is needed before the DKL settles.
"""
function explore_transient_ngibbs()
    patterns = [-1 -1 -1; -1 1 1; 1 -1 1; 1 1 -1]
    n_XOR, N = size(patterns)
    
    eta_0 = 0.05        # Initial learning rate
    eta_end = 0.001     # End learning rate
    vmax = 25_000       
    sigma = 1           # STD of initially normally distributed weights
    k = 10              # CD-k
    n_gibbs = 100_000   # How many steps to Gibbs sample after training
    p0 = 4              # How many samples of patterns per "epoch"
    M = 2
    weights, threshold_hidden, threshold_visible = constrastive_divergence(M, vmax, eta_0, eta_end, p0, k, sigma)
    
    Q = zeros(n_gibbs, n_XOR)
    dkl_array = zeros(n_gibbs)
    visible = rand((-1,1), N)

    epochs = collect(2:n_gibbs)
    for t in epochs
        # println(visible)
        localfield_hidden = (weights * visible) - threshold_hidden 
        hidden = [rand() < x ? 1 : -1 for x in p(localfield_hidden)]
        
        localfield_visible = (weights' * hidden) - threshold_visible
        visible = [rand() < x ? 1 : -1 for x in p(localfield_visible)]

        match_idx = findfirst(row -> row == visible, eachrow(patterns))
        Q[t, :] = Q[t-1, :]
        if match_idx !== nothing
            Q[t, match_idx] += 1
        end
        c = Q[t, :]./t
        dkl_array[t] = -0.25*sum(log.(4*(c .+ 1e-6)))
    end

    plt = scatter(epochs, dkl_array, markersize=1, markerstrokewidth=0, markercolor=:black, alpha=0.5, ylims=(0.1, 0.5))
    savefig(plt, "/home/sacredeux/Documents/Chalmers/FFR135/OpenTA/Homework2/julia_v1/transient_ngibbs.png")


end


function run()

    eta_0 = 0.05        # Initial learning rate
    eta_end = 0.001     # End learning rate
    vmax = 25_000       
    sigma = 1          # STD of initially normally distributed weights
    k = 10              # CD-k steps
    n_gibbs = 1_000_000   # How many steps to Gibbs sample after training
    p0 = 4              # How many samples of patterns per "epoch"

    M_hidden_neurons = [1,2,4,8]      # How many hidden neurons
    # M_hidden_neurons = [8]      # How many hidden neurons
    n_runs_per_M = 30   
    results = zeros(length(M_hidden_neurons), n_runs_per_M)
    open("/home/sacredeux/Documents/Chalmers/FFR135/OpenTA/Homework2/julia_v1/delim_file_final.txt", "w") do io
        for (i, M) ∈ enumerate(M_hidden_neurons)
            n_failed_runs = 0 
            for j ∈ 1:n_runs_per_M
                weights, threshold_hidden, threshold_visible = constrastive_divergence(M, vmax, eta_0, eta_end, p0, k, sigma)
                dkl = discover_dynamics(weights, threshold_hidden, threshold_visible, n_gibbs)

                n_failed_runs += dkl > 0.68 ? 1 : 0
                @printf("For M = %.0f\t j = %.0f\t DKL = %.3f\n", M, j, dkl)
                res = [i, M, dkl]
                results[i, j] = dkl
                writedlm(io, res')
            end # j
            @printf("For M = %.0f\t The number of failed runs was %.0f\n\n", M, n_failed_runs)
        end # i, M

    end # Write to file

end

"""
Reads the results from run() that are stored in the "delim_file_final.txt" file. 
"""
function plot_results()
    data = readdlm("/home/sacredeux/Documents/Chalmers/FFR135/OpenTA/Homework2/julia_v1/delim_file_final.txt", '\t', Float64)
    println("The size of data is = ", size(data))
    
    Ms = unique(data[:,2])
    println("The unique values of M are = ", Ms)

    p = plot(size=(600, 300))
    for M ∈ Ms
        M_mask = findall(x -> x == M, data[:, 2]) 
        dkls = data[M_mask, 3]
        ms = data[M_mask, 2] .+ randn(length(M_mask)) / 20

        legend_entry = "M = "*string(floor(Int8, M))
        scatter!(p, ms, dkls, label=legend_entry, markerstrokewidth = 0, alpha=0.75, markersize=2.2)

    end

    # Plot 4.40 equation
    dkl_theory = zeros(8)
    N = 3
    for M ∈ 1:8
        d = floor(log2(M + 1))
        if M < (2^(N-1) - 1)
            dkl_theory[M] = log(2)*(N - d - (M+1)/2^d)
        else
            dkl_theory[M] = 0
        end
    end
    plot!(p, collect(1:8), dkl_theory, label="Theory", linewidth=2.1, alpha=0.7, color="red", framestyle = :axis)
    xlabel!(p, L"M")
    ylabel!(p, L"D_{\mathrm{KL}}")
    title!(p, L"\mathrm{XOR\ Kullback-Leibler\ divergence}")

    savefig(p, "/home/sacredeux/Documents/Chalmers/FFR135/OpenTA/Homework2/julia_v1/scatterplot_of_m.pdf")
end

# explore_transient_ngibbs()
run()
plot_results()

