using Random
using LinearAlgebra
using Plots
using Statistics
using LaTeXStrings
using Printf
# Let us try in Julia instead:



function periodic_boundaries(coords, N)
        i = coords[1]
        j = coords[2]

        i = i > N ? 1 : i < 1 ? N : i
        j = j > N ? 1 : j < 1 ? N : j
        
    return CartesianIndex(i, j)
end


"""
This function simulates forest fires given parameters\n
    N - the side length of the 2D grid (i.e. the size of the grid is N²) 
    p - the probability of a sprouting 
    f - the probability lightning strikes in a randomly selected cell
    n_fires - maximum number of fires to observe \n
    Returns:
    - The size of the fires as an array
"""
function forest_fires(N, p, f, n_fires)

    S = zeros(Int, N, N) # Status array, 0: No trees, 1: Trees 2: Burned  3: Expanding fire  
    fire_sizes = zeros(Int, n_fires)
    fire_count = 0

    grid_indices = CartesianIndices(S)
    offsets = [CartesianIndex(i, j) for i in -1:1 for j in -1:1 if !(i == 0 && j == 0)]
    # offsets = [CartesianIndex(i, j) for i in (-1, 1) for j in (-1, 1)]

    # anim = Animation()
    # colors = [:black, :green, :red]
    # plt = heatmap(S, aspect_ratio=1, title="n fires = $fire_count", size=(400,400), color = colors, colorbar=false, clim=(0, 2))
    # frame(anim, plt)
    # for t in 1:t_end
    while fire_count < n_fires
        # Sprout new trees:
        S[rand(N, N) .< p .&& S .== 0] .= 1
        
        # Destroy trees depending on lightning:
        lightning_location = rand(grid_indices)
        if rand() < f && S[lightning_location] == 1
            S[lightning_location] = 3
            fire_count += 1

            while sum(S .== 3) > 0
                for burning_location in findall(==(3), S)
                    # i,j = Tuple(burning_location)
                    for offset in offsets
                        neighbor = periodic_boundaries(burning_location + offset, N)
                        if S[neighbor] == 1
                            S[neighbor] = 3
                        end
                    end
                    S[burning_location] = 2
                end # for burning
            end # while
            fire_sizes[fire_count] = sum(S .== 2)
        end # if lightning

        # plt = heatmap(S, aspect_ratio=1, title="n fires = $fire_count", size=(400,400), color = colors, colorbar=false, clim=(0, 2))
        # frame(anim, plt)
        # @assert sum(S .== 3) == 0
        fire_locations = findall(==(2), S)
        S[fire_locations] .= 0
        
    end # t_end
    # gif(anim, "forest_fires_N=$N.gif", fps=10)

    return fire_sizes


end # function

# periodic_boundaries(CartesianIndex(65, 15), 64)


function power_fit(x, y; limit = 0.1)
    # Only use positive data
    mask = x .< limit
    x, y = x[mask], y[mask]

    X = [ones(length(x)) log.(x)]
    z = log.(y)
    β = X \ z  # equivalent to (X'X)\(X'z)
    a = exp(β[1])
    alpha = β[2]
    return a, alpha
end


function run()
    p = 0.01 # probability of tree sprouting
    f = 0.2 # probability of lightning striking per iteration
    n_fires = 500 # Run dynamics until the number of occured fires matches this value, i.e. max number of fires. 
    n_repeats = 10
    k = n_fires*n_repeats
    Ns = [16, 32, 64, 128, 256, 512]
    # Ns = [16, 32, 64, 128]
    # Ns = [16, 64]

    # Cn = zeros(Float16, length(Ns), n_fires)
    alphas = Dict(N => zeros(n_repeats) for N in Ns)
    bs = Dict(N => zeros(n_repeats) for N in Ns)

    fire_sizes = Dict(N => zeros(n_repeats, n_fires) for N in Ns)
    
    C_n = map(x -> (n_fires - x)/n_fires, 0:(n_fires-1))
    C_k = map(x -> (k - x)/k, 0:(k-1))

    for N in Ns
        for r in 1:n_repeats
            fires = forest_fires(N, p, f, n_fires)
            fire_sizes[N][r, :] .= fires
            # fires = vec(fire_sizes[N])
            sort!(fires)
            fires /= N^2
            
            b, beta = power_fit(fires, C_n)
            bs[N][r] = b
            alphas[N][r] = 1 - beta
            
        end
        vec_fires = sort(vec(fire_sizes[N]))/N^2
        # plt = histogram(vec_fire, size=(400,400), bins=25)
        # savefig(plt, "histogram_N=$N.pdf")

        mean_alpha = mean(alphas[N])
        mean_b = mean(bs[N])

        pwr_line = map(x -> mean_b*x^(1 - mean_alpha), vec_fires)
        plt2 = scatter(vec_fires, C_k, xaxis=:log, yaxis=:log, markersize=1, ylabel=L"C(n)", xlabel=L"n/N^2", title=L"N="*"$N", label="Simulation", markerstrokewidth=0,
                        size=(300, 300), legend_position=:bottomleft, color=:blue)

        label_str = @sprintf "%.2f x^{(1 - %.2f)}" mean_b mean_alpha

        plot!(plt2, vec_fires, pwr_line, xaxis=:log, yaxis=:log, label=latexstring(label_str), color=:green, alpha=0.75, linewidth=1)
        vline!(plt2, [0.1], color=:red, linestyle=:dash, label="Fit cut-off", linewidth=1)
        #TODO: Kurvanpassning alpha
        # plot!(plt2, vec_fires, )
        savefig(plt2, "q3_pics\\ccdf_N=$N.pdf")
        
    end
    mean_alphas = [mean(alphas[N]) for N in Ns]
    std_alphas = [std(alphas[N]) for N in Ns]
    
    println("The mean alphas are$mean_alphas")
    println("The std alphas are$std_alphas")
    

    xvals = [1//N for N in Ns]

    plt3 = scatter(
    xvals, mean_alphas;
    yerror=std_alphas,
    xrotation = -60,
    markersize=3,
    markerstrokewidth=0,
    markercolor=:green,
    markerstrokecolor=:green,
    ylabel=L"\alpha",
    xlabel=L"1/N",
    xticks = (xvals, ["1/$(N)" for N in Ns]),  # "1//64" -> "1//64"
    size=(600, 350),
    label="Simulation"
    )

    
    x = 1 ./ Ns[3:end]   # independent variable: 1/N
    y = mean_alphas[3:end]
    X = [ones(length(x)) x]   # design matrix [1  1/N]
    β = X \ y                 # least-squares solution
    β0, β1 = β

    label_str2 = @sprintf "%.2f x + %.2f" β1 β0 
    plot!(plt3, xvals[3:end], β1.*x .+ β0, color=:black, linestyle=:dash, label=latexstring(label_str2))


    # plt3 = scatter([1/N for N in Ns], mean_alphas, yerror=std_alphas,  markersize=1, ylabel=L"\alpha", xlabel=L"1/N", markerstrokewidth=0, markercolor=:black)
    savefig(plt3, "q3_pics\\alphas.pdf")

end

@time run()