using Random
using LinearAlgebra
using Plots
# Let us try in Julia instead:

"""
This function simulates forest fires given parameters\n
    N - the side length of the 2D grid (i.e. the size of the grid is N²) 
    p - the probability of a sprouting 
    f - the probability lightning strikes in a randomly selected cell
    t_end - the number of time steps to take, because the time step size is 1. \n
    Returns:
    - The number of forest fires
    - The size of the fires as an array
"""


function periodic_boundaries(coords, N)
        i = coords[1]
        j = coords[2]

        i = i > N ? 1 : i < 1 ? N : i
        j = j > N ? 1 : j < 1 ? N : j
        
    return CartesianIndex(i, j)
end


function forest_fires(N, p, f, n_fires)

    S = zeros(Int, N, N) # Status array, 0: No trees, 1: Trees 2: Burned  3: Expanding fire  
    fire_sizes = zeros(Int, n_fires)
    fire_count = 0

    grid_indices = CartesianIndices(S)
    offsets = [CartesianIndex(i, j) for i in -1:1 for j in -1:1 if !(i == 0 && j == 0)]

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

function run()
    p = 0.01 # probability of tree sprouting
    f = 0.2 # probability of lightning striking per iteration
    n_fires = 500 # Run dynamics until the number of occured fires matches this value, i.e. max number of fires. 
    n_repeats = 10
    Ns = [64]

    fire_sizes = Dict(N => zeros(n_repeats, n_fires) for N in Ns)

    for N in Ns
        fire_sizes = forest_fires(N, p, f, n_fires)
    end
    plt = histogram(fire_sizes, size=(400,400), bins=25)
    savefig(plt, "histogram.pdf")

end

run()