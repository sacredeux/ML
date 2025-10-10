using LinearAlgebra
using Random
using Distributions
using StatsBase
using Plots
using Printf
using DelimitedFiles
using LaTeXStrings


function plot_data()
    training_dataset = readdlm("C:\\Users\\H677232\\OneDrive - Husqvarna Group\\Documents\\privat2\\project\\ffr135\\hw3\\lorenz\\training-set.csv", ',', Float32)
    println("The size of data is = ", size(training_dataset))
    x = training_dataset[1,:]
    y = training_dataset[2,:]
    z = training_dataset[3,:]

    plt = plot3d(
    x,y,z,
    xlim = (-30, 30),
    ylim = (-30, 30),
    zlim = (0, 60),
    title = "Lorenz Attractor",
    legend = false,
    marker = 1,
    )
    savefig(plt, "C:\\Users\\H677232\\OneDrive - Husqvarna Group\\Documents\\privat2\\project\\ffr135\\hw3\\lorenz\\lorenz.pdf")

end

function reservoir_computer()
    dt = 0.02
    k = 0.01

    N = 3
    M = 500

    mu = 0
    sigma_in = sqrt(0.002)
    sigma = sqrt(0.004)
    normal_distribution_in = Normal(mu, sigma_in)
    normal_distribution = Normal(mu, sigma)
    weights_in = rand(normal_distribution_in, (M, N))
    weights = rand(normal_distribution, (M, M))
    weights_out = rand(normal_distribution_in, (N, M))

    # Train the network: 

    # Run the test data:

    # Continue 500 steps: 

end

"""
Function that runs the functions above in the correct order
"""
function run()
    plot_data()
    # reservoir_computer()
end

run()
