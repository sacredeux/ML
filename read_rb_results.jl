using Random
using Distributions
using StatsBase
using Plots
using Printf
using DelimitedFiles
using Statistics


function read_results()
    data = readdlm("delim_file_zoom5.txt", '\t', Float64)

    goodmask = [all(isfinite, row) for row in eachrow(data)]
    cleaned = data[goodmask, :]

    col7 = cleaned[:, 7]
    # print(col7[1:10])
    cors = [cor(cleaned[:, j], col7) for j in axes(cleaned,2)]

    for j in eachindex(cors)
    println("Column $j vs 7: ", cors[j])
end

end 


read_results()