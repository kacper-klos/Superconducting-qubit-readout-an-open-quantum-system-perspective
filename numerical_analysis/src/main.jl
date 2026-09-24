using LinearAlgebra
using Plots

include("numerical_analysis.jl")
using .TwoLevelTransmonAnalysis
using .TwoLevelTransmon
using .ArbitraryLevelTransmonAnalysis

const IMAGES_PATH = joinpath(@__DIR__, "..", "..", "images")

# leakage_heatmap = ArbitraryLevelSystemAnalysis.plot_leakage_probability_on_gap(7, kron(1.0 .* (1:3 .== 2), 1.0 .* (1:7 .== 2)), 0.0:0.001:200.0, range(0.0, 0.5, length=80))
# savefig(leakage_heatmap, normpath(joinpath(IMAGES_PATH, "leakage_heatmap.png")))

qubit_expectation_plot = TwoLevelTransmonAnalysis.plot_expected_values(
    7,
    kron(TwoLevelTransmon.excited, 1.0 .* (1:7 .== 1)),
    0.0:0.001:200.0,
)
savefig(
    qubit_expectation_plot,
    normpath(joinpath(IMAGES_PATH, "qubit_expectation_values.png")),
)
