module Utils

using LinearAlgebra

function create_anihilator(dim::Integer)
    dim >= 1 || throw(ArgumentError("dim must be positive"))

    a = zeros(Float64, dim, dim)
    for n = 1:(dim-1)
        a[n, n+1] = sqrt(n)
    end
    return a
end

function create_N(dim::Integer)
    dim >= 1 || throw(ArgumentError("dim must be positive"))
    return Diagonal(Float64.(0:(dim-1)))
end

"""
Evolve a state ψ_0 specified at time t0. Column j of the returned matrix is the state at times[j].
"""
function time_evolution(
    H::Hermitian,
    ψ_0::AbstractVector,
    times::AbstractVector{<:Real};
    t_0::Real = 0.0,
)
    d = size(H, 1)
    length(ψ_0) == d || throw(DimensionMismatch("ψ_0 must have length $d"))

    eigenvalues, eigenvectors = eigen(H)

    # Change to the basis of eigenvectors.
    c0 = eigenvectors' * ComplexF64.(ψ_0)
    c = similar(c0)

    states = Matrix{ComplexF64}(undef, d, length(times))

    for (j, t) in enumerate(times)
        Δt = t - t_0
        @. c = c0 * cis(-eigenvalues * Δt)
        # Change to the standard basis.
        mul!(view(states, :, j), eigenvectors, c)
    end

    return states
end
end

# Simulates hamiltonian H = ω_q/2 * σ_z + ω_a a†a + g σ_x (a + a†)
module TwoLevelTransmon

using LinearAlgebra
using ..Utils

const σ_z = [1.0 0.0; 0.0 -1.0]
const σ_x = [0.0 1.0; 1.0 0.0]
const excited = [1.0, 0.0]
const ground = [0.0, 1.0]
const dim_transmon = 2

function hamiltonian(dim::Integer, ω_q::Real, ω_a::Real, g::Real)
    dim >= 1 || throw(ArgumentError("dim must be positive"))

    a = Utils.create_anihilator(dim)
    N = Utils.create_N(dim)

    I_q = Matrix{Float64}(I, dim_transmon, dim_transmon)
    I_a = Matrix{Float64}(I, dim, dim)

    H_q = (ω_q / 2) * kron(σ_z, I_a)
    H_a = ω_a * kron(I_q, N)
    H_int = g * kron(σ_x, a + a')

    return Hermitian(H_q + H_a + H_int)
end

function expected_value_N(dim::Integer, ψ::AbstractVector)
    dim >= 1 || throw(ArgumentError("dim must be positive"))
    length(ψ) == dim_transmon * dim ||
        throw(DimensionMismatch("ψ must have length $(dim_transmon * dim)"))

    value = 0.0
    for n = 0:(dim-1)
        value += n * (abs2(ψ[n+1]) + abs2(ψ[dim+n+1]))
    end
    return value
end

function expected_value_σ_z(dim::Integer, ψ::AbstractVector)
    dim >= 1 || throw(ArgumentError("dim must be positive"))
    length(ψ) == dim_transmon * dim ||
        throw(DimensionMismatch("ψ must have length $(dim_transmon * dim)"))

    value = 0.0
    for j = 1:dim
        value += abs2(ψ[j]) - abs2(ψ[dim+j])
    end
    return value
end

end


module TwoLevelTransmonAnalysis

using LinearAlgebra
using Plots
using LaTeXStrings
import ..TwoLevelTransmon
import ..Utils

function expected_values(
    dim::Integer,
    ψ_0::AbstractVector,
    times::AbstractVector{<:Real};
    ω_q::Real = 1.0,
    ω_a::Real = 1.0,
    g::Real = 0.05,
    t_0::Real = 0.0,
)
    dim >= 1 || throw(ArgumentError("dim must be positive"))
    isempty(times) && throw(ArgumentError("values cannot be empty"))
    length(ψ_0) == TwoLevelTransmon.dim_transmon * dim || throw(
        DimensionMismatch("ψ must have length $(TwoLevelTransmon.dim_transmon * dim)"),
    )

    H = TwoLevelTransmon.hamiltonian(dim, ω_q, ω_a, g)
    samples = Utils.time_evolution(H, ψ_0, times; t_0 = t_0)

    σ_z = TwoLevelTransmon.expected_value_σ_z.(dim, eachcol(samples))
    N = TwoLevelTransmon.expected_value_N.(dim, eachcol(samples))
    return σ_z, N
end

"""
Simulates the dynamics of two level transmon and plots the expected values of operators σ_z and N.
"""
function plot_expected_values(
    dim::Integer,
    ψ_0::AbstractVector,
    times::AbstractVector{<:Real};
    ω_q::Real = 1.0,
    ω_a::Real = 1.0,
    g::Real = 0.05,
    t_0::Real = 0.0,
)
    σ_z, N = expected_values(dim, ψ_0, times; ω_q = ω_q, ω_a = ω_a, g = g, t_0 = t_0)
    return plot(
        times,
        [σ_z, N],
        dpi = 300,
        size = (1200, 800),
        label = [L"<\hat{σ}_z>" L"<\hat{N}>"],
        xlabel = L"\textnormal{Time} \quad [s/\hbar]",
        title = L"ω_q = %$ω_q, ω_a = %$ω_a, g = %$g",
        linewidth = 2,
    )
end

function plot_excited_probability(
    dim::Integer,
    ψ_0::AbstractVector,
    times::AbstractVector{<:Real};
    ω_q::Real = 1.0,
    ω_a::Real = 1.0,
    g::Real = 0.05,
    t_0::Real = 0.0,
)
    H = TwoLevelTransmon.hamiltonian(dim, ω_q, ω_a, g)
    samples = Utils.time_evolution(H, ψ_0, times; t_0 = t_0)

    reshaped = reshape(samples, dim, TwoLevelTransmon.dim_transmon, length(times))
    excited_qubit = @view reshaped[:, 1, :]

    probabilities = vec(sum(abs2, excited_qubit; dims = 1))

    return plot(
        times,
        probabilities,
        dpi = 300,
        size = (1200, 800),
        label = L"P(|e>)",
        xlabel = L"\textnormal{Time} \quad [s/\hbar]",
        title = L"ω_q = %$ω_q, ω_a = %$ω_a, g = %$g",
        linewidth = 2,
    )
end
end

# Simulates hamiltonian H = \sum_j(ω_j b†b)  + ω_a a†a + g (b† + b) (a + a†)
module ArbitraryLevelTransmon

using LinearAlgebra
using ..Utils

function hamiltonian(
    dim_resonator::Integer,
    energies::AbstractVector{<:Real},
    ω_a::Real,
    g::Real,
)
    dim_resonator >= 1 || throw(ArgumentError("dim_resonator must be positive"))
    isempty(energies) && throw(ArgumentError("energies cannot be empty"))

    dim_transmon = length(energies)

    a = Utils.create_anihilator(dim_resonator)
    b = Utils.create_anihilator(dim_transmon)
    N_a = Utils.create_N(dim_resonator)

    I_a = Matrix{Float64}(I, dim_resonator, dim_resonator)
    I_b = Matrix{Float64}(I, dim_transmon, dim_transmon)

    H_b = kron(Diagonal(energies), I_a)
    H_a = ω_a * kron(I_b, N_a)
    H_int = g * kron(b + b', a + a')

    return Hermitian(H_b + H_a + H_int)
end
end

module ArbitraryLevelTransmonAnalysis

using LinearAlgebra
using Plots
using LaTeXStrings
import ..ArbitraryLevelTransmon
import ..Utils

function leakage_probability_evolution(
    dim_resonator::Integer,
    ψ_0::AbstractVector,
    times::AbstractVector{<:Real};
    energies::AbstractVector{<:Real},
    ω_a::Real = 1.0,
    g::Real = 0.05,
    t_0::Real = 0.0,
)
    dim_transmon = length(energies)

    dim_resonator >= 1 || throw(ArgumentError("dim_resonator must be positive"))
    isempty(times) && throw(ArgumentError("times cannot be empty"))
    dim_transmon >= 3 || throw(ArgumentError("At least three transmon levels are required"))
    length(ψ_0) == dim_transmon * dim_resonator ||
        throw(DimensionMismatch("Incorrect initial-state length"))
    isapprox(norm(ψ_0), 1.0; atol = 1e-10, rtol = 1e-10) ||
        throw(ArgumentError("ψ_0 must be normalized"))

    H = ArbitraryLevelTransmon.hamiltonian(dim_resonator, energies, ω_a, g)
    samples = Utils.time_evolution(H, ψ_0, times; t_0 = t_0)
    # Probability of being outside qubit subsystem.
    reshaped = reshape(samples, dim_resonator, dim_transmon, length(times))
    outside_qubit = @view reshaped[:, 3:end, :]

    return dropdims(sum(abs2, outside_qubit; dims = (1, 2)); dims = (1, 2))
end

function plot_leakage_probability_on_gap(
    dim_resonator::Integer,
    ψ_0::AbstractVector,
    times::AbstractVector{<:Real},
    α_values::AbstractVector{<:Real};
    ω_1::Real = 1.5,
    ω_a::Real = 1.0,
    g::Real = 0.05,
    t_0::Real = 0.0,
)
    dim_resonator >= 1 || throw(ArgumentError("dim_resonator must be positive"))
    isempty(times) && throw(ArgumentError("times cannot be empty"))
    isempty(α_values) && throw(ArgumentError("α_values cannot be empty"))
    length(ψ_0) == 3 * dim_resonator ||
        throw(DimensionMismatch("ψ_0 must have length $(3 * dim_resonator)"))
    all(α -> 0 <= α < ω_1, α_values) || throw(ArgumentError("Require 0 ≤ α < ω_1"))

    leakages = Matrix{Float64}(undef, length(α_values), length(times))

    for (j, α) in enumerate(α_values)
        energies = [0.0, ω_1, 2 * ω_1 - α]

        leakages[j, :] = leakage_probability_evolution(
            dim_resonator,
            ψ_0,
            times;
            energies = energies,
            ω_a = ω_a,
            g = g,
            t_0 = t_0,
        )
    end

    return heatmap(
        times,
        α_values,
        leakages;
        xlabel = L"\textnormal{Time} [s \, \hbar]",
        ylabel = L"\textnormal{Anharmonicity α} [1/(s \, \hbar)]",
        colorbar_title = "Leakage probability",
        title = L"ω_1 = %$ω_1, ω_a = %$ω_a, g = %$g",
        c = :viridis,
        dpi = 300,
        size = (1200, 800),
        clims = (0, 1),
        aspect_ratio = :auto,
    )
end
end
