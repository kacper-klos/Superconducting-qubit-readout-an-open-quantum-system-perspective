module Utils

using LinearAlgebra

function create_a(dim::Integer)
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
Evolve a state ψ_0 specified at time t0.
Column j of the returned matrix is the state at times[j].
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

module TwoLevelTransmon

using LinearAlgebra
using ..Utils

const σ_z = [1.0 0.0; 0.0 -1.0]
const σ_x = [0.0 1.0; 1.0 0.0]
const excited = [1.0, 0.0]
const ground = [0.0, 1.0]

function hamiltonian(dim::Integer, ω_q::Real, ω_a::Real, g::Real)
    a = Utils.create_a(dim)
    N = Utils.create_N(dim)

    I_q = Matrix{Float64}(I, 2, 2)
    I_a = Matrix{Float64}(I, dim, dim)

    H_q = (ω_q / 2) * kron(σ_z, I_a)
    H_a = ω_a * kron(I_q, N)
    H_int = g * kron(σ_x, a + a')

    return Hermitian(H_q + H_a + H_int)
end

function expected_value_N(dim::Integer, ψ::AbstractVector)
    dim >= 1 || throw(ArgumentError("dim must be positive"))
    length(ψ) == 2 * dim || throw(DimensionMismatch("ψ must have length $(2 * dim)"))

    value = 0.0
    for n = 0:(dim-1)
        value += n * (abs2(ψ[n+1]) + abs2(ψ[dim+n+1]))
    end
    return value
end

function expected_value_σ_z(dim::Integer, ψ::AbstractVector)
    dim >= 1 || throw(ArgumentError("dim must be positive"))
    length(ψ) == 2 * dim || throw(DimensionMismatch("ψ must have length $(2 * dim)"))

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
    length(ψ_0) == 2 * dim || throw(DimensionMismatch("ψ must have length $(2 * dim)"))

    H = TwoLevelTransmon.hamiltonian(dim, ω_q, ω_a, g)
    samples = Utils.time_evolution(H, ψ_0, times; t_0 = t_0)

    σ_z = TwoLevelTransmon.expected_value_σ_z.(dim, eachcol(samples))
    N = TwoLevelTransmon.expected_value_N.(dim, eachcol(samples))
    return σ_z, N
end

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
        linewidth = 2,
    )
end
end
