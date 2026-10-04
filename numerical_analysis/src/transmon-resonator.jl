module TransmonResonator

using LinearAlgebra

function hamiltonian(
    charge_cutoff::Integer,
    resonator_levels::Integer,
    E_C::Real,
    E_J::Real,
    ω::Real,
    g::Real,
)
    H_s = hamiltonian_subsystems_combined(charge_cutoff, resonator_levels, E_C, E_J, ω)
    H_i = hamiltonian_interaction(charge_cutoff, resonator_levels, g)
    return Hermitian(H_s + H_i)
end

function hamiltonian_subsystems_combined(
    charge_cutoff::Integer,
    resonator_levels::Integer,
    E_C::Real,
    E_J::Real,
    ω::Real,
)
    H_t = hamiltonian_transmon_subsystem(charge_cutoff, E_C, E_J)
    H_r = hamiltonian_resonator_subsystem(resonator_levels, ω)
    I_t = Matrix{Float64}(I, size(H_t, 1), size(H_t, 1))
    I_r = Matrix{Float64}(I, resonator_levels, resonator_levels)
    return Hermitian(kron(H_t, I_r) + kron(I_t, H_r))
end

"""Charge-basis Hamiltonian at zero offset charge, with n = -charge_cutoff:charge_cutoff."""
function hamiltonian_transmon_subsystem(charge_cutoff::Integer, E_C::Real, E_J::Real)
    charge_cutoff >= 1 || throw(ArgumentError("charge_cutoff must be positive"))
    E_C > 0 && E_J > 0 || throw(ArgumentError("E_C and E_J must be positive"))
    charges = Float64.((-charge_cutoff):charge_cutoff)
    diagonal = 4 * E_C .* charges .^ 2
    offdiagonal = fill(-E_J / 2, length(charges) - 1)
    return Hermitian(Matrix(SymTridiagonal(diagonal, offdiagonal)))
end
"""Quantum simple harmonic oscillator with ℏ = 1."""
function hamiltonian_resonator_subsystem(resonator_levels::Integer, ω::Real)
    resonator_levels >= 1 || throw(ArgumentError("resonator_levels must be positive"))
    ω > 0 || throw(ArgumentError("ω must be positive"))
    return Hermitian(Diagonal(ω .* Float64.(0:(resonator_levels-1))))
end

"""Capacitive interaction i*g*n ⊗ (a† - a), in charge ⊗ Fock basis. ℏ = 1."""
function hamiltonian_interaction(charge_cutoff::Integer, resonator_levels::Integer, g::Real)
    charge_cutoff >= 1 || throw(ArgumentError("charge_cutoff must be positive"))
    resonator_levels >= 1 || throw(ArgumentError("resonator_levels must be positive"))
    a = diagm(1 => sqrt.(Float64.(1:(resonator_levels-1))))
    charge = Diagonal(Float64.((-charge_cutoff):charge_cutoff))
    return Hermitian(im * g .* kron(charge, a' - a))
end

end

module TransmonResonatorAnalysis

include("matching.jl")
using ..TransmonResonator
using .Matching
using LinearAlgebra
using Plots
using LaTeXStrings

function transmon_lowest_eigenvalues(
    basis_number::Integer,
    charge_cutoff::Integer,
    E_C::Real,
    E_J::Real,
)
    H = TransmonResonator.hamiltonian_transmon_subsystem(charge_cutoff, E_C, E_J)
    1 <= basis_number <= size(H, 1) || throw(ArgumentError("Invalid transmon basis size"))
    # Hermitian eigenvalues are returned in ascending order.
    eigenval, eigenvec = eigen(H)
    return eigenval[1:basis_number], eigenvec[:, 1:basis_number]
end

"""
Track dressed joint-system energies from bare transmon ⊗ Fock states.
λ scales the interaction from zero to g. Labels identify the initial states. 
Energies use ℏ = 1.
"""
function plot_energy_evolution(
    transmon_basis_number::Integer,
    resonator_basis_number::Integer,
    charge_cutoff::Integer,
    resonator_levels::Integer,
    E_C::Real,
    E_J::Real,
    ω::Real,
    g::Real;
    steps::Integer = 201,
)
    _, transmon_basis =
        transmon_lowest_eigenvalues(transmon_basis_number, charge_cutoff, E_C, E_J)
    resonator_basis = Matrix{Float64}(I, resonator_levels, resonator_basis_number)
    basis = kron(transmon_basis, resonator_basis)

    H_s = TransmonResonator.hamiltonian_subsystems_combined(
        charge_cutoff,
        resonator_levels,
        E_C,
        E_J,
        ω,
    )
    H_i = TransmonResonator.hamiltonian_interaction(charge_cutoff, resonator_levels, g)
    λ_range, energies, _ = Matching.eigen_matching(H_s, H_i, basis; steps = steps)

    labels = [
        latexstring("|$(i),$(j)\\rangle_{\\lambda=0}") for
        i = 0:(transmon_basis_number-1) for j = 0:(resonator_basis_number-1)
    ]
    labels = reshape(labels, 1, :)
    parameters = "E_C=$E_C, E_J=$E_J, ω=$ω, g=$g\ncharge cutoff=$charge_cutoff, resonator levels=$resonator_levels"
    shift_plot = plot(
        λ_range,
        transpose(energies);
        label = labels,
        xlabel = L"\lambda",
        ylabel = L"E(\lambda)",
        title = parameters,
        linewidth = 2,
        dpi = 300,
        size = (1200, 800),
        aspect_ratio = :auto,
    )
    return shift_plot
end

end
