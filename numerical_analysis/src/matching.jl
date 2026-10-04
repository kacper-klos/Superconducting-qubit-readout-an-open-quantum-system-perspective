module Matching

using LinearAlgebra
using Hungarian

"""
Track k initial eigenvectors as the interaction is turned on from λ = 0 to 1.
The columns of vectors must be orthonormal eigenvectors of H_subsystems,
with size n × k and 1 ≤ k ≤ n. At each step, maximize the total squared
overlap with distinct new eigenvectors.

Return (λ_range, energies_evolution, states_evolution), with sizes
(steps,), (k, steps), and (n, k, steps), respectively. The history includes
λ = 0 and λ = 1 and preserves input column labels. 
"""
function eigen_matching(
    H_subsystems::Hermitian,
    H_interaction::Hermitian,
    vectors::AbstractMatrix;
    steps::Integer = 100,
)
    n = size(H_subsystems, 1)
    k = size(vectors, 2)

    size(H_interaction) == size(H_subsystems) ||
        throw(DimensionMismatch("Hamiltonians must have the same size"))
    size(vectors, 1) == n && 1 <= k <= n ||
        throw(DimensionMismatch("vectors must have size n × k, with 1 ≤ k ≤ n"))
    steps >= 2 || throw(ArgumentError("steps must be at least 2"))

    λ_range = range(0.0, 1.0, length = steps)
    energies_evolution = zeros(Float64, k, steps)
    states_evolution = zeros(ComplexF64, n, k, steps)
    # Starts from the given states.
    states_evolution[:, :, 1] = vectors
    for j = 1:k
        v = @view vectors[:, j]
        energies_evolution[j, 1] = real(dot(v, H_subsystems * v))
    end

    for i = 2:steps
        λ = λ_range[i]
        H = Hermitian(H_subsystems + (λ * H_interaction))

        eigenval, eigenvec = eigen(H)

        previous = @view states_evolution[:, :, i-1]
        overlaps = abs2.(previous' * eigenvec)
        assignment, _ = hungarian(1.0 .- overlaps)

        states_evolution[:, :, i] = eigenvec[:, assignment]
        energies_evolution[:, i] = eigenval[assignment]

    end

    return λ_range, energies_evolution, states_evolution
end
end
