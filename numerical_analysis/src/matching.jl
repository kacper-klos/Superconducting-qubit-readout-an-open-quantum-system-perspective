module Matching

"""
Performs a matching of the initial, non interacting vectors
with the one where the interaction is completely turn on.
"""
using LinearAlgebra
using Hungarian

function eigen_matching(
    H_subsystems::Hermitian,
    H_interaction::Hermitian,
    vectors::AbstractMatrix;
    step::Float64 = 0.05,
)
    size(H_subsystems) == size(H_interaction) == size(vectors) ||
        throw(ArgumentError("All inputted matrices must have the same size"))

    prev_eigenvec = vectors
    prev_eigenval = zeros(Float64, size(vectors, 1))

    λ = step
    while true
        H = Hermitian(H_subsystems + (λ * H_interaction))

        eigenval, eigenvec = eigen(H)

        cost_matrix = abs2.(eigenvec' * prev_eigenvec)
        # Avoid float approximations for the Hungarian algorithm
        cost_matrix_int = round.(Int, (1.0 .- cost_matrix) .* 1e8)
        assignment, _ = hungarian(cost_matrix_int)

        # Invert the permutation to correctly reorder columns to match the old states
        p = invperm(assignment)
        prev_eigenvec = eigenvec[:, p]
        prev_eigenval = eigenval[p]

        # Ensure we evaluate exactly at λ = 1.0 before breaking
        if λ ≈ 1.0
            break
        end
        λ = min(λ + step, 1.0)
    end

    return prev_eigenval, prev_eigenvec
end
end
