# TensorOps/decompositions.jl
using LinearAlgebra

export svd_truncate, entropy, truncation_error

"""
    svd_truncate(A::Matrix, chi_max::Int, cutoff::Float64)

Perform SVD with truncation based on maximum bond dimension and singular value cutoff.
Works for both MPS and MPDO since they're reshaped to matrices before calling this.
"""
function svd_truncate(A::Matrix{T}, chi_max::Int, cutoff::Float64) where T
    # Same implementation as before
    F = svd(A, alg=LinearAlgebra.QRIteration())
    
    # Normalize and truncate
    S_normalized = F.S / norm(F.S)
    chi_cut = findfirst(x -> x < cutoff, S_normalized)
    chi_cut = isnothing(chi_cut) ? length(S_normalized) : chi_cut - 1
    chi = min(chi_cut, chi_max, length(S_normalized))

    U = F.U[:, 1:chi]
    S = S_normalized[1:chi] 
    V = F.Vt[1:chi, :]
    
    return U, S, V
end


# ===== Utility functions =====

"""
    entropy(S::Vector)

Calculate entanglement entropy from singular values.
"""
function entropy(S::Vector{T}) where T
    S_normalized = S ./ norm(S)
    S_squared = S_normalized .^ 2
    
    # Remove zero values to avoid log(0)
    S_squared = S_squared[S_squared .> eps(T)]
    
    return -sum(s2 * log(s2) for s2 in S_squared)
end

"""
    truncation_error(S::Vector, chi::Int)

Calculate truncation error from discarding singular values beyond chi.
"""
function truncation_error(S::Vector{T}, chi::Int) where T
    if chi >= length(S)
        return zero(T)
    end
    
    S_normalized = S ./ norm(S)
    return sum(S_normalized[chi+1:end].^2)
end