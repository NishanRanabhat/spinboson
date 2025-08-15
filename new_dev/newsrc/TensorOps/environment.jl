using LinearAlgebra
using TensorOperations

# ============= Basic Tensor Contractions =============
"""
    contract_left_environment(L::Array{T,3}, A::Array{T,3}, W::Array{T,4}) where T

Contract left environment tensor with MPS and MPO tensors.
L[a,b,c] * A[a,s,d] * W[b,e,s,s'] * conj(A)[c,s',d] -> L'[d,e,d]
"""
function contract_left_environment(L::Array{T,3}, A::Array{T,3}, W::Array{T,4}) where T
    @tensoropt L_new[-1,-2,-3] := L[5,6,7] * conj(A)[5,4,-1] * W[6,-2,4,8] * A[7,8,-3]
    return L_new
end

"""
    contract_right_environment(R::Array{T,3}, B::Array{T,3}, W::Array{T,4}) where T

Contract right environment tensor with MPS and MPO tensors.
"""
function contract_right_environment(R::Array{T,3}, B::Array{T,3}, W::Array{T,4}) where T
    @tensoropt R_new[-1,-2,-3] := conj(B)[-1,4,5] * W[-2,6,4,8] * B[-3,8,7]* R[5,6,7]
    return R_new
end

# ============= Environment Building =============
"""
    build_environment(mps::MPS{T}, mpo::MPO{T}, center::Int) where T

Build environment tensors around the orthogonality center.
- env[1:center-1] contains left environments
- env[center+2:N+1] contains right environments
- env[center] and env[center+1] are nothing (at the orthogonality center)
"""
function build_environment(mps::MPS{T}, mpo::MPO{T}, center::Int) where T
    N = length(mps.tensors)
    @assert 1 <= center <= N "Center must be between 1 and N"
    
    # Initialize environment array
    env_tensors = Vector{Union{Array{T,3}, Nothing}}(nothing, N+1)
    
    # Right/Left boundary
    env_tensors[N+1] = ones(T, 1, 1, 1)
    
    # Build left environments up to (but not including) center
    for i in 1:center-1
        env_tensors[i] = contract_left_environment(
            env_tensors[i-1 == 0 ? N+1 : i-1], mps.tensors[i], mpo.tensors[i]
        )
    end
    
    # Build right environments from center+1 to N
    for i in N:-1:center+1
        env_tensors[i] = contract_right_environment(
            env_tensors[i+1], mps.tensors[i], mpo.tensors[i]
        )
    end
    
    # env_tensors[center] remain nothing
    
    return Environment{T}(env_tensors)
end

# ============= Environment Updates =============
"""
    update_left_environment!(env::Environment, site::Int, mps_tensor, mpo_tensor)

Update environment at site i by contracting from site i-1.
Used when moving orthogonality center rightward.
"""
function update_left_environment!(env::Environment, site::Int, mps_tensor::Array{T,3}, mpo_tensor::Array{T,4}) where T
    N = length(env.tensors) - 1  # Number of sites
    env.tensors[site] = contract_left_environment(
        env.tensors[site-1 == 0 ? N+1 : site-1], mps_tensor, mpo_tensor
    )
end

"""
    update_right_environment!(env::Environment, site::Int, mps_tensor, mpo_tensor)

Update environment at site i by contracting from site i+1.
Used when moving orthogonality center leftward.
"""
function update_right_environment!(env::Environment, site::Int, mps_tensor::Array{T,3}, mpo_tensor::Array{T,4}) where T
    N = length(env.tensors) - 1  # Number of sites
    
    env.tensors[site] = contract_right_environment(
        env.tensors[site+1], mps_tensor, mpo_tensor
    )
end
