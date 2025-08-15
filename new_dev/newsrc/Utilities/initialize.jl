"""
This file takes the initial state and initializes the system for DMRG and TDVP
sweeps. The outputs are the canonical MPS, and the Environment tensors.
"""

using LinearAlgebra
using TensorOperations

export Initialize

function right_normalize(psi,N)

"""
This function puts the input MPS into right canonical form starting from site N to
site 2. The site 1 is non- canonical as putting it in right canonical will result to
loss of information.
"""

    @inbounds for i in reverse(2:N)

        left_index = size(psi[i])[1]
        center_index = size(psi[i])[2]
        right_index = size(psi[i])[3]

        Mat = reshape(psi[i],(left_index,center_index*right_index))

        F = svd(Mat)

        psi[i] = reshape(F.Vt,(:,center_index,right_index))

        US = F.U*diagm(F.S)
        @tensoropt psi[i-1][-1,-2,-3] := psi[i-1][-1,-2,4]*US[4,-3]/norm(F.S)
    end

    return psi
end

function move_orthogonality_left(A_left::Array{T,3}, A_right::Array{T,3}; normalize=true) where T
    left_index, center_index, right_index = size(A_right)
    
    F = svd(reshape(A_right, left_index, center_index * right_index))
    A_right_new = reshape(F.Vt, (:, center_index, right_index))
    US = F.U * Diagonal(F.S)
    
    # Absorb US into left tensor
    if normalize
        @tensoropt A_left_new[-1,-2,-3] := A_left[-1,-2,4] * US[4,-3] / norm(F.S)
    else
        @tensoropt A_left_new[-1,-2,-3] := A_left[-1,-2,4] * US[4,-3]
    end
    return A_left_new, A_right_new
end

function left_normalize(psi,N)

"""
This function accordingly puts the MPS in right canonical form.
"""
    @inbounds for i in 1:N-1
        left_index = size(psi[i])[1]
        center_index = size(psi[i])[2]
        right_index = size(psi[i])[3]

        Mat = reshape(psi[i],(left_index*center_index,right_index))

        F = svd(Mat)

        psi[i] = reshape(F.U,(left_index,center_index,:))

        SV = diagm(F.S)*F.Vt

        @tensor psi[i+1][-1,-2,-3] := SV[-1,4]*psi[i+1][4,-2,-3]/norm(F.S)
    end

    return psi
end

function move_orthogonality_right(A_left::Array{T,3}, A_right::Array{T,3}; normalize=true) where T
    left_index, center_index, right_index = size(A_left)
    
    # SVD of left tensor
    F = svd(reshape(A_left, left_index * center_index, right_index))
    A_left_new = reshape(F.U, (left_index, center_index, :))
    SV = Diagonal(F.S) * F.Vt
    
    # Absorb SV into right tensor
    if normalize
        @tensoropt A_right_new[-1,-2,-3] := SV[-1,4] * A_right[4,-2,-3] / norm(F.S)
    else
        @tensoropt A_right_new[-1,-2,-3] := SV[-1,4] * A_right[4,-2,-3]
    end
    
    return A_left_new, A_right_new
end

# ============= Canonicalization Methods =============
"""
    canonicalize!(mps::MPS, center::Int)

Put MPS into mixed canonical form with orthogonality center at `center`.
Sites 1:center-1 are left-orthogonal, sites center+1:N are right-orthogonal.
"""
#function canonicalize!(mps::MPS, center::Int)
#    N = length(mps.tensors)
    
    # Right-canonicalize from right to center+1
#    for site in N:-1:center+1
#        mps.tensors[site-1], mps.tensors[site] = 
#            move_orthogonality_left(mps.tensors[site-1], mps.tensors[site])
#    end
    
    # Left-canonicalize from left to center-1
#    for site in 1:center-1
#        mps.tensors[site], mps.tensors[site+1] = 
#            move_orthogonality_right(mps.tensors[site], mps.tensors[site+1])
#    end
    
#    return mps
#end

# ============= Canonicalization Methods =============
"""
    canonicalize!(mps::MPS, center::Int)

Put MPS into mixed canonical form with orthogonality center at `center`.
Sites 1:center-1 are left-orthogonal, sites center+1:N are right-orthogonal.
"""
function canonicalize(mps, center)
    N = length(mps)
    
    # Right-canonicalize from right to center+1
    for site in N:-1:center+1
        mps[site-1], mps[site] = 
            move_orthogonality_left(mps[site-1], mps[site])
    end
    
    # Left-canonicalize from left to center-1
    for site in 1:center-1
        mps[site], mps[site+1] = 
            move_orthogonality_right(mps[site], mps[site+1])
    end
    
    return mps
end

function contract_right(B,R,W,object::String)

if object == "MPDO"

    @tensoropt fin[-1,-2,-3] := conj(B)[-1,4,5,6]*R[6,7,8]*W[-2,7,4,9]*B[-3,9,5,8]
    return fin

elseif object == "MPS"

    @tensoropt fin[-1,-2,-3] := conj(B)[-1,4,5]*R[5,6,7]*W[-2,6,4,8]*B[-3,8,7]
    return fin

end
end

function Initialize(N,psi,Ham,object::String)
    
"""
This function initializes the state at time zero in right canonical form and the environment tensors.
Since a string of MPS cannot be put into complete left/right canonical form (i.e. all the individual
MPS are canonical) without losing some information, we have chosen to represent our state as an MPS with all
but the edge tensors ( "M" ) in the canonical form. 
"""
    
    if object == "MPS"

        psi = right_normalize(psi,N)
        M = psi[1]

        Env = Array{Any,1}(undef,N+1)
        Env[N+1] = ones(1,1,1)
        @inbounds for i in reverse(2:N)
            
            Env[i] = contract_right(psi[i],Env[i+1],Ham[i],"MPS")
        end
    
    elseif object == "MPDO"
        
        M = psi[1]

        Env = Array{Any,1}(undef,N+1)
        Env[N+1] = ones(1,1,1)

        @inbounds for i in reverse(2:N)
            Env[i] = contract_right(psi[i],Env[i+1],Ham[i],"MPDO")
        end
        
    end
    return M,psi,Env
end

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


# Your original approach is actually quite sensible!
function indx(i::Int, N::Int)
    return i == 0 ? N+1 : i
end

# ============= Environment Building =============
"""
    build_environment(mps::MPS{T}, mpo::MPO{T}, center::Int) where T

Build environment tensors around the orthogonality center.
- env[1:center-1] contains left environments
- env[center+2:N+1] contains right environments
- env[center] and env[center+1] are nothing (at the orthogonality center)
"""

function build_environment(mps, mpo, center::Int)
    N = length(mps)
    @assert 1 <= center <= N "Center must be between 1 and N"
    
    # Initialize environment array
    env_tensors = Vector{Union{Array{Float64,3}, Nothing}}(nothing, N+1)
    
    # Right boundary
    env_tensors[N+1] = ones(1, 1, 1)

    # Build left environments up to (but not including) center
    for i in 1:center-1
        env_tensors[i] = contract_left_environment(
            env_tensors[i-1 == 0 ? N+1 : i-1], mps[i], mpo[i]
        )
    end
    
    # Build right environments from center+1 to N
    for i in N:-1:center+1
        env_tensors[i] = contract_right_environment(
            env_tensors[i+1], mps[i], mpo[i]
        )
    end
    
    # env_tensors[center] remain nothing
    
    return env_tensors
end


psi = Array{Array{Float64,3},1}(undef,10)
psi[1] = rand(1,2,8)
psi[10] = rand(8,2,1)
for i in 2:9
    psi[i] = rand(8,2,8)
end

mps = deepcopy(psi)

Ham = Array{Array{Float64,4},1}(undef,10)
Ham[1] = rand(1,4,2,2)
Ham[10] = rand(4,1,2,2)
for i in 2:9
    Ham[i] = rand(4,4,2,2)
end

M,psi,Env = Initialize(10,psi,Ham,"MPS")

mps = canonicalize(mps,3)


println(M==mps[1])




