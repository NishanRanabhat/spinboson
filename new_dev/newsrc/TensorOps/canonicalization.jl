#TensorOps/canonicalization.jl
using LinearAlgebra
using TensorOperations

export canonicalize, is_left_orthogonal, is_right_orthogonal, is_orthogonal

function _move_orthogonality_left(A_left::Array{T,3}, A_right::Array{T,3}; normalize=true) where T
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

function _move_orthogonality_right(A_left::Array{T,3}, A_right::Array{T,3}; normalize=true) where T
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

function canonicalize(mps::MPS, center::Int)
    N = length(mps.tensors)
    
    # Right-canonicalize from right to center+1
    for site in N:-1:center+1
        mps.tensors[site-1], mps.tensors[site] = 
            _move_orthogonality_left(mps.tensors[site-1], mps.tensors[site])
    end
    
    # Left-canonicalize from left to center-1
    for site in 1:center-1
        mps.tensors[site], mps.tensors[site+1] = 
            _move_orthogonality_right(mps.tensors[site], mps.tensors[site+1])
    end
    
    return mps
end

"""
Test if tensor is left-orthogonal: ∑_s A†[i,s,j] * A[i,s,k] = δ[j,k]
"""
function is_left_orthogonal(A::Array{T,3}; tol=1e-12) where T
    left_index, center_index, right_index = size(A)
    
    # Contract over left and physical indices
    @tensor check[-1,-2] := conj(A)[i,s,-1] * A[i,s,-2]
    
    # Should be identity matrix
    I_expected = Matrix{T}(I, right_index, right_index)
    return norm(check - I_expected) < tol
end

"""
Test if tensor is right-orthogonal: ∑_s A[i,s,j] * A†[k,s,j] = δ[i,k]
"""
function is_right_orthogonal(A::Array{T,3}; tol=1e-12) where T
    left_index, center_index, right_index = size(A)
    
    # Contract over right and physical indices
    @tensor check[-1,-2] := A[-1,s,j] * conj(A)[-2,s,j]
    
    # Should be identity matrix
    I_expected = Matrix{T}(I, left_index, left_index)
    return norm(check - I_expected) < tol
end

"""
Test orthogonality of MPS/MPSState
"""
function is_orthogonal(mps::MPS, center::Int)
    N = length(mps.tensors)
    
    println("Testing orthogonality with center at site $center:")
    
    # Test left-orthogonal tensors
    for i in 1:center-1
        is_left = is_left_orthogonal(mps.tensors[i])
        println("  Site $i: left-orthogonal = $is_left")
    end
    
    # Center should not be orthogonal
    println("  Site $center: center (not orthogonal)")
    
    # Test right-orthogonal tensors  
    for i in center+1:N
        is_right = is_right_orthogonal(mps.tensors[i])
        println("  Site $i: right-orthogonal = $is_right")
    end
end
