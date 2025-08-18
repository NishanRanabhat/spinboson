# Algorithms/solvers.jl
using LinearAlgebra
using TensorOperations

#export LanczosSolver, KrylovExponential, solve, evolve
#export OneSiteEffectiveHamiltonian, TwoSiteEffectiveHamiltonian, ZeroSiteEffectiveHamiltonian

# ============= Effective Hamiltonian Types =============

"""
Effective Hamiltonian for one-site optimization in DMRG
"""
struct OneSiteEffectiveHamiltonian{T}
    left_env::Array{T,3}
    mpo_tensor::Array{T,4}
    right_env::Array{T,3}
end

"""
Effective Hamiltonian for two-site optimization in DMRG/TDVP
"""
struct TwoSiteEffectiveHamiltonian{T}
    left_env::Array{T,3}
    mpo_tensor1::Array{T,4}
    mpo_tensor2::Array{T,4}
    right_env::Array{T,3}
end

"""
Effective Hamiltonian for zero-site (bond) optimization in TDVP
"""
struct ZeroSiteEffectiveHamiltonian{T}
    left_env::Array{T,3}
    right_env::Array{T,3}
end

# ============= Apply Methods (Hamiltonian-Vector Products) =============

function apply(H::OneSiteEffectiveHamiltonian{T}, v::Vector{T}) where T
    chi_l = size(H.left_env, 3)
    chi_r = size(H.right_env, 3)
    d = size(H.mpo_tensor, 4)  # Extract dimension from MPO
    
    # Reshape vector to tensor
    M = reshape(v, chi_l, d, chi_r)
    
    # Contract - matching your MpoToMpsOneSite logic
    @tensoropt M_new[-1,-2,-3] := 
        H.left_env[-1,4,5] * M[5,6,8] * 
        H.mpo_tensor[4,7,-2,6] * 
        H.right_env[-3,7,8]
    
    return vec(M_new)
end

function apply(H::TwoSiteEffectiveHamiltonian{T}, v::Vector{T}) where T
    chi_l = size(H.left_env, 3)
    chi_r = size(H.right_env, 3)
    d1 = size(H.mpo_tensor1, 4)
    d2 = size(H.mpo_tensor2, 4)
    
    # Reshape vector to tensor
    Psi2 = reshape(v, chi_l, d1, d2, chi_r)
    
    # Contract - matching your MpoToMpsTwoSite logic
    @tensoropt Psi2_new[-1,-2,-3,-4] := 
        H.left_env[-1,5,6] * Psi2[6,7,9,11] * 
        H.mpo_tensor1[5,8,-2,7] * 
        H.mpo_tensor2[8,10,-3,9] * 
        H.right_env[-4,10,11]
    return vec(Psi2_new)
end

function apply(H::ZeroSiteEffectiveHamiltonian{T}, v::Vector{T}) where T
    chi_l = size(H.left_env, 3)
    chi_r = size(H.right_env, 3)
    
    # Reshape vector to matrix
    C = reshape(v, chi_l, chi_r)
    
    # Contract - matching your MpoToMpsOneSiteKeff logic
    @tensoropt C_new[-1,-2] := 
        H.left_env[-1,3,4] * C[4,5] * 
        H.right_env[-2,3,5]
    
    return vec(C_new)
end


# ============= Solver Types =============

"""
#Lanczos solver for eigenvalue problems (DMRG)
"""
#returns default value if not given new
struct LanczosSolver
    krylov_dim::Int
    max_iter::Int
    tol::Float64
    
    function LanczosSolver(; krylov_dim=4, max_iter=100, tol=1e-12)
        new(krylov_dim, max_iter, tol)
    end
end

"""
#Krylov exponential solver for time evolution (TDVP)
"""
struct KrylovExponential
    krylov_dim::Int
    tol::Float64
    
    function KrylovExponential(; krylov_dim=30, tol=1e-12)
        new(krylov_dim, tol)
    end
end

# ============= Eigenvalue Solver =============

"""
#   solve(solver::LanczosSolver, H, v_init)

#Find the lowest eigenvalue and eigenvector of H using Lanczos algorithm.
"""
function solve(solver::LanczosSolver, H, v_init::Vector{T}) where T
    # Handle zero initial vector
    if norm(v_init) == 0
        v_init = randn(T, length(v_init))
    end
    
    # Initialize
    n = length(v_init)
    V = zeros(T, n, solver.krylov_dim + 1)
    H_mat = zeros(T, solver.krylov_dim, solver.krylov_dim)
    
    eigenval = real(T)(Inf)
    eigenvec = v_init / norm(v_init)
        
    # Lanczos iterations
    for iter in 1:solver.max_iter
        # Restart with current best eigenvector
        V[:, 1] = eigenvec / norm(eigenvec)
        # Build Krylov subspace
        for p = 2:solver.krylov_dim+1
            # Apply Hamiltonian
            V[:,p] = apply(H,V[:,p-1])
            # Orthogonalize using modified Gram-Schmidt
            for g = p-2:1:p-1
                if g >= 1
                    H_mat[p-1,g] = dot(V[:,p],V[:,g]);
                    H_mat[g,p-1] = conj(H_mat[p-1,g]);# Maintain symmetry
                end
            end
            for g = 1:1:p-1
                V[:,p] = V[:,p] - dot(V[:,g],V[:,p])*V[:,g];
                V[:,p] = V[:,p]/max(norm(V[:,p]),1e-16);
            end
        end
        
        G = eigen(0.5*(H_mat+H_mat'));
        eigenval, xloc = findmin(G.values);
        eigenvec = V[:,1:solver.krylov_dim]*G.vectors[:,xloc[1]];
    end
    return eigenvec / norm(eigenvec), eigenval
end

# ============= Time Evolution Solver =============

"""
#    evolve(solver::KrylovExponential, H, v_init, dt)

#Evolve state v_init by time dt under Hamiltonian H using Krylov method.
#For real-time evolution use complex dt = -im*t, for imaginary time use real dt.
"""

function evolve(solver::KrylovExponential, H, v_init::Vector{T}, dt::Number) where T
    # Handle zero initial vector
    if norm(v_init) == 0
        v_init = randn(T, length(v_init))
    end
    
    # Determine output type based on dt
    Tout = promote_type(T, typeof(dt))

    n = length(v_init)
    V = zeros(Tout, n, solver.krylov_dim + 1)
    A_mat = zeros(Tout, solver.krylov_dim, solver.krylov_dim)
    output_vec = zeros(Tout,n)
    transit_vec = zeros(Tout,n)

    # Normalize and store norm
    v_norm = norm(v_init)
    V[:, 1] = v_init / v_norm

    for p = 2:solver.krylov_dim+1
        output_vec = 0*output_vec
        V[:,p] = apply(H,V[:,p-1]) 
        for g = p-2:1:p-1
            if g >= 1
                A_mat[p-1,g] = dot(V[:,p],V[:,g]);
                A_mat[g,p-1] = conj(A_mat[p-1,g]);
            end
        end
        for g = 1:1:p-1
            V[:,p] = (V[:,p] - dot(V[:,g],V[:,p])*V[:,g])
        end
        V[:,p] = V[:,p]/max(norm(V[:,p]),1e-16);
        if p > 3
            output_vec = _evol(A_mat[1:p-1,1:p-1],V,dt,output_vec) 
            c = _closeness(transit_vec,output_vec,solver.tol)
            if c == length(output_vec)
                break
            else
                transit_vec = output_vec
            end
        end
    end
    
    return v_norm * output_vec
end

function _evol(mat,vect,dt,output_vec)
    c =  exp(-dt*mat)*I(length(mat[:,1]))[:,1]
    for i in 1:length(c)
        output_vec += c[i]*vect[:,i]
    end
    return output_vec
end  

function _closeness(list1,list2,cutoff)
    c = 0
    for i in 1:length(list1)
        if abs(list1[i]-list2[i]) <= cutoff
            c += 1
        end
    end
    return c
end


# ============= Helper Functions =============

"""
#Create effective Hamiltonians from state components
"""

"""
function OneSiteEffectiveHamiltonian(state::MPSState, site::Int)
    N = length(state.mps.tensors)
    return OneSiteEffectiveHamiltonian(
        state.env.tensors[site-1 == 0 ? N+1 : site-1],
        state.mpo.tensors[site],
        state.env.tensors[site+1]
    )
end

function TwoSiteEffectiveHamiltonian(state::MPSState, site::Int)
    N = length(state.mps.tensors)
    return TwoSiteEffectiveHamiltonian(
        state.env.tensors[site-1 == 0 ? N+1 : site-1],
        state.mpo.tensors[site],
        state.mpo.tensors[site+1],
        state.env.tensors[site+2]
    )
end

function ZeroSiteEffectiveHamiltonian(state::MPSState, site::Int)
    N = length(state.mps.tensors)
    return ZeroSiteEffectiveHamiltonian(
        state.env.tensors[site == 0 ? N+1 : site],
        state.env.tensors[site+1]
    )
end
"""
L = rand(10,4,10)
R = rand(10,4,10)
W1 = rand(4,4,2,2)
W2 = rand(4,4,2,2)
d = 2
psivec = rand(10*2*2*10)
linfunct = MpoToMpsTwoSite
functArgs = (L,W1,W2,R,d)
krydim = 16
maxit = 5
object = "MPS"
dt = 0.02
close_cutoff = 0.0000001
Heff = TwoSiteEffectiveHamiltonian{ComplexF64}(L,W1,W2,R)
#Heff = OneSiteEffectiveHamiltonian(L,W1,R)

#solver = LanczosSolver(krylov_dim=krydim, max_iter=maxit,tol=1e-12)
solver = KrylovExponential(krylov_dim=krydim,tol=close_cutoff)

#vec0,val0 = EigenLancz(psivec,linfunct,functArgs,krydim,maxit,object)
#vec1,val1 = solve(solver,Heff,psivec)

vec0 =  ExpLancz(psivec,linfunct,functArgs,krydim,dt,close_cutoff,"MPS","real")
vec1 = evolve(solver,Heff,psivec,im*dt)

#println(val0 == val1)
println(vec0 == vec1)
