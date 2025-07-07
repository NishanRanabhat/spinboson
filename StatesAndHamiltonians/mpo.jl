using LinearAlgebra
using Kronecker
using BlockDiagonals
using TensorOperations
using Random

function boson_annihilator(nmax::Integer)
    @assert nmax ≥ 0 "nmax must be non-negative"
    dB = nmax + 1
    A = zeros(Float64, dB, dB)
    for k in 1:nmax                 # super-diagonal entries
        A[k, k+1] = sqrt(k)              # √k = √(n) with n=k
    end
    return A
end

function boson_identity(nmax::Integer)
    dB = nmax + 1
    I = zeros(Float64, dB, dB)          # or zeros(n,n) if Float64 is fine
    for k in 1:dB                    # i ↔ n in the formula above
        I[k, k] = 1.0          # diagonal entry
    end
    return I
end 

"""
    spin_ops(d::Integer)

Return (Sx,Sy,Sz,Id) for spin‑S with d = 2S+1.
For d=2 this gives the usual Pauli matrices/2.
"""
function spin_ops(d::Integer)
    @assert d ≥ 1 "d must be ≥ 1"
    # total spin S and its m‐values
    S = (d - 1)/2
    m_vals = collect(S:-1:-S)   # [S, S-1, …, -S]

    # Sz is just diagonal of m_vals
    Sz = Diagonal(m_vals)

    # Build S+ and S– by placing coef on the super/sub‐diagonal
    Sp = zeros(Float64, d, d)
    for i in 1:d-1
        m_lower = m_vals[i+1]   # THIS is the m of the state being raised
        coef = sqrt((S - m_lower)*(S + m_lower + 1))
        Sp[i, i+1] = coef
    end
    Sm = Sp'  # adjoint

    # Now the cartesian components
    Sx = (Sp + Sm)/2
    Sy = (Sp - Sm) / (2im)

    return Sx, Sy, Sz, Matrix{Float64}(I, d, d)
end


function direct_sum(mpo1,mpo2)

    # shape of both MPOs
    shape1 = size(mpo1)
    shape2 = size(mpo2)

    @assert shape1[3] == shape1[4] == shape2[3] == shape2[4] #Physical dims must match

    """
    direct sum of two mpo of shape (1,n1,d,d) and (1,n2,d,d) is (1,n1+n2,d,d)
    """
    if shape1[1] == 1 && shape2[1] == 1
        #initialize a larger MPO to accomodate the direct sum
        mpo = zeros(1,shape1[2]+shape2[2],shape1[3],shape1[4])

        @views mpo[1:1,1:shape1[2],:,:] .= mpo1 #if we use [1,...] julia will drop that index reducing its rank along that index so we need to use [1:1,...] 
        @views mpo[1:1,shape1[2]+1:end,:,:] .= mpo2

    """
    direct sum of two mpo of shape (n1,1,d,d) and (n2,1,d,d) is (n1+n2,1,d,d)
    """
    elseif shape1[2] ==1 && shape2[2] == 1
        #initialize a larger MPO to accomodate the direct sum
        mpo = zeros(shape1[1]+shape2[1],1,shape1[3],shape1[4])

        @views mpo[1:shape1[1],1:1,:,:] .= mpo1
        @views mpo[shape1[1]+1:end,1:1,:,:] .= mpo2   

    else
        #initialize a larger MPO to accomodate the direct sum
        mpo = zeros(shape1[1]+shape2[1],shape1[2]+shape2[2],shape1[3],shape1[4])

        @views mpo[1:shape1[1],1:shape1[2],:,:] .= mpo1
        @views mpo[shape1[1]+1:end,shape1[2]+1:end,:,:] .= mpo2
    end

    return mpo 
end

function hamiltonian_tc(N::Integer,nmax::Integer,d::Integer,w::Float64,g::Float64,longdir::String)

    X,Y,Z,I = spin_ops(d)
    plus = real((X+im*Y)/2.0)
    minus = real((X-im*Y)/2.0)
    a = boson_annihilator(nmax)

    if longdir == "X"
        Long  = X
        Trans = Z
    elseif longdir == "Z"
        Long  = Z
        Trans = X
    end

    Hbulk = zeros(3,3,d,d)
    H0 = zeros(1,3,nmax+1,nmax+1)

    Hbulk[1,1,:,:] = I; 
    Hbulk[2,1,:,:] = minus;
    Hbulk[3,1,:,:] = plus;
    Hbulk[2,2,:,:] = I;
    Hbulk[3,3,:,:] = I;

    H0[1,1,:,:] = w*a'*a
    H0[1,2,:,:] = (g/sqrt(N))*a'
    H0[1,3,:,:] = (g/sqrt(N))*a

    return H0,Hbulk
end

function hamiltonian_dickie(N::Integer,nmax::Integer,d::Integer,w::Float64,g::Float64,longdir::String)

    X,Y,Z,I = spin_ops(d)
    a = boson_annihilator(nmax)

    if longdir == "X"
        Long  = X
        Trans = Z
    elseif longdir == "Z"
        Long  = Z
        Trans = X
    end

    Hbulk = zeros(2,2,d,d)
    H0 = zeros(1,2,nmax+1,nmax+1)

    Hbulk[1,1,:,:] = I; 
    Hbulk[2,1,:,:] = Trans;
    Hbulk[2,2,:,:] = I;

    H0[1,1,:,:] = w*a'*a
    H0[1,2,:,:] = (g/sqrt(N))*(a+a')

    return H0,Hbulk
end

function hamiltonian_transverse_ising(N::Integer,d::Integer,J::Float64,h::Float64,longdir::String)

    X,Y,Z,I = spin_ops(d)

    if longdir == "X"
        Long  = X
        Trans = Z
    elseif longdir == "Z"
        Long  = Z
        Trans = X
    end

    H = zeros(3,3,d,d)
    H[1,1,:,:] = I; H[3,3,:,:] = I; H[3,1,:,:] = h*Trans
    H[2,1,:,:] = Long; H[3,2,:,:] = J*Long

    return H
end

function hamiltonian_longitudinal_ising(N::Integer,d::Integer,J::Float64,h::Float64,longdir::String)

    X,Y,Z,I = spin_ops(d)

    if longdir == "X"
        Long  = X
    elseif longdir == "Z"
        Long  = Z
    end

    H = zeros(3,3,d,d)
    H[1,1,:,:] = I; H[3,3,:,:] = I; H[3,1,:,:] = h*Long
    H[2,1,:,:] = Long; H[3,2,:,:] = J*Long

    return H
end

function power_law_to_exp(a::Float64,n::Integer,N::Integer)

    """

    function gives (x_i,lambda_i) such that

    1/r^a = Sum_{i=1-->n} x_i * (lambda_i)^r + errors

    a : interaction strength, a -> infinity is nearest neighbor Ising
        a -> 0 is fully connected Ising.

    n : number of exponential sums. Refer to SciPostPhys.12.4.126 appendix C
        for further details.

    N : lattice size.
    """

    F = Array{Float64,1}(undef,N)

    @inbounds for k in 1:N
        F[k] = 1/k^a
    end

    M = zeros(N-n+1,n)

    @inbounds for j in 1:n
        @inbounds for i in 1:N-n+1
            M[i,j] = F[i+j-1]
        end
    end

    F1 = qr(M)

    Q1 = F1.Q[1:N-n,1:n]
    Q1_inv = pinv(Q1)
    Q2 = F1.Q[2:N-n+1,1:n]

    V = Q1_inv*Q2

    lambda = real(eigvals(V))

    lam_mat = zeros(N,n)

    @inbounds for i in 1:length(lambda)
        @inbounds for k in 1:N
            lam_mat[k,i] = lambda[i]^k
        end
    end

    x = lam_mat\F

    return x, lambda
end

function Kac_norm(a::Float64,N::Int64)

    Kac = 0.0

    for i in 1:N
        Kac += (N-i)/i^a
    end

    Kac = Kac/(N-1)

    return Kac
end

function hamiltonian_lr_transverse_ising(a::Float64,J::Float64,h::Float64,N::Integer,d::Integer,n::Integer,kac::String,longdir::String)

    """
    a : interaction strength
    h : Transverse field
    N : system size
    n : number of exponential sums. Also called the MPO bond dimension.
    kac : "true" if kac normalization is used, "false" otherwise
    longdir : longitudinal direction, "X" or "Z"

    For given "n" the bulk MPO is of size (n+2,n+2,d,d). Refer to Phys. Rev. B 78, 035116 (2008)
    for further details on the construction of long range Hamiltonians.
    """

    X,Y,Z,I = spin_ops(d)

    if longdir == "X"
        Long  = X
        Trans = Z
    elseif longdir == "Z"
        Long  = Z
        Trans = X
    end

    if kac == "true"

        Kac = Kac_norm(a,N)

    elseif kac == "false"

        Kac = 1.0
    end

    x, lambda = power_law_to_exp(a,n,N)

    #building the local bulk MPO
    H = zeros(n+2,n+2,2,2)

    H[1,1,:,:] = I; H[n+2,n+2,:,:] = I; H[n+2,1,:,:] = h*Trans

    @inbounds for i in 2:n+1
        H[i,1,:,:] = (x[i-1]/Kac)*Long
        H[i,i,:,:] = lambda[i-1]*I
    end

    @inbounds for j in 2:n+1
        H[n+2,j,:,:] = J*lambda[j-1]*Long
    end

    return H
end

function hamiltonian_lr_longitudinal_ising(a::Float64,J::Float64,h::Float64,N::Integer,d::Integer,n::Integer,kac::String,longdir::String)

    """
    a : interaction strength
    h : longitudinal field
    N : system size
    n : number of exponential sums. Also called the MPO bond dimension.
    kac : "true" if kac normalization is used, "false" otherwise
    longdir : longitudinal direction, "X" or "Z"

    For given "n" the bulk MPO is of size (n+2,n+2,d,d). Refer to Phys. Rev. B 78, 035116 (2008)
    for further details on the construction of long range Hamiltonians.
    """

    X,Y,Z,I = spin_ops(d)

    if longdir == "X"
        Long  = X
    elseif longdir == "Z"
        Long  = Z
    end

    if kac == "true"

        Kac = Kac_norm(a,N)

    elseif kac == "false"

        Kac = 1.0
    end

    x, lambda = power_law_to_exp(a,n,N)

    #building the local bulk MPO
    H = zeros(n+2,n+2,2,2)

    H[1,1,:,:] = I; H[n+2,n+2,:,:] = I; H[n+2,1,:,:] = h*Long

    @inbounds for i in 2:n+1
        H[i,1,:,:] = (x[i-1]/Kac)*Long
        H[i,i,:,:] = lambda[i-1]*I
    end

    @inbounds for j in 2:n+1
        H[n+2,j,:,:] = J*lambda[j-1]*Long
    end

    return H
end

"""
To build the spin-boson MPO given spin and boson MPOs and their boundaries
"""
function cavity_mpo(Ns::Integer,MPO_boson_0,MPO_boson_bulk,MPO_spin_0,MPO_spin_bulk,vR_boson,vR_spin)

    MPO_bulk = direct_sum(MPO_boson_bulk,MPO_spin_bulk)
    MPO_0 = direct_sum(MPO_boson_0,MPO_spin_0)
    vR = vcat(vR_boson,vR_spin)
    @tensor MPO_R[-1,-2,-3,-4] := MPO_bulk[-1,5,-3,-4]*vR[5,-2] 

    Ham = Array{Any,1}(undef,Ns+1)

    Ham[1] = MPO_0
    Ham[Ns+1] = MPO_R

    for i in 2:Ns
        Ham[i] = copy(MPO_bulk)
    end

    return Ham
end
