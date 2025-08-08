"""
one point (expectations) and two points (correlators) observables calculated on a MPS/MPDO
"""

using LinearAlgebra


"""
Inner product of a quantum state <psi|O_i|psi>
Will be 1 if the states are normalized
"""

function inner_product(psi,N,object::String)

    """
    psi: quantum state (MPS/MPDO)
    object: either MPS or MPDO
    """

    R = ones(1,1)

    if object == "MPS"

        @inbounds for i in reverse(1:N)
            R = contract_right_noop(psi[i],R,"MPS")
        end

    elseif object == "MPDO"

        @inbounds for i in reverse(1:N)
            R = contract_right_noop(psi[i],R,"MPDO")
        end       
    end
    return R[1]
end
    
"""
calculates single site expectation <psi|O_i|psi>
"""
function expect_single_site(j,psi,O,N,object::String)

    """
    psi1 : input state1 in right canonical form
    psi2 : input state2 in right canonical form
    object : nature of psi, MPS or MPDO
    """

    R = ones(1,1)
                
    if object == "MPS"

        @inbounds for i in reverse(j+1:N)
            R = contract_right_noop(psi[i],R,"MPS")
        end

        R = contract_right_nompo(psi[j],R,O,"MPS")
    
        @inbounds for i in reverse(1:j-1)
            R = contract_right_noop(psi[i],R,"MPS")
        end
    
    elseif object == "MPDO"

        @inbounds for i in reverse(j+1:N)
            R = contract_right_noop(psi[i],R,"MPDO")
        end

        R = contract_right_nompo(psi[i],R,O,"MPDO")
    
        @inbounds for i in reverse(1:j-1)
            R = contract_right_noop(psi[i],R,"MPDO")
        end         
    end

    return R[1]
end


"""
calculates single site expectation <psi|O_i|psi>
"""

function expect_single_site_updated(i,psi,O,object::String)

    """
    i : site where expectation is calculated
    psi : input state in right canonical form
    O : operator of dimension (d,d)
    object : nature of psi, MPS or MPDO
    """
                
    if object == "MPS"

        siz = size(psi[i])[3]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[i],R,O,"MPS")
    
        @inbounds for i in reverse(1:i-1)
        
            R = contract_right_noop(psi[i],R,"MPS")
        end
        
        return real(R[1])
    
    elseif object == "MPDO"

        siz = size(psi[i])[4]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[i],R,O,"MPDO")
    
        @inbounds for i in reverse(1:i-1)
        
            R = contract_right_noop(psi[i],R,"MPDO")
        end
        
        return R[1]
    end
end

#Note : the inner product is calculated with this function by setting O as a (d,d) unit matrix, i.e. <psi|1|psi> = <psi||psi> 

"""
Calculates subsystem expectation, <psi|SUM_{i=1:l} O_i |psi>
"""

function expect_subsystem(N,l,psi,O,object::String)

    """
    N : system size
    l : subsystem size 
    psi : input state in right canonical form
    O : operator of dimension (d,d)
    object : nature of psi, MPS or MPDO
    """
    
    k = trunc(Int,N/2) - trunc(Int,l/2)

    expect_val = 0.0
    
    if object == "MPS"

        for i in k+1:k+l

            val = expect_single_site(i,psi,O,N,"MPS")

            expect_val += val
        end

        return expect_val/l

    elseif object == "MPDO"

            for i in k+1:k+l
    
                val = expect_single_site(i,psi,O,N,"MPDO")
    
                expect_val += val
            end
    
        return expect_val/l
    end
end

"""
Calculates two site expectation, <psi|O1_j O2_k|psi>
"""

function expect_two_site_updated(j,k,Oj,Ok,psi,object::String)

    """
    j,k : sites with operators Oj and Ok
    psi : input state in right canonical form
    object : nature of psi, MPS or MPDO
    Assume k>j
    """

    if object == "MPS"
    
        siz = size(psi[k])[3]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[k],R,Ok,"MPS")
    
        @inbounds for i in reverse(1:k-1)
        
            if i == j
                R = contract_right_nompo(psi[j],R,Oj,"MPS")
            else
                R = contract_right_noop(psi[i],R,"MPS")
            end
        end

    elseif object == "MPDO"

        siz = size(psi[k])[4]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[k],R,Ok,"MPDO")
    
        @inbounds for i in reverse(1:k-1)
        
            if i == j
                R = contract_right_nompo(psi[j],R,Oj,"MPDO")
            else
                R = contract_right_noop(psi[i],R,"MPDO")
            end
        end
    end        
    return R[1]
end

function kink_full(N,psi,r,O,object::String)
    kink_num = 0
    for i in 1:N-r
        kink_num += expect_two_site_updated(i,i+1,O,O,psi,object)
    end
    return ((N-1)-4*kink_num)/2
end

"""
Calculates equal time correlated function, <psi|O_j O_k|psi>
"""

function corr_func(j,k,psi,O,object::String)

    """
    j,k : sites with operators O
    psi : input state in right canonical form
    O : operator of dimension (d,d)
    object : nature of psi, MPS or MPDO
    """

    if object == "MPS"
    
        siz = size(psi[k])[3]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[k],R,O,"MPS")
    
        @inbounds for i in reverse(1:k-1)
        
            if i == j
                R = contract_right_nompo(psi[j],R,O,"MPS")
            else
                R = contract_right_noop(psi[i],R,"MPS")
            end
        end

        return R[1]

    elseif object == "MPDO"

        siz = size(psi[k])[4]
    
        R = Matrix{Float64}(I,siz,siz)
    
        R = contract_right_nompo(psi[k],R,O,"MPDO")
    
        @inbounds for i in reverse(1:k-1)
        
            if i == j
                R = contract_right_nompo(psi[j],R,O,"MPDO")
            else
                R = contract_right_noop(psi[i],R,"MPDO")
            end
        end

        return R[1]
    end         
end

"""
Calculates energy density of the given state (MPS or MPDO)
"""
function Average_Energy(state,Ham,N,object::String)

    """
    state : input MPS or MPDO
    Ham : Hamiltonian as MPO
    N : system size
    object : type of state
    """

    X = ones(1,1,1)

    if object == "MPS"
        for i in 1:N
            X = contract_left(state[i],X,Ham[i],"MPS")
        end

        return reshape(X,1)[1]

    elseif object == "MPDO"
        for i in 1:N
            X = contract_left(state[i],X,Ham[i],"MPDO")
        end

        return real(reshape(X,1)[1])
    end        
end

"""
Calculates subsystem expectation of the form <psi|PROD_{i=1:l} O_i |psi>
Here we have assumed few things:
1) O_i is independent of i, i.e. same operator at every sites of subsystem, can be modified to take list of operators
2) The state psi 
"""

function product_opt_updated(N,l,psi,theta,Oprt,object::String)

    k = trunc(Int,N/2) - trunc(Int,l/2)

    siz = size(psi[k+l+1])[1]
    
    R = Matrix{Float64}(I,siz,siz)

    O = exp(im*theta*Oprt);

    if object == "MPS"
        
        @inbounds for i in reverse(k+1:k+l)
            R = contract_right_nompo(psi[i],R,O,"MPS")
        end
    
        @inbounds for i in reverse(1:k)
            R = contract_right_noop(psi[i],R,"MPS")
        end

    elseif object == "MPDO"

        @inbounds for i in reverse(k+1:k+l)
            R = contract_right_nompo(psi[i],R,O,"MPDO")
        end
    
        @inbounds for i in reverse(1:k)
            R = contract_right_noop(psi[i],R,"MPDO")
        end
    end

    return reshape(R,1)[1]
end
