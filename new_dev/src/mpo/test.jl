using LinearAlgebra

# 1. Build FSM transitions from coupling list
function build_fsm(coupling_terms::Vector{Tuple{Int,String,String,Float64}})
    num_states = 1                 # state 0 = idle
    transitions = Tuple{Int,Int,String,Int,Float64}[]  # (start,end,op,offset,weight)

    for (dx, op1, op2, w) in coupling_terms
        path = (num_states + 1):(num_states + dx)
        num_states += dx

        println(path)
        # idle → first state : emit op1
        push!(transitions, (0, path[1], op1, 0, w))
        # intermediate identity hops
        for i in 2:length(path)
            push!(transitions, (path[i-1], path[i], "I", i-1, 1.0))
        end
        # last state → idle : emit op2
        push!(transitions, (path[end], 0, op2, dx, 1.0))
    end

    return num_states, transitions
end

"""
creates the FSM for two site interaction term
"""
function build_fsm1(coupling_terms::Vector{Tuple{Int,String,String,Float64}})
    num_states = 1                 
    transitions = Tuple{Int,Int,String,Float64}[] 
    nodes = 2 + sum([couplings[i][1] for i in 1:length(couplings)])
    for (dx, op1, op2, w) in coupling_terms
        path = num_states+1:(num_states + dx)
        num_states += dx 

        # idle → first state : emit op1
        push!(transitions, (1, path[1], op1, w))
        # intermediate identity hops
        for i in 2:length(path)
            push!(transitions, (path[i-1], path[i], "I", 1.0))
        end
        # last state → final idle : emit op2
        push!(transitions, (path[end],nodes, op2, 1.0))
    end

    return num_states, transitions
end

# 2. Initialize empty grids
function init_grids(chi::Int, d::Int, L::Int)
    # 5D: (χ_in, χ_out, phys_in, phys_out, site)
    return zeros(Float64, chi, chi, d, d, L)
end

# 2. Initialize empty grids
function init_grids1(chi::Int, d::Int,T::Type=Float64)
    # 4D: (χ_in, χ_out, phys_in, phys_out)
    return zeros(T, chi, chi, d, d)
end

# 3. Populate grids from FSM transitions
function populate_grids!(
    grids::Array{Float64,5},
    transitions::Vector{Tuple{Int,Int,String,Int,Float64}},
    phys_ops::Dict{String,Matrix{Float64}},
    L::Int
)
    for (s, e, opname, offset, w) in transitions
        op_mat = phys_ops[opname]
        for site in 1:(L - offset)
            @inbounds grids[s, e, :, :, site + offset] .+= w * op_mat
        end
    end
    return grids
end

function populate_grids1!(
    grids::AbstractArray{G,4},
    transitions::Vector{Tuple{Int,Int,String,W}},
    phys_ops::Dict{String,Matrix},
) where {G<:Number, W<:Number}

    grids[1,1,:,:] = phys_ops["I"]
    grids[lastindex(grids,1),lastindex(grids,2),:,:] = phys_ops["I"]
    for (row,col,opname,w) in transitions 
        op_mat = phys_ops[opname]
        grids[row,col,:,:] = w*op_mat 
    end 
    return grids 
end

# 4. Assemble MPO tensors from grids
function assemble_mpo(grids::Array{Float64,5})
    _, _, d, _, L = size(grids)
    chi = size(grids, 1)
    MPO = Vector{Array{Float64,4}}(undef, L)
    for i in 1:L
        # slice out (χ_in, χ_out, d, d)
        slice = grids[:, :, :, :, i]
        # permute to (χ_in, phys_in, phys_out, χ_out)
        MPO[i] = permutedims(slice, (1, 3, 4, 2))
    end
    return MPO
end

# Top‑level: build the full MPO
function build_mpo(
        coupling_terms::Vector{Tuple{Int,String,String,Float64}},
        phys_ops::Dict{String,Matrix{Float64}},
        L::Int)
    χ, transitions = build_fsm(coupling_terms)
    # assume phys_ops["I"] exists
    d = size(phys_ops["I"], 1)
    grids = init_grids(χ, d, L)
    populate_grids!(grids, transitions, phys_ops, L)
    return assemble_mpo(grids)
end

# --- Example usage ---
sx = [0.0 1.0; 1.0 0.0]
sy = [0.0 -1.0*im; 1.0*im  0.0]
sz = [1.0 0.0; 0.0  -1.0]
I2 = Matrix{Float64}(I, 2, 2)

couplings = [(1, "X", "X", 1.0), (2, "Z", "Z", 0.5)]
ops = Dict("X"=>sx, "Y"=>sy, "Z"=>sz,"I"=>I2)

num_states, transitions = build_fsm1(couplings)
chi = num_states + 1 
d = 2
grids = init_grids1(chi,d,Float64)
mpo = populate_grids1!(grids,transitions,ops)

#grids[0,0,:,:] = sz
#println(sz)

#println(size(mpo))
for i in 1:chi
    for j in 1:chi 
        println(i,",",j,",",mpo[i,j,:,:])
    end 
end
#println("Built MPO with bond dim=$(size(mpo[1],1)), length=$L")
