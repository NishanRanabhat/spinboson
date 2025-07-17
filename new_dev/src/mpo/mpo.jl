using LinearAlgebra

function build_fsm(coupling_terms::Vector{Tuple{Int,String,String,Float64}})
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
function init_grids(chi::Int, d::Int,T::Type=Float64)
    # 4D: (χ_in, χ_out, phys_in, phys_out)
    return zeros(T, chi, chi, d, d)
end

function populate_grids!(
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

# --- Example usage ---
sx = [0.0 1.0; 1.0 0.0]
sy = [0.0 -1.0*im; 1.0*im  0.0]
sz = [1.0 0.0; 0.0  -1.0]
I2 = Matrix{Float64}(I, 2, 2)

couplings = [(1, "X", "X", 1.0), (2, "Z", "Z", 0.5)]
ops = Dict("X"=>sx, "Y"=>sy, "Z"=>sz,"I"=>I2)

num_states, transitions = build_fsm(couplings)
chi = num_states + 1 
d = 2
grids = init_grids(chi,d,Float64)
mpo = populate_grids!(grids,transitions,ops)

