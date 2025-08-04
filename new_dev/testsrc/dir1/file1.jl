module Sub1

    export MPS

    abstract type TensorNetwork{T} end

    struct MPS{T} <: TensorNetwork{T}
      tensors::Vector{Array{T,3}}
    end

end  # module Sub1
