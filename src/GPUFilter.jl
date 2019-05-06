module GPUFilter

using Statistics, CUDAnative, CuArrays, Distributions

include("algorithms.jl")
include("utils.jl")

export
    gpu_imROF,
    compute_λ_filt,
    filter_rof
end # module
