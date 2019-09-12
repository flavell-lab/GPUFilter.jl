function gpu_imROF(img::Array{Float32,2}, λ, maxitr)
    # GPU version of imROF of Images package
    size_x, size_y = size(img)

    block_n_x = Int(ceil(size_x / 16))
    block_n_y = Int(ceil(size_y / 16))

    d_img = CuArray(img)
    d_p = CuArray(zeros(Float32, size_x, size_y, 2))
    d_p_div = CuArray(zeros(Float32, size_x, size_y))
    d_u = CuArray(zeros(Float32, size_x, size_y))
    d_grad_u = CuArray(zeros(Float32, size_x, size_y, 2))
    d_grad_u_mag = CuArray(zeros(Float32, size_x, size_y))

    τ = 0.25 # see 2nd remark after proof of Theorem 3.1.

    # This iterates Eq. (9) of the Chambolle citation
    for k = 1:maxitr
        @cuda threads=(16,16) blocks=(block_n_x,block_n_y) kernel_rof(d_p, d_p_div, d_u, d_grad_u, d_grad_u_mag, d_img, λ, τ, size_x, size_y)
    end
    Array(d_u)
end
