function gpu_imROF(img::Array{Float32,2}, λ, maxitr)
    # GPU version of imROF of Images package
    size_x, size_y = size(img)

    n_th = 16 # number of thread

    b_n_x = Int(ceil(size_x / n_th)) # blocks x
    b_n_y = Int(ceil(size_y / n_th)) # blocks y

    d_img = CuArray(img)
    d_p = CuArrays.zeros(Float32, size_x, size_y, 2)
    d_p_div = CuArrays.zeros(Float32, size_x, size_y)
    d_u = CuArrays.zeros(Float32, size_x, size_y)
    d_grad_u = CuArrays.zeros(Float32, size_x, size_y, 2)
    d_grad_u_mag = CuArrays.zeros(Float32, size_x, size_y)

    τ = 0.25 # see 2nd remark after proof of Theorem 3.1.

    # This iterates Eq. (9) of the Chambolle citation
    for k = 1:maxitr
        @cuda threads=(n_th,n_th) blocks=(b_n_x,b_n_y) kernel_div(d_p_div, d_p, size_x, size_y)
        @cuda threads=(n_th,n_th) blocks=(b_n_x,b_n_y) kernel_u(d_u, d_img, λ, d_p_div, size_x, size_y) # multiply term inside ∇ by -λ. Thm. 3.1 relates this to u via Eq. 7.
        @cuda threads=(n_th,n_th) blocks=(b_n_x,b_n_y) kernel_grad_u(d_grad_u, d_u, size_x, size_y)
        @cuda threads=(n_th,n_th) blocks=(b_n_x,b_n_y) kernel_grad_u_mag(d_grad_u_mag, d_grad_u, size_x, size_y)
        @cuda threads=(n_th,n_th) blocks=(b_n_x,b_n_y) kernel_p_update(d_p, d_grad_u, d_grad_u_mag, λ, τ, size_x, size_y)
    end

    Array(d_u)
end
