function kernel_rof(p, p_div, u, grad_u, grad_u_mag, img, λ, τ, size_x, size_y)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > size_x || j > size_y
        return nothing
    end

    ## p_div
    if i == size_x
        p_div[i,j] = - p[i-1,j,1]
    elseif i == 1
        p_div[i,j] = p[i,j,1]
    else
        p_div[i,j] = p[i,j,1] - p[i-1,j,1]
    end

    if j == size_y
        p_div[i,j] += - p[i,j-1,2]
    elseif j == 1
        p_div[i,j] += p[i,j,2]
    else
        p_div[i,j] += p[i,j,2] - p[i,j-1,2]
    end

    sync_threads()

    ## u
    u[i,j] = img[i,j] - λ * p_div[i,j]
    sync_threads()

    ## grad_u
    # forwarddiffy
    if i < size_x
        grad_u[i,j,1] = u[i+1,j] - u[i,j]
    else
        grad_u[i,j,1] = u[size_x,j] - u[i,j]
    end

    # forwarddiffx
    if j < size_y
        grad_u[i,j,2] = u[i,j+1] - u[i,j]
    else
        grad_u[i,j,2] = u[i,size_y] - u[i,j]
    end

    sync_threads()

    ## grad_u_mag
    grad_u_mag[i,j] = CUDAnative.sqrt(abs2(grad_u[i,j,1]) + abs2(grad_u[i,j,2]))

    sync_threads()

    ## p_update
    p[i,j,1] = (p[i,j,1] - (τ/λ) * grad_u[i,j,1]) / (1 + (τ/λ) * grad_u_mag[i,j])
    p[i,j,2] = (p[i,j,2] - (τ/λ) * grad_u[i,j,2]) / (1 + (τ/λ) * grad_u_mag[i,j])

    sync_threads()

    nothing
end

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
