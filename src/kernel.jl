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
