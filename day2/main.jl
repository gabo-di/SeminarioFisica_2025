using LinearAlgebra, SparseArrays 
using CairoMakie  
using Printf

"""
Finds and solves the eigensystem for EM wave equation
using Dirichlet boundary condition
"""
function main_EM_wave_Dirichlet()
    T = Float64
    p = (L = 1,
        N = 100,
        c = 1,
        )
    A = make_EM_wave_Dirichlet_matrix(T, p)
    x = LinRange(T(0), T(p.L), p.N+2)
    F = eigen(Matrix(A), sortby=abs)
    
    function analytical_eigenstate(i_mode)
        return x -> sin(x *pi *i_mode/ p.L)
    end

    function plot_sol_and_analytical(i_mode)
        fig_p = (
            savedir = @sprintf("./day2"),
            basename = @sprintf("EM_Dirichlet_eigenstate_%03d.png", i_mode),
            L0 = p.L,
        )
        outdir = fig_p.savedir
        basename = fig_p.basename
        L = fig_p.L0

        f = analytical_eigenstate(i_mode) 
        u = F.vectors[:,i_mode]
        u_m = maximum(u)
        u ./= u_m

        fig = Figure(size=(900,300))
        ax1 = Axis(fig[1,1], title=@sprintf("Eigen mode %03d",i_mode),
                    limits = (-0.2, L+0.2, -1.1, 1.1),
                    xlabel = "x",
                    ylabel = "ϕ")
        # walls
        poly!(ax1, Rect(-0.2, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        poly!(ax1, Rect(L, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        scatter!(ax1, x[2:end-1], u, label="Numerical solution")
        lines!(ax1, x, f.(x), color=:coral, label="Analytical solution")
        axislegend(position = :rt)
        save(joinpath(outdir, basename), fig)
    end

    plot_sol_and_analytical(1)
    plot_sol_and_analytical(3)


    fig_p = (
        savedir = @sprintf("./day2"),
        basename = @sprintf("EM_Dirichlet_eigenvalues.png"),
    )
    outdir = fig_p.savedir
    basename = fig_p.basename
    n_range = 1:p.N

    fig = Figure(size=(900,300))
    ax1 = Axis(fig[1,1], title=@sprintf("Eigen values"),
                xlabel = "n",
                ylabel = "eigenvalue")
    scatter!(ax1, n_range, F.values, label="Numerical solution")
    lines!(ax1, n_range, -(n_range*pi/p.L).^2, color=:coral, label="Analytical solution")
    axislegend(position = :rt)
    save(joinpath(outdir, basename), fig)
end

function make_EM_wave_Dirichlet_matrix(T, p)
    n = p.N #this is number of nodes
    dx = p.L/(p.N+1)

    I = Int[]
    J = Int[]
    V = T[]
    for i in 1:n
        push!(I, i)
        push!(J, i)
        push!(V, T(-2/dx^2))
        i_m = i - 1 
        i_p = i + 1
        if i_m >= 1
            push!(I, i)
            push!(J, i_m)
            push!(V, T(1/dx^2))
        end
        if i_p <= n
            push!(I, i)
            push!(J, i_p)
            push!(V, T(1/dx^2))
        end
    end
    
    return sparse(I,J,V, n, n)
end

"""
Finds and solves the eigensystem for diffusion equation
using Neumann boundary condition
"""
function main_Diffusion_eq_Neumann()
    T = Float64
    p = (L = 1,
        N = 100,
        )
    A = make_Diffusion_eq_Neumann_matrix(T, p)
    x = LinRange(T(0), T(p.L), p.N+2)
    F = eigen(Matrix(A), sortby=abs)
    
    function analytical_eigenstate(i_mode)
        return x -> cos(x *pi *i_mode/ p.L)
    end

    function plot_sol_and_analytical(i_mode)
        fig_p = (
            savedir = @sprintf("./day2"),
            basename = @sprintf("Diffusion_eq_eigenstate_%03d.png", i_mode),
            L0 = p.L,
        )
        outdir = fig_p.savedir
        basename = fig_p.basename
        L = fig_p.L0

        f = analytical_eigenstate(i_mode-1) # the eigenmodes start in 0 
        u = F.vectors[:,i_mode]
        u_m = u[1]#maximum(u)
        u ./= u_m

        fig = Figure(size=(900,300))
        ax1 = Axis(fig[1,1], title=@sprintf("Eigen mode %03d",i_mode),
                    limits = (-0.2, L+0.2, -1.1, 1.1),
                    xlabel = "x",
                    ylabel = "ϕ")
        # walls
        poly!(ax1, Rect(-0.2, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        poly!(ax1, Rect(L, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        scatter!(ax1, x[2:end-1], u, label="Numerical solution")
        lines!(ax1, x, f.(x), color=:coral, label="Analytical solution")
        axislegend(position = :rt)
        save(joinpath(outdir, basename), fig)
    end

    plot_sol_and_analytical(1)
    plot_sol_and_analytical(3)
end

function make_Diffusion_eq_Neumann_matrix(T, p)
    n = p.N #this is number of nodes
    dx = p.L/(p.N+1)

    I = Int[]
    J = Int[]
    V = T[]
    for i in 2:n-1
        push!(I, i)
        push!(J, i)
        push!(V, T(-2/dx^2))
        i_m = i - 1 
        i_p = i + 1
        if i_m >= 1
            push!(I, i)
            push!(J, i_m)
            push!(V, T(1/dx^2))
        end
        if i_p <= n
            push!(I, i)
            push!(J, i_p)
            push!(V, T(1/dx^2))
        end
    end

    # left boundary
    push!(I, 1)
    push!(J, 1)
    push!(V, T(-1/dx^2))
    push!(I, 1)
    push!(J, 2)
    push!(V, T(1/dx^2))
    
    # right boundary
    push!(I, n)
    push!(J, n)
    push!(V, T(-1/dx^2))
    push!(I, n)
    push!(J, n-1)
    push!(V, T(1/dx^2))
    
    return sparse(I,J,V, n, n)
end

"""
Finds and solves the eigensystem for Schrodinger equation with finite well 
using periodic boundary condition
"""
function main_Schrodinger_eq_finite_well()
    T = Float64
    p = (L = 1,
        N = 1000,
        pot_ = -100,
        L_extent = 4
        )
    x = LinRange( -T(p.L_extent*p.L), T(p.L*p.L_extent), p.N+2)
    A = make_Schrodinger_Periodic_matrix(T, p, x)

    F = eigen(Matrix(A))
    

    function plot_sol_and_analytical(i_mode)
        fig_p = (
            savedir = @sprintf("./day2"),
            basename = @sprintf("Schr_finitewell_eigenstate_%03d.png", i_mode),
            L0 = p.L,
        )
        outdir = fig_p.savedir
        basename = fig_p.basename
        L = fig_p.L0

        u = F.vectors[:,i_mode]
        u_m = maximum(abs.(u))
        u ./= u_m
        @show F.values[i_mode]

        fig = Figure(size=(900,300))
        ax1 = Axis(fig[1,1], title=@sprintf("Eigen mode %03d with eigen energy %6.2f",i_mode, F.values[i_mode]),
        limits = (-0.2 - 1, 1+0.2, -1.1, 1.1),
                    xlabel = "x",
                    ylabel = "ϕ")
        # hard walls
        poly!(ax1, Rect(-0.2 - 1, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        poly!(ax1, Rect(1, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        # soft walls
        poly!(ax1, Rect(-1, -1, 1 - 1/p.L_extent, 2), color=(:navajowhite, 0.7), strokecolor=:gray, strokewidth=2)
        poly!(ax1, Rect(1/p.L_extent, -1, 1-1/p.L_extent, 2), color=(:navajowhite, 0.7), strokecolor=:gray, strokewidth=2)

        lines!(ax1, x[2:end-1]./p.L_extent, u, label="Numerical solution")
        axislegend(position = :rt)
        save(joinpath(outdir, basename), fig)
    end

    plot_sol_and_analytical(1)
    plot_sol_and_analytical(2)
    plot_sol_and_analytical(3)
    plot_sol_and_analytical(4)
end

"""
Finds and solves the eigensystem for Schrodinger equation with finite barrier
using Dirichlet boundary condition
"""
function main_Schrodinger_eq_finite_barrier()
    T = Float64
    p = (L = 1,
        N = 1000,
        pot_ = 100*2,
        L_extent = 4,
        τ = 3
        )
    x = LinRange( -T(p.L_extent*p.L), T(p.L*p.L_extent), p.N+2)
    A = make_Schrodinger_Dirichlet_matrix(T, p, x)

    F = eigen(Matrix(A))
    

    function plot_sol_and_analytical(i_mode)
        fig_p = (
            savedir = @sprintf("./day2"),
            basename = @sprintf("Schr_finitebarrier_eigenstate_%03d.png", i_mode),
            L0 = p.L,
        )
        outdir = fig_p.savedir
        basename = fig_p.basename

        u = F.vectors[:,i_mode]
        u_m = maximum(abs.(u))
        u ./= u_m
        @show F.values[i_mode]

        fig = Figure(size=(900,300))
        ax1 = Axis(fig[1,1], title=@sprintf("Eigen mode %03d with eigen energy %7.3f",i_mode, F.values[i_mode]),
        limits = (-0.2 - 1, 1+0.2, -1.1, 1.1),
                    xlabel = "x",
                    ylabel = "ϕ")
        # hard walls
        poly!(ax1, Rect(-0.2 - 1, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        poly!(ax1, Rect(1, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        # soft walls
        poly!(ax1, Rect(- 1/p.L_extent, -1, 2/p.L_extent , 2), color=(:navajowhite, 0.7), strokecolor=:gray, strokewidth=2)

        lines!(ax1, x[2:end-1]./p.L_extent, u, label="Numerical solution")
        axislegend(position = :rt)
        save(joinpath(outdir, basename), fig)
    end

    plot_sol_and_analytical(1)
    plot_sol_and_analytical(2)
    
    function plot_ev_two_eigenstates(i_mode)
        fig_p = (
            savedir = @sprintf("./day2/frames_Schr_finitebarrier"),
            basename = @sprintf("sum_modes_%02d", i_mode),
            L0 = p.L,
            nframes = 100
        )
        mkpath(fig_p.savedir)
        outdir = fig_p.savedir
        basename = fig_p.basename
        
        u, u1, u2 = make_sum_two_wave(F, i_mode)
        @show F.values[i_mode]
        @show F.values[i_mode+1]
        min_omega = F.values[i_mode]
        max_omega = F.values[i_mode+1]
        tspan= (T(0), T(2pi/min_omega) * p.τ)

        fig = Figure(size=(900,300))
        ax1 = Axis(fig[1,1], title="Sum of two eigenmodes",
        limits = (-0.2 - 1, 1+0.2, -1.1, 1.1),
                    xlabel = "x",
                    ylabel = "ϕ")
        # hard walls
        poly!(ax1, Rect(-0.2 - 1, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        poly!(ax1, Rect(1, -1, 0.2, 2), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        # soft walls
        poly!(ax1, Rect(- 1/p.L_extent, -1, 2/p.L_extent , 2), color=(:navajowhite, 0.7), strokecolor=:gray, strokewidth=2)

        u_obs = Observable(u)
        x_pos = x[2:end-1] ./ p.L_extent
        lines!(ax1, x_pos, u_obs) 
        for (i,t) in enumerate(range(tspan[1], tspan[2], fig_p.nframes))
            u_obs[] = real.(u1 * exp(-im*min_omega*t) + u2 * exp(-im*max_omega*t))
            save(joinpath(outdir, @sprintf("%s-%04d.png", basename, i)), fig)
        end
    end

    plot_ev_two_eigenstates(1)
end

function make_sum_two_wave(F, i_mode)
    u1 = F.vectors[:,i_mode]
    u_m = maximum(abs2.(u1))
    u1 ./= u_m

    u2 = F.vectors[:,i_mode+1]
    u_m = maximum(abs2.(u2))
    u2 ./= u_m

    u = u2 + u1
    u_m = maximum(abs2.(u))
    u ./= sqrt(u_m)

    u1 ./= sqrt(u_m)
    u2 ./= sqrt(u_m)

    return u, u1, u2
end

function make_Schrodinger_Periodic_matrix(T, p, x)
    n = p.N #this is number of nodes
    dx = 2/(p.N+1)

    I = Int[]
    J = Int[]
    V = T[]
    for i in 1:n
        push!(I, i)
        push!(J, i)
        push!(V, T(2/dx^2))
        i_m = i - 1 
        i_p = i + 1
        if i_m >= 1
            push!(I, i)
            push!(J, i_m)
            push!(V, T(-1/dx^2))
        else
            push!(I, i)
            push!(J, n)
            push!(V, T(-1/dx^2))
        end
        if i_p <= n
            push!(I, i)
            push!(J, i_p)
            push!(V, T(-1/dx^2))
        else
            push!(I, i)
            push!(J, 1)
            push!(V, T(-1/dx^2))
        end
    end
    
    A = sparse(I,J,V, n, n)
    actualize_A_pot!(A, x, p)
    return A
end

function make_Schrodinger_Dirichlet_matrix(T, p, x)
    n = p.N #this is number of nodes
    dx = 2/(p.N+1)

    I = Int[]
    J = Int[]
    V = T[]
    for i in 1:n
        push!(I, i)
        push!(J, i)
        push!(V, T(2/dx^2))
        i_m = i - 1 
        i_p = i + 1
        if i_m >= 1
            push!(I, i)
            push!(J, i_m)
            push!(V, T(-1/dx^2))
        end
        if i_p <= n
            push!(I, i)
            push!(J, i_p)
            push!(V, T(-1/dx^2))
        end
    end
    
    A = sparse(I,J,V, n, n)
    actualize_A_pot!(A, x, p)
    return A
end

function make_Schrodinger_Neumann_matrix(T, p, x)
    n = p.N #this is number of nodes
    dx = 2/(p.N+1)

    I = Int[]
    J = Int[]
    V = T[]
    for i in 2:n-1
        push!(I, i)
        push!(J, i)
        push!(V, T(2/dx^2))
        i_m = i - 1 
        i_p = i + 1
        if i_m >= 1
            push!(I, i)
            push!(J, i_m)
            push!(V, T(-1/dx^2))
        end
        if i_p <= n
            push!(I, i)
            push!(J, i_p)
            push!(V, T(-1/dx^2))
        end
    end

    # left boundary
    push!(I, 1)
    push!(J, 1)
    push!(V, T(1/dx^2))
    push!(I, 1)
    push!(J, 2)
    push!(V, T(-1/dx^2))
    
    # right boundary
    push!(I, n)
    push!(J, n)
    push!(V, T(1/dx^2))
    push!(I, n)
    push!(J, n-1)
    push!(V, T(-1/dx^2))
    
    A = sparse(I,J,V, n, n)
    actualize_A_pot!(A, x, p)
    return A
end

function actualize_A_pot!(A, x, p)
    n = p.N #this is number of nodes
    dx = 2/(p.N+1)
    n = length(x)
    for i in 2:n-1
        if  p.L >= x[i] >= -p.L
            A[i-1,i-1] += p.pot_
        end
    end
    return nothing
end
