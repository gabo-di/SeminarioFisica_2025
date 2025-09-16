using LinearAlgebra, SparseArrays 
using CairoMakie  
using DifferentialEquations
using Printf

"""
    harmonic_oscillator!(du, u, p, t)

Simple harmonic oscillator:

    ẍ = - ω² x

assume that ω = p.ω and u = [x, ẋ]
"""
function harmonic_oscillator!(du, u ,p, t)
    ω = p.ω
    du[1] = u[2]
    du[2] = - ω^2*u[1]
    return nothing
end

function main_simple_harmonic_oscillator()
    # Simulation of a simple harmonic oscillator
    T = Float64
    p = (ω = 2pi, 
         τ = 3)
    u0 = T[1, 0]
    tspan = (T(0), T(2pi/p.ω * p.τ))
    prob = ODEProblem(harmonic_oscillator!, u0, tspan, p)
    sol = solve(prob)


    # plotting result
    fig_p = (
        nframes = 100,
        savedir = "./day1/frames_simple_oscillator",
        basename = "simple_ho",
        L = 1.5
    )

    mkpath(fig_p.savedir)
    L = fig_p.L
    outdir = fig_p.savedir
    basename = fig_p.basename
    fig = Figure(size=(900,300))
    c = [:coral]

    # first plot
    ax1 = Axis(fig[1,1], title="Mass position",
                limits = (-2, 2, -1, 1),
                xlabel = "x(t)")
    hideydecorations!(ax1, ticks = false)

    # walls
    poly!(ax1, Rect(-L-0.2, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
    # poly!(ax1, Rect(L, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)

    # particle
    x_obs = Observable(sol(tspan[1])[1])
    scatter!(ax1, lift(x->[x], x_obs), [0.0], markersize=22, color=c[1])

    # second plot
    ax2 = Axis(fig[2, 1],
               title = "Position vs time",
               limits = (tspan[1], tspan[2], -L, L),
               xlabel = "time t",
               ylabel = "x(t)")
    t_trace = Observable(T[])
    x_trace = Observable(T[])
    lines!(ax2, t_trace, x_trace, linewidth=3)
    scatter!(ax2, lift(v -> isempty(v) ? [0.0] : [last(v)], t_trace),
                lift(v -> isempty(v) ? [sol(tspan[1])[1]] : [last(v)], x_trace),
                markersize=12, color=c[1])

    for (i,t) in enumerate(range(tspan[1], tspan[2], fig_p.nframes))
        x = sol(t)[1]
        x_obs[] = x
        t_trace[] = [t_trace[]; t]
        x_trace[] = [x_trace[]; x]
        save(joinpath(outdir, @sprintf("%s-%04d.png", basename, i)), fig)
    end
end

"""
    two_harmonic_oscillator!(du, u, p, t)

Two oscillators simple:

    ẍ₁ = -ω² (x₁ + (x₁ - x₂))  
    ẍ₂ = -ω² (x₂ + (x₂ - x₁))  

assume that ω = p.ω, u = [x₁, x₂, ẋ₁, ẋ₂]
"""
function two_harmonic_oscillator!(du, u, p, t)
    ω = p.ω
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -ω^2 * (2*u[1] - u[2])
    du[4] = -ω^2 * (2*u[2] - u[1])
    return nothing
end

function main_two_harmonic_oscillator()
    # Simulation of two harmonic oscillator
    T = Float64
    p = (ω = 1, 
         τ = 3)

    # eigen modes
    ω = p.ω
    A = T[
         0     0    1   0;
         0     0    0   1;
      -2ω^2   ω^2   0   0;
        ω^2 -2ω^2   0   0
    ]
    F = eigen(A, sortby=abs)
    min_omega = abs(imag(F.values[1]))
    max_omega = abs(imag(F.values[end]))
    @show min_omega
    @show max_omega
    tspan = (T(0), T(2pi/min_omega * p.τ))


    # first eigen mode
    u0_1 = T.(real.(F.vectors[:,1] + F.vectors[:,2]) )
    prob_1 = ODEProblem(two_harmonic_oscillator!, u0_1, tspan, p)
    sol_1 = solve(prob_1)

    # second eigen mode
    u0_2 = T.(real.(F.vectors[:,3] + F.vectors[:,4]) )
    prob_2 = ODEProblem(two_harmonic_oscillator!, u0_2, tspan, p)
    sol_2 = solve(prob_2)

    # plotting result
    fig_p = (
        nframes = 100,
        savedir = "./day1/frames_two_oscillators",
        basename = "two_ho",
        L0 = 1,
    )

    mkpath(fig_p.savedir)
    L0 = fig_p.L0
    L = L0
    outdir = fig_p.savedir
    basename = fig_p.basename
    fig = Figure(size=(900,300))
    c = [:coral1, :red4]

    # First eigen mode            
   
    # first plot
    ax1 = Axis(fig[1,1], title="First eigen mode",
                limits = (-L-0.2, L+0.2, -1, 1),
                xlabel = "x(t)")
    hideydecorations!(ax1, ticks = false)

    # walls
    poly!(ax1, Rect(-L-0.2, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
    poly!(ax1, Rect(L, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)

    # particle
    x11_obs = Observable(sol_1(tspan[1])[1].*0.4-L0/2)
    x21_obs = Observable(sol_1(tspan[1])[2].*0.4+L0/2)
    scatter!(ax1, lift(x->[x], x11_obs), [0.0], markersize=22, color=c[1])
    scatter!(ax1, lift(x->[x], x21_obs), [0.0], markersize=22, color=c[2])

    # second plot
    ax2 = Axis(fig[2, 1],
               title = "Position vs time",
               limits = (tspan[1], tspan[2], -L, L),
               xlabel = "time t",
               ylabel = "x(t)")
    t_trace = Observable(T[])
    x11_trace = Observable(T[])
    x21_trace = Observable(T[])
    lines!(ax2, t_trace, x11_trace, linewidth=3)
    lines!(ax2, t_trace, x21_trace, linewidth=3)
    scatter!(ax2, lift(v -> isempty(v) ? [0.0] : [last(v)], t_trace),
                lift(v -> isempty(v) ? [sol_1(tspan[1])[1]] : [last(v)], x11_trace),
                markersize=12, color=c[1])
    scatter!(ax2, lift(v -> isempty(v) ? [0.0] : [last(v)], t_trace),
                lift(v -> isempty(v) ? [sol_1(tspan[1])[2]] : [last(v)], x21_trace),
                markersize=12, color=c[2])
                
            

    # Second eigen mode            
   
    # first plot
    ax3 = Axis(fig[1,2], title="Second eigen mode",
                limits = (-L-0.2, L+0.2, -1, 1),
                xlabel = "x(t)")
    hideydecorations!(ax3, ticks = false)

    # walls
    poly!(ax3, Rect(-L-0.2, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
    poly!(ax3, Rect(L, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)

    # particle
    x12_obs = Observable(sol_2(tspan[1])[1].*0.4-L0/2)
    x22_obs = Observable(sol_2(tspan[1])[2].*0.4+L0/2)
    scatter!(ax3, lift(x->[x], x12_obs), [0.0], markersize=22, color=c[1])
    scatter!(ax3, lift(x->[x], x22_obs), [0.0], markersize=22, color=c[2])

    # second plot
    ax4 = Axis(fig[2, 2],
               title = "Position vs time",
               limits = (tspan[1], tspan[2], -L, L),
               xlabel = "time t",
               ylabel = "x(t)")
    x12_trace = Observable(T[])
    x22_trace = Observable(T[])
    lines!(ax4, t_trace, x12_trace, linewidth=3)
    lines!(ax4, t_trace, x22_trace, linewidth=3)
    scatter!(ax4, lift(v -> isempty(v) ? [0.0] : [last(v)], t_trace),
                lift(v -> isempty(v) ? [sol_2(tspan[1])[1]] : [last(v)], x12_trace),
                markersize=12, color=c[1])
    scatter!(ax4, lift(v -> isempty(v) ? [0.0] : [last(v)], t_trace),
                lift(v -> isempty(v) ? [sol_2(tspan[1])[2]] : [last(v)], x22_trace),
                markersize=12, color=c[2])


    for (i,t) in enumerate(range(tspan[1], tspan[2], fig_p.nframes))
        x11, x21 = sol_1(t)[1:2]
        x12, x22 = sol_2(t)[1:2]
        x11_obs[] = x11.*0.4 - L0/2
        x21_obs[] = x21.*0.4 + L0/2
        x12_obs[] = x12.*0.4 - L0/2
        x22_obs[] = x22.*0.4 + L0/2
        t_trace[] = [t_trace[]; t]
        x11_trace[] = [x11_trace[]; x11.*0.4 - L0/2]
        x21_trace[] = [x21_trace[]; x21.*0.4 + L0/2]
        x12_trace[] = [x12_trace[]; x12.*0.4 - L0/2]
        x22_trace[] = [x22_trace[]; x22.*0.4 + L0/2]
        save(joinpath(outdir, @sprintf("%s-%04d.png", basename, i)), fig)
    end
end

"""
    n_harmonic_oscillator!(du, u, p, t)

N oscillators simple:
    
    du = A*u 

assume that A = p.A
"""
function n_harmonic_oscillator!(du, u, p, t)
    mul!(du, p.A, u) 
    return nothing
end

function make_A_simple_matrix(T, p)
    n = p.N #this is number of particles, so matrix is size 2N x 2N
    ω = p.ω

    I = Int[]
    J = Int[]
    V = T[]
    for i in 1:n
        # velocities 
        push!(I, i)
        push!(J, i + n)
        push!(V, T(1))
        # accelerations
        push!(I, i+n)
        push!(J, i)
        push!(V, T(-2*ω^2))
        i_m = i - 1 
        i_p = i + 1
        if i_m >= 1
            push!(I, i+n)
            push!(J, i_m)
            push!(V, T(ω^2))
        end
        if i_p <= n
            push!(I, i+n)
            push!(J, i_p)
            push!(V, T(ω^2))
        end
    end
    
    return sparse(I,J,V, 2*n, 2*n)
end

function main_n_harmonic_oscillator()
    # Simulation of N harmonic oscillator
    T = Float64
    p = (ω = 1, 
         τ = 3,
         N = 20)

    A = make_A_simple_matrix(T, p)
    F = eigen(Matrix(A), sortby=abs)
    omegas = abs.(imag.(F.values))
    p_omegas = sortperm(omegas)
    min_omega = omegas[p_omegas[1]]
    max_omega = omegas[p_omegas[end]]
    @show min_omega
    @show max_omega
    tspan = (T(0), T(2pi/min_omega * p.τ))

    function solve_plot_mode(i_mode)
        l_mode = i_mode*2 -1
        # Mode evolution
        z = F.vectors[1,p_omegas[l_mode]]
        if abs(imag(z)) > abs(real(z)) # choose position over velocity
            u0 = T.(imag.(F.vectors[:,p_omegas[l_mode]] -  F.vectors[:,p_omegas[l_mode+1]] ))
        else
            u0 = T.(real.(F.vectors[:,p_omegas[l_mode]] +  F.vectors[:,p_omegas[l_mode+1]] ))
        end
        max_u0 = maximum(abs.(u0))
        u0 ./= max_u0
        prob = ODEProblem(n_harmonic_oscillator!, u0, tspan, (A=A,))
        sol = solve(prob)

        fig_p = (
            nframes = 100,
            savedir = @sprintf("./day1/frames_%03d_oscillators_%03d_mode", p.N, i_mode),
            basename = "N_ho",
            L0 = 1,
        )

        # Plot results
        mkpath(fig_p.savedir)
        L0 = fig_p.L0
        L = L0
        outdir = fig_p.savedir
        basename = fig_p.basename
        fig = Figure(size=(900,300))
        c = [:coral]

        ax1 = Axis(fig[1,1], title=@sprintf("Eigen mode %03d",i_mode),
                    limits = (-L-0.2, L+0.2, -1.1, 1.1),
                    xlabel = "x",
                    ylabel = "A(t)")
        # walls
        poly!(ax1, Rect(-L-0.2, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        poly!(ax1, Rect(L, -0.5, 0.2, 1), color=:lightskyblue1, strokecolor=:black, strokewidth=2)

        # particles
        x_pos = range(-L*(1 - 1/p.N), L*(1 - 1/p.N), p.N) 
        A_obs = Observable(sol(tspan[1])[1:p.N])
        scatter!(ax1, x_pos, A_obs, markersize=10, color=c[1])

        for (i,t) in enumerate(range(tspan[1], tspan[2], fig_p.nframes))
            A_obs[] = sol(t)[1:p.N]
            save(joinpath(outdir, @sprintf("%s-%04d.png", basename, i)), fig)
        end
    end

    solve_plot_mode(1)
    solve_plot_mode(3)

    function plot_energy_modes(A,F)
        B = make_B_simple_matrix(T, p)
        n = length(F.values)
        omegas = T[]
        energies = T[]
        modes = Int[]
        for i in 1:2:n
            z = F.vectors[1,i]
            if abs(imag(z)) > abs(real(z)) # choose position over velocity
                u0 = T.(imag.(F.vectors[:,i] -  F.vectors[:,i+1] ))
            else
                u0 = T.(real.(F.vectors[:,i] +  F.vectors[:,i+1] ))
            end
            max_u0 = norm(u0)
            u0 ./= max_u0

            omega = abs(imag(F.values[i]))
            push!(omegas, omega)
            push!(energies, energy_har(A,B,u0))
            push!(modes, div(i,2))
        end
        fig_p = (
            savedir = "./day1",
            basename = @sprintf("eigen_energies_%03d.png", p.N),
            L0 = 1,
        )

        # Plot results
        L0 = fig_p.L0
        L = L0
        outdir = fig_p.savedir
        basename = fig_p.basename

        c = [:coral, :lightskyblue]

        fig = Figure(size = (900, 300))

        # Left axis: energies (left y-axis)
        axL = Axis(fig[1, 1];
            title  = @sprintf("Eigen energies & frequencies chain of %03d oscillators", p.N),
            xlabel = "mode",
            ylabel = "Energy",
            yticklabelcolor = c[1],
            ylabelcolor     = c[1],
            ytickcolor      = c[1],
            leftspinecolor  = c[1]
        )
        scatter!(axL, modes, energies; markersize=15, color=(c[1], 0.7), strokecolor=:black, strokewidth=1)

        # Right axis: omegas (right y-axis), overlaid in the same grid cell
        axR = Axis(fig[1, 1];
            yaxisposition   = :right,
            ylabel          = "Frequency",
            yticklabelcolor = c[2],
            ylabelcolor     = c[2],
            ytickcolor      = c[2],
            rightspinecolor = c[2],
            xgridvisible    = false,
            ygridvisible    = false,
            backgroundcolor = :transparent,   # so panels don't stack
        )
        linkxaxes!(axR, axL)
        hidexdecorations!(axR, grid=false)
        hidespines!(axR, :l, :t, :b)
        hidespines!(axL, :r)

        scatter!(axR, modes, omegas; markersize=15, color=(c[2], 0.7), strokecolor=:black, strokewidth=1)

        save(joinpath(outdir, basename), fig)
    end

    plot_energy_modes(A,F)
end

function energy_har(A,B,u0)
    (u0' * B * A * u0) / 2
end

function make_B_simple_matrix(T, p)
    n = p.N #this is number of particles, so matrix is size 2N x 2N
    ω = p.ω

    I = Int[]
    J = Int[]
    V = T[]
    for i in 1:n
        # velocities 
        push!(I, i)
        push!(J, i + n)
        push!(V, T(-1))
        # accelerations
        push!(I, i+n)
        push!(J, i)
        push!(V, T(1))
    end
    
    return sparse(I,J,V, 2*n, 2*n)
end
