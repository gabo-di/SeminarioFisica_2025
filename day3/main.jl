using LinearAlgebra, SparseArrays 
using CairoMakie  
using Printf
using DifferentialEquations
using Revise
using Random
using Sundials
using SparseArrays

"""
solves the Nonlinear Schrodinger Equation with a finite barrier
note this takes some minutes to run
"""
function main_NLSE_eq_finite_barrier()
    T = Float64
    p = (L = 1,
        N = 1000,
        pot_ = 10*2,
        L_extent = 4,
        τ = 3,
        g = -10*4
        )
    x = LinRange( -T(p.L_extent*p.L), T(p.L*p.L_extent), p.N+2)
    A = make_Schrodinger_Dirichlet_soft_barrier_matrix(T, p, x)

    F = eigen(Matrix(A))
    i_mode = 1
    u = make_initial_wave(F, i_mode, x, p) 
    min_omega = F.values[i_mode]
    @show min_omega
    max_omega = F.values[i_mode+1]
    tspan= (T(0), T(2pi/min_omega) * p.τ)
    u0 = vcat(u', zeros(eltype(u), size(u,1))')

    common =(
        reltol = 1e-7,
        abstol = 1e-10,
        dtmax = 0.001,
        maxiters=Int(1e6),
    )

    function plot_nlse(fig_p)
        mkpath(fig_p.savedir)
        outdir = fig_p.savedir
        basename = fig_p.basename
        g = fig_p.g
        title = fig_p.title


        fu = ODEFunction(nlse!; jvp=nlse_jvp!)
        prob = ODEProblem(fu, u0, tspan, (A=A, g=g))
        sol = solve(prob, KenCarp5(); common...)

        fig = Figure(size=(900,300))
        ax1 = Axis(fig[1,1], title=title,
                    limits = fig_p.limits,
                    xlabel = "x",
                    ylabel = "ϕ")
        # hard walls
        poly!(ax1, Rect(-0.2 - 1, 0, 0.2, fig_p.limits[4]), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        poly!(ax1, Rect(1, 0, 0.2, fig_p.limits[4]), color=:lightskyblue1, strokecolor=:black, strokewidth=2)
        # soft walls
        poly!(ax1, Rect(- 1/p.L_extent, 0, 2/p.L_extent , fig_p.limits[4]), color=(:navajowhite, 0.7), strokecolor=:gray, strokewidth=2)

        u_obs = Observable(u)
        x_pos = x[2:end-1] ./ p.L_extent
        lines!(ax1, x_pos, u_obs) 
        for (i,t) in enumerate(range(tspan[1], tspan[2], fig_p.nframes))
            u_ = sol(t)
            u_obs[] = (u_[1,:].^2 + u_[2,:].^2) 
            save(joinpath(outdir, @sprintf("%s-%04d.png", basename, i)), fig)
        end
        @show(norm(sol(tspan[1])))
        @show(norm(sol(tspan[2])))
    end

    fig_p = (
        savedir = @sprintf("./day3/frames_NLSE_finitebarrier"),
        basename = @sprintf("wave_change"),
        title="NLSE",
        L0 = p.L,
        nframes = 500,
        g = p.g,
        limits = (-0.2 - 1, 1+0.2, -0.1, 4.1),
    )
    plot_nlse(fig_p)

    fig_p = (
        savedir = @sprintf("./day3/frames_SE_finitebarrier"),
        basename = @sprintf("wave_change"),
        title="SE",
        L0 = p.L,
        nframes = 500,
        g = 0,
        limits = (-0.2 - 1, 1+0.2, -0.1, 2.1),
    )
    plot_nlse(fig_p)
end

function nlse!(du, u, p, t)
    A = p.A
    g = p.g
    r_du, i_du = eachslice(du, dims=1)
    r_u, i_u = eachslice(u, dims=1)

    # linear part 
    mul!(r_du, A, i_u)
    mul!(i_du, A, -r_u)

    # non linear part 
    map!((z,x,y)-> z + g*(x^2+y^2)*y, r_du, r_du, r_u, i_u)
    map!((z,x,y)-> z - g*(x^2+y^2)*x, i_du, i_du, r_u, i_u)
    return nothing
end

function nlse_jvp!(Jv,v,u,p,t)
    A = p.A
    g = p.g
    r_Jv, i_Jv = eachslice(Jv, dims=1)
    r_v, i_v = eachslice(v, dims=1)
    r_u, i_u = eachslice(u, dims=1)

    # linear part 
    mul!(r_Jv, A, i_v)
    mul!(i_Jv, A, -r_v)

    # non linear part 
    map!((z,x,y,dx,dy)-> z + g*((x^2+y^2)*dy + 2*dx*x*y + 2*dy*y*y), r_Jv, r_Jv, r_u, i_u, r_v, i_v)
    map!((z,x,y,dx,dy)-> z - g*((x^2+y^2)*dx + 2*dx*x*x + 2*dy*y*x), i_Jv, i_Jv, r_u, i_u, r_v, i_v)
end

function make_Schrodinger_Dirichlet_soft_barrier_matrix(T, p, x)
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

function actualize_A_pot!(A, x, p)
    n = length(x)
    for i in 2:n-1
        A[i-1,i-1] += p.pot_/2 * (-tanh((x[i] - p.L)*sqrt(p.N)) + tanh((x[i] + p.L)*sqrt(p.N)))
    end
    return nothing
end

function make_initial_wave(F, i_mode, x, p)
    u1 = F.vectors[:,i_mode]
    u_m = maximum(abs2.(u1))
    u1 ./= u_m

    u2 = F.vectors[:,i_mode+1]
    u_m = maximum(abs2.(u2))
    u2 ./= u_m
    

    u = 0.01*(u2 + u1) .+ semi_uniform(x, p) 
    u_m = maximum(abs2.(u))
    u ./= sqrt(u_m)
    return u
end

function semi_uniform(x, p)
    n = length(x)
    a = zeros(eltype(x), n-2)
    j = round(Int, cbrt(p.N))
    for i in 2:n-1
        a[i-1] += 1/2 * (-tanh((x[i] + x[1+j])*cbrt(p.N)) + tanh((x[i] + x[end-j])*cbrt(p.N)))
    end
    return a
end

"""
solves the chain of N nonlinear oscillators
"""
function main_CNLO_eq()
    # Simulation of N harmonic oscillator
    T = Float64
    p = (ω = 4, 
         τ = 1,
         N = 100,
         a = 10
         )

    A = make_A_simple_matrix(T, p)
    F = eigen(Matrix(A), sortby=abs)
    omegas = abs.(imag.(F.values))
    p_omegas = sortperm(omegas)
    min_omega = omegas[p_omegas[3]]
    max_omega = omegas[p_omegas[end]]
    @show min_omega
    @show max_omega
    tspan = (T(0), T(2pi/min_omega * p.τ))
    Random.seed!(42)
    u0 = 0.5 .* vcat(2*rand(T, p.N) .- 1, zeros(T, p.N))

    common =(
        reltol = 1e-7,
        abstol = 1e-10,
        dtmax = 0.001,
        maxiters=Int(1e6),
    )

    fu = ODEFunction(n_harmonic_oscillator!, jac=(J,u,p,t)->map!(identity, J, p.A), jvp=(Jv,v,u,p,t)->n_harmonic_oscillator!(Jv,v,p,t) )

    function plot_linear_chain_evol()
        prob = ODEProblem(fu, u0, tspan, (A=A,))
        sol = solve(prob, KenCarp4(); common...)

        fig_p = (
            nframes = 100,
            savedir = @sprintf("./day3/frames_lc"),
            basename = "N_ho",
            L0 = 1,
        )

        # Plot results
        mkpath(fig_p.savedir)
        L = fig_p.L0
        outdir = fig_p.savedir
        basename = fig_p.basename
        fig = Figure(size=(900,300))
        c = [:coral]

        ax1 = Axis(fig[1,1], title=@sprintf("Linear chain of %03d oscillators", p.N),
                    limits = (-L-0.2, L+0.2, -2.1, 2.1),
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

    function plot_nonlinear_osc_evol()
        prob = ODEProblem(n_nonlinear_oscillator!, u0, tspan, (a=p.a, b=-p.a/2, n=p.N))
        sol = solve(prob, KenCarp4(); common...)

        fig_p = (
            nframes = 100,
            savedir = @sprintf("./day3/frames_no"),
            basename = "N_ho",
            L0 = 1,
        )

        # Plot results
        mkpath(fig_p.savedir)
        L = fig_p.L0
        outdir = fig_p.savedir
        basename = fig_p.basename
        fig = Figure(size=(900,300))
        c = [:coral]

        ax1 = Axis(fig[1,1], title=@sprintf("%03d nonlinear oscillators", p.N),
                    limits = (-L-0.2, L+0.2, -2.1, 2.1),
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

    function plot_chain_nonlinear_osc_evol()
        prob = SplitODEProblem(fu, n_nonlinear_oscillator!, u0, tspan, (A=A, a=p.a, b=-p.a/2, n=p.N) )
        sol = solve(prob, KenCarp4(); common...)

        fig_p = (
            nframes = 100,
            savedir = @sprintf("./day3/frames_nloc"),
            basename = "N_ho",
            L0 = 1,
        )

        # Plot results
        mkpath(fig_p.savedir)
        L = fig_p.L0
        outdir = fig_p.savedir
        basename = fig_p.basename
        fig = Figure(size=(900,300))
        c = [:coral]

        ax1 = Axis(fig[1,1], title=@sprintf("Chain of %03d nonlinear oscillators", p.N),
                    limits = (-L-0.2, L+0.2, -2.1, 2.1),
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

    plot_linear_chain_evol()
    plot_nonlinear_osc_evol()
    plot_chain_nonlinear_osc_evol()
end

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
        else
            push!(I, i+n)
            push!(J, n)
            push!(V, T(ω^2))
        end
        if i_p <= n
            push!(I, i+n)
            push!(J, i_p)
            push!(V, T(ω^2))
        else
            push!(I, i+n)
            push!(J, 1)
            push!(V, T(ω^2))
        end
    end
    
    return sparse(I,J,V, 2*n, 2*n)
end

function n_nonlinear_oscillator!(du, u, p, t)
    a = p.a
    b = p.b
    n = p.n
    x = @view u[1:n]
    v = @view u[n+1:end]
    dx = @view du[1:n]
    dv = @view du[n+1:end]
    map!((x)-> -a*x^3-b*x, dv, x)
    map!(identity, dx, v)
    return nothing
end


"""
solves the Reaction Diffusion equation
note this takes some minutes to run
"""
function main_ReactionDiffusion_eq()
    T = Float64
    p = (
        N = 256,
        L = 256,
        ϵ = 0.02,
        a = 0.75,
        b = 0.01,
        diff_u = 1,
        diff_v = 0,
    )
    D2 = make_lap2d_periodic(T, p)
    u0 = make_initial_condition_rdeq(T,p)
    rd_eq! = make_reactiondiffusion_eq(p, D2)
    tspan = (T(0), T(30))

    common =(
        reltol = 1e-4,
        abstol = 1e-6,
        maxiters=Int(1e6),
    )

    prob = SplitODEProblem(rd_eq!, u0, tspan)
    alg = ARKODE(Sundials.Implicit(), order=3, linear_solver=:GMRES)
    # prob = ODEProblem(rd_eq!, u0, tspan)
    # alg = CVODE_BDF( linear_solver=:GMRES)
    sol = solve(prob, alg; common...)

    fig_p = (
        savedir = @sprintf("./day3/frames_reactiondiffusion"),
        basename = @sprintf("wave_change"),
        title="Barkley spiral waves",
        nframes = 200,
        limits = ( 0, p.L, 0, p.L),
    )
    function plot_rdeq(fig_p, sol)
        mkpath(fig_p.savedir)
        outdir = fig_p.savedir
        basename = fig_p.basename
        title = fig_p.title

        fig = Figure(size=(900,300))
        ax1 = Axis(fig[1,1], title=title,
                    # limits = fig_p.limits,
                    xlabel = "x",
                    ylabel = "y",
                    aspect = DataAspect())
        u = Observable(reshape(sol[1][1,:], p.N, p.N))
        hm1  = heatmap!(ax1, u; colormap=:lajolla, colorrange=(0,1))
        Colorbar(fig[1,2], hm1)

        ax2 = Axis(fig[1,3], title=title,
                    # limits = fig_p.limits,
                    xlabel = "x",
                    ylabel = "y",
                    aspect = DataAspect())
        v = Observable(reshape(sol[1][2,:], p.N, p.N))
        hm2  = heatmap!(ax2, v; colormap=:lajolla, colorrange=(0,1))
        Colorbar(fig[1,4], hm2)
        for (i,t) in enumerate(range(tspan[1], tspan[2], fig_p.nframes))
            u[] = reshape(sol(t)[1,:], p.N, p.N)
            v[] = reshape(sol(t)[2,:], p.N, p.N)
            save(joinpath(outdir, @sprintf("%s-%04d.png", basename, i)), fig)
        end
    end
    plot_rdeq(fig_p, sol)
end

function make_reactiondiffusion_eq(par, D2)
    ϵ = par.ϵ
    a = par.a
    b = par.b

    L_u = par.diff_u * (D2)
    L_v = par.diff_v * (D2)

    function f_rdeq(u,v)
        1/ϵ * u*(1-u)*(u - (v+b)/a) 
    end

    function g_rdeq(u,v)
        u - v
    end

    function reactiondiffusion!(du, u, p, t)
        u_f, v_f = eachslice(u, dims=1)
        du_f, dv_f = eachslice(du, dims=1)
        mul!(du_f, L_u, u_f)
        mul!(dv_f, L_v, v_f)
        map!((z,x,y)->z+g_rdeq(x,y), dv_f, dv_f, u_f, v_f)
        map!((z,x,y)->z+f_rdeq(x,y), du_f, du_f, u_f, v_f)
        return nothing
    end

    function reaction!(du, u, p, t)
        u_f, v_f = eachslice(u, dims=1)
        du_f, dv_f = eachslice(du, dims=1)
        map!((x,y)->g_rdeq(x,y), dv_f, u_f, v_f)
        map!((x,y)->f_rdeq(x,y), du_f, u_f, v_f)
        return nothing
    end

    function diffusion!(du, u, p, t)
        u_f, v_f = eachslice(u, dims=1)
        du_f, dv_f = eachslice(du, dims=1)
        mul!(du_f, L_u, u_f)
        mul!(dv_f, L_v, v_f)
        return nothing 
    end

    function diffusion_jvp!(Jv, v, u, p, t)
        diffusion!(Jv, v, p, t)     
    end

    fu = ODEFunction(diffusion!, jvp=diffusion_jvp!)
    return SplitFunction(fu, reaction!, jvp=fu.jvp)
    # return reactiondiffusion!
end

function make_lap1d_periodic(T, p)
    L = p.L
    N = p.N
    dx = T(L) / N
    main = fill(T(-2) / dx^2, N)
    off  = fill(T( 1) / dx^2, N-1)
    A = spdiagm(-1 => off, 0 => main, 1 => off)
    A[1, end] = T(1) / dx^2
    A[end, 1] = T(1) / dx^2
    return A
end

function make_lap2d_periodic(T, p)
    N = p.N
    A = make_lap1d_periodic(T, p)
    i = sparse(I,N,N)
    return kron(i, A) + kron(A, i)
end

function make_initial_condition_rdeq(T,p)
    v = zeros(T, p.N*p.N) 
    u = zeros(T, p.N*p.N)
    delta = max(2, round(Int, cbrt(p.N)))
    # wave front
    for i in 1:p.N
        for j in 1:p.N
            idx = i + (j-1)*p.N
            u[idx] = soft_one(i, p.N, p.N/2 - delta, p.N/2 + delta) * soft_one(j, p.N, 1, p.N)
            v[idx] = soft_one(i, p.N, 1, p.N) * soft_one(j, p.N, p.N/2 - delta, p.N/2 + delta)
        end
    end
    
    u0 = permutedims(cat(u, v, dims=2), (2,1))
    return u0
end

function soft_one(i, n, n_min, n_max)
    -1/2*(-tanh((i-n_min)/cbrt(n)) + tanh((i-n_max)/cbrt(n)))
end
