using Random
using GLMakie
using Printf
using Statistics
using FFTW
using MCMCDiagnosticTools


abstract type AbstractUpdateMode end

struct SingleUpdate <: AbstractUpdateMode end
struct CheckerboardUpdate <: AbstractUpdateMode end

abstract type AbstractUpdateAlgorithm end

struct MetropolisUpdate <: AbstractUpdateAlgorithm end
struct GlauberUpdate <: AbstractUpdateAlgorithm end

function main_test_movie()
    Random.seed!(42)
    rng = Random.default_rng()

    # 1 D
    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_1D_single_metropolis.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, SingleUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_1D_single_glauber.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, SingleUpdate(), GlauberUpdate(), p; rng=rng)

    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_1D_checkerboard_metropolis.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, CheckerboardUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_1D_checkerboard_glauber.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, CheckerboardUpdate(), GlauberUpdate(), p; rng=rng)
    
    # 2 D
    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_2D_single_metropolis.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, SingleUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_2D_single_glauber.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, SingleUpdate(), GlauberUpdate(), p; rng=rng)

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_2D_checkerboard_metropolis.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, CheckerboardUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 0.5,
          frames = 400,
          out = "./day4/ising_2D_checkerboard_glauber.mp4"
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_movie!(S, CheckerboardUpdate(), GlauberUpdate(), p; rng=rng)
end

function simulate_ising_movie!(S, u_mode::M, u_alg::T, p; rng=Random.default_rng()) where {M<:AbstractUpdateMode, T<:AbstractUpdateAlgorithm}
    fig = Figure(size=(1100, 800))

    axL = Axis(fig[1,1],
        title = @sprintf("Ising %dx%d   T=%.3f   J=%.2f   h=%.2f",
                         p.Nx, p.Ny, 1/p.β, p.J, p.h),
        xlabel = "x", ylabel = "y", aspect = DataAspect()
    )
    xlims!(axL, 0.5, p.Nx + 0.5)
    ypad_hi = p.Ny > 1 ? p.Ny + 0.5 : 1.5
    ylims!(axL, 0.5, ypad_hi)

    Z = Observable(S)
    heatmap!(axL, Z; colormap=:grays, colorrange=(-1,1), interpolate=false)


    axM = Axis(fig[2,1], xlabel="Monte Carlo updates", ylabel="m")
    xlims!(axM, 0, p.frames)
    ylims!(axM, -1.05, 1.05)
    hlines!(axM, [0.0], color=:gray, linestyle=:dash)
    hlines!(axM, [1.0], color=:red, linestyle=:dash)
    hlines!(axM, [-1.0], color=:blue, linestyle=:dash)
    
    m_text = @lift(@sprintf("m = %.3f", magnetization($(Z))))
    text!(axM, 0.02, 0.98; text=m_text, space=:relative,
          align=(:left, :top), color=:tomato, fontsize=18)

    t_obs = Observable(Float32[0.0])
    m_obs = Observable(Float32[magnetization(S)])
    lines!(axM, t_obs, m_obs; color=:green, linewidth=2)

    record(fig, p.out, 1:p.frames) do f
        # update spins
        spin_update!(S, u_mode, u_alg, p, rng)

        # update spins plot
        Z[] = S  

        # update magnetization and time plot
        push!(t_obs[], Float32(f))
        push!(m_obs[], Float32(magnetization(S)))
        notify(t_obs); notify(m_obs)
    end
    res = ess_rhat(reshape(m_obs[],:,1); kind=:basic )

    res_text = @sprintf("ess = %.3f, rhat = %3.f", res.ess, res.rhat) 
    text!(axM, 0.8, 0.2; text=res_text, space=:relative,
          align=(:left, :top), color=:coral, fontsize=18)
    save(p.out[1:end-4]*".png", fig)

    println("Simulation saved in ", p.out)
end

function neigh_L(i, L)
    # periodic boundary condition
    if i == 1
        return L 
    else
        return i-1
    end
end

function neigh_R(i, L)
    # periodic boundary condition
    if i == L
        return 1 
    else
        return i+1
    end
end

function magnetization(S)
    return sum(S)/length(S)
end

function local_field(S, i, j, p)
    # sum in x direction
    s = S[neigh_L(i, p.Nx), j] + S[neigh_R(i, p.Nx), j]
    # sum in y direction
    if p.Ny > 1
        s += S[i, neigh_L(j, p.Ny)] + S[i, neigh_R(j, p.Ny)]
    end
    # total energy
    return p.J*s + p.h
end

function initial_spin_chain(p, rng)
    Nx = p.Nx
    Ny = p.Ny
    S = ones(Int8, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        if rand(rng) < 0.5
            S[i,j] = 1
        else
            S[i,j] = -1
        end
    end
    return S
end

function spin_update!(S, u_mode::SingleUpdate, u_alg::T, p, rng) where T<:AbstractUpdateAlgorithm
    for _ in 1:(p.Nx * p.Ny)
        # random position
        # WRITE YOUR CODE HERE
        i = rand(rng, 1:p.Nx) 
        j = rand(rng, 1:p.Ny) 
        s = S[i,j]
        if update_condition(u_alg, s, S, i, j, p, rng) 
            S[i,j] = -s
        end
    end
end

function spin_update!(S, u_mode::CheckerboardUpdate, u_alg::T, p, rng) where T<:AbstractUpdateAlgorithm
    parities = rand(rng, Bool) ? (0, 1) : (1, 0)
    @inbounds for mm in parities 
        for j in axes(S,2)
            for i in axes(S,1)
                if mod(i + j, 2) != mm 
                    continue
                end
                s = S[i,j]
                if update_condition(u_alg, s, S, i, j, p, rng) 
                    S[i,j] = -s
                end
            end
        end
    end
end

function update_condition(u_alg::MetropolisUpdate, s, S, i, j, p, rng)
    # WRITE YOUR CODE HERE
    # energy change
    ΔE = 2.0 * s * local_field(S, i, j, p)
    # update probability
    return rand(rng) < exp(-p.β * ΔE)
end

function update_condition(u_alg::GlauberUpdate, s, S, i, j, p, rng)
    # WRITE YOUR CODE HERE
    # energy change
    ΔE = 2.0 * s * local_field(S, i, j, p)
    # update probability
    return rand(rng) < 1/(1+exp(p.β * ΔE))
end

function simulate_ising_meanfields!(S, u_mode::M, u_alg::T, p; rng=Random.default_rng()) where {M<:AbstractUpdateMode, T<:AbstractUpdateAlgorithm}
    # warm up
    if p.verbose
        m_warmup = Float32[]
    end
    for _ in 1:p.n_warmup
        # update spins
        spin_update!(S, u_mode, u_alg, p, rng)
        if p.verbose
            push!(m_warmup, magnetization(S))
        end
    end

    if p.verbose
        res_warmup = ess_rhat(reshape(m_warmup,:,1); kind=:basic )
        @printf("\nResults for u_mode = %s and u_alg = %s", string(T), string(M))
        @printf("\nIsing %dx%d   T=%.3f   J=%.2f   h=%.2f",
                         p.Nx, p.Ny, 1/p.β, p.J, p.h),
        println("")
        @printf("\n\tWarm up stats for <m> ess = %.3f, rhat = %3.f", res_warmup.ess, res_warmup.rhat) 
    end

    # actual run
    m = Float32[]
    for _ in 1:p.n_iter
        # update spins
        spin_update!(S, u_mode, u_alg, p, rng)
        push!(m, magnetization(S))
    end
    μ, mcse_μ, μ2, mcse_μ2, ess1, ess2 = m_stats(m)
    if p.verbose
        println("")
        @printf("\n\tRun stats ess(m): %.0f   ess(m²): %.0f\n", ess1, ess2)
        @printf("\t<m>   = %.5f  ± %.5f (MCSE)\n",  μ,  mcse_μ)
        @printf("\t<m^2> = %.5f  ± %.5f (MCSE)\n", μ2, mcse_μ2)
    end

    return μ, mcse_μ, μ2, mcse_μ2
end

function m_stats(m)
    x   = reshape(m, :, 1)
    x2  = reshape(m.^2, :, 1)
    res1 = ess_rhat(x;  kind=:basic)
    res2 = ess_rhat(x2; kind=:basic)
    ess1 = res1.ess
    ess2 = res2.ess

    μ   = mean(m)
    μ2  = mean(m.^2)

    v1 = var(m, mean=μ)
    v2 = var(m.^2, mean=μ2)
    mcse_μ  = sqrt(v1 / max(ess1, 1.0))
    mcse_μ2 = sqrt(v2 / max(ess2, 1.0))
    return μ, mcse_μ, μ2, mcse_μ2, ess1, ess2
end

function main_test_meanfields()
    Random.seed!(42)
    rng = Random.default_rng()

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 1/0.3,
          out = "",
          n_iter = 1000,
          n_warmup = 1000,
          verbose = true
    )
    S = initial_spin_chain(p, rng)
    simulate_ising_meanfields!(S, CheckerboardUpdate(), GlauberUpdate(), p; rng=rng)

    return nothing
end

function simulate_ising_phasetransition!(S, u_mode::M, u_alg::T, p; rng=Random.default_rng()) where {M<:AbstractUpdateMode, T<:AbstractUpdateAlgorithm}
    # gather data for all the temperatures
    tmin = p.tmin
    tmax = p.tmax
    n_t = p.n_t

    m = Float32[]
    m_err = Float32[]
    m2 = Float32[]
    m2_err = Float32[]
    temps = range(tmin, tmax, n_t)
    for (ii, temp) in enumerate(temps) 
        _p = merge(p, (β=1/temp,))
        _m = Float32[]
        _m_err = Float32[]
        _m2 = Float32[]
        _m2_err = Float32[]
        for i in 1:p.n_rep
            μ, mcse_μ, μ2, mcse_μ2 = simulate_ising_meanfields!(S, u_mode, u_alg, _p; rng=rng)
            push!(_m, abs(μ))
            push!(_m_err, mcse_μ)
            push!(_m2, μ2)
            push!(_m2_err, mcse_μ2)
            if (sqrt(mean(_m_err.^2)) < p.m_err_tol) || (i == p.n_rep) 
                push!(m, mean(_m))
                push!(m_err, sqrt(mean(_m_err.^2)))
                push!(m2, mean(_m2))
                push!(m2_err, sqrt(mean(_m2_err.^2)))
                break
            end
        end
        if mod(ii-1,n_t/10)==0
            println((ii-1)/length(temps))
        end
    end

    # plot data
    fig = Figure(size=(900,300))

    ax1 = Axis(fig[1,1],
                xlabel = "Temperature",
                ylabel = "Magnetization")
    errors = m_err 
    errorbars!(ax1, temps, m, errors,
        color = range(0, 1, length = length(temps)),
        whiskerwidth = 10)
    scatter!(ax1, temps, m, markersize = 5, color = :black)

    ax2 = Axis(fig[1,2],
                xlabel = "Temperature",
                ylabel = "Magnetic Susceptibility")
    scatter!(ax2, temps, m2 - m.^2, markersize=5, color=:coral)

    save(p.out, fig)
    return fig 
end

function main_test_phasetransition()
    Random.seed!(42)
    rng = Random.default_rng()

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          n_iter = 500,
          n_warmup = 10,
          verbose = false,
          tmin = 2.0,
          tmax = 3.0,
          n_t = 200,
          n_rep = 10,
          m_err_tol = 0.03,
          out = "./day4/ising_2d_phasetransition.png"
    )
    
    # start spin chain in +1 state
    Nx = p.Nx
    Ny = p.Ny
    S = ones(Int8, Nx, Ny)
    simulate_ising_phasetransition!(S, CheckerboardUpdate(), MetropolisUpdate(), p; rng=rng)
end
