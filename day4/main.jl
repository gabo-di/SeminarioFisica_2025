using Random
using GLMakie
using Printf


abstract type AbstractUpdateMode end

struct SingleUpdate <: AbstractUpdateMode end
struct CheckerboardUpdate <: AbstractUpdateMode end

abstract type AbstractUpdateAlgorithm end

struct MetropolisUpdate <: AbstractUpdateAlgorithm end
struct GlauberUpdate <: AbstractUpdateAlgorithm end

function main_test()
    Random.seed!(31)
    rng = Random.default_rng()

    # 1 D
    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_1D_single_metropolis.mp4"
    )
    simulate_ising(SingleUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_1D_single_glauber.mp4"
    )
    simulate_ising(SingleUpdate(), GlauberUpdate(), p; rng=rng)

    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_1D_checkerboard_metropolis.mp4"
    )
    simulate_ising(CheckerboardUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 256,
          Ny = 1,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_1D_checkerboard_glauber.mp4"
    )
    simulate_ising(CheckerboardUpdate(), GlauberUpdate(), p; rng=rng)
    
    # 2 D
    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_2D_single_metropolis.mp4"
    )
    simulate_ising(SingleUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_2D_single_glauber.mp4"
    )
    simulate_ising(SingleUpdate(), GlauberUpdate(), p; rng=rng)

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_2D_checkerboard_metropolis.mp4"
    )
    simulate_ising(CheckerboardUpdate(), MetropolisUpdate(), p; rng=rng)

    p = ( Nx = 128,
          Ny = 128,
          h = 0,
          J = 1,
          β = 2,
          frames = 400,
          out = "./day4/ising_2D_checkerboard_glauber.mp4"
    )
    simulate_ising(CheckerboardUpdate(), GlauberUpdate(), p; rng=rng)
end

function simulate_ising(u_mode::M, u_alg::T, p; rng=Random.default_rng()) where {M<:AbstractUpdateMode, T<:AbstractUpdateAlgorithm}

    S = initial_spin_chain(p, rng)

    
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

        # update magnetiation and time plot
        push!(t_obs[], Float32(f))
        push!(m_obs[], Float32(magnetization(S)))
        notify(t_obs); notify(m_obs)
    end
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

function local_energy(S, i, j, p)
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
    # return true if the spin must be flipped
end

function update_condition(u_alg::GlauberUpdate, s, S, i, j, p, rng)
    # WRITE YOUR CODE HERE
    # return true if the spin must be flipped
end

