############################################################
# Day 4 — Ising model (1D/2D) + GLMakie MP4 + Magnetization
############################################################

using Random
using GLMakie
using Printf

# # ------------------------
# # Parameters & utilities
# # ------------------------
# struct IsingParams
#     Lx::Int
#     Ly::Int        # set Ly=1 for 1D
#     J::Float64
#     h::Float64
#     β::Float64     # 1/(k_B T)
# end
#
# function init_spins(Lx, Ly; p_up=0.5, rng=Random.GLOBAL_RNG)
#     S = fill(Int8(1), Lx, Ly)
#     @inbounds for j in 1:Ly, i in 1:Lx
#         S[i,j] = rand(rng) < p_up ? Int8(1) : Int8(-1)
#     end
#     S
# end
#
# @inline left(i, L)  = (i == 1 ? L : i-1)
# @inline right(i, L) = (i == L ? 1 : i+1)
#
# @inline function local_field(S::AbstractMatrix{<:Integer}, i::Int, j::Int, J::Real, h::Real)
#     Lx, Ly = size(S)
#     s = S[left(i,Lx), j] + S[right(i,Lx), j]
#     if Ly > 1
#         s += S[i, left(j,Ly)] + S[i, right(j,Ly)]
#     end
#     J * s + h
# end
#
# magnetization(S) = sum(S) / length(S)
#
# # ------------------------------------------
# # 1) Single-spin Metropolis (one attempt)
# # ------------------------------------------
# """
#     metropolis_single_spin_update!(S, p; rng)
#
# Pick a random site and attempt a Metropolis flip.
# (Students can re-implement this.)
# """
# function metropolis_single_spin_update!(S::AbstractMatrix{<:Integer}, p::IsingParams; rng=Random.GLOBAL_RNG)
#     i = rand(rng, 1:p.Lx); j = rand(rng, 1:p.Ly)
#     s = S[i,j]
#     ΔE = 2.0 * s * local_field(S, i, j, p.J, p.h)
#     if ΔE <= 0 || rand(rng) < exp(-p.β * ΔE)
#         S[i,j] = -s
#     end
#     return nothing
# end
#
# # ---------------------------------------------------------
# # 2) Checkerboard Metropolis sweep (two parities)
# # ---------------------------------------------------------
# """
#     metropolis_checkerboard_sweep!(S, p; rng)
#
# Update all black sites then all white sites (parity (i+j)%2).
# (Students can re-implement this.)
# """
# function metropolis_checkerboard_sweep!(S::AbstractMatrix{<:Integer}, p::IsingParams; rng=Random.GLOBAL_RNG)
#     Lx, Ly = size(S)
#     @inbounds for parity in (0, 1)
#         for j in 1:Ly, i in 1:Lx
#             ((i + j) & 1) == parity || continue
#             s = S[i,j]
#             ΔE = 2.0 * s * local_field(S, i, j, p.J, p.h)
#             if ΔE <= 0 || rand(rng) < exp(-p.β * ΔE)
#                 S[i,j] = -s
#             end
#         end
#     end
#     return nothing
# end
#
# # ------------------------
# # Animation with magnetization trace
# # ------------------------
# """
#     record_ising!(S, p; out, frames, sweeps_per_frame, mode)
#
# Make an MP4 showing the spin field (heatmap) and a growing plot of
# magnetization m vs sweep index.
#
# - mode = :checkerboard  -> uses `metropolis_checkerboard_sweep!` per sweep
# - mode = :single        -> ≈ Lx*Ly random single-spin attempts per sweep
# """
# function record_ising!(S::AbstractMatrix{<:Integer}, p::IsingParams;
#                        out="ising.mp4", frames=300, sweeps_per_frame=1,
#                        mode::Symbol=:checkerboard, rng=Random.GLOBAL_RNG)
#
#     GLMakie.activate!()
#
#     fig = Figure(size=(1100, 800))
#
#     # ---- left panel: spins
#     axL = Axis(fig[1,1],
#         title = @sprintf("Ising %dx%d   T=%.3f   J=%.2f   h=%.2f",
#                          p.Lx, p.Ly, 1/p.β, p.J, p.h),
#         xlabel = "x", ylabel = "y", aspect = DataAspect()
#     )
#     # pad y so a single-row (Ly=1) still has some height
#     xlims!(axL, 0.5, p.Lx + 0.5)
#     ypad_hi = p.Ly > 1 ? p.Ly + 0.5 : 1.5
#     ylims!(axL, 0.5, ypad_hi)
#
#     Z = Observable(Float32.(S))
#     hm = heatmap!(axL, Z; colormap=:grays, colorrange=(-1,1), interpolate=false)
#
#
#     # ---- bottom panel: magnetization vs sweeps
#     axM = Axis(fig[2,1], xlabel="Monte Carlo sweeps", ylabel="m")
#     xlims!(axM, 0, frames*sweeps_per_frame)
#     ylims!(axM, -1.05, 1.05)
#     hlines!(axM, [0.0], color=:gray, linestyle=:dash)
#     
#     # magnetization readout (place in relative axis coords: top-left)
#     m_text = @lift(@sprintf("m = %.3f", magnetization($(Z))))
#     text!(axM, 0.02, 0.98; text=m_text, space=:relative,
#           align=(:left, :top), color=:tomato, fontsize=18)
#
#     t_obs = Observable(Float32[0.0])
#     m_obs = Observable(Float32[magnetization(S)])
#     lines!(axM, t_obs, m_obs; linewidth=2)
#
#     # ---- one sweep (same as before)
#     function one_sweep!()
#         if mode === :checkerboard
#             metropolis_checkerboard_sweep!(S, p; rng)
#         elseif mode === :single
#             for _ in 1:(p.Lx * p.Ly)
#                 metropolis_single_spin_update!(S, p; rng)
#             end
#         else
#             error("mode must be :checkerboard or :single")
#         end
#     end
#
#     # ---- record
#     record(fig, out, 1:frames) do f
#         for _ in 1:sweeps_per_frame
#             one_sweep!()
#         end
#         Z[] = Float32.(S)  # update spins
#
#         # append to magnetization buffers (mutate, then notify)
#         push!(t_obs[], Float32(f*sweeps_per_frame))
#         push!(m_obs[], Float32(magnetization(S)))
#         notify(t_obs); notify(m_obs)
#     end
#
#     return nothing
# end
#

# # ------------------------
# # Example runs
# # ------------------------
# if abspath(PROGRAM_FILE) == @__FILE__
#     Random.seed!(42)
#
#     # 2D: above and below Tc (Tc/J ≈ 2.269…)
#     L = 128
#     S0 = init_spins(L, L; p_up=0.5)
#     p_hi = IsingParams(L, L, 1.0, 0.0, 1/3.50)   # T > Tc
#     p_lo = IsingParams(L, L, 1.0, 0.0, 1/2.00)   # T < Tc
#
#     record_ising!(copy(S0), p_hi;
#                   out="ising2d_T3p50_checkerboard.mp4",
#                   frames=400, sweeps_per_frame=1, mode=:checkerboard)
#
#     record_ising!(copy(S0), p_lo;
#                   out="ising2d_T2p00_checkerboard.mp4",
#                   frames=400, sweeps_per_frame=1, mode=:checkerboard)
#
#     # 1D chain demo (Ly=1)
#     L1 = 256
#     S1 = init_spins(L1, 1; p_up=0.5)
#     p1 = IsingParams(L1, 1, 1.0, 0.0, 1/0.50)
#     record_ising!(S1, p1;
#                   out="ising1d_T0p50_single.mp4",
#                   frames=400, sweeps_per_frame=1, mode=:single)
# end

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
    # energy change
    ΔE = 2.0 * s * local_energy(S, i, j, p)
    # update probability
    return rand(rng) < exp(-p.β * ΔE)
end

function update_condition(u_alg::GlauberUpdate, s, S, i, j, p, rng)
    # WRITE YOUR CODE HERE
    # energy change
    ΔE = 2.0 * s * local_energy(S, i, j, p)
    # update probability
    return rand(rng) < 1/(1+exp(p.β * ΔE))
end

