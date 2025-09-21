using QuantumToolbox
using CairoMakie

function main_JaynesCumming()
    p = (
        M = 2, # Fock space truncated dimension
        ωa = 1, # atomic frequency
        ωc = 1, # light frequency
        Ω  = 0.05
    )

    simulation_JaynesCumming(p)
end

function JaynesCumming(p)
    M = p.M
    ωa = p.ωa
    ωc = p.ωc
    Ω = p.Ω

    σz = sigmaz() ⊗ qeye(M) 
    a  = qeye(2)  ⊗ destroy(M)  
    σ  = sigmam() ⊗ qeye(M) # σ₋ 

    Ha = ωa / 2 * σz
    Hc = ωc * a' * a 
    Hint = Ω * (σ * a' + σ' * a)

    Htot  = Ha + Hc + Hint
    return Htot, a, σ
end

function initialstate_JC(p)
    M = p.M
    e_ket = basis(2,0) 
    return e_ket ⊗ fock(M, 0), e_ket
end

function simulation_JaynesCumming(p)
    M = p.M
    ωa = p.ωa

    Htot, a, σ = JaynesCumming(p)
    ψ0, e_ket = initialstate_JC(p)


    tlist = 0:0.5:100 ./ ωa # a list of time points of interest

    # define a list of operators whose expectation value dynamics exhibit Rabi oscillation
    eop_ls = [
        a' * a,                      # number operator of cavity
        (e_ket * e_ket') ⊗ qeye(M), # excited state population in atom
    ]

    sol = sesolve(Htot , ψ0, tlist; e_ops = eop_ls)


    n = real.(sol.expect[1, :])
    e = real.(sol.expect[2, :])
    fig_se = Figure()
    ax_se = Axis(
        fig_se[1, 1],
        xlabel = L"time $[1/\omega_a]$", 
        ylabel = "expectation value", 
        xlabelsize = 15, 
        ylabelsize = 15,
    )
    xlims!(ax_se, 0, max(tlist...))
    lines!(ax_se, tlist, n, label = L"$\langle a^\dagger a \rangle$")
    lines!(ax_se, tlist, e, label = L"$P_e$")
    axislegend(ax_se; position = :rt, labelsize = 15)
    return fig_se
end

function main_DissipativeJaynesCumming()
    p = (
        M = 2, # Fock space truncated dimension
        ωa = 1,
        ωc = 1,
        Ω  = 0.05,
        γ = 4e-3,
        κ = 7e-3,
        KT = 0.3 # thermal field at finite temperature
    )

    simulation_DissipativeJaynesCumming(p)
end

function simulation_DissipativeJaynesCumming(p)
    M = p.M
    ωa = p.ωa
    ωc = p.ωc
    γ = p.γ
    κ = p.κ
    KT = p.KT

    Htot, a, σ = JaynesCumming(p)
    ψ0, e_ket = initialstate_JC(p)

    cop_ls(_γ, _κ, _KT) = (
        √(_κ * n_thermal(ωc, _KT)) * a', 
        √(_κ * (1 + n_thermal(ωc, _KT))) * a, 
        √(_γ * n_thermal(ωa, _KT)) * σ', 
        √(_γ * (1 + n_thermal(ωa, _KT))) * σ, 
    )

    tlist = 0:0.5:400 ./ ωa # a list of time points of interest

    # define a list of operators whose expectation value dynamics exhibit Rabi oscillation
    eop_ls = [
        a' * a,                      # number operator of cavity
        (e_ket * e_ket') ⊗ qeye(M), # excited state population in atom
    ]

    sol_me  = mesolve(Htot,  ψ0, tlist, cop_ls(γ, κ, KT), e_ops = eop_ls)

    n_me = real.(sol_me.expect[1, :])
    e_me = real.(sol_me.expect[2, :])
    fig_me = Figure()
    ax_me = Axis(
        fig_me[1, 1],
        xlabel = L"time $[1/\omega_a]$", 
        ylabel = "expectation value", 
        xlabelsize = 15, 
        ylabelsize = 15,
    )
    lines!(ax_me, tlist, n_me, label = L"\langle a^\dagger a \rangle")
    lines!(ax_me, tlist, e_me, label = L"$P_e$")
    axislegend(ax_me; position = :rt, labelsize = 15)
    return fig_me
end

function Dicke(p)
    ωc = p.ωc
    ωa = p.ωa
    N = p.N # N: number of atoms
    M = p.M # M: cavity Hilbert space truncation 
    λ = p.λ # interaction of co-rotating term 
    λ_ = p.λ_ # interaction of counter-rotating term

    j = N / 2

    Jz = (jmat(N/2, :z) ⊗ qeye(M))
    a = (qeye(N+1) ⊗ destroy(M))
    Jp = jmat(j, :+) ⊗ qeye(M)
    Jm = jmat(j, :-) ⊗ qeye(M) 

    H0 = ωc * a' * a + ωa * Jz
    H1 = λ/ sqrt(N) * (Jp*a + Jm*a') 
    H2 = λ_/ sqrt(N) * (Jp*a' + Jm*a) 

    return (H0 + H1 + H2)
end

function simulate_Dicke(p)

    ψGs = map(gs) do g
        H = H0(g)
        vals, vecs = eigenstates(H)
        vecs[1]
    end

    nvec = expect(a0'*a0, ψGs)
    Jzvec = expect(Jz0, ψGs)
end
