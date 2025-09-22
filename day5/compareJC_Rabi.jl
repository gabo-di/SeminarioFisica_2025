using QuantumToolbox
using CairoMakie
using Printf

function JaynesCumming(p, Ω)
    # WRITE YOUR CODE HERE
    # Ha = ωa / 2 * σz
    # Hc = ωc * a' * a 
    # Hint = Ω * (σ * a' + σ' * a)
    #
    # Htot  = Ha + Hc + Hint
    return Htot, a, σ
end

function simulation_EnJaynesCumming(p)
    M = p.M
    ωa = p.ωa
    ωc = p.ωc

    gs = range(p.gmin, p.gmax; length=p.gnum)

    En = []

    for g in gs
        Htot, a, σ = JaynesCumming(p, g)
        vals, _ = eigenstates(Htot)
        vals = sort(real(vals))
        push!(En, vals)
    end
    En = hcat(En...)
    return En
end

function compare_JC_Rabi()
    p = (
        M = 10,
        ωa = 0.5,
        ωc =  2,
        gmin = 0.0,
        gmax = 6.0,
        gnum = 100,
    )
    simulation_Compare_JC_Rabi(p)
end

function simulation_Compare_JC_Rabi(p)
    En_JC = simulation_EnJaynesCumming(p)
    En_Rabi = simulation_Rabi(p)
    fig = Figure()
    ax1 = Axis(
        fig[1, 1],
        xlabel = L"g / \omega_a", 
        ylabel = L"E_n / \omega_a", 
        title = "Jaynes-Cumming Model",
        xlabelsize = 15, 
        ylabelsize = 15,
    )
    ax2 = Axis(
        fig[1, 2],
        xlabel = L"g / \omega_a", 
        ylabel = L"E_n / \omega_a", 
        title = "Rabi Model",
        xlabelsize = 15, 
        ylabelsize = 15,
    )
    gs = range(p.gmin, p.gmax; length=p.gnum)
    for n in 1:size(En_JC,1)
        lines!(ax1, gs/p.ωc, En_JC[n, :]/p.ωc, color = :teal)
        lines!(ax2, gs/p.ωc, En_Rabi[n, :]/p.ωc, color = :orange)
    end
    return fig
end

function Rabi(p, Ω)
    M = p.M
    ωa = p.ωa
    ωc = p.ωc

    σz = sigmaz() ⊗ qeye(M)
    a  = qeye(2)  ⊗ destroy(M)
    σ = sigmam() ⊗ qeye(M) # σ₋

    Ha = ωa / 2 * σz
    Hc = ωc * a' * a
    Hint = Ω * (σ + σ') * (a + a')

    Htot  = Ha + Hc + Hint
    return Htot, a, σ
end

function simulation_Rabi(p)
    M = p.M
    ωa = p.ωa
    ωc = p.ωc

    gs = range(p.gmin, p.gmax; length=p.gnum)

    En = []

    for g in gs
        Htot, a, σx = Rabi(p, g)
        vals, _ = eigenstates(Htot)
        vals = sort(real(vals))
        push!(En, vals)
    end
    En = hcat(En...)

    return En
end
