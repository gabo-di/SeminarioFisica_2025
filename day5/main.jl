using QuantumToolbox
using CairoMakie
using Printf

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
    # WRITE YOUR CODE HERE
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
    # WRITE YOUR CODE HERE
    ωc = p.ωc
    ωa = p.ωa
    N = p.N # N: number of atoms
    M = p.M # M: cavity Hilbert space truncation 

    j = N / 2

    Jz = (jmat(N/2, :z) ⊗ qeye(M))
    a = (qeye(N+1) ⊗ destroy(M))
    Jp = jmat(j, :+) ⊗ qeye(M)
    Jm = jmat(j, :-) ⊗ qeye(M) 

    H0 = ωc * a' * a + ωa * Jz
    H1 = 1/ sqrt(N) * (Jp*a + Jm*a') 
    H2 = 1/ sqrt(N) * (Jp*a' + Jm*a) 

    return H0, H1, H2, a, Jz
end

function simulation_Dicke(p)
    N = p.N # N: number of atoms
    M = p.M # M: cavity Hilbert space truncation 

    H0, H1, H2, a, Jz = Dicke(p)

    gs = 0.0:0.05:1.0
    ψGs = QuantumObject[]
    for g in gs 
        H = H0 + g*(H1 + H2)
        vals, vecs = eigenstates(H)
        push!(ψGs, vecs[1])
    end

    nvec = expect(a'*a, ψGs)
    Jzvec = expect(Jz, ψGs)

    # the indices in coupling strength list (gs)
    # to display wigner and fock distribution
    cases = 1:5:21

    function plot_atoms_data()
        fig_1 = Figure(size = (900,650))
        for (hpos, idx) in enumerate(cases)
            g = gs[idx] # coupling strength
            ρcav = ptrace(ψGs[idx], 1) # atoms reduced state
            
            # plot wigner
            _, ax, hm = plot_wigner(ρcav, location = fig_1[1,hpos])
            ax.title = "g = $g"
            ax.aspect = 1
            
            # plot fock distribution
            _, ax2 = plot_fock_distribution(ρcav, location = fig_1[2,hpos])

            ax2.xticks = (0:1:(size(ρcav,1)-1), string.(collect(range(N/2,-N/2,step=-1))))
            ax2.xlabel = L"\hat{J}_{z}"
            
            if hpos != 1
                hideydecorations!(ax, ticks=false)
                hideydecorations!(ax2, ticks=false)
                if hpos == 5 # Add colorbar with the last returned heatmap (_hm) 
                    Colorbar(fig_1[1,6], hm)
                end
            end    
        end

        # plot average Jz 
        ax3 = Axis(fig_1[3,1:6], height=200, xlabel=L"g", ylabel=L"\langle \hat{J}_{z} \rangle")
        xlims!(ax3, -0.02, 1.02)
        lines!(ax3, gs, real(Jzvec), color=:teal)
        ax3.xlabelsize, ax3.ylabelsize = 20, 20
        vlines!(ax3, gs[cases], color=:orange, linestyle = :dash, linewidth = 4)
        return fig_1
    end
    fig_1 = plot_atoms_data()


    function plot_cavity_data()
        fig_2 = Figure(size = (900,650))
        for (hpos, idx) in enumerate(cases)
            g = gs[idx] # coupling strength
            ρcav = ptrace(ψGs[idx], 2) # cavity reduced state
            
            # plot wigner
            _, ax, hm = plot_wigner(ρcav, location = fig_2[1,hpos])
            ax.title = "g = $g"
            ax.aspect = 1
            
            # plot fock distribution
            _, ax2 = plot_fock_distribution(ρcav, location = fig_2[2,hpos])

            ax2.xticks = (0:2:size(ρcav,1)-1, string.(0:2:size(ρcav,1)-1))
            
            if hpos != 1
                hideydecorations!(ax, ticks=false)
                hideydecorations!(ax2, ticks=false)
                if hpos == 5 # Add colorbar with the last returned heatmap (_hm) 
                    Colorbar(fig_2[1,6], hm)
                end
            end    
        end

        # plot average photon number with respect to coupling strength
        ax3 = Axis(fig_2[3,1:6], height=200, xlabel=L"g", ylabel=L"\langle \hat{n} \rangle")
        xlims!(ax3, -0.02, 1.02)
        lines!(ax3, gs, real(nvec), color=:teal)
        ax3.xlabelsize, ax3.ylabelsize = 20, 20
        vlines!(ax3, gs[cases], color=:orange, linestyle = :dash, linewidth = 4)
        return fig_2
    end
    fig_2 = plot_cavity_data()

    
    function plot_expectations_evol()
        # initial state: all spins down ⊗ vacuum
        ψ_init = spin_state(N/2, -N/2) ⊗ fock(M,0) 

        g2 = gs[cases[end]]
        g1 = gs[cases[2]]
        H_g2 = H0 + g2*(H1 + H2)
        H_g1 = H0 + g1*(H1 + H2)

        tlist = range(0, 50; length=400)
        res_g2 = sesolve(H_g2, ψ_init, tlist; e_ops=[a'*a, Jz])
        res_g1 = sesolve(H_g1, ψ_init, tlist; e_ops=[a'*a, Jz])

        n_of_t_2  = real.(res_g2.expect[1,:])
        jz_of_t_2 = real.(res_g2.expect[2,:])
        n_of_t_1  = real.(res_g1.expect[1,:])
        jz_of_t_1 = real.(res_g1.expect[2,:]) 

        fig_3 = Figure(size=(900, 350))
        axn = Axis(
         fig_3[1,1],
         xlabel = "time",
         ylabel = L"\langle \hat{n} \rangle"
        )
        axJz = Axis(
         fig_3[1,2],
         xlabel = "time",
         ylabel = L"\langle \hat{J}_{z} \rangle"
        )

        xlims!(axn, 0, max(tlist...))
        xlims!(axJz, 0, max(tlist...))

        lines!(axn, tlist, n_of_t_1, label = @sprintf("g = %.3f",g1))
        lines!(axn, tlist, n_of_t_2, label = @sprintf("g = %.3f",g2))

        lines!(axJz, tlist, jz_of_t_1, label = @sprintf("g = %.3f",g1))
        lines!(axJz, tlist, jz_of_t_2, label = @sprintf("g = %.3f",g2))

        axislegend(axn; position = :rt, labelsize = 15)
        axislegend(axJz; position = :rt, labelsize = 15)
        return fig_3
    end
    fig_3 = plot_expectations_evol()

    return fig_1, fig_2, fig_3
end

function main_Dicke()
    p = (
        M = 10, # Fock space truncated dimension
        N = 4, # number of atoms
        ωa = 1, # atomic frequency
        ωc = 1, # light frequency
    )

    simulation_Dicke(p)
end
