# ============================================================================
# High-Precision NLS Simulation using BigFloat Arithmetic
# ============================================================================

using Pkg
Pkg.activate("/home/kunet.ae/100060615/RCfiles/NLS_project2/NLS_bigfloat")

# Set high precision arithmetic (113 bits ≈ quadruple precision)
setprecision(BigFloat, 113)

# Import required packages
using DelimitedFiles, GenericFFT, LinearAlgebra
using NLS_bigfloat
using LoggingExtras, Dates

# Logging Setup

const date_format = "yyyy-mm-dd HH:MM:SS"

logger = FormatLogger("out.log"; append=true) do io, args
    println(io, args._module, " | ", "[", args.level, ": ", 
            Dates.format(now(), date_format), "] ", args.message)
end

global_logger(logger)
@info "Computing with BigFloat with $(precision(BigFloat)) bits and machine epsilon: $(eps(BigFloat))"

# ============================================================================
# Main Simulation Function
# ============================================================================
function main()
    # Simulation Parameters
    a = BigFloat(1) / BigFloat(3)          # Modulation parameter
    N = 128                                # Number of spatial grid points
    M = 4                                  # Taylor series order in time
    tᵣ = 0 => 50                          # Time range
    t0 = BigFloat(-10)                     # Initial time for Akhmediev Breather
    dt = BigFloat(1) / BigFloat(100000)    # Time step size
    saveat = 400                           # Save every nth time step
    
 
    # Calculate Akhmediev Breather Parameters
    λ, L, Ω = calculate_params(a)
    x, k, t, Nₜ = calculate_grid(tᵣ, L, dt, N, n_periods = 1)
    δ = imag(λ) * Ω                       # Growth rate parameter

    # Initialize Solution Arrays
    ψ₀ = Array{Complex{BigFloat}}(undef, N)    # Current solution
    ψᵢ = Array{Complex{BigFloat}}(undef, N)    # Next solution
    
    # Generate initial condition using analytical Akhmediev Breather
    # ψ₀ .= readdlm("init_sim_part2.csv", ',', Complex{BigFloat})  # Alternative: load from file
    ψ₀ .= AB_ex.(x, t0, δ, Ω, a)
    
    @info "Number of threads: $(Threads.nthreads())"

    # ========================================================================
    # Main Time Integration Loop
    # ========================================================================
    uall = Array{Complex{BigFloat}, 2}(undef, N, cld(Nₜ, saveat))
    uall[:, 1] = ψ₀ 
    p = plan_fft(ψ₀) # Pre-compute FFT plan for efficiency
    thres = 5 # Progress tracking
    
    for i in 2:Nₜ 
        if i == 2
            @info "Starting Taylor integration with order $M in time, $(Nₜ-1) iterations total"
        end       
        # Progress reporting
        if thres < (i / Nₜ * 100)
            @info "Progress: $(thres)%"
            thres += 5
        end      
        # Advance solution by one time step using high-order Taylor method
        ψᵢ .= Taylor_thread_n(ψ₀, dt, k, p, M)         
        # Check for numerical instabilities
        if any(isnan, ψᵢ)
            @error "Computation failed at iteration $(i-1)"
            break
        end        
        # Save solution at specified intervals
        if mod(i, saveat) == 1
            uall[:, cld(i, saveat)] = ψᵢ 
        end       
        ψ₀ = ψᵢ # Update solution for next iteration
    end

    # Save Results
    if any(isnan, uall)
        @error "Computation failed: output contains NaN values"
    else
        dNsave = 1
        @info "Saving data with dt_save = $(saveat * dt)"
        writedlm("NLS_8order_128x_part1.csv", uall[:, 1:dNsave:end], ',')
        @info "Data saved successfully!"
    end
end

# ============================================================================
# Execute Main Function
# ============================================================================
main()