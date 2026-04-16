"""
NLS_bigfloat Module

by Abrari Noor Hasmi, 2023
This module provides high-precision numerical methods for solving the 
Nonlinear Schrödinger (NLS) equation using arbitrary precision arithmetic. 
It includes functions for generating Akhmediev Breather solutions and 
implementing Taylor series time-stepping methods with extended precision.

The module is designed to study numerical precision effects in rogue wave 
simulations and provides tools for "Clean Numerical Simulation" as 
described in the dissertation.

Key features:
- Arbitrary precision arithmetic support via BigFloat
- Analytical Akhmediev Breather solutions for validation
- High-order Taylor series time integration methods
- Thread-safe parallel computation support
- FFT-based spectral differentiation
"""
module NLS_bigfloat

using DelimitedFiles,GenericFFT, Base.Threads

export AB_ex,calculate_params,Taylor_n,Taylor_thread_n ,calculate_grid


function AB_ex(t::Real,x::Real, δ::Real, Ω::Real,a::Real)
    """
    AB_ex(t::Real, x::Real, δ::Real, Ω::Real, a::Real) -> Complex
    
    The Akhmediev Breather represents a spatially periodic, temporally 
    localized solution that models rogue wave formation through modulation 
    instability. It is an exact solution of the focusing NLS equation and 
    serves as a benchmark for numerical methods.

    # Arguments
    - `t::Real`: Time variable
    - `x::Real`: Spatial variable  
    - `δ::Real`: Growth rate parameter δ = √(2a(1-2a))
    - `Ω::Real`: Modulation wavenumber Ω = 2√(1-2a)
    - `a::Real`: Modulation parameter (0 < a < 1/2)

    # Returns
    - `Complex`: Akhmediev Breather amplitude ψ(x,t)

    # Mathematical Formula
    The function computes the Akhmediev Breather solution:
    ```
    ψ(x,t) = [1 + ((1-4a)cosh(δx) + √(2a)cos(Ωt) + iδsinh(δx)) / 
              (√(2a)cos(Ωt) - cosh(δx))] e^(it)
    ```
    """
    # Compute the modulation envelope components
    numerator = (1-4*a)*cosh(δ*x) + sqrt(2*a)*cos(Ω*t) + im*δ*sinh(δ*x)
    denominator = sqrt(2*a)*cos(Ω*t) - cosh(δ*x)
    
    # Return full Akhmediev Breather with carrier wave phase e^(it)
    # Note: Fixed phase factor - should be exp(im*t) not exp(im*x)
    return (numerator/denominator + 1) * exp(im*t);
end

    
function calculate_params(a::Real)
    """
    calculate_params(a::Real) -> (Complex, Real, Real)

    This function computes the key parameters that characterize the 
    Akhmediev Breather solution based on the fundamental modulation 
    parameter `a`.

    # Arguments
    - `a::Real`: Modulation parameter (0 < a < 1/2)

    # Returns
    - `λ::Complex`: Complex eigenvalue λ = i√(2a)
    - `T::Real`: Temporal period T = π/√(1-2a)  
    - `Ω::Real`: Modulation frequency Ω = 2π/T = 2√(1-2a)
    """
    # Calculate complex eigenvalue for Akhmediev Breather
    λ = im * sqrt(2 * a)
    
    # Calculate temporal period (related to modulation instability)
    T = π/sqrt(1 - imag(λ)^2)  # = π/√(1-2a)
    
    # Calculate modulation frequency
    Ω = 2π/T  # = 2√(1-2a)
    
    return λ, T, Ω
end


function calculate_grid(tᵣ, L::Real, dt::Real, Nₓ::Int64; n_periods = 1)  
    """
    calculate_grid(tᵣ, L::Real, dt::Real, Nₓ::Int64; n_periods=1) 
                   -> (Vector, Vector, Vector, Int)

    This function creates spatial and temporal grids for numerical 
    simulation of the NLS equation, with support for arbitrary precision 
    arithmetic. It handles both range-based and linspace-based grid 
    generation depending on the arithmetic type.

    # Arguments
    - `tᵣ`: Time range (tuple, range, or similar iterable with 
            .first and .second)
    - `L::Real`: Fundamental spatial domain length
    - `dt::Real`: Time step size
    - `Nₓ::Int64`: Number of spatial grid points
    - `n_periods=1`: Number of breather periods to include in spatial domain

    # Returns
    - `x::Vector`: Spatial grid points (centered at origin)
    - `k::Vector`: Fourier wavenumber grid (for spectral methods)
    - `t::Vector`: Time grid points
    - `Nₜ::Int`: Number of time steps

    # Grid Construction
    - Spatial grid: x ∈ [-L/2, L/2) with spacing dx = L/Nₓ
    - Wavenumber grid: k = 2π/L [-Nₓ/2, Nₓ/2-1] (FFT ordering)
    - Time grid: t ∈ [tᵣ.first, tᵣ.second] with spacing dt

    # Notes
    - Uses periodic boundary conditions in space
    - Supports arbitrary precision types through generic programming
    - Fallback to LinRange if range arithmetic fails (common with BigFloat)
    - Grid points are pre-allocated with correct precision type
    """
    @info "Initializing grid with $n_periods period(s) and dt = $dt, Nₓ = $Nₓ."
    
    # Scale spatial domain to include multiple breather periods
    L = n_periods * L
    @info "Longitudinal range is [$(tᵣ.first), $(tᵣ.second)], transverse range is [$(-L/2), $(L/2))"
    
    # Calculate grid spacings
    dx = L / Nₓ
    Nₜ = Int(floor((tᵣ.second - tᵣ.first) / dt)) + 1
    
    # Pre-allocate arrays with correct precision type
    t = Array{typeof(dt)}(undef, Nₜ)
    x = Array{typeof(dt)}(undef, Nₓ)
    k = similar(x)  # Same type as x
    
    try 
        # Try range-based construction (works for standard Float64)
        t .= collect(tᵣ.first:dt:tᵣ.second)
        x .= dx * collect((-Nₓ/2:Nₓ/2-1))
        k .= 2π/L * collect((-Nₓ/2:Nₓ/2-1))
    catch err 
        # Fallback to LinRange (necessary for BigFloat and other extended types)
        @info "Range construction failed, using LinRange fallback"
        t .= collect(LinRange(tᵣ.first, tᵣ.second, Nₜ))
        x .= dx * collect(LinRange(-Nₓ/2, Nₓ/2-1, Nₓ))
        k .= 2π/L * collect(LinRange(-Nₓ/2, Nₓ/2-1, Nₓ))
    end
    
    @info "Done computing x, k, t grids"
    return x, k, t, Nₜ
end

function Taylor_thread_n(u0::Vector{Complex{TT}}, Δt, ω, p, M) where TT<: Real    
    """
    Taylor_thread_n(u0::Vector{Complex{TT}}, Δt, ω, p, M) where TT<:Real 
                     -> Vector{Complex{TT}}
    
    High-order Taylor series time integration for the NLS equation 
    (threaded parallel version). This is the parallelized version that uses 
    multi-threading to accelerate the computation of nonlinear terms. 
    
    # Arguments
    - `u0::Vector{Complex{TT}}`: Initial condition at current time step
    - `Δt`: Time step size
    - `ω`: Frequency array (for linear dispersion term)  
    - `p`: FFT plan or similar operator for spectral differentiation
    - `M`: Maximum Taylor series order

    # Returns
    - `Vector{Complex{TT}}`: Solution advanced by time step Δt

    # Performance Notes
    - Uses Julia's @threads macro for shared-memory parallelization
    - Most effective for large spatial grids (N >> number of threads)
    - FFT operations remain serial (handled by FFTW's internal threading)
    - Nonlinear term computation is parallelized

    # Threading Considerations
    - Set JULIA_NUM_THREADS environment variable before starting Julia
    - Optimal thread count depends on system architecture and problem size
    - Memory bandwidth can become limiting factor for very fine grids
    """
    N = length(u0)
    
    # Storage for Taylor coefficients
    up = Array{Complex{TT},2}(undef, N, M+1)
    p_f = Array{Complex{TT}}(undef, N)
    
    # Initialize
    up[:,1] .= u0
    u = copy(u0)
    
    # Pre-compute linear operator coefficients
    ws2 = -ω.^2 / convert(Complex{TT}, 2)
    
    # Compute Taylor coefficients iteratively
    for m in 1:M
        p_f .= ifft(p * (@view(up[:,m])) .* ws2) # Linear part (serial - FFT operations)
        dtM = convert(Complex{TT}, Δt)^m # Power of time step
        
        # Compute nonlinear contributions in parallel
        @threads for i in 1:N
            # Each thread computes nonlinear term for its assigned spatial points
            nl_term = calc_nl_term(up[i,1:m], m)
            up[i,m+1] = im / convert(TT, m) * (p_f[i] + nl_term)
        end
        
        # Add contribution to solution
        u += @view(up[:,m+1]) * dtM
    end
    
    return u
end

function calc_nl_term(up_t::Vector{Complex{TT}}, m::Int) where TT<:Real
    
    """
    calc_nl_term(up_t::Vector{Complex{TT}}, m::Int) where TT<:Real 
                  -> Complex{TT}
    
    This function computes the mth Taylor coefficient of the nonlinear 
    term |u|²u using previously computed Taylor coefficients. This is a 
    core component of the high-order Taylor series time integration method.

    # Arguments
    - `up_t::Vector{Complex{TT}}`: Taylor coefficients u₀, u₁, ..., uₘ 
                                   at a single spatial point
    - `m::Int`: Current Taylor series order being computed

    # Returns
    - `Complex{TT}`: The mth Taylor coefficient of |u|²u

    """
    # Initialize accumulator with proper precision
    nl_term = zero(Complex{TT})
    for j in 1:m+1
        for n in 1:m+1-j
            # Note: up_t[j] corresponds to (j-1)th Taylor coefficient
            nl_term += conj(up_t[j]) * up_t[n] * up_t[m+2-j-n]
        end
    end
    
    return nl_term
end

end # module NLS_bigfloat
