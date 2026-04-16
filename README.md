# HighOrderTaylorNLS

High-order Taylor series solver for the nonlinear Schrödinger equation with support for arbitrary precision.

## Implementation (Julia + BigFloat)

The solver is implemented in Julia and uses `BigFloat` arithmetic to reduce round-off error beyond standard double precision. In practice, the solution and intermediate Taylor coefficients are stored as `Complex{BigFloat}`, and you can control the working precision globally with:

- `setprecision(BigFloat, bits)` (e.g. 113 bits $\approx$ quadruple precision)

Spatial derivatives are computed spectrally using FFTs in arbitrary precision via `GenericFFT` (so FFT-based differentiation works with `BigFloat`). For efficiency, FFT plans (e.g. `plan_fft`) are precomputed and reused, and the nonlinear convolution sum in the Taylor recursion can be parallelized with Julia threads.

## Method (Taylor-in-time, spectral-in-space)

We consider the (focusing) nonlinear Schrödinger equation

$$
\frac{\partial \psi}{\partial t}=i\left(\frac{1}{2}\frac{\partial^2 \psi}{\partial\xi^2}+\left|\psi\right|^{2}\psi\right),
$$

and advance the solution $\psi(\xi,t)$ over one time step $\Delta t$ by a truncated Taylor expansion in time,

$$
\psi(\xi,t+\Delta t)=\sum_{m=0}^{M}\psi^{[m]}(\xi,t)(\Delta t)^{m},
$$

where $M$ is the Taylor order and

$$
\psi^{[m]}(\xi,t)=\frac{1}{m!}\frac{\partial^{m}\psi(\xi,t)}{\partial t^{m}}.
$$

Substituting this expansion into the (focusing) nonlinear Schrödinger equation and matching powers of $(\Delta t)^m$ yields the explicit recursion for the Taylor coefficients:

$$
\psi^{[m+1]}=\frac{i}{m+1}\left(\frac{1}{2}\psi_{\xi\xi}^{[m]}+\sum_{j=0}^{m}\sum_{n=0}^{m-j}\psi^{[j]*}\,\psi^{[n]}\,\psi^{[m-j-n]}\right),
$$

where $\psi^*$ denotes complex conjugation. The spatial second derivative is computed spectrally in Fourier space using the FFT (for periodic domains), e.g.

$$
\psi_{\xi\xi}^{[m]} = \mathcal{F}^{-1}\left[-k^2\,\mathcal{F}\left(\psi^{[m]}\right)\right].
$$

The scheme is explicit because $\psi^{[m+1]}$ depends only on previously computed orders $\{\psi^{[j]}\}_{j=0}^{m}$. The local temporal truncation error is $O\big((\Delta t)^{M+1}\big)$ while retaining spectral accuracy in space.

## References

This code is inspired by the Clean Numerical Simulation (CNS) strategy (high-order time integration combined with multiple-precision arithmetic) as presented in:

- T. Hu and S. Liao, “On the risks of using double precision in numerical simulations of spatio-temporal chaos,” *Journal of Computational Physics*, **418** (2020), 109629. https://doi.org/10.1016/j.jcp.2020.109629

Related work:

- A. Hasmi and H. Susanto, “Reliability of Numerical Rogue Wave Simulations,” *Wave Motion*, in revision.

BibTeX:

```bibtex
@article{hu_risks_2020,
  title   = {On the risks of using double precision in numerical simulations of spatio-temporal chaos},
  author  = {Hu, Tianli and Liao, Shijun},
  journal = {Journal of Computational Physics},
  volume  = {418},
  pages   = {109629},
  year    = {2020},
  doi     = {10.1016/j.jcp.2020.109629},
  url     = {https://www.sciencedirect.com/science/article/pii/S0021999120304034}
}

@unpublished{hasmi_susanto_reliability_rogue_waves,
  title  = {Reliability of Numerical Rogue Wave Simulations},
  author = {Hasmi, Abrari Noor and Susanto, Hadi},
  note   = {Manuscript in revision for Wave Motion}
}
```
