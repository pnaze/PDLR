import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

R_VALUES = [0.01, 0.1, 1, 10, 100]
EPS_VALUES = np.logspace(-5, -1, 5)

EPS_FIG1 = 1e-5
N_TRAJECTORIES = 60000
N_STEPS = 1500

rng = np.random.default_rng(12345)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "pdf.fonttype": 42
})


# ============================================================
# LINEAR RESPONSE THEORY
# ============================================================

def wirr_dimensionless(R, eps):
    """Dimensionless irreversible work: beta * W_irr."""

    return (
        eps**2
        * (R + np.expm1(-R))
        / (2 * R**2)
    )


def shift_lrt(R, eps):
    """Dissipative translation in the rescaled work variable."""

    return wirr_dimensionless(R, eps) / eps


def shift_exact(eps):
    """Exact free-energy translation."""

    return 0.5 - np.log1p(eps) / (2 * eps)


# ============================================================
# LANGEVIN SIMULATION
# ============================================================

def simulate_work(R, eps,
                  n_trajectories=N_TRAJECTORIES,
                  n_steps=N_STEPS):

    """
    Dimensionless Langevin equation:

    dxi = -(R/2)(1 + eps*v)*xi*dv + sqrt(R)*dB

    Rescaled irreversible work:

    y = (1/2)*integral xi(v)^2 dv
        - log(1+eps)/(2*eps).
    """

    dv = 1.0 / n_steps

    # Initial canonical distribution
    xi = rng.normal(size=n_trajectories)

    U = np.zeros(n_trajectories)

    for j in range(n_steps):

        v_mid = (j + 0.5) * dv

        stiffness = 1.0 + eps * v_mid
        k = R * stiffness

        # Gaussian Ornstein-Uhlenbeck transition
        a = np.exp(-k * dv / 2)

        variance = (
            -np.expm1(-k * dv) / stiffness
        )

        xi_new = (
            a * xi
            + np.sqrt(variance)
            * rng.normal(size=n_trajectories)
        )

        # Trapezoidal integration of xi^2/2
        U += dv * (xi**2 + xi_new**2) / 4

        xi = xi_new

    y = U - np.log1p(eps) / (2 * eps)

    return y


# ============================================================
# FEYNMAN-KAC CHARACTERISTIC FUNCTION
# ============================================================

def characteristic_function(omega, R, eps, M=128):

    """
    Characteristic function of

    U = (1/2)*integral_0^1 xi(v)^2 dv.

    Calculated through piecewise constant propagation
    of the Feynman-Kac Riccati equation.
    """

    s = -1j * omega
    dv = 1.0 / M

    h = np.zeros_like(s, dtype=complex)
    logJ = np.zeros_like(s, dtype=complex)

    for j in range(M):

        v = (j + 0.5) * dv

        k = R * (1 + eps * (1 - v))
        q = np.sqrt(k**2 + 4 * R * s)

        E = np.exp(-q * dv)

        a = (1 + E) / 2
        b = -np.expm1(-q * dv) / q

        D = a + b * (k / 2 + R * h)

        h_new = (
            a * h + b * (s - k * h / 2)
        ) / D

        logJ += (
            (q - k) * dv / 2
            + np.log(D)
        )

        h = h_new

    return np.exp(
        -0.5 * (logJ + np.log1p(h))
    )


# ============================================================
# FOURIER INVERSION
# ============================================================

def fourier_grid(R, L=16.0):

    N = 2**19 if R <= 0.01 else 2**17

    dy = L / N

    omega = (
        2 * np.pi
        * np.fft.rfftfreq(N, d=dy)
    )

    return omega, N, dy


def window_function(omega):

    """Smooth high-frequency cutoff."""

    r = omega / omega[-1]

    window = np.ones_like(r)

    mask = r > 0.7

    window[mask] = 0.5 * (
        1 + np.cos(
            np.pi * (r[mask] - 0.7) / 0.3
        )
    )

    return window


def inverse_fourier(phi, N, dy):

    density = (
        np.fft.irfft(np.conj(phi), n=N)
        / dy
    )

    return np.fft.fftshift(density)


def work_densities(R, eps, M=128):

    omega, N, dy = fourier_grid(R)

    # Equilibrium characteristic function
    phi_eq = characteristic_function(
        omega, R, 0.0, M=1
    )

    # Exact driven characteristic function
    phi_ex = characteristic_function(
        omega, R, eps, M=M
    )

    # Translation of the work distributions
    phi_lrt = (
        phi_eq
        * np.exp(1j * omega * shift_lrt(R, eps))
    )

    phi_exact = (
        phi_ex
        * np.exp(1j * omega * shift_exact(eps))
    )

    window = window_function(omega)

    p_lrt = inverse_fourier(
        phi_lrt * window, N, dy
    )

    p_exact = inverse_fourier(
        phi_exact * window, N, dy
    )

    delta_p = inverse_fourier(
        (phi_exact - phi_lrt) * window, N, dy
    )

    y = (np.arange(N) - N // 2) * dy - 0.5

    return y, p_exact, p_lrt, delta_p


# ============================================================
# FIGURE 1: WORK PROBABILITY DISTRIBUTIONS
# ============================================================

def reproduce_figure1():

    fig, axes = plt.subplots(
        5, 2,
        figsize=(7.0, 8.0)
    )

    left_limits = [
        (-0.5, 4.0),
        (-0.5, 4.0),
        (-0.5, 3.0),
        (-0.45, 1.2),
        (-0.16, 0.30)
    ]

    right_limits = [
        (-0.499, -0.478),
        (-0.49, -0.30),
        (-0.45, 1.2),
        (-0.45, 1.2),
        (-0.16, 0.30)
    ]

    for i, R in enumerate(R_VALUES):

        print(f"Figure 1: R = {R}")

        y, p_ex, p_lrt, dp = work_densities(
            R, EPS_FIG1
        )

        samples = simulate_work(
            R, EPS_FIG1
        )

        # Left panel
        ax = axes[i, 0]

        ax.hist(
            samples,
            bins=200,
            range=left_limits[i],
            weights=np.full(
                len(samples),
                1.0 / len(samples)
            ) / (
                (left_limits[i][1]
                 - left_limits[i][0]) / 200
            ),
            color="#b5c7d6",
            alpha=0.8,
            label="Simulation"
        )

        ax.plot(
            y, p_ex,
            "k-",
            linewidth=1.0,
            label="Exact"
        )

        ax.plot(
            y, p_lrt,
            "--",
            color="#c23b22",
            linewidth=1.0,
            label="LRT"
        )

        ax.set_xlim(left_limits[i])
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r"$p(y)$")

        ax.text(
            0.97, 0.90,
            rf"$R={R:g}$",
            transform=ax.transAxes,
            ha="right",
            va="top"
        )

        ax.text(
            0.03, 0.90,
            f"({chr(97 + 2*i)})",
            transform=ax.transAxes
        )

        if i == 0:
            ax.legend(
                frameon=False,
                loc="upper right",
                bbox_to_anchor=(1, 0.82)
            )

        # Right panel: amplified discrepancy
        ax = axes[i, 1]

        pmax = np.max(p_ex)

        discrepancy = (
            dp / (EPS_FIG1 * pmax)
        )

        ax.plot(
            y,
            discrepancy,
            color="#245785",
            linewidth=1.0
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.6,
            linestyle="--"
        )

        ax.set_xlim(right_limits[i])

        ax.set_ylabel(
            r"$\Delta p/(\epsilon p_{\max})$"
        )

        ax.text(
            0.96, 0.08,
            f"({chr(98 + 2*i)})",
            transform=ax.transAxes,
            ha="right"
        )

        # Save simulation results
        np.savez_compressed(
            f"figure1_R_{R:g}.npz",
            y=y,
            p_exact=p_ex,
            p_lrt=p_lrt,
            simulation=samples
        )

    axes[-1, 0].set_xlabel(
        r"$y=\beta w/\epsilon$"
    )

    axes[-1, 1].set_xlabel(
        r"$y=\beta w/\epsilon$"
    )

    fig.tight_layout()

    fig.savefig("figure1.pdf")
    fig.savefig("figure1.png", dpi=600)

    plt.close(fig)


# ============================================================
# TOTAL VARIATION DISTANCE
# ============================================================

def total_variation(R, eps, M=128):

    omega, N, dy = fourier_grid(R)

    phi_eq = characteristic_function(
        omega, R, 0.0, M=1
    )

    # Restrict calculation to useful Fourier frequencies
    indices = np.flatnonzero(
        np.abs(phi_eq) > 1e-18
    )

    end = min(len(omega), indices[-1] + 100)

    u = omega[:end]

    phi_ex = characteristic_function(
        u, R, eps, M=M
    )

    delta_phi = np.zeros_like(phi_eq)

    delta_phi[:end] = (
        np.exp(1j * u * shift_exact(eps))
        * phi_ex
        -
        np.exp(1j * u * shift_lrt(R, eps))
        * phi_eq[:end]
    )

    delta_phi *= window_function(omega)

    delta_p = inverse_fourier(
        delta_phi, N, dy
    )

    return 0.5 * dy * np.sum(np.abs(delta_p))


# ============================================================
# FIGURE 2: CONVERGENCE WITH DRIVING AMPLITUDE
# ============================================================

def reproduce_figure2():

    results = np.zeros(
        (len(R_VALUES), len(EPS_VALUES))
    )

    for i, R in enumerate(R_VALUES):

        for j, eps in enumerate(EPS_VALUES):

            results[i, j] = total_variation(
                R, eps
            )

            print(
                f"R = {R:6g}, "
                f"epsilon = {eps:.2e}, "
                f"D_TV = {results[i,j]:.8e}"
            )

    # Export numerical data
    header = "epsilon," + ",".join(
        f"R={R:g}" for R in R_VALUES
    )

    np.savetxt(
        "tv_convergence.csv",
        np.column_stack(
            (EPS_VALUES, results.T)
        ),
        delimiter=",",
        header=header,
        comments=""
    )

    # Publication-style figure
    fig, ax = plt.subplots(
        figsize=(3.4, 2.75)
    )

    colors = [
        "#245785",
        "#b84435",
        "#28876e",
        "#73529c",
        "#bc8600"
    ]

    markers = ["o", "s", "^", "D", "v"]

    for i, R in enumerate(R_VALUES):

        ax.loglog(
            EPS_VALUES,
            results[i],
            color=colors[i],
            marker=markers[i],
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.0,
            label=rf"$R={R:g}$"
        )

    # Reference linear scaling
    ax.loglog(
        EPS_VALUES,
        0.1 * EPS_VALUES,
        "k--",
        linewidth=0.8,
        label=r"$\propto\epsilon$"
    )

    ax.set_xlabel(
        r"$\epsilon=\delta\lambda/\lambda_0$"
    )

    ax.set_ylabel(r"$D_{\mathrm{TV}}$")

    ax.set_xlim(6e-6, 0.16)
    ax.set_ylim(1e-7, 0.1)

    ax.tick_params(
        which="both",
        top=True,
        right=True
    )

    ax.legend(
        loc="upper left",
        frameon=False
    )

    fig.subplots_adjust(
        left=0.19,
        right=0.975,
        bottom=0.19,
        top=0.96
    )

    fig.savefig("tv_convergence.pdf")
    fig.savefig("tv_convergence.png", dpi=600)

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    reproduce_figure1()
    reproduce_figure2()