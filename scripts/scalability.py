import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Data
data = {
    "5 cycles\n(106k DoF)": {
        "N": np.array([1, 2, 4, 6, 8, 10, 14, 20, 28]),
        "T": np.array([355.352, 312.358, 165.7, 154.769, 120.593, 105.233, 79.053, 74.4776, 61.0269]),
        "dof": 106341,
    },
    "6 cycles\n(421k DoF)": {
        "N": np.array([1, 4, 16, 28]),
        "T": np.array([4940.66, 1814.89, 802.885, 515.543]),
        "dof": 421573,
    },
}

colors  = ["#1f77b4", "#2ca02c"]
markers = ["o", "s"]

for key in data:
    T1 = data[key]["T"][0]
    N  = data[key]["N"]
    T  = data[key]["T"]
    data[key]["speedup"]    = T1 / T
    data[key]["efficiency"] = T1 / T / N * 100
    data[key]["dof_per_proc"] = data[key]["dof"] / N

N_ideal = np.linspace(1, 28, 200)

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.grid":      True,
    "grid.alpha":     0.35,
    "grid.linestyle": "--",
})

# Figure1 1: wall time, speedup, and efficiency vs number of processors
fig1 = plt.figure(figsize=(15, 5))
gs   = gridspec.GridSpec(1, 3, wspace=0.38)
ax1  = fig1.add_subplot(gs[0])
ax2  = fig1.add_subplot(gs[1])
ax3  = fig1.add_subplot(gs[2])

ax2.plot(N_ideal, N_ideal,            "--", color="gray", lw=1.5, label="Ideal")
ax3.plot(N_ideal, np.ones_like(N_ideal)*100, "--", color="gray", lw=1.5, label="Ideal (100%)")

for (label, d), color, marker in zip(data.items(), colors, markers):
    N = d["N"]
    ax1.plot(N, d["T"],          f"{marker}-", color=color, lw=2, ms=7, label=label)
    ax2.plot(N, d["speedup"],    f"{marker}-", color=color, lw=2, ms=7, label=label)
    ax3.plot(N, d["efficiency"], f"{marker}-", color=color, lw=2, ms=7, label=label)

ax1.set_xlabel("Number of processors")
ax1.set_ylabel("Wall time [s]")
ax1.set_xticks([1, 4, 8, 14, 20, 28])
ax1.legend(fontsize=9)

ax2.set_xlabel("Number of processors")
ax2.set_ylabel("Speedup $S(N) = T(1)/T(N)$")
ax2.set_xticks([1, 4, 8, 14, 20, 28])
ax2.legend(fontsize=9)

ax3.set_xlabel("Number of processors")
ax3.set_ylabel("Efficiency $E(N)=S(N)/N$ [%]")
ax3.set_xticks([1, 4, 8, 14, 20, 28])
ax3.set_ylim(0, 120)
ax3.axhline(100, color="gray", lw=1.5, ls="--")
ax3.legend(fontsize=9)

fig1.savefig("strong_scaling_comparison.png", bbox_inches="tight", dpi=150)

# Figure 2: efficiency vs DoFs per process
fig2, ax = plt.subplots(figsize=(7, 5))
ax.axhline(100, color="gray", lw=1.5, ls="--", label="Ideal (100%)")

for (label, d), color, marker in zip(data.items(), colors, markers):
    ax.plot(d["dof_per_proc"], d["efficiency"],
            f"{marker}-", color=color, lw=2, ms=7, label=label)
    for n, dpp, eff in zip(d["N"], d["dof_per_proc"], d["efficiency"]):
        ax.annotate(f"N={n}", (dpp, eff), textcoords="offset points",
                    xytext=(5, 4), fontsize=8, color=color)

ax.set_xlabel("DoFs per process")
ax.set_ylabel("Efficiency $E(N)=S(N)/N$ [%]")
ax.set_xscale("log")
ax.set_ylim(0, 120)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35, linestyle="--")
plt.tight_layout()
fig2.savefig("efficiency_vs_dof_per_proc.png", bbox_inches="tight", dpi=150)