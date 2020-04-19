import pymc3 as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


# rcParams["font.family"] = "sans-serif"
# rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"font.size": 10})

SEED = 123
n_samples = 8_000

default_sample_options = {
    "tune": 2_000,
    "draws": n_samples,
    "chains": 2,
    "cores": 2,
    "nuts_kwargs": {"target_accept": 0.95},
    "random_seed": SEED,
}

np.random.seed(SEED)

default_prior = {
    "β0μ": 12,
    "β0σ": 20,
    "β1μ": 0,
    "β1σ": 20,
    "β2μ": 0,
    "β2σ": 20,
    "β3μ": 0,
    "β3σ": 20,
    "σ": 1,
}


# THE BAYESIAN MODEL IN PyMC3 ============================================


class BayesianModeration:
    def __init__(
        self,
        y,
        x,
        m,
        prior=default_prior,
        sample_options=default_sample_options,
        xlabel="x",
        ylabel="y",
        mlabel="moderator",
    ):
        self.y = y
        self.x = x
        self.m = m
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.mlabel = mlabel
        self.prior = prior
        self.sample_options = sample_options
        self.scalarMap = self._make_colormap()

        self.model = self._make_model()
        self.trace = self._infer()

    def _make_colormap(self):
        return ScalarMappable(
            norm=Normalize(vmin=np.min(self.m), vmax=np.max(self.m)), cmap="viridis",
        )

    def _make_model(self):
        print(f"Prior parameters are:\n{self.prior}")
        print(
            "WARNING: Ensure your priors are appropriate for your data and problem at hand."
        )
        y, x, m = self.y, self.x, self.m
        with pm.Model() as model:
            β0 = pm.Bound(pm.Normal, lower=0.0)(
                "β0", mu=self.prior["β0μ"], sigma=self.prior["β0σ"], testval=20
            )
            β1 = pm.Normal(
                "β1", mu=self.prior["β1μ"], sd=self.prior["β1σ"], testval=0.4
            )
            β2 = pm.Normal(
                "β2", mu=self.prior["β2μ"], sd=self.prior["β2σ"], testval=-1.2
            )
            β3 = pm.Normal(
                "β3", mu=self.prior["β3μ"], sd=self.prior["β3σ"], testval=0.05
            )
            σ = pm.HalfCauchy("σ", self.prior["σ"], testval=0.5)
            y = pm.Cauchy(
                "y", alpha=β0 + (β1 * x) + (β2 * m) + (β3 * x * m), beta=σ, observed=y
            )

        return model

    def _infer(self):
        with self.model:
            trace = pm.sample(**self.sample_options)
        return trace

    def plot(self, ax=None, percentile_list=[2.5, 25, 50, 75, 97.5]):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        self.plot_posterior_prediction(ax, percentile_list=percentile_list)
        self.plot_data(ax)
        return ax

    def plot_multipanel(
        self,
        figsize=(12, 8),
        percentile_list=[2.5, 50, 97.5],
        moderation_multiplier=1,
        kind="interval",
    ):
        """Moderation plot with data + posterior prediction, as well as posterior distributions over parameters"""

        # gs0 = matplotlib.gridspec.GridSpec(2, 4, height_ratios=[2, 1])
        gs0 = GridSpec(3, 4, height_ratios=[4, 1, 3])
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(gs0[0, :])
        # ax1 = fig.add_subplot(gs0[0, 0:2])
        ax1 = self.plot(ax=ax1, percentile_list=percentile_list)
        # ax1.set(
        #     title=r"$BMI = \beta_0 + (\beta_1 age) + (\beta_2 \ln(k)) + (\beta_3 age \cdot \ln(k))$"
        # )

        # plot moderation effect
        # ax2 = fig.add_subplot(gs0[0, 2:4])
        ax2 = fig.add_subplot(gs0[2, :])
        ax2 = self.plot_moderation_effect(
            ax=ax2,
            percentile_list=percentile_list,
            moderation_multiplier=moderation_multiplier,
            kind=kind,
        )

        axb0 = fig.add_subplot(gs0[1, 0])
        axb0 = az.plot_posterior(
            self.trace, var_names="β0", ax=axb0, credible_interval=0.95, textsize=9
        )
        axb0.set(title="intercept", xlabel=r"$\beta_0$")

        axb1 = fig.add_subplot(gs0[1, 1])
        az.plot_posterior(
            self.trace, var_names="β1", ax=axb1, credible_interval=0.95, textsize=9
        )
        axb1.set(title="main effect of age", xlabel=r"$\beta_1$")

        axb2 = fig.add_subplot(gs0[1, 2])
        az.plot_posterior(
            self.trace, var_names="β2", ax=axb2, credible_interval=0.95, textsize=9
        )
        axb2.set(title="main effect of $\ln(k)$", xlabel=r"$\beta_2$")

        axb3 = fig.add_subplot(gs0[1, 3])
        az.plot_posterior(
            self.trace, var_names="β3", ax=axb3, credible_interval=0.95, textsize=9
        )
        axb3.set(title="interaction term", xlabel=r"$\beta_3$")

        # increase spacing between rows
        plt.subplots_adjust(hspace=0.35)
        return (ax1, ax2, axb0, axb1, axb2, axb3)

    def plot_data(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()

        h = ax.scatter(self.x, self.y, c=self.m, cmap=self.scalarMap.cmap)
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        # colourbar for moderator
        cbar = fig.colorbar(h)
        cbar.ax.set_ylabel(self.mlabel)
        return ax

    def plot_posterior_prediction(
        self, ax=None, percentile_list=[2.5, 25, 50, 75, 97.5]
    ):
        """Plot posterior predicted `y` for the defined moderator percentiles"""
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # PLOT POSTERIOR PREDICTED `y` FOR DEFINED MODERATOR LEVELS
        xi = np.linspace(np.min(self.x), np.max(self.x), 200)
        m_levels = np.percentile(self.m, percentile_list)

        for p, m in zip(percentile_list, m_levels):
            β0 = np.expand_dims(self.trace["β0"], axis=1)
            β1 = np.expand_dims(self.trace["β1"], axis=1)
            β2 = np.expand_dims(self.trace["β2"], axis=1)
            β3 = np.expand_dims(self.trace["β3"], axis=1)
            _y = β0 + (β1 * xi) + (β2 * m) + (β3 * xi * m)
            region = np.percentile(_y, [2.5, 50, 95], axis=0)

            ax.fill_between(
                xi,
                region[0, :],
                region[2, :],
                alpha=0.2,
                color=self.scalarMap.to_rgba(m),
                edgecolor="w",
            )
            ax.plot(
                xi,
                region[1, :],
                color=self.scalarMap.to_rgba(m),
                linewidth=2,
                label=f"{p}th percentile of moderator",
            )

        ax.legend(fontsize=9)

        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        return ax

    def plot_moderation_effect(
        self,
        ax=None,
        # true_β1=None,
        # true_β2=None,
        percentile_list=[0, 2.5, 25, 50, 75, 97.5, 100],
        samples_to_plot=100,
        moderation_multiplier=1,
        kind="spaghetti",
    ):
        """Plot the slope of y=slope*x + c where slope = β1 + β3 * moderator
        `moderation_multiplier` is used if you want to change the units of the slope (eg. per year to per decade"""

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        m = np.linspace(np.min(self.m), np.max(self.m), 2)

        # sample from posterior
        n_samples = len(self.trace["β1"])
        if samples_to_plot < n_samples:
            ind = np.random.choice(range(n_samples), samples_to_plot)

        β1 = np.expand_dims(self.trace["β1"], axis=1)[ind, :]
        β3 = np.expand_dims(self.trace["β3"], axis=1)[ind, :]
        rate = moderation_multiplier * (β1 + β3 * m)

        if kind is "spaghetti":
            for n in range(samples_to_plot):
                ax.plot(m, rate[n, :], "k", alpha=0.1)
        elif kind is "interval":
            # TODO: can I extract this "spaghetti to interval" as a function? It's a common pattern I use.
            mi = np.linspace(np.min(self.m), np.max(self.m), 200)
            β1 = np.expand_dims(self.trace["β1"], axis=1)
            β3 = np.expand_dims(self.trace["β3"], axis=1)
            rate = moderation_multiplier * (β1 + β3 * mi)
            region = np.percentile(rate, [2.5, 50, 95], axis=0)

            ax.fill_between(
                mi, region[0, :], region[2, :], alpha=0.1, color="k", edgecolor="w",
            )
            ax.plot(
                mi, region[1, :], color="k", linewidth=2,
            )

        # # plot true model, if known
        # if (true_β1 is not None) and (true_β2 is not None):
        #     true = true_β1 + true_β2 * m
        #     ax.plot(m, true, "r", lw=3, label="true")

        # plot points at each percentile of m
        m_levels = np.percentile(self.m, percentile_list)
        for m in m_levels:
            ax.plot(
                m,
                moderation_multiplier * (np.mean(β1) + np.mean(β3) * m),
                "o",
                c=self.scalarMap.to_rgba(m),
                markersize=10,
                markeredgecolor="w",
            )

        #     ax.legend()

        ax.axhline(y=0, linewidth=1, c="k", ls="--")

        ax.set(
            title="moderation effect",
            xlabel=self.mlabel,
            ylabel=r"$\beta_1 + \beta_3 \cdot moderator$",
        )
        return ax
