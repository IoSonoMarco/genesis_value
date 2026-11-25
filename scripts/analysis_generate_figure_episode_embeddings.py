from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


OUTPUT_DIR = Path("output/figure_episode_geometry")
DATA_DIR = OUTPUT_DIR/"episode_embeddings.npy"


def tg_mds(D, n_components=1):
    """
    Torgerson-Guttman Multidimensional Scaling

    D: distance matrix (symmetric)
    """
    # Double-centering
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])

def l2norm(X):
    d = np.linalg.norm(X, axis=-1)[:,None]
    return X/d


###


if __name__ == "__main__":

    data = np.load(DATA_DIR, allow_pickle=True)

    stimulus = []
    reward = []
    time = []
    embeddings = {
        'z': [], 
        'object_shape': [], 
        'object_color': [], 
        'floor_color': [], 
        'wall_color': [],
        'reward': [],
        'time': []
    }
    metadata = []

    for d in data:
        
        for key in d["stimulus"]:
            embeddings[key].append(d["stimulus"][key])
            
        embeddings["reward"].append(d["reward"]["reward_embedding"])
        
        embeddings["time"].append(d["time"])
            
        metadata.append(d["metadata"])

    embeddings = {k:np.vstack(embeddings[k]) for k in embeddings}
    metadata = pd.DataFrame(metadata).astype(int)

    def make_tensor(*args):
        X = []
        for a in args:
            x = l2norm(embeddings[a])
            X.append(x)
        return np.hstack(X)

    n = embeddings["z"].shape[0]

    D, L = [], []
    for i, k in enumerate(embeddings):
        x = l2norm(embeddings[k])
        dim = x.shape[-1]
        X = np.vstack([np.zeros((n*i, dim)), np.vstack([x]*(7-i))])
        labels = [k]*n    
        
        D.append(X)
        L.append(labels)
        
    D = np.hstack(D)
    L = np.hstack(L)


    ###


    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    rho = pairwise_distances(D[L=="z"], metric="euclidean")
    coord = tg_mds(rho, n_components=2)
    df = pd.concat(
        [pd.DataFrame(dict(x=coord[:, 0], y=coord[:, 1])), metadata],
        axis=1
    )
    sns.scatterplot(df, x="x", y="y", ax=axes[0], legend=False)

    rho = pairwise_distances(D[L=="object_shape"], metric="euclidean")
    coord = tg_mds(rho, n_components=2)
    df = pd.concat(
        [pd.DataFrame(dict(x=coord[:, 0], y=coord[:, 1])), metadata],
        axis=1
    )
    sns.scatterplot(
        df, x="x", y="y",
        hue="object_shape",
        ax=axes[1],
        legend=True
    )

    rho = pairwise_distances(D[L=="object_color"], metric="euclidean")
    coord = tg_mds(rho, n_components=2)
    df = pd.concat(
        [pd.DataFrame(dict(x=coord[:, 0], y=coord[:, 1])), metadata],
        axis=1
    )
    sns.scatterplot(
        df, x="x", y="y",
        hue="object_color",
        ax=axes[2],
        legend=True
    )

    rho = pairwise_distances(D[L=="reward"], metric="euclidean")
    coord = tg_mds(rho, n_components=2)
    df = pd.concat(
        [pd.DataFrame(dict(x=coord[:, 0], y=coord[:, 1])), metadata],
        axis=1
    )
    sns.scatterplot(
        df, x="x", y="y",
        hue="scalar_reward",
        ax=axes[3],
        legend=True
    )

    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("")
        ax.set_ylabel("")

        # Remove tick marks
        ax.tick_params(axis="both", which="both", length=0)

        # Keep tick labels only on first subplot
        if i > 0:
            ax.tick_params(
                axis="both",
                labelbottom=False,
                labelleft=False
            )

        legend = ax.get_legend()

        if legend is not None:
            title = legend.get_title().get_text()
            handles, labels = ax.get_legend_handles_labels()

            patch_handles = []

            for handle in handles:
                if isinstance(handle, Line2D):
                    color = handle.get_color()
                else:
                    color = handle.get_facecolor()
                    if len(color) > 0:
                        color = color[0]

                patch_handles.append(
                    Patch(
                        facecolor=color,
                        edgecolor="none"
                    )
                )

            legend.remove()

            ax.legend(
                patch_handles,
                labels,
                title=title,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(labels),
                frameon=False,
                borderaxespad=0,
                handlelength=1.8,
                handleheight=0.7,
            )

    plt.subplots_adjust(top=0.8, wspace=0.15)

    fig.savefig(OUTPUT_DIR/"figure.pdf")