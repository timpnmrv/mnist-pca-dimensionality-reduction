
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
# Для быстрого доклада можно оставить 12000.
# Для полного MNIST поставьте SAMPLE_SIZE = None.
SAMPLE_SIZE: int | None = 12000
FIG_DIR = Path("figures")


def load_mnist(sample_size: int | None = SAMPLE_SIZE):
    print("Загружаю MNIST через OpenML...")
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        data_home="data",
        cache=True,
    )

    X = X.astype("float32") / 255.0
    y = y.astype("int64")

    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )

    print(f"Форма X: {X.shape}; форма y: {y.shape}")
    return X, y


def plot_digit_examples(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
    for digit, ax in enumerate(axes.ravel()):
        idx = np.flatnonzero(y == digit)[0]
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(f"label={digit}")
        ax.axis("off")
    fig.suptitle("Примеры изображений MNIST")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pca_2d(X_train: np.ndarray, y_train: np.ndarray, path: Path) -> None:
    """Строит 2D-проекцию PCA."""
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_train)

    # Чтобы график не был слишком тяжёлым, рисуем не более 5000 точек.
    n_plot = min(5000, len(X_2d))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_2d), size=n_plot, replace=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        c=y_train[idx],
        cmap="tab10",
        s=8,
        alpha=0.65,
    )
    ax.set_title("MNIST после PCA: первые две главные компоненты")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Цифра")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print("Доля дисперсии в PC1 и PC2:", pca2.explained_variance_ratio_)
    print("Суммарно две компоненты объясняют:", pca2.explained_variance_ratio_.sum())


def fit_full_pca(X_train: np.ndarray) -> PCA:
    print("Обучаю PCA со всеми компонентами...")
    pca = PCA(n_components=None, svd_solver="full")
    pca.fit(X_train)
    return pca


def components_for_variance(pca: PCA, threshold: float) -> int:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumulative >= threshold) + 1)


def plot_explained_variance(pca: PCA, path: Path) -> None:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    k90 = components_for_variance(pca, 0.90)
    k95 = components_for_variance(pca, 0.95)
    k99 = components_for_variance(pca, 0.99)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, linewidth=2)
    for threshold, k in [(0.90, k90), (0.95, k95), (0.99, k99)]:
        ax.axhline(threshold, linestyle="--", linewidth=1)
        ax.axvline(k, linestyle="--", linewidth=1)
        ax.text(k + 3, threshold - 0.035, f"{int(threshold*100)}%: k={k}")

    ax.set_title("Накопленная объяснённая дисперсия PCA")
    ax.set_xlabel("Число компонент k")
    ax.set_ylabel("Доля объяснённой дисперсии")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"k для 90% дисперсии: {k90}")
    print(f"k для 95% дисперсии: {k95}")
    print(f"k для 99% дисперсии: {k99}")
    print(f"Сжатие при 95%: 784 -> {k95}, то есть в {784 / k95:.2f} раза меньше признаков")


def reconstruct_with_first_k(pca: PCA, X: np.ndarray, k: int) -> np.ndarray:
    W = pca.components_[:k]
    X_centered = X - pca.mean_
    Z = X_centered @ W.T
    X_rec = Z @ W + pca.mean_
    return np.clip(X_rec, 0.0, 1.0)


def plot_reconstructions(pca: PCA, X_test: np.ndarray, y_test: np.ndarray, path: Path) -> None:
    k95 = components_for_variance(pca, 0.95)
    k_values = [2, 10, 30, 80, k95]

    chosen_indices = []
    for digit in [0, 2, 3, 5, 8]:
        chosen_indices.append(np.flatnonzero(y_test == digit)[0])
    X_sel = X_test[chosen_indices]
    y_sel = y_test[chosen_indices]

    rows = 1 + len(k_values)
    cols = len(X_sel)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.55, rows * 1.55))

    for col in range(cols):
        axes[0, col].imshow(X_sel[col].reshape(28, 28), cmap="gray")
        axes[0, col].set_title(f"orig: {y_sel[col]}")
        axes[0, col].axis("off")

    for row, k in enumerate(k_values, start=1):
        X_rec = reconstruct_with_first_k(pca, X_sel, k)
        mse = np.mean((X_sel - X_rec) ** 2)
        for col in range(cols):
            axes[row, col].imshow(X_rec[col].reshape(28, 28), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"k={k}\nMSE={mse:.4f}", rotation=0, labelpad=35, va="center")

    fig.suptitle("Восстановление MNIST после сжатия PCA")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_principal_components(pca: PCA, path: Path, n_components: int = 12) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(7, 5.2))
    for i, ax in enumerate(axes.ravel()[:n_components]):
        comp = pca.components_[i].reshape(28, 28)
        vmax = np.abs(comp).max()
        ax.imshow(comp, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"PC{i+1}")
        ax.axis("off")
    fig.suptitle("Первые главные компоненты: 'шаблоны' изменения цифр")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    X, y = load_mnist(SAMPLE_SIZE)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    plot_digit_examples(X_train, y_train, FIG_DIR / "01_mnist_examples.png")
    plot_pca_2d(X_train, y_train, FIG_DIR / "02_pca_2d_scatter.png")

    pca_full = fit_full_pca(X_train)
    plot_explained_variance(pca_full, FIG_DIR / "03_explained_variance.png")
    plot_reconstructions(pca_full, X_test, y_test, FIG_DIR / "04_reconstructions.png")
    plot_principal_components(pca_full, FIG_DIR / "05_principal_components.png")

    print("\nГотово. Графики сохранены в папке figures/.")


if __name__ == "__main__":
    main()
