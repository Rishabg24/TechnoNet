import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import torch
import warnings
from typing import Tuple, Dict, Optional


class AnomolyMetrics:
    def __init__(
        self,
        combine_weight,
        reduction: str = "mean",
        epsilon: float = 1e-8,
        flatten_mode: str = "mean",
        normalize: str = "robust",
    ):
        assert flatten_mode in ("mean", "flatten")
        assert normalize in ("robust", "zscore")
        assert reduction in ("mean", "max")

        self.mode = flatten_mode
        self.norm = normalize
        self.weight = combine_weight
        self.epsilon = epsilon
        self.reduction = reduction
        self.isFitted = False
        self.mean_vec = None
        self.cov_inv = None

    def to_numpy(self, input):
        if torch.is_tensor(input):
            return input.detach().cpu().numpy()
        return np.asanyarray(input)

    def flatten_data(self, embedding):
        """
        Reducing Latent space embedding to a fixed size vector.
        This allows for fitting the latent embedding to a latent distribution later on.

        """
        if embedding.ndim == 2:
            return embedding
        elif embedding.ndim == 3:
            if self.mode == "mean":
                return embedding.mean(axis=1)
            if self.mode == "flatten":
                N, D, T = embedding.shape
                return embedding.reshape(N, D * T)

        else:
            raise ValueError(" Embedding needs to have 2 or 3 dimensions ")

    def compute_mahalanobis_distance(self, embedding):
        """
        Compute Mahalanobis Distance using:
        D_m = √(z-mu)^T covariance(Z-mu)
        """
        if not self.isFitted:
            raise ValueError(
                "Model Must be fit to the data before computing Mahalanobis distance"
            )

        flattened_emb = self.flatten_data(embedding)
        emb = self.to_numpy(flattened_emb)

        diff = emb - self.mean_vec
        M_d = np.sqrt(np.einsum("ij,jk,ik->i", diff, self.cov_inv, diff))

        return M_d

    def fit(self, embedding, recon_errors=None):
        """
        Computing Mean and Covariance matrix of the latent space to learn the nominal distribution of normal light curve data
        """
        embed = self.flatten_data(embedding)
        emb = self.to_numpy(embed)
        covariance_mat = LedoitWolf().fit(emb)

        self.mean_vec = covariance_mat.location_
        self.cov_inv = covariance_mat.precision_

        # Set fitted to True NOW, before computing Mahalanobis distances
        self.isFitted = True

        M_d = self.compute_mahalanobis_distance(
            embedding=embedding
        )  # ← Now this works!

        # Initialize all variables
        median = mad = mean = std = None

        if self.norm == "robust":
            median = np.median(M_d)
            mad = np.median(np.abs(M_d - median))
            mean = None
            std = None

        elif self.norm == "zscore":
            mean = np.mean(M_d)
            std = np.std(M_d)
            median = None
            mad = None

        self.md_stats = {
            "median": median,
            "mean_abs_deviation": mad,
            "mean": mean,
            "std": std,
        }

        if recon_errors is not None:
            recon_errors = self.to_numpy(recon_errors)

            median_r = mad_r = mean_recon = std_recon = None

            if self.norm == "robust":
                median_r = np.median(recon_errors)
                mad_r = np.median(np.abs(recon_errors - median_r))
                mean_recon = None
                std_recon = None
            elif self.norm == "zscore":
                mean_recon = np.mean(recon_errors)
                std_recon = np.std(recon_errors)
                median_r = None
                mad_r = None

            self.recon_errors_stats = {
                "median": median_r,
                "mean_abs_deviation": mad_r,
                "mean": mean_recon,
                "std": std_recon,
            }
        else:
            warnings.warn(
                "No reconstruction error specified. Fitting and anomoly dection may be impacted later because of this"
            )

        return self.md_stats, self.recon_errors_stats

    def reconstruction_MAE(self, x, x_recon):
        """computes and returns mean absolute error from reconstructed x(t) input from Temporal Autoencoder"""
        diff = np.abs(x - x_recon)
        if self.reduction == "mean":
            E_i = np.mean(diff, axis=(1, 2))
        elif self.reduction == "max":
            E_i = diff.max(axis=(1, 2))
        else:
            raise ValueError("choose mean or max for reduction")
        return E_i

    def normalize_scores(self, mahal_dist, recon_Mae):
        """
        take in mahalanobis distance and MAE reconstructed error to normalize to fitted data scale
        outputs normalized arrays of both.

        z -> normalized mahalanobis distance
        y -> normalized recon_MAE

        """
        if self.norm == "robust":
            z = (mahal_dist - self.md_stats["median"]) / (
                self.md_stats["mean_abs_deviation"] + self.epsilon
            )
            y = (recon_Mae - self.recon_errors_stats["median"]) / (
                self.recon_errors_stats["mean_abs_deviation"] + self.epsilon
            )

        if self.norm == "zscore":
            z = (mahal_dist - self.md_stats["mean"]) / (
                self.md_stats["std"] + self.epsilon
            )
            y = (recon_Mae - self.md_stats["mean"]) / (
                self.recon_errors_stats["mean_abs_deviation"] + self.epsilon
            )

        return z, y

    def combine_signals(self, norm_MD, norm_recon_err):
        """uses weighting factor to combine signals and return anomoly score"""
        raw_score = self.weight * norm_MD + (1 - self.weight) * norm_recon_err

        median = mad = mean = std = None

        if self.norm == "robust":
            median = np.median(raw_score)
            mad = np.median(np.abs(raw_score - median))
            final_score = (raw_score - median) / (mad + self.epsilon)
        elif self.norm == "zscore":  # zscore
            mean = np.mean(raw_score)
            std = np.std(raw_score)
            final_score = (raw_score - mean) / (std + self.epsilon)
        else:
            raise ValueError("Unknown Normalization")

        score_metrics = {
            "score": final_score,
            "median": median,
            "mean_abs_dev": mad,
            "mean": mean,
            "std": std,
        }
        return score_metrics

    def adaptive_thresholding(self, k, score_metrics: dict = None):
        if score_metrics is not None:
            if self.norm == "robust":
                score = score_metrics["score"]
                median = score_metrics["median"]
                MAD = score_metrics["mean_abs_dev"]
                flag = score > median + k * MAD
            elif self.norm == "zscore":
                score = score_metrics = ["score"]
                mean = score_metrics["mean"]
                std = score_metrics["std"]
                flag = score > mean + k * std
            else:
                raise ValueError("Unknown Normalization")
        else:
            raise ValueError(
                "Score_metrics must be a dict returned from Combine signals"
            )

        if "score" not in score_metrics:
            raise ValueError("Score is not found in score_metrics dict")

        return flag
