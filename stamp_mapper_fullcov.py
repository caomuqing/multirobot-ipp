"""
Full-covariance variant of the GP-Kalman mapper, for the diagonal-approximation
ablation (reviewer-requested: diagonal storage vs. full covariance).

Subclasses STAMP_FR_GPF_IK and overrides ONLY the covariance representation:
a dense V x V posterior covariance P is maintained and used exactly where the
parent reconstructs P from marginal variances + the fixed kernel correlation
template. Footprint geometry, measurement semantics, noise model, node-feature
accessors, and the process-update form are all inherited unchanged, so a
comparison against the parent isolates the diagonal approximation itself.

Memory: 32x32 grid -> P is 1024x1024 float64 = 8 MB per mapper.
Intended for the centralized deployment (the decentralized one exchanges
diagonal packets by design; full-covariance exchange is not proposed).
"""

import numpy as np
from scipy.linalg import solve_triangular

from stamp_mapper_multi import STAMP_FR_GPF_IK, integral_kernel_fcn


class STAMP_FR_GPF_IK_FullCov(STAMP_FR_GPF_IK):

    def init_v_map(self):
        # Parent sets prior_diag, corr (template), state_var, compatibility cov.
        super().init_v_map()

        # Rebuild the full prior covariance exactly as the parent did internally
        # (same kernel call + symmetrization + obstruction zeroing).
        K = integral_kernel_fcn(self.grid_corners, self.grid_corners, self.v_prior, self.l)
        K = 0.5 * (K + K.T) + 1e-10 * np.eye(K.shape[0])
        if self.obstructed_grids_mask is not None and np.any(self.obstructed_grids_mask):
            idx = np.where(self.obstructed_grids_mask)[0]
            K[idx, :] = 0.0
            K[:, idx] = 0.0
            K[idx, idx] = 0.0

        self.K_full = K
        self.P_full = K.copy()

    def _sync_state_var(self):
        self.state_var = np.maximum(np.diag(self.P_full).copy(), 1e-12)
        self._sync_covariance_diag()

    def update_map(
        self,
        m,
        Hm,
        noise_std=0.1,
        regularization=1e-6,
        max_condition_number=1e8,
        distances=None,
        noise_distance_scale=1.0,
        pos_noise_scale=1.0,
        neg_noise_scale=1.0,
    ):
        """Exact Kalman measurement update on the dense posterior covariance."""
        if len(Hm) == 0:
            return

        # Same obstruction filtering as the parent
        valid_idx, valid_m, valid_dist = [], [], []
        for i, cell_idx in enumerate(Hm):
            if not self.obstructed_grids_mask[cell_idx]:
                valid_idx.append(cell_idx)
                valid_m.append(m[i])
                if distances is not None and i < len(distances):
                    valid_dist.append(distances[i])
        if len(valid_idx) == 0:
            return

        H = np.asarray(valid_idx, dtype=int)
        z = np.asarray(valid_m, dtype=float).reshape(-1)

        # Same noise model as the parent
        if distances is not None and len(valid_dist) == len(z):
            dist = np.asarray(valid_dist, dtype=float)
            noise_stds = noise_std * (1.0 + noise_distance_scale * dist)
        else:
            noise_stds = noise_std * np.ones_like(z)
        pos_mask = z > 0.0
        noise_stds = noise_stds * np.where(pos_mask, pos_noise_scale, neg_noise_scale)
        r = noise_stds**2

        mu = self.state_value
        P = self.P_full

        # Exact covariance blocks (the parent reconstructs these from diag+template)
        P_xH = P[:, H]
        P_HH = P[np.ix_(H, H)]

        Sigma = 0.5 * (P_HH + P_HH.T)
        Sigma[np.diag_indices_from(Sigma)] += r + regularization

        # Same condition-number guard as the parent
        try:
            eigvals = np.linalg.eigvalsh(Sigma)
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))
            cond = max_eig / min_eig if min_eig > 0 else np.inf
            if cond > max_condition_number:
                target_min = max_eig / max_condition_number
                extra = max(0.0, target_min - min_eig)
                Sigma[np.diag_indices_from(Sigma)] += extra
        except Exception:
            pass

        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(Sigma.shape[0]))

        nu = z - mu[H]

        # W = P_xH L^{-T}: mean gain and full covariance downdate share it
        tmp = solve_triangular(L, P_xH.T, lower=True)      # (m, V)
        gain_T = solve_triangular(L.T, tmp, lower=False)   # (m, V)
        gain = gain_T.T                                    # (V, m)
        W = tmp.T                                          # (V, m)

        self.state_value = mu + gain @ nu

        # P_post = P - W W^T  (exact, symmetric downdate)
        P_new = P - W @ W.T
        P_new = 0.5 * (P_new + P_new.T)
        d = np.diag(P_new).copy()
        np.fill_diagonal(P_new, np.maximum(d, 1e-12))
        self.P_full = P_new

        self.exploration_mask[H] = True
        self._sync_state_var()
        self._reset_region_grids()

    def process_update(self, alpha=None):
        """GP-consistent full-covariance prediction: P <- a^2 P + (1-a^2) K."""
        if alpha is None:
            alpha = self.temporal_alpha

        self.state_value = self.prior_mean_map + alpha * (self.state_value - self.prior_mean_map)
        self.P_full = (alpha**2) * self.P_full + (1.0 - alpha**2) * self.K_full

        self._sync_state_var()
        self._reset_region_grids()
