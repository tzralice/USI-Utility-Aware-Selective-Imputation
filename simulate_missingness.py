import numpy as np
import pandas as pd
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))

def get_beta_params(mean, concentration):
    if not (0 < mean < 1):
        raise ValueError("mean must be in (0,1)")
    if concentration <= 0:
        raise ValueError("concentration must be > 0")
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    return alpha, beta

def simulate_missing_df(
    cleaned_df: pd.DataFrame,
    target_col: str,
    mechanism: str = "MCAR",            # "MCAR" | "MAR" | "MNAR"
    missing_row_rate: float = 0.3,     # fraction of rows that become incomplete (row gate)
    observed_feature_frac: float = 0.3,# fraction of features always observed (HyperImpute-style)
    beta_mean: float = 0.2,            # mean of Beta for column base-rates (baseline level, NOT calibrated)
    beta_concentration: float = 5.0,   # higher => less heterogeneity across columns
    w_scale: float = 0.5,              # std of random weights in MAR logistic model
    mnar_lambda_scale: float = 1.0,    # strength of self-masking in MNAR (numeric cols)
    random_state: int = 42,
):
    """
    Simulation with row-level control only (no cell-level calibration):

    - Row gate: exactly ~missing_row_rate fraction of rows are gated, and we ENFORCE
      that each gated row has at least one missing entry among maskable columns.
      (So achieved row-incomplete rate ~= missing_row_rate by construction.)

    - Column heterogeneity: for each maskable column j, draw a base rate:
        pi_j ~ Beta(alpha, beta)  with mean=beta_mean and concentration=beta_concentration
      This controls feature-wise heterogeneity.

    - MCAR:
        For all rows i and maskable columns j:
          P_ij = pi_j
        Then missingness is applied only to gated rows (and forced >=1 missing per gated row).

    - MAR:
        Choose an always-observed feature set O (never masked).
        For maskable columns j:
          b_j = logit(pi_j)
          w_j ~ N(0, w_scale^2 I)
          P_ij = sigmoid(b_j + w_j^T x_iO)

    - MNAR (self-masking, numeric only):
        Same as MAR but for numeric maskable cols:
          P_ij = sigmoid(b_j + w_j^T x_iO + lambda_j * s(x_ij))
        where s(x_ij) = |z-score(x_ij)| and lambda_j ~ N(0, mnar_lambda_scale^2).
        For categorical maskable columns, keep MAR logits.
    """
    if target_col not in cleaned_df.columns:
        raise ValueError("`target_col` must be present in the DataFrame.")
    if mechanism not in {"MCAR", "MAR", "MNAR"}:
        raise ValueError("mechanism must be one of {'MCAR','MAR','MNAR'}")
    if mechanism in {"MAR", "MNAR"} and not (0 < observed_feature_frac < 1):
        raise ValueError("observed_feature_frac must be in (0,1) for MAR/MNAR")
    if not (0 <= missing_row_rate <= 1):
        raise ValueError("missing_row_rate must be in [0,1]")
    if beta_concentration <= 0:
        raise ValueError("beta_concentration must be > 0")

    rng = np.random.default_rng(random_state)

    X_df = cleaned_df.drop(columns=[target_col]).copy()
    y_df = cleaned_df[[target_col]].copy()
    n_rows, n_cols = X_df.shape

    if n_cols == 0 or n_rows == 0:
        out_df = cleaned_df.copy()
        diag = {
            "mechanism": mechanism,
            "requested_row_incomplete_rate": float(missing_row_rate),
            "achieved_row_incomplete_rate": float(out_df.drop(columns=[target_col]).isna().any(axis=1).mean())
            if n_rows > 0 and n_cols > 0
            else 0.0,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "random_state": int(random_state),
        }
        return out_df, diag

    # -------------------------
    # Step 1: Row gate G_i (exactly n_incomplete rows)
    # -------------------------
    n_incomplete = int(round(missing_row_rate * n_rows))
    n_incomplete = max(0, min(n_rows, n_incomplete))
    selected_rows = (
        rng.choice(X_df.index.to_numpy(), size=n_incomplete, replace=False)
        if n_incomplete > 0
        else np.array([], dtype=X_df.index.to_numpy().dtype)
    )

    # -------------------------
    # Step 0: Choose always-observed O (only for MAR/MNAR)
    # -------------------------
    if mechanism == "MCAR":
        observable_cols = []                   # not used for MCAR
        maskable_cols = X_df.columns.tolist()  # all can be masked
    else:
        if n_cols <= 1:
            observable_cols = []
            maskable_cols = X_df.columns.tolist()
        else:
            n_observable = max(1, int(round(n_cols * observed_feature_frac)))
            n_observable = min(n_observable, n_cols - 1)  # keep at least 1 maskable col
            observable_cols = rng.choice(
                X_df.columns.to_list(), size=n_observable, replace=False
            ).tolist()
            maskable_cols = [c for c in X_df.columns if c not in observable_cols]

    m = len(maskable_cols)
    mask_df = pd.DataFrame(False, index=X_df.index, columns=X_df.columns)

    # -------------------------
    # Step 2: Beta base-rate per maskable column -> intercept b_j
    # -------------------------
    mu = float(np.clip(beta_mean, 1e-6, 1.0 - 1e-6))
    alpha, beta_ = get_beta_params(mu, beta_concentration)

    pi = rng.beta(alpha, beta_, size=m).astype(float)   # (m,)
    b = logit(pi)                                       # (m,)

    # -------------------------
    # Step 3: Build probabilities P for all rows & maskable cols
    # -------------------------
    if mechanism == "MCAR":
        P = np.tile(pi[None, :], (n_rows, 1))           # (n_rows, m)
        W = None
        Lambda = None
        p_dim = 0

    else:
        # One-hot encode + normalize observed drivers X_{i,O}
        X_obs_oh = pd.get_dummies(X_df.loc[:, observable_cols], dummy_na=False)
        X_obs = X_obs_oh.to_numpy(dtype=float)
        X_obs_mean = X_obs.mean(axis=0, keepdims=True)
        X_obs_std = X_obs.std(axis=0, keepdims=True) + 1e-8
        X_obs = (X_obs - X_obs_mean) / X_obs_std
        p_dim = X_obs.shape[1]

        W = rng.normal(loc=0.0, scale=w_scale, size=(m, p_dim))   # (m, p_dim)
        Z = b[None, :] + (X_obs @ W.T)                            # (n_rows, m)

        if mechanism == "MNAR":
            Lambda = rng.normal(loc=0.0, scale=mnar_lambda_scale, size=m)  # (m,)
            Z_mnar = Z.copy()
            for j, col in enumerate(maskable_cols):
                if pd.api.types.is_numeric_dtype(X_df[col]):
                    v = X_df[col].to_numpy(dtype=float)
                    mu_v = np.nanmean(v)
                    sd_v = np.nanstd(v) + 1e-8
                    s = np.abs((v - mu_v) / sd_v)                 # |z-score|
                    Z_mnar[:, j] = Z_mnar[:, j] + Lambda[j] * s
            P = sigmoid(Z_mnar)
        elif mechanism == "MAR":
            Lambda = None
            P = sigmoid(Z)

    # -------------------------
    # Step 4: Apply masking only within gated rows
    # -------------------------
    U = rng.random(P.shape)
    mask_block = (U < P)  # (n_rows, m)

    row_positions = X_df.index.get_indexer(selected_rows)
    if n_incomplete > 0:
        mask_df.loc[selected_rows, maskable_cols] = mask_block[row_positions, :]

    # ENFORCE: each gated row has at least one missing among maskable cols
    if n_incomplete > 0:
        gated_no_missing = ~mask_df.loc[selected_rows, maskable_cols].any(axis=1)
        if gated_no_missing.any():
            weights = pi / pi.sum() if pi.sum() > 0 else np.full(m, 1.0 / m)
            bad_rows = gated_no_missing.index[gated_no_missing]
            forced_cols = rng.choice(maskable_cols, size=len(bad_rows), replace=True, p=weights)
            for ridx, c in zip(bad_rows, forced_cols):
                mask_df.loc[ridx, c] = True

    # Apply mask to X and return
    X_masked = X_df.mask(mask_df, other=np.nan)
    out_df = pd.concat([X_masked, y_df], axis=1)

    diag = {
        "mechanism": mechanism,
        "requested_row_incomplete_rate": float(missing_row_rate),
        "achieved_row_incomplete_rate": float(X_masked.isna().any(axis=1).mean()),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_incomplete_rows": int(n_incomplete),
        "observable_cols": list(observable_cols),
        "maskable_cols": list(maskable_cols),
        "beta_mean": float(mu),
        "beta_concentration": float(beta_concentration),
        "pi_maskable": {c: float(p) for c, p in zip(maskable_cols, pi)},
        "w_scale": float(w_scale),
        "mnar_lambda_scale": float(mnar_lambda_scale),
        "achieved_col_missingness": X_masked.isna().mean().to_dict(),
        "random_state": int(random_state),
    }
    return out_df, diag
