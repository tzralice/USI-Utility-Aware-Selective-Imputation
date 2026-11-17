import numpy as np
import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from hyperimpute.plugins.imputers import Imputers


class DataPreprocessor:
    def __init__(self, config):
        self.train_df = config.get("train_df")
        self.val_df = config.get("val_df")
        self.test_df = config.get("test_df")
        self.dev_df = config.get("dev_df")

        self.target_col = config.get("target_col")
        self.categorical_cols = config.get("categorical_cols") or []
        self.numerical_cols = config.get("numerical_cols") or []

        # imputation_method:
        # - dict -> Pipeline A (mean/median/MICE + categorical placeholder)
        # - str  -> Pipeline B (encoded-space imputers like "hyper_gain", "hyper_missforest", ...)
        self.imputation_method = config.get("imputation_method")

        self.create_variant = config.get("create_variant")
        self.process_incomplete_val_test = config.get("process_incomplete_val_test", False)
        self.random_state = config.get("random_state",42)

    def preprocess(self):
        """
        Returns dataset variants WITH index preservation.
        - complete_samples: complete-only rows, encoded with frozen preprocessor + metadata
        - incomplete_with_indicators: raw incomplete rows + missingness indicators (no encoding)
        - imputed_all: all rows, imputed per method, returned as numeric matrix
        - per_feature_marginals: marginals estimated from complete training rows (for stochastic imputation, etc.)
        """
        trnX, trnY, valX, valY, devX, devY, tstX, tstY = self.separate_features_target()

        # Create dataset variants: complete_samples, incomplete_with_indicators, per_feature_marginals
        if self.create_variant is True:
            trn_complete_mask = ~trnX.isna().any(axis=1)
            val_complete_mask = ~valX.isna().any(axis=1)
            dev_complete_mask = ~devX.isna().any(axis=1)
            tst_complete_mask = ~tstX.isna().any(axis=1)

            trnX_complete, trnY_complete = trnX[trn_complete_mask], trnY[trn_complete_mask]
            trnX_incomplete, trnY_incomplete = trnX[~trn_complete_mask], trnY[~trn_complete_mask]
            trn_complete_idx = trnX.index[trn_complete_mask].tolist()
            trn_incomplete_idx = trnX.index[~trn_complete_mask].tolist()

            valX_complete, valY_complete = valX[val_complete_mask], valY[val_complete_mask]
            valX_incomplete, valY_incomplete = valX[~val_complete_mask], valY[~val_complete_mask]
            val_complete_idx = valX.index[val_complete_mask].tolist()
            val_incomplete_idx = valX.index[~val_complete_mask].tolist()

            devX_complete, devY_complete = devX[dev_complete_mask], devY[dev_complete_mask]
            devX_incomplete, devY_incomplete = devX[~dev_complete_mask], devY[~dev_complete_mask]
            dev_complete_idx = devX.index[dev_complete_mask].tolist()
            dev_incomplete_idx = devX.index[~dev_complete_mask].tolist()

            tstX_complete, tstY_complete = tstX[tst_complete_mask], tstY[tst_complete_mask]
            tstX_incomplete, tstY_incomplete = tstX[~tst_complete_mask], tstY[~tst_complete_mask]
            tst_complete_idx = tstX.index[tst_complete_mask].tolist()
            tst_incomplete_idx = tstX.index[~tst_complete_mask].tolist()

            print(
                f"Complete rows - Train: {len(trnX_complete)}, Val: {len(valX_complete)}, "
                f"Dev: {len(devX_complete)}, Test: {len(tstX_complete)}"
            )
            print(
                f"Incomplete rows - Train: {len(trnX_incomplete)}, Val: {len(valX_incomplete)}, "
                f"Dev: {len(devX_incomplete)}, Test: {len(tstX_incomplete)}"
            )

            print("Step 3: Building per-feature marginals...")
            per_feature_marginals = self.build_marginals_from_complete(
                trnX_complete, self.numerical_cols, self.categorical_cols
            )

            print("Step 4: Processing dataset variants...")

            print("  Processing complete_samples datasets...")
            complete_samples_datasets = self._process_complete_samples_da(
                trnX_complete, trnY_complete, valX_complete, valY_complete,
                devX_complete, devY_complete, tstX_complete, tstY_complete,
                trn_complete_idx, val_complete_idx, dev_complete_idx, tst_complete_idx
            )

            print("  Processing incomplete datasets with indicators...")
            if self.process_incomplete_val_test is False:
                valX_incomplete, valY_incomplete, val_incomplete_idx = valX_complete, valY_complete, val_complete_idx
                tstX_incomplete, tstY_incomplete, tst_incomplete_idx = tstX_complete, tstY_complete, tst_complete_idx

            incomplete_with_indicators = self._process_incomplete_data(
                trnX_incomplete, trnY_incomplete, valX_incomplete, valY_incomplete,
                devX_incomplete, devY_incomplete, tstX_incomplete, tstY_incomplete,
                trn_incomplete_idx, val_incomplete_idx, dev_incomplete_idx, tst_incomplete_idx
            )

            return {
                "complete_samples": complete_samples_datasets,
                "incomplete_with_indicators": incomplete_with_indicators,
                "per_feature_marginals": per_feature_marginals,
            }

        # create_variant == False: only return the chosen processing result
        return self._process_imputed_data(
            trnX, trnY, valX, valY, devX, devY, tstX, tstY,
            trnX.index.tolist(), valX.index.tolist(), devX.index.tolist(), tstX.index.tolist()
        )

    def _process_complete_samples_da(self, trnX, trnY, valX, valY, devX, devY, tstX, tstY,
                           trn_idx, val_idx, dev_idx, tst_idx):
        """Complete-only rows: fit preprocessor on train, transform all, return frozen artifacts."""
        if len(trnX) == 0:
            print("    Warning: No complete training data found!")
            return None

        trnX_enc, valX_enc, devX_enc, tstX_enc, feature_names, preprocessor, meta = \
            self.encode_categorical_scale_numerical(
                trnX.copy(), valX.copy(), devX.copy(), tstX.copy(),
                self.categorical_cols, self.numerical_cols,
                return_preprocessor=True,
                scale_numericals=True,          # complete_samples baseline uses scaled numericals
                missing_token="__MISSING__",
                drop=None
            )

        print(f"    Complete datasets final shape - Train: {trnX_enc.shape}")
        return {
            "trnX": trnX_enc, "trnY": trnY, "trnX_idx": trn_idx,
            "valX": valX_enc, "valY": valY, "valX_idx": val_idx,
            "devX": devX_enc, "devY": devY, "devX_idx": dev_idx,
            "tstX": tstX_enc, "tstY": tstY, "tstX_idx": tst_idx,
            "feature_names": feature_names,
            "preprocessor": preprocessor,
            "preprocessor_metadata": meta,
        }

    def _process_incomplete_data(self, trnX, trnY, valX, valY, devX, devY, tstX, tstY,
                                 trn_idx, val_idx, dev_idx, tst_idx):
        """Return raw incomplete rows + missing indicators (no imputation/encoding here)."""
        if len(trnX) == 0:
            print("    Warning: No incomplete training data found!")
            return None

        trnX_final, valX_final, devX_final, tstX_final = trnX.copy(), valX.copy(), devX.copy(), tstX.copy()
        trn_missing, val_missing, dev_missing, tst_missing = self.add_missing_indicators(
            trnX_final, valX_final, devX_final, tstX_final
        )
        print(f"    Incomplete datasets final shape - Train: {trnX_final.shape}")
        return {
            "trnX": trnX_final, "trnY": trnY, "trnX_idx": trn_idx,
            "valX": valX_final, "valY": valY, "valX_idx": val_idx,
            "devX": devX_final, "devY": devY, "devX_idx": dev_idx,
            "tstX": tstX_final, "tstY": tstY, "tstX_idx": tst_idx,
            "trnX_missing_indicators": trn_missing,
            "valX_missing_indicators": val_missing,
            "devX_missing_indicators": dev_missing,
            "tstX_missing_indicators": tst_missing,
            "feature_names": trnX.columns.tolist(),
        }

    def _process_imputed_data(self, trnX, trnY, valX, valY, devX, devY, tstX, tstY,
                              trn_idx, val_idx, dev_idx, tst_idx):
        """
        Pipeline A (dict): impute(raw typed) -> encode+scale (safe, no NaNs into scaler)
        Pipeline B (str): encode(cats only + raw nums) -> impute(encoded) -> scale numeric block
        """
        method = self.imputation_method

        # ---------- Pipeline A: statistical imputers in raw space ----------
        if isinstance(method, dict):
            trnX_imp, valX_imp, devX_imp, tstX_imp = self.statistical_imputer(
                method, trnX, valX, devX, tstX, self.categorical_cols, self.numerical_cols, self.random_state
            )

            trnX_enc, valX_enc, devX_enc, tstX_enc, feature_names = \
                self.encode_categorical_scale_numerical(
                    trnX_imp, valX_imp, devX_imp, tstX_imp,
                    self.categorical_cols, self.numerical_cols,
                    return_preprocessor=False,
                    scale_numericals=True,                         # scale happens here for Pipeline A
                    missing_token=method.get("categorical", "__MISSING__"),
                    drop=None
                )

            print(f"    Imputed(all)-rawspace->encoded final shape - Train: {trnX_enc.shape}")
            return {
                "trnX": trnX_enc, "trnY": trnY, "trnX_idx": trn_idx,
                "valX": valX_enc, "valY": valY, "valX_idx": val_idx,
                "devX": devX_enc, "devY": devY, "devX_idx": dev_idx,
                "tstX": tstX_enc, "tstY": tstY, "tstX_idx": tst_idx,
                "feature_names": feature_names,
            }
        # ---------- Pipeline B: encoded-space imputers (HyperImpute plugins) ----------
        if isinstance(method, str):
            if not method.startswith("hyper_") and not method=="remasker":
                raise ValueError(f"Unknown string imputation_method: {method}")
            # Step 1: OHE cats + passthrough numericals (NaNs allowed), NO SCALING here
            trnX_enc, valX_enc, devX_enc, tstX_enc, feature_names, preprocessor, meta = \
                self.encode_categorical_scale_numerical(
                    trnX.copy(), valX.copy(), devX.copy(), tstX.copy(),
                    self.categorical_cols, self.numerical_cols,
                    return_preprocessor=True,
                    scale_numericals=False,        # key difference
                    missing_token="__MISSING__",
                    drop=None
                )
            if method.startswith("hyper_"):
                # Step 2: encoded-space imputation
                plugin = method.replace("hyper_", "")
                trnX_imp, valX_imp, devX_imp, tstX_imp = self._hyperimpute_on_encoded(
                    trnX_enc, valX_enc, devX_enc, tstX_enc,
                    plugin_name=plugin,
                    random_state=self.random_state
                )
            # Step 3: scale numeric block only (post-imputation)
            trnX_final, valX_final, devX_final, tstX_final = \
                self._scale_numeric_block_after_ohe(
                    trnX_imp, valX_imp, devX_imp, tstX_imp,
                    n_ohe=meta["n_ohe"], n_num=meta["n_num"]
                )
            print(f"    Imputed(all)-encoded->{method} final shape - Train: {trnX_final.shape}")
            return {
                "trnX": trnX_final, "trnY": trnY, "trnX_idx": trn_idx,
                "valX": valX_final, "valY": valY, "valX_idx": val_idx,
                "devX": devX_final, "devY": devY, "devX_idx": dev_idx,
                "tstX": tstX_final, "tstY": tstY, "tstX_idx": tst_idx,
                "feature_names": feature_names,
                "preprocessor": preprocessor,
                "preprocessor_metadata": meta,
            }

        raise ValueError(f"Unsupported imputation_method type: {type(method)}")


    @staticmethod
    def _hyperimpute_on_encoded(trnX_enc, valX_enc, devX_enc, tstX_enc,
                               plugin_name: str, random_state: int, **plugin_kwargs):
        """HyperImpute expects DataFrame-like; return numpy arrays (same shapes)."""
        cols = [f"f{i}" for i in range(trnX_enc.shape[1])]
        trn_df = pd.DataFrame(trnX_enc, columns=cols)
        val_df = pd.DataFrame(valX_enc, columns=cols)
        dev_df = pd.DataFrame(devX_enc, columns=cols)
        tst_df = pd.DataFrame(tstX_enc, columns=cols)

        imp = Imputers().get(plugin_name, random_state=random_state, **plugin_kwargs)
        imp.fit(trn_df)

        trn_out = imp.transform(trn_df)
        dev_out = imp.transform(dev_df)
        if val_df.isna().any().any():
            val_out = imp.transform(val_df)
        else:
            val_out = val_df
        if tst_df.isna().any().any():
            tst_out = imp.transform(tst_df)
        else:
            tst_out = tst_df

        trn_out = trn_out.to_numpy() if hasattr(trn_out, "to_numpy") else np.asarray(trn_out)
        val_out = val_out.to_numpy() if hasattr(val_out, "to_numpy") else np.asarray(val_out)
        dev_out = dev_out.to_numpy() if hasattr(dev_out, "to_numpy") else np.asarray(dev_out)
        tst_out = tst_out.to_numpy() if hasattr(tst_out, "to_numpy") else np.asarray(tst_out)

        return trn_out, val_out, dev_out, tst_out

    def separate_features_target(self):
        trnX = self.train_df.drop(columns=[self.target_col])
        trnY = self.train_df[self.target_col].values

        valX = self.val_df.drop(columns=[self.target_col])
        valY = self.val_df[self.target_col].values

        devX = self.dev_df.drop(columns=[self.target_col])
        devY = self.dev_df[self.target_col].values

        tstX = self.test_df.drop(columns=[self.target_col])
        tstY = self.test_df[self.target_col].values

        # print(f"Feature shapes - Train: {trnX.shape}, Val: {valX.shape}, Test: {tstX.shape}, Dev: {devX.shape}")
        return trnX, trnY, valX, valY, devX, devY, tstX, tstY

    @staticmethod
    def statistical_imputer(imputation_method, trnX, valX, devX, tstX, categorical_cols, numerical_cols, random_state):
        """Raw-space imputers. Ensures no NaNs in numeric cols after this step for Pipeline A."""
        trnX_imp = trnX.copy()
        valX_imp = valX.copy()
        devX_imp = devX.copy()
        tstX_imp = tstX.copy()

        # numerical
        if numerical_cols:
            num_method = (imputation_method.get("numerical") or "")

            if "mice" in num_method:
                imputer = IterativeImputer(max_iter=10, random_state=random_state)
                imputer.fit(trnX[numerical_cols])

                trnX_imp.loc[:, numerical_cols] = imputer.transform(trnX[numerical_cols])
                valX_imp.loc[:, numerical_cols] = imputer.transform(valX[numerical_cols])
                devX_imp.loc[:, numerical_cols] = imputer.transform(devX[numerical_cols])
                tstX_imp.loc[:, numerical_cols] = imputer.transform(tstX[numerical_cols])
            else:
                if "median" in num_method:
                    fill_vals = trnX[numerical_cols].median()
                else:
                    fill_vals = trnX[numerical_cols].mean()

                trnX_imp[numerical_cols] = trnX[numerical_cols].fillna(fill_vals)
                valX_imp[numerical_cols] = valX[numerical_cols].fillna(fill_vals)
                devX_imp[numerical_cols] = devX[numerical_cols].fillna(fill_vals)
                tstX_imp[numerical_cols] = tstX[numerical_cols].fillna(fill_vals)

        # categorical
        if categorical_cols:
            placeholder = imputation_method.get("categorical", "__MISSING__")
            trnX_imp[categorical_cols] = trnX[categorical_cols].fillna(placeholder)
            valX_imp[categorical_cols] = valX[categorical_cols].fillna(placeholder)
            devX_imp[categorical_cols] = devX[categorical_cols].fillna(placeholder)
            tstX_imp[categorical_cols] = tstX[categorical_cols].fillna(placeholder)

        return trnX_imp, valX_imp, devX_imp, tstX_imp

    @staticmethod
    def encode_categorical_scale_numerical(
        trnX, valX, devX, tstX,
        categorical_cols, numerical_cols,
        preprocessor=None,
        return_preprocessor=False,
        missing_token="__MISSING__",
        drop=None,
        scale_numericals=True,
    ):
        """
        - Always OHE categoricals (fit on train).
        - If scale_numericals=True: MinMax-scale numericals inside the preprocessor (Pipeline A / complete_samples).
        - If scale_numericals=False: numericals are passthrough (can contain NaNs), for encoded-space imputers (Pipeline B).
        """
        # make categoricals safe
        for df in (trnX, valX, devX, tstX):
            for col in categorical_cols:
                df[col] = df[col].astype("string").fillna(missing_token)

        if preprocessor is None:
            cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop)
            num_transformer = MinMaxScaler(feature_range=(0, 1)) if scale_numericals else "passthrough"

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", cat_transformer, categorical_cols),
                    ("num", num_transformer, numerical_cols),
                ],
                remainder="drop"
            )
            preprocessor.fit(trnX)

        trnX_enc = preprocessor.transform(trnX)
        valX_enc = preprocessor.transform(valX)
        devX_enc = preprocessor.transform(devX)
        tstX_enc = preprocessor.transform(tstX)

        feature_names = preprocessor.get_feature_names_out()

        if return_preprocessor:
            all_cols = trnX.columns.tolist()
            cat_indices = [all_cols.index(c) for c in categorical_cols]
            num_indices = [all_cols.index(c) for c in numerical_cols]

            n_num = len(numerical_cols)
            n_ohe = len(feature_names) - n_num  # because ("cat", ...) then ("num", ...)

            meta = {
                "categorical_cols": list(categorical_cols),
                "numerical_cols": list(numerical_cols),
                "cat_indices": cat_indices,
                "num_indices": num_indices,
                # "cat_vocab": cat_vocab,
                "all_cols": all_cols,
                "feature_names": feature_names,
                "missing_token": missing_token,
                "drop": drop,
                "scale_numericals": scale_numericals,
                "n_ohe": n_ohe,
                "n_num": n_num,
            }
            return trnX_enc, valX_enc, devX_enc, tstX_enc, feature_names, preprocessor, meta
        return trnX_enc, valX_enc, devX_enc, tstX_enc, feature_names

    @staticmethod
    def _scale_numeric_block_after_ohe(trnX, valX, devX, tstX, n_ohe, n_num):
        """
        Post-imputation scaling for Pipeline B (and reusable for other encoded-space benchmarks).
        Scales ONLY the numeric block [n_ohe : n_ohe+n_num], leaves OHE block untouched.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        s = slice(n_ohe, n_ohe + n_num)

        scaler.fit(trnX[:, s])

        trnX2 = trnX.copy()
        valX2 = valX.copy()
        devX2 = devX.copy()
        tstX2 = tstX.copy()

        trnX2[:, s] = scaler.transform(trnX[:, s])
        valX2[:, s] = scaler.transform(valX[:, s])
        devX2[:, s] = scaler.transform(devX[:, s])
        tstX2[:, s] = scaler.transform(tstX[:, s])

        return trnX2, valX2, devX2, tstX2

    def add_missing_indicators(self, trnX, valX, devX, tstX):
        trn_missing = trnX.isna().values.astype(int)
        val_missing = valX.isna().values.astype(int)
        dev_missing = devX.isna().values.astype(int)
        tst_missing = tstX.isna().values.astype(int)
        return trn_missing, val_missing, dev_missing, tst_missing

    @staticmethod
    def build_marginals_from_complete(complete_trainX_df, numerical_cols, categorical_cols):
        cols = list(numerical_cols) + list(categorical_cols)
        _ = cols  # (kept for readability; not strictly needed)

        marg = {"num": {}, "cat": {}}

        for c in numerical_cols:
            mu = float(complete_trainX_df[c].mean())
            sd = float(complete_trainX_df[c].std(ddof=1)) if complete_trainX_df[c].shape[0] > 1 else 0.0
            sd = max(sd, 1e-6)
            marg["num"][c] = {"mean": mu, "std": sd}

        for c in categorical_cols:
            vc = complete_trainX_df[c].value_counts(normalize=True, complete_samples=True)
            if vc.shape[0] == 0:
                marg["cat"][c] = {"cats": np.array(["-999"]), "probs": np.array([1.0])}
            else:
                marg["cat"][c] = {"cats": vc.index.to_numpy(), "probs": vc.values}

        return marg