"""
LEVEL 2: Surrogate Model — The Steady-State Brain

Two implementations:
    A) InterpSurrogate  — Deterministic linear interpolation (baseline)
    B) MLPSurrogate     — Multi-output MLP regression (advanced)

Both share the SAME interface:
    surrogate.predict(heat_duty) -> {"XD_ss", "TTOP_ss", "TMID_ss", "TBOTTOM_ss"}

The rest of the pipeline (Env, RL, Evaluation) does NOT need to change.

════════════════════════════════════════════════════════════════
MLPSurrogate Training Protocol:
    1. 80/20 random train/test split (stratified by index shuffle)
    2. StandardScaler on input AND output (fitted on TRAIN only)
    3. MLP trained on TRAIN set with early stopping
    4. R² evaluated on TEST set, reported per output variable
    5. Post-prediction physical clamping (Route A)
════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.interpolate import interp1d


# ==================================================================
#  A) Interpolation Surrogate (Baseline — unchanged)
# ==================================================================

class InterpSurrogate:
    """
    Maps: HeatDuty -> (XD_ss, TTOP_ss, TMID_ss, TBOTTOM_ss)
    Implementation: Linear interpolation with edge clamping.
    """

    XD_UPPER_CLIP = 0.9999

    def __init__(self, df):
        heat = df["HeatDuty"].values
        sort_idx = np.argsort(heat)
        self.heat_sorted = heat[sort_idx]

        self.heat_min = float(self.heat_sorted.min())
        self.heat_max = float(self.heat_sorted.max())

        self._interp = {}
        for col in ["XD", "TTOP", "TMID", "TBOTTOM"]:
            values = df[col].values[sort_idx]
            self._interp[col] = interp1d(
                self.heat_sorted, values,
                kind="linear",
                bounds_error=False,
                fill_value=(values[0], values[-1])
            )

        print("[SURROGATE-INTERP] Deterministic interpolation model built")
        print(f"   Method: scipy.interp1d (linear, edge-clamped)")
        print(f"   Input range: HeatDuty in [{self.heat_min:.2f}, {self.heat_max:.2f}]")
        print(f"   Data points: {len(self.heat_sorted)}")

    def predict(self, heat_duty: float) -> dict:
        heat_duty = np.clip(heat_duty, self.heat_min, self.heat_max)
        xd = float(self._interp["XD"](heat_duty))
        xd = np.clip(xd, 0.0, self.XD_UPPER_CLIP)

        result = {
            "XD_ss": xd,
            "TTOP_ss": float(self._interp["TTOP"](heat_duty)),
            "TMID_ss": float(self._interp["TMID"](heat_duty)),
            "TBOTTOM_ss": float(self._interp["TBOTTOM"](heat_duty)),
        }

        for k, v in result.items():
            assert np.isfinite(v), f"[SURROGATE-INTERP] Non-finite: {k}={v}"

        return result


# ==================================================================
#  B) MLP Surrogate (Advanced — with proper train/test evaluation)
# ==================================================================

class MLPSurrogate:
    """
    Maps: HeatDuty -> (XD_ss, TTOP_ss, TMID_ss, TBOTTOM_ss)

    Training protocol:
        1. Data is split 80/20 RANDOMLY (shuffled, seeded)
        2. StandardScaler is fitted on TRAIN set ONLY
           (test set is transformed using train-fitted scaler)
        3. MLP is trained on TRAIN set with early stopping
           (early stopping uses an internal validation split
            carved from the 80% train portion)
        4. R² is computed on the HELD-OUT 20% TEST set
        5. R² is reported PER OUTPUT VARIABLE separately
        6. Post-prediction outputs are clamped to physical
           bounds derived from FULL dataset (Route A)

    For report / defense:
        "The MLP surrogate was trained on a random 80/20 split.
         Scalers were fitted exclusively on training data to prevent
         data leakage. R² was evaluated on the held-out test set
         for each output variable independently. Physical feasibility
         is enforced by clamping predictions to data-derived bounds."
    """

    XD_UPPER_CLIP = 0.9999

    def __init__(self, df,
                 hidden_layers=(64, 64),
                 activation="relu",
                 alpha=1e-4,
                 max_iter=2000,
                 random_state=42,
                 test_size=0.2,
                 bound_margin_pct=0.05):
        """
        Build, train, and evaluate the MLP surrogate.

        Args:
            df:                DataFrame with HeatDuty, XD, TTOP, TMID, TBOTTOM
            hidden_layers:     Tuple of hidden layer sizes
            activation:        'relu' or 'tanh'
            alpha:             L2 regularization strength
            max_iter:          Maximum training iterations
            random_state:      Seed for split AND MLP (reproducibility)
            test_size:         Fraction held out for testing (default 0.20)
            bound_margin_pct:  Margin added to data bounds for clamping
        """
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        # ── Store heat range ──
        self.heat_min = float(df["HeatDuty"].min())
        self.heat_max = float(df["HeatDuty"].max())

        # ── Output columns ──
        self._output_cols = ["XD", "TTOP", "TMID", "TBOTTOM"]

        # ── Physical bounds for clamping (from FULL dataset) ──
        self._bounds = {}
        for col in self._output_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            margin = (col_max - col_min) * bound_margin_pct
            if col == "XD":
                self._bounds[col] = (0.0, self.XD_UPPER_CLIP)
            else:
                self._bounds[col] = (col_min - margin, col_max + margin)

        # ── Prepare full arrays ──
        X_all = df[["HeatDuty"]].values.astype(np.float64)
        Y_all = df[self._output_cols].values.astype(np.float64)

        # ══════════════════════════════════════════════════════
        #  STEP 1: Random 80/20 split
        # ══════════════════════════════════════════════════════
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_all, Y_all,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        n_total = len(X_all)
        n_train = len(X_train)
        n_test = len(X_test)

        print("[SURROGATE-MLP] Data split:")
        print(f"   Total samples: {n_total}")
        print(f"   Train samples: {n_train} ({n_train/n_total*100:.0f}%)")
        print(f"   Test samples:  {n_test} ({n_test/n_total*100:.0f}%)")

        # ══════════════════════════════════════════════════════
        #  STEP 2: Scaling — fitted on TRAIN only
        # ══════════════════════════════════════════════════════
        self._scaler_X = StandardScaler()
        self._scaler_Y = StandardScaler()

        X_train_sc = self._scaler_X.fit_transform(X_train)
        Y_train_sc = self._scaler_Y.fit_transform(Y_train)

        X_test_sc = self._scaler_X.transform(X_test)
        # Y_test is NOT scaled — we compare in original units

        # ══════════════════════════════════════════════════════
        #  STEP 3: Train MLP on TRAIN set
        #  early_stopping carves a validation portion from
        #  the 80% train set (NOT from the 20% test set)
        # ══════════════════════════════════════════════════════
        self._mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver="adam",
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
            n_iter_no_change=50,
            tol=1e-6,
        )

        self._mlp.fit(X_train_sc, Y_train_sc)

        # ══════════════════════════════════════════════════════
        #  STEP 4: Evaluate on TEST set — R² per output
        # ══════════════════════════════════════════════════════
        Y_test_pred_sc = self._mlp.predict(X_test_sc)
        Y_test_pred = self._scaler_Y.inverse_transform(Y_test_pred_sc)

        # Also compute train metrics for comparison
        Y_train_pred_sc = self._mlp.predict(X_train_sc)
        Y_train_pred = self._scaler_Y.inverse_transform(Y_train_pred_sc)

        print("")
        print("[SURROGATE-MLP] Architecture: %s | Activation: %s | "
              "L2 alpha: %s" % (str(hidden_layers), activation, alpha))
        print("[SURROGATE-MLP] Training iterations: %d | "
              "Final loss: %.6f" % (self._mlp.n_iter_, self._mlp.loss_))
        print("")
        print("[SURROGATE-MLP] ═══ EVALUATION RESULTS ═══")
        print("")
        print("   %-10s │ %12s %12s │ %12s %12s │ %s" %
              ("Output", "Train RMSE", "Train R²",
               "Test RMSE", "Test R²", "Clamp Range"))
        print("   " + "─" * 85)

        self._test_r2 = {}
        self._test_rmse = {}
        self._train_r2 = {}
        self._train_rmse = {}

        for i, col in enumerate(self._output_cols):
            # Train metrics
            train_rmse = float(np.sqrt(np.mean(
                (Y_train[:, i] - Y_train_pred[:, i]) ** 2)))
            train_r2 = float(r2_score(Y_train[:, i], Y_train_pred[:, i]))

            # Test metrics
            test_rmse = float(np.sqrt(np.mean(
                (Y_test[:, i] - Y_test_pred[:, i]) ** 2)))
            test_r2 = float(r2_score(Y_test[:, i], Y_test_pred[:, i]))

            self._train_rmse[col] = train_rmse
            self._train_r2[col] = train_r2
            self._test_rmse[col] = test_rmse
            self._test_r2[col] = test_r2

            lo, hi = self._bounds[col]
            print("   %-10s │ %12.6f %12.4f │ %12.6f %12.4f │ [%.4f, %.4f]" %
                  (col, train_rmse, train_r2, test_rmse, test_r2, lo, hi))

        print("")

        # ── Overall summary ──
        avg_test_r2 = float(np.mean(list(self._test_r2.values())))
        print("   Average Test R²: %.4f" % avg_test_r2)

        # ── Warnings ──
        for col in self._output_cols:
            if self._test_r2[col] < 0.90:
                print("   ⚠ WARNING: %s Test R²=%.4f is below 0.90 — "
                      "model may be unreliable for this output!" %
                      (col, self._test_r2[col]))

        overfits = []
        for col in self._output_cols:
            gap = self._train_r2[col] - self._test_r2[col]
            if gap > 0.05:
                overfits.append((col, gap))
        if overfits:
            print("   ⚠ OVERFITTING WARNING:")
            for col, gap in overfits:
                print("      %s: Train R² - Test R² = %.4f" % (col, gap))

        print("")
        print("[SURROGATE-MLP] Input range: HeatDuty in "
              "[%.2f, %.2f]" % (self.heat_min, self.heat_max))
        print("[SURROGATE-MLP] Ready for inference")

        # ══════════════════════════════════════════════════════
        #  STEP 5: Retrain on FULL data for production use
        #  The test-set evaluation above is for REPORTING only.
        #  For the actual surrogate used in RL, we want maximum
        #  accuracy, so we retrain on all data.
        # ══════════════════════════════════════════════════════
        print("")
        print("[SURROGATE-MLP] Retraining on FULL dataset for "
              "production inference...")

        self._scaler_X = StandardScaler()
        self._scaler_Y = StandardScaler()

        X_all_sc = self._scaler_X.fit_transform(X_all)
        Y_all_sc = self._scaler_Y.fit_transform(Y_all)

        self._mlp_production = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver="adam",
            alpha=alpha,
            max_iter=max_iter,
            early_stopping=False,
            random_state=random_state,
            n_iter_no_change=50,
            tol=1e-6,
        )

        self._mlp_production.fit(X_all_sc, Y_all_sc)

        # Verify production model on full data
        Y_all_pred_sc = self._mlp_production.predict(X_all_sc)
        Y_all_pred = self._scaler_Y.inverse_transform(Y_all_pred_sc)

        print("[SURROGATE-MLP] Production model (full data) RMSE:")
        for i, col in enumerate(self._output_cols):
            prod_rmse = float(np.sqrt(np.mean(
                (Y_all[:, i] - Y_all_pred[:, i]) ** 2)))
            print("   %-10s  RMSE=%.6f" % (col, prod_rmse))

        print("[SURROGATE-MLP] Production model ready")

    def predict(self, heat_duty: float) -> dict:
        """
        Unified interface: HeatDuty -> steady-state outputs.
        Uses the production model (trained on full data).
        Clamps input and outputs to physical bounds (Route A).
        """
        heat_duty = np.clip(heat_duty, self.heat_min, self.heat_max)

        X_raw = np.array([[heat_duty]], dtype=np.float64)
        X_scaled = self._scaler_X.transform(X_raw)

        Y_scaled = self._mlp_production.predict(X_scaled)
        Y_pred = self._scaler_Y.inverse_transform(Y_scaled)[0]

        result = {}
        for i, col in enumerate(self._output_cols):
            lo, hi = self._bounds[col]
            val = float(np.clip(Y_pred[i], lo, hi))
            result[col + "_ss"] = val

        for k, v in result.items():
            assert np.isfinite(v), f"[SURROGATE-MLP] Non-finite: {k}={v}"

        return result

    def get_test_metrics(self) -> dict:
        """Return test-set evaluation metrics for reporting."""
        return {
            "test_r2": dict(self._test_r2),
            "test_rmse": dict(self._test_rmse),
            "train_r2": dict(self._train_r2),
            "train_rmse": dict(self._train_rmse),
        }


# ==================================================================
#  BACKWARD COMPATIBILITY
# ==================================================================
SurrogateModel = InterpSurrogate