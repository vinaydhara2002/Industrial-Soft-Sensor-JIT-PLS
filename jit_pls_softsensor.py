import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


class JITPLSSoftSensor:
    def __init__(self, n_neighbors=20, max_degree=3):
        self.n_neighbors = n_neighbors
        self.max_degree = max_degree

    # -----------------------------
    # Utility: convert to numpy
    # -----------------------------
    def _to_numpy(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.values
        return X

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(self, X, y):
        # Store column names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = None

        X = self._to_numpy(X)
        y = self._to_numpy(y).reshape(-1, 1)

        self.X = X
        self.y = y

    # -----------------------------
    # Polynomial transformation
    # -----------------------------
    def _transform_polynomial(self, X, X_query, degree):
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        X_poly = poly.fit_transform(X)
        Xq_poly = poly.transform(X_query)

        return X_poly, Xq_poly

    # -----------------------------
    # Mahalanobis distance
    # -----------------------------
    def _mahalanobis_distance(self, Xs, x_q, cov_inv):
        diff = Xs - x_q
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    # -----------------------------
    # Compute weights
    # -----------------------------
    def _compute_weights(self, distances, sigma):
        return np.exp(-(distances**2) / (2 * sigma**2))

    # -----------------------------
    # Select neighbors
    # -----------------------------
    def _select_neighbors(self, distances):
        idx = np.argsort(distances)[:self.n_neighbors]
        return idx

    # -----------------------------
    # Select optimal components
    # -----------------------------
    def _select_optimal_components(self, X_local, y_local, weights, degree):
        if np.std(y_local) < 1e-6:
          return None

        best_r2 = -np.inf
        best_model = None

        max_comp_formula = int((X_local.shape[1] + 1) * 7 / (7 + degree))
        max_allowed = X_local.shape[0] - 1
        max_comp = min(max_comp_formula, max_allowed)
        for n_comp in range(1, max_comp):
            pls = PLSRegression(n_components=n_comp)

            # Weighted PLS trick
            Xw = X_local * np.sqrt(weights[:, None])
            yw = y_local * np.sqrt(weights[:, None])

            pls.fit(Xw, yw)

            # y_pred_local = pls.predict(X_local)
            # r2 = r2_score(y_local, y_pred_local)
            yw_pred = pls.predict(Xw)
            r2 = r2_score(yw, yw_pred)

            if r2 > best_r2 + 1e-2:
                best_r2 = r2
                best_model = pls
            else:
                break

        return best_model

    # -----------------------------
    # JIT prediction (batch)
    # -----------------------------
    def _jit_predict_batch(self, Xs, y, Xq_s, degree):
        cov = np.cov(Xs, rowvar=False) + 1e-6 * np.eye(Xs.shape[1])
        cov_inv = np.linalg.pinv(cov)

        predictions = []

        for x_q in Xq_s:

            # Distance
            distances = self._mahalanobis_distance(Xs, x_q, cov_inv)

            #sigma = np.std(distances) + 1e-8
            sigma = max(np.median(distances), 1e-3)

            # Neighbors
            idx = self._select_neighbors(distances)

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_local = scaler_X.fit_transform(Xs[idx])
            x_q = scaler_X.transform(x_q.reshape(1, -1))
            y_local = scaler_y.fit_transform(y[idx])
            d_local = distances[idx]

            # Weights
            weights = self._compute_weights(d_local, sigma)

            # Model selection
            best_model = self._select_optimal_components(X_local, y_local, weights, degree)

            if best_model is None:
              # fallback: weighted average (VERY IMPORTANT)
              y_pred = np.sum(weights * y_local.ravel()) / (np.sum(weights) + 1e-8)
            else:
              y_pred = best_model.predict(x_q.reshape(1, -1))[0, 0]

            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

            predictions.append(y_pred)

        return np.array(predictions)

    # -----------------------------
    # Public: Predict with degrees
    # -----------------------------
    def predict(self, X_query):
        # Convert input
        X_query = self._to_numpy(X_query)

        # Handle single sample
        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)

        results = {}

        for degree in range(1, self.max_degree + 1):

            # Transform once per degree
            Xs, Xq_s = self._transform_polynomial(self.X, X_query, degree)

            # JIT prediction
            y_pred = self._jit_predict_batch(Xs, self.y, Xq_s, degree)

            results[degree] = y_pred

        return results


def plot_scatter(ax, y_real, y_pred):

    all_y = np.concatenate([y_real, y_pred])
    min_y, max_y = all_y.min(), all_y.max()

    # Results
    ax.scatter(y_real, y_pred, alpha=0.4, edgecolors='black', lw=0.5)
    ax.plot([min_y, max_y], [min_y, max_y], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Scatter Plot\nActual vs Predicted", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_line(ax, y_real, y_pred, window_size):

    time = [i+1+window_size for i in range(len(y_real))]

    # Results
    ax.plot(time, y_real, label='Actual', color='blue', lw=0.5)
    ax.plot(time, y_pred, label='Predicted', color='red', lw=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Butane Concentration")
    ax.set_title(f"Time Series Comparison", fontsize=13)
    ax.legend()
    ax.grid(True)

def plot_error(ax, y_real, y_pred):

    errors = y_real - y_pred
    ax.hist(errors, bins=30, alpha=0.7, edgecolor='black', lw=0.5)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Error Distribution", fontsize=12)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":

    df = pd.read_csv("debutanizer_data(1).txt", sep=r"\s+", header=None)

    df.columns = [f"x{i}" for i in range(1, 8)] + ["y"]

    X = df[[f"x{i}" for i in range(1, 8)]]
    y = df[["y"]]

    X = X.values
    y = y.values

    print("\n"+"*"*48+"  Rolling Window JIT Predicting  "+"*"*48+"\n")

    window_size = 200
    points = 50
    degree = 2
    # l = 300
    l = len(X) - window_size

    reals_all = []

    model = JITPLSSoftSensor(
        n_neighbors=points,
        max_degree=degree
    )

    results = {}

    for i in range(1, degree+1):
      results[i] = []

    r2 = {}

    rmse = {}

    for t in range(window_size, len(X)):

        X_db = X[t - window_size:t]
        y_db = y[t - window_size:t]

        X_test = X[t].reshape(1, -1)
        y_true = y[t][0]

        model.fit(X_db, y_db)

        result = model.predict(X_test)

        for i in range(1, degree+1):
          results[i] += result[i][0].tolist()

        reals_all.append(y_true)

    print("*"*48+" "*12+"Predicted"+" "*12+"*"*48)

s = 5*2
actual_all = reals_all
print("Actual vs Smoothed Output Data R²:", round(r2_score(reals_all, actual_all), 4))
print("\nNo. Of Data Points Predicted:", len(reals_all), "\n")

for i in range(1, degree+1):
      pred_all = uniform_filter1d(results[i], size=s+10, mode='nearest')
      # pred_all = results[i]
      r2[i] = round(r2_score(actual_all, pred_all), 4)
      rmse[i] = round(np.sqrt(mean_squared_error(actual_all, pred_all)), 4)
      print(f"\nDegree {i}:")
      print(f"R²: {r2[i]}")
      print(f"RMSE: {rmse[i]}\n")

      fig = plt.figure(figsize=(12, 10))
      gs = fig.add_gridspec(2, 2)
      ax00 = fig.add_subplot(gs[0, 0])
      plot_scatter(ax00, actual_all, pred_all)
      ax01 = fig.add_subplot(gs[0, 1])
      plot_error(ax01, actual_all, pred_all)
      ax10 = fig.add_subplot(gs[1, :])
      plot_line(ax10, actual_all[:l], pred_all[:l], window_size)
      fig.suptitle(f"Rolling Window JIT (Degree = {i})", fontsize=20, fontweight='bold')
      fig.tight_layout(rect=[0, 0, 1, 0.99])
      fig.savefig(f"Rolling_Window_JIT_Degree_{i}.png", dpi=1200)
      plt.close(fig)