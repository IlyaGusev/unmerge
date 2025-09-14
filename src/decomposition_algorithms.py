import torch
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.decomposition import SparseCoder
from scipy.optimize import nnls


class DecompositionAlgorithms:
    def __init__(self, dictionary_matrix):
        self.dictionary = dictionary_matrix.float()
        self.n_atoms, self.n_features = self.dictionary.shape

    def dot_product_similarity(self, target_vector):
        target_vector = target_vector.float()
        similarities = torch.matmul(self.dictionary, target_vector)
        coefficients = similarities / torch.norm(self.dictionary, dim=1)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def lasso_regression(self, target_vector, alpha=0.01, max_iter=1000):
        target_vector = target_vector.float()
        lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
        lasso.fit(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(lasso.coef_, dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def ridge_regression(self, target_vector, alpha=1.0):
        target_vector = target_vector.float()
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(ridge.coef_, dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def elastic_net_regression(
        self, target_vector, alpha=0.01, l1_ratio=0.5, max_iter=1000
    ):
        target_vector = target_vector.float()
        elastic_net = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=False
        )
        elastic_net.fit(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(elastic_net.coef_, dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def orthogonal_matching_pursuit(
        self, target_vector, n_nonzero_coefs=None, tol=1e-4
    ):
        target_vector = target_vector.float()
        if n_nonzero_coefs is None:
            n_nonzero_coefs = min(self.n_atoms, int(0.1 * self.n_atoms))

        coder = SparseCoder(
            dictionary=self.dictionary.T.numpy().transpose(),
            transform_algorithm="omp",
            transform_n_nonzero_coefs=n_nonzero_coefs,
            transform_alpha=tol,
        )
        coefficients = coder.transform(target_vector.numpy().reshape(1, -1))
        coefficients = torch.tensor(coefficients.flatten(), dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def non_negative_least_squares(self, target_vector):
        target_vector = target_vector.float()
        coefficients_np, _ = nnls(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(coefficients_np, dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def iterative_hard_thresholding(
        self, target_vector, sparsity_level=None, max_iter=100, step_size=0.001
    ):
        if sparsity_level is None:
            sparsity_level = min(self.n_atoms, int(0.1 * self.n_atoms))

        sparsity_level = max(1, min(sparsity_level, self.n_atoms))

        x = torch.zeros(self.n_atoms)
        best_x = x.clone()
        best_error = float("inf")

        for iteration in range(max_iter):
            residual = target_vector - torch.matmul(x, self.dictionary)

            if torch.norm(residual) < 1e-8:
                break

            gradient = torch.matmul(self.dictionary, residual)
            x_new = x + step_size * gradient

            x_new = torch.clamp(x_new, -1e6, 1e6)

            if torch.isnan(x_new).any() or torch.isinf(x_new).any():
                break

            if torch.sum(torch.abs(x_new) > 0) == 0:
                break

            _, indices = torch.topk(torch.abs(x_new), sparsity_level)
            x_thresholded = torch.zeros_like(x_new)
            x_thresholded[indices] = x_new[indices]

            current_error = torch.norm(
                target_vector - torch.matmul(x_thresholded, self.dictionary)
            )
            if current_error < best_error:
                best_error = current_error
                best_x = x_thresholded.clone()

            if torch.norm(x_thresholded - x) < 1e-8:
                break
            x = x_thresholded

        reconstruction = torch.matmul(best_x, self.dictionary)
        return best_x, reconstruction

    def compressive_sampling_matching_pursuit(
        self, target_vector, sparsity_level=None, max_iter=50
    ):
        if sparsity_level is None:
            sparsity_level = min(self.n_atoms, int(0.1 * self.n_atoms))

        sparsity_level = max(1, min(sparsity_level, self.n_atoms))

        x = torch.zeros(self.n_atoms)
        residual = target_vector.clone()

        for _ in range(max_iter):
            correlations = torch.abs(torch.matmul(self.dictionary, residual))

            k_select = min(2 * sparsity_level, self.n_atoms)
            _, new_indices = torch.topk(correlations, k_select)

            support = torch.unique(torch.cat([torch.nonzero(x).flatten(), new_indices]))

            if len(support) == 0:
                break

            A_support = self.dictionary[support]

            try:
                x_support = torch.linalg.lstsq(A_support.T, target_vector).solution
                x_temp = torch.zeros(self.n_atoms)
                x_temp[support] = x_support[: len(support)]

                if torch.sum(torch.abs(x_temp) > 0) >= sparsity_level:
                    _, final_indices = torch.topk(torch.abs(x_temp), sparsity_level)
                    x = torch.zeros(self.n_atoms)
                    x[final_indices] = x_temp[final_indices]
                else:
                    x = x_temp

                residual = target_vector - torch.matmul(x, self.dictionary)

                if torch.norm(residual) < 1e-6:
                    break
            except Exception:
                break

        reconstruction = torch.matmul(x, self.dictionary)
        return x, reconstruction

    def admm_lasso(self, target_vector, alpha=0.01, rho=1.0, max_iter=100):
        A = self.dictionary.T
        b = target_vector
        n = self.n_atoms

        x = torch.zeros(n)
        z = torch.zeros(n)
        u = torch.zeros(n)

        AtA = torch.matmul(A.T, A)
        Atb = torch.matmul(A.T, b)
        L = AtA + rho * torch.eye(n)

        try:
            L_inv = torch.inverse(L)
        except Exception:
            L_inv = torch.pinverse(L)

        for _ in range(max_iter):
            x = torch.matmul(L_inv, Atb + rho * (z - u))

            z_old = z.clone()
            z = self._soft_threshold(x + u, alpha / rho)

            u = u + x - z

            if torch.norm(x - z) < 1e-6 and torch.norm(z - z_old) < 1e-6:
                break

        reconstruction = torch.matmul(x, self.dictionary)
        return x, reconstruction

    def _soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.maximum(
            torch.abs(x) - threshold, torch.zeros_like(x)
        )
