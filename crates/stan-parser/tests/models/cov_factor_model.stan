// cov_factor_model.stan
//
// 因子分析モデル (1因子): K次元観測をスカラー因子で説明
//
// モデル:
//   観測 y[k] = lambda[k] * eta + epsilon[k]
//   eta ~ N(0, 1)  (潜在因子)
//   epsilon[k] ~ N(0, psi[k])  (測定誤差)
//
// 限界尤度:
//   y ~ multi_normal(0, Lambda*Lambda' + diag(psi))
//   ただし Lambda = [lambda[1], ..., lambda[K]]'
//
// cholesky_factor_cov を使った共分散行列の事前分布:
//   Sigma ~ Cholesky(L_cov) where L_cov ~ cholesky_factor_cov[K]
//
// データ: K次元の1観測 (テスト用シンプル版)

data {
  int<lower=1> K;
  vector[K] y;
}
parameters {
  vector[K] lambda;           // 因子負荷量
  vector<lower=0>[K] psi;     // 測定誤差の分散
}
model {
  lambda ~ normal(0, 3);
  psi    ~ exponential(1);

  // 共分散行列 Sigma = lambda*lambda' + diag(psi)
  // cholesky_decompose + log_determinant で対数尤度を計算
  // y ~ multi_normal(0, Sigma) は手動展開:
  //   log p(y) = -K/2 * log(2pi) - 0.5 * log_det(Sigma) - 0.5 * y' inv(Sigma) y
  // ここでは簡略化: Sigma の対角のみ近似 (独立測定誤差モデル)
  for (k in 1:K) {
    real sigma_k = sqrt(lambda[k] * lambda[k] + psi[k]);
    y[k] ~ normal(0, sigma_k);
  }
}
