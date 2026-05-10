// multivariate_lkj.stan
//
// K次元多変量正規分布 with LKJ相関事前分布
//
// 用途: 多変量データの平均と相関構造を同時推定する基本モデル
//
// パラメータ:
//   mu[K]  : 平均ベクトル (各成分に N(0, 5) 事前分布)
//   L[K,K] : 相関行列のCholesky因子 (LKJ(2) 事前分布 → 対角に近い構造を選好)
//
// 尤度: y ~ multi_normal_cholesky(mu, L)
//   (スケールパラメータなし: Lは相関行列のCholesky因子)
//
// 使い方:
//   data { K=2, y=[y1, y2] }   ← Kは次元数、yは観測ベクトル（1観測）

data {
  int<lower=1> K;
  vector[K] y;
}
parameters {
  vector[K] mu;
  cholesky_factor_corr[K] L;
}
model {
  mu ~ normal(0, 5);
  L  ~ lkj_corr_cholesky(2.0);
  y  ~ multi_normal_cholesky(mu, L);
}
