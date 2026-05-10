// corr_random_effects.stan
//
// 階層ベイズ線形回帰: グループごとの切片と傾きが相関する変量効果
//
// モデル:
//   y[j] ~ normal(alpha[j] + beta[j] * x[j], sigma)
//
//   切片・傾きの階層事前分布 (非心化パラメータ化):
//     alpha[j] = mu_alpha + tau[1] * z1[j]
//     beta[j]  = mu_beta  + tau[2] * (L[2,1]*z1[j] + L[2,2]*z2[j])
//
//   L は2次元相関行列のCholesky因子 (LKJ(2) 事前分布)
//   L[2,1] = rho (相関), L[2,2] = sqrt(1-rho^2)
//
// 非心化パラメータ化の利点:
//   - z1, z2 ~ N(0,1) は観測データとほぼ独立 → MCMCの収束が速い
//   - パラメータ空間がシンプル (funnel問題を回避)

data {
  int<lower=0> J;     // グループ数
  vector[J] y;        // 応答変数
  vector[J] x;        // 説明変数
}
parameters {
  real mu_alpha;               // 切片の平均
  real mu_beta;                // 傾きの平均
  real<lower=0> sigma;         // 残差標準偏差
  vector<lower=0>[2] tau;      // 切片・傾きのスケール
  cholesky_factor_corr[2] L;   // 切片-傾き相関のCholesky因子
  vector[J] z1;                // 切片成分の標準正規偏差
  vector[J] z2;                // 傾き成分の標準正規偏差
}
model {
  // 超事前分布
  mu_alpha ~ normal(0, 10);
  mu_beta  ~ normal(0, 10);
  sigma    ~ exponential(1);
  tau      ~ exponential(1);
  L        ~ lkj_corr_cholesky(2.0);

  // 非心化標準正規偏差 (これが効率的な階層構造を実現する)
  z1 ~ normal(0, 1);
  z2 ~ normal(0, 1);

  // グループごとの尤度
  // L[1,1]=1 (相関行列Choleskyの(1,1)要素は常に1)
  // L[2,1]=rho, L[2,2]=sqrt(1-rho^2)
  for (j in 1:J) {
    real alpha_j = mu_alpha + tau[1] * z1[j];
    real beta_j  = mu_beta  + tau[2] * (L[2,1] * z1[j] + L[2,2] * z2[j]);
    y[j] ~ normal(alpha_j + beta_j * x[j], sigma);
  }
}
