data {
  int<lower=0> N;
  int<lower=1> K;
  vector[N] y;
}
parameters {
  ordered[K] mu;
  vector<lower=0>[K] sigma;
  simplex[K] theta;
}
model {
  for (k in 1:K) {
    mu[k] ~ normal(0, 10);
    sigma[k] ~ normal(0, 2);
  }
  theta ~ dirichlet(rep_vector(1.0, K));
  for (i in 1:N) {
    array[K] real log_lik;
    for (k in 1:K) {
      log_lik[k] = log(theta[k]) + normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    target += log_sum_exp(log_lik);
  }
}
