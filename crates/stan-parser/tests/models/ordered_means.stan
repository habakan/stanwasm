// Ordered means model: estimate K ordered means from observed group means
data {
  int<lower=2> K;
  vector[K] y_bar;
  vector[K] se;
}
parameters {
  ordered[K] mu;
}
model {
  for (k in 1:K) {
    mu[k] ~ normal(0, 10);
    y_bar[k] ~ normal(mu[k], se[k]);
  }
}
