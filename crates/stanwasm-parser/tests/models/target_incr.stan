// Demonstrates target += syntax: same as linear regression but using
// manual log-likelihood accumulation instead of ~ notation.
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ normal(0, 2);
  for (i in 1:N) {
    target += -log(sigma) - 0.5 * ((y[i] - alpha - beta * x[i]) / sigma)^2;
  }
}
