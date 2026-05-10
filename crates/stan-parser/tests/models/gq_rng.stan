// Test model for generated quantities RNG functions.
data {
  int<lower=0> N;
  vector[N] x;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu    ~ normal(0, 5);
  sigma ~ exponential(1);
  for (i in 1:N) {
    y[i] ~ normal(mu, sigma);
  }
}
generated quantities {
  real y_ln  = lognormal_rng(mu, sigma);
  real y_exp = exponential_rng(1.0);
  real y_unif = uniform_rng(0.0, 1.0);
  real y_gam = gamma_rng(2.0, 1.0);
}
