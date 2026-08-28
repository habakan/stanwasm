// Piecewise constant regression: tests if/else in model block.
// y[i] ~ normal(mu_high, sigma) if x[i] > 0, else normal(mu_low, sigma)
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real mu_low;
  real mu_high;
  real<lower=0> sigma;
}
model {
  mu_low  ~ normal(0, 10);
  mu_high ~ normal(0, 10);
  sigma   ~ normal(0, 2);
  for (i in 1:N) {
    if (x[i] > 0) {
      y[i] ~ normal(mu_high, sigma);
    } else {
      y[i] ~ normal(mu_low, sigma);
    }
  }
}
