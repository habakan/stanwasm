// Lognormal regression: log(y) ~ Normal(alpha + beta * x, sigma)
data {
  int<lower=0> N;
  vector[N] x;
  vector<lower=0>[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  sigma ~ normal(0, 2);
  for (i in 1:N) {
    y[i] ~ lognormal(alpha + beta * x[i], sigma);
  }
}
