data {
  int<lower=0> N;
  vector[N] x;
  array[N] int<lower=0> y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> phi;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  phi   ~ gamma(2, 1);
  for (i in 1:N) {
    y[i] ~ neg_binomial_2(exp(alpha + beta * x[i]), phi);
  }
}
