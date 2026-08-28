data {
  int<lower=0> N;
  vector[N] x;
  array[N] int<lower=0> y;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  for (i in 1:N) {
    y[i] ~ poisson_log(alpha + beta * x[i]);
  }
}
