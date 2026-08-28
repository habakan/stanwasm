data {
  int<lower=0> N;
  vector[N] x;
  array[N] int<lower=0,upper=1> y;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  y     ~ bernoulli_logit(alpha + beta * x);
}
