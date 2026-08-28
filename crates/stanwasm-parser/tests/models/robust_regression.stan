// Robust linear regression with Student-t likelihood (nu=3 fixed)
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
    y[i] ~ student_t(3, alpha + beta * x[i], sigma);
  }
}
