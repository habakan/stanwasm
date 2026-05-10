// Gamma GLM with log link: E[y_i] = exp(alpha + beta * x_i)
// shape fixed as data; rate = shape / E[y]
data {
  int<lower=0> N;
  vector[N] x;
  vector<lower=0>[N] y;
  real<lower=0> shape;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  for (i in 1:N) {
    y[i] ~ gamma(shape, shape / exp(alpha + beta * x[i]));
  }
}
