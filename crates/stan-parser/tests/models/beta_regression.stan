// Beta regression for proportions in (0,1) with logit link
// mu_i = inv_logit(alpha + beta * x_i), phi = precision
data {
  int<lower=0> N;
  vector[N] x;
  vector<lower=0,upper=1>[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> phi;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 5);
  phi   ~ gamma(2, 0.5);
  for (i in 1:N) {
    real mu = inv_logit(alpha + beta * x[i]);
    y[i] ~ beta(mu * phi, (1 - mu) * phi);
  }
}
