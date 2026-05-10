data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_tilde;
}
transformed parameters {
  vector[J] theta = mu + tau * theta_tilde;
}
model {
  mu          ~ normal(0, 10);
  tau         ~ normal(0, 1);   // half-normal via <lower=0> constraint
  theta_tilde ~ normal(0, 1);
  y           ~ normal(theta, sigma);
}
