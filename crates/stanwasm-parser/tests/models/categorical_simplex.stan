// Categorical likelihood with simplex parameter.
// Posterior is Dirichlet(alpha + counts) where counts[k] = sum(y==k).
data {
  int<lower=2> K;
  int<lower=0> N;
  array[N] int<lower=1,upper=K> y;
  vector[K] alpha;
}
parameters {
  simplex[K] theta;
}
model {
  // Dirichlet prior: theta ~ dirichlet(alpha)
  for (k in 1:K) {
    target += (alpha[k] - 1) * log(theta[k]);
  }
  // Categorical likelihood
  for (i in 1:N) {
    y[i] ~ categorical(theta);
  }
}
