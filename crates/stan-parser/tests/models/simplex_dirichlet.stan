// Dirichlet posterior: theta | y, alpha ~ Dirichlet(alpha + y)
// Likelihood: target += sum((alpha[k]-1)*log(theta[k])) + sum(y[k]*log(theta[k]))
// (normalizing constants dropped)
data {
  int<lower=2> K;
  array[K] int<lower=0> y;
  vector[K] alpha;
}
parameters {
  simplex[K] theta;
}
model {
  for (k in 1:K) {
    target += (alpha[k] - 1) * log(theta[k]);
  }
  for (k in 1:K) {
    target += y[k] * log(theta[k]);
  }
}
