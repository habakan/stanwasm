export interface Preset {
  name: string;
  description: string;
  stanCode: string;
  data: Record<string, number | number[]>;
  init: number[];
  /** Column names expected when uploading a CSV. */
  csvColumns: string[];
  /** Scalar field that should be set to the CSV row count (e.g. N, J). */
  rowCountScalar: string;
}

export const PRESETS: Record<string, Preset> = {
  linear_regression: {
    name: "Linear regression",
    description:
      "30 synthetic points along y ≈ 1.8x + 0.45 with small noise.",
    stanCode: `data {
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
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}`,
    data: {
      N: 30,
      x: Array.from({ length: 30 }, (_, i) => -1.5 + i * 0.1),
      y: Array.from({ length: 30 }, (_, i) => -1.3 + i * 0.18),
    },
    init: [0, 1, 0],
    csvColumns: ["x", "y"],
    rowCountScalar: "N",
  },

  poisson_regression: {
    name: "Poisson regression",
    description: "Counts that grow exponentially with x.",
    stanCode: `data {
  int<lower=0> N;
  vector[N] x;
  array[N] int y;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 1);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}`,
    data: {
      N: 5,
      x: [0, 1, 2, 3, 4],
      y: [1, 2, 5, 12, 30],
    },
    init: [0, 1],
    csvColumns: ["x", "y"],
    rowCountScalar: "N",
  },

  eight_schools: {
    name: "Eight schools (non-centered)",
    description:
      "Classic hierarchical model — does coaching affect SAT scores?",
    stanCode: `data {
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
  mu ~ normal(0, 5);
  tau ~ half_normal(5);
  theta_tilde ~ normal(0, 1);
  y ~ normal(theta, sigma);
}`,
    data: {
      J: 8,
      y: [28, 8, -3, 7, -1, 1, 18, 12],
      sigma: [15, 10, 16, 11, 9, 11, 10, 18],
    },
    init: Array(10).fill(0.1),
    csvColumns: ["y", "sigma"],
    rowCountScalar: "J",
  },
};
