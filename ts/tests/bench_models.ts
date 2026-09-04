// The models the gradient benchmark runs, as plain Stan plus JSON data, so the
// same definitions can be handed to any other implementation.
//
// Each model is evaluated at a fixed point in the *unconstrained* space, which
// is the space `logProbGrad` takes and the one another implementation's
// log_prob method takes too — so the log density and the gradient at that
// point are comparable, not just the time to compute them.

export type BenchModel = {
  name: string;
  src: string;
  data: Record<string, unknown>;
  /// Unconstrained parameter values to evaluate at; length must be n_params.
  init: number[];
};

/// Deterministic and the same for every model, so a run is reproducible.
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/// Moderate values in every direction: small enough that `exp` of one stays
/// finite, large enough that no gradient is zero by accident.
const at = (n: number): number[] =>
  Array.from({ length: n }, (_, i) => 0.2 * Math.sin(i + 1));

export function benchModels(n: number): BenchModel[] {
  const r = rng(20260903);
  const x = Array.from({ length: n }, (_, i) => -1.5 + (3 * i) / n);
  const noise = Array.from({ length: n }, () => r() - 0.5);
  const y = x.map((v, i) => 1.0 + 1.8 * v + 0.4 * noise[i]);
  const counts = x.map((v, i) => Math.max(0, Math.round(Math.exp(0.5 + 0.4 * v) + noise[i])));
  const bits = y.map((v) => (v > 1.0 ? 1 : 0));
  const g = Array.from({ length: n }, () => 1 + Math.floor(r() * 8));
  const h = Array.from({ length: n }, () => 1 + Math.floor(r() * 3));
  const mat = (k: number) =>
    Array.from({ length: n }, () => Array.from({ length: k }, () => r() * 2 - 1));

  const models: BenchModel[] = [];
  const add = (name: string, src: string, data: Record<string, unknown>, np: number) =>
    models.push({ name, src, data, init: at(np) });

  add(
    "linreg",
    `data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; }
model {
  alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}`,
    { N: n, x, y },
    3,
  );

  add(
    "logistic",
    `data { int<lower=0> N; vector[N] x; array[N] int<lower=0,upper=1> y; }
parameters { real alpha; real beta; }
model {
  alpha ~ normal(0, 5); beta ~ normal(0, 5);
  y ~ bernoulli_logit(alpha + beta * x);
}`,
    { N: n, x, y: bits },
    2,
  );

  add(
    "poisson",
    `data { int<lower=0> N; vector[N] x; array[N] int<lower=0> y; }
parameters { real alpha; real beta; }
model {
  alpha ~ normal(0, 5); beta ~ normal(0, 5);
  for (i in 1:N) y[i] ~ poisson(exp(alpha + beta * x[i]));
}`,
    { N: n, x, y: counts },
    2,
  );

  add(
    "neg_binomial",
    `data { int<lower=0> N; vector[N] x; array[N] int<lower=0> y; }
parameters { real alpha; real beta; real<lower=0> phi; }
model {
  alpha ~ normal(0, 5); beta ~ normal(0, 5); phi ~ exponential(1);
  y ~ neg_binomial_2(exp(alpha + beta * x), phi);
}`,
    { N: n, x, y: counts },
    3,
  );

  add(
    "student_t",
    `data { int<lower=0> N; vector[N] x; vector[N] y; }
parameters { real alpha; real beta; real<lower=0> sigma; real<lower=1> nu; }
model {
  alpha ~ normal(0, 10); beta ~ normal(0, 10);
  sigma ~ exponential(1); nu ~ exponential(1);
  y ~ student_t(nu, alpha + beta * x, sigma);
}`,
    { N: n, x, y },
    4,
  );

  add(
    "gather",
    `data { int<lower=0> N; int<lower=1> G; array[N] int<lower=1> g; vector[N] y; }
parameters { vector[G] mu; real<lower=0> sigma; }
model {
  mu ~ normal(0, 5); sigma ~ exponential(1);
  for (i in 1:N) y[i] ~ normal(mu[g[i]], sigma);
  // the same gather written as one array index, which is its own code path
  target += -0.5 * dot_self(y - mu[g]);
}`,
    { N: n, G: 8, g, y },
    9,
  );

  add(
    "two_level",
    `data {
  int<lower=0> N; int<lower=1> G; int<lower=1> H;
  array[N] int<lower=1> g; array[N] int<lower=1> h; vector[N] y;
}
parameters { vector[G] mu; vector[H] delta; real<lower=0> sigma; }
model {
  mu ~ normal(0, 5); delta ~ normal(0, 2); sigma ~ exponential(1);
  for (i in 1:N) y[i] ~ normal(mu[g[i]] + delta[h[i]], sigma);
}`,
    { N: n, G: 8, H: 3, g, h, y },
    12,
  );

  for (const k of [4, 16]) {
    add(
      `matrix_k${k}`,
      `data { int<lower=0> N; int<lower=0> K; matrix[N,K] X; vector[N] y; }
parameters { vector[K] beta; real<lower=0> sigma; }
model {
  beta ~ normal(0, 1); sigma ~ exponential(1);
  y ~ normal(X * beta, sigma);
}`,
      { N: n, K: k, X: mat(k), y },
      k + 1,
    );
  }

  // Small on purpose: the shapes it exercises are the constraint transforms and
  // the multivariate density, not the length of a vectorised loop.
  //
  // The loop form is the only one this runtime takes — an array of vectors is a
  // load-time error pointing at it — and it is also the form that flatters a
  // trace-once implementation most, since the per-observation setup is
  // identical at every observation and recorded once. Read its timing as the
  // best case for that advantage, not as a like-for-like density comparison.
  const d = 4;
  const rows = Array.from({ length: 400 }, () =>
    Array.from({ length: d }, () => r() * 2 - 1),
  );
  add(
    "mvn_cholesky",
    `data { int<lower=0> M; int<lower=0> D; array[M] vector[D] y; }
parameters { vector[D] mu; cholesky_factor_corr[D] L; real<lower=0> tau; }
model {
  mu ~ normal(0, 5); L ~ lkj_corr_cholesky(2); tau ~ exponential(1);
  for (m in 1:M) y[m] ~ multi_normal_cholesky(mu, L);
}`,
    { M: rows.length, D: d, y: rows },
    d + (d * (d - 1)) / 2 + 1,
  );

  add(
    "eight_schools",
    `data { int<lower=0> J; vector[J] y; vector<lower=0>[J] sigma; }
parameters { real mu; real<lower=0> tau; vector[J] theta_raw; }
model {
  mu ~ normal(0, 5); tau ~ normal(0, 5); theta_raw ~ std_normal();
  y ~ normal(mu + tau * theta_raw, sigma);
}`,
    {
      J: 8,
      y: [28, 8, -3, 7, -1, 1, 18, 12],
      sigma: [15, 10, 16, 11, 9, 11, 10, 18],
    },
    10,
  );

  // Not a shape anyone would fit, but every distribution and function added
  // for the posteriordb sweep is in one of these two, so CmdStan checks each
  // formula rather than only that it runs.
  add(
    "binomial",
    `data { int<lower=0> N; array[N] int<lower=0> trials; array[N] int<lower=0> hits; array[N] int<lower=0,upper=1> flags; vector[N] x; }
parameters { real<lower=0, upper=1> theta; real alpha; real beta; }
model {
  theta ~ uniform(0, 1);
  alpha ~ normal(0, 5); beta ~ normal(0, 5);
  hits ~ binomial(trials, theta);
  flags ~ bernoulli(theta);
  target += binomial_logit_lpmf(hits | trials, alpha + beta * x);
}`,
    { N: n, trials: counts.map((c) => c + 5), hits: counts, flags: bits, x },
    3,
  );

  add(
    "count_mix",
    `data { int<lower=0> N; vector[N] x; array[N] int<lower=0> y; array[N] int<lower=1,upper=3> label; }
parameters { real alpha; real beta; real<lower=0> sigma; real<lower=0, upper=1> lambda; vector[3] gamma; }
model {
  sigma ~ inv_gamma(3, 2);
  lambda ~ uniform(0, 1);
  alpha ~ normal(0, sigma); beta ~ normal(0, 5);
  gamma ~ normal(0, 2);
  y ~ poisson_log(alpha + beta * x);
  label ~ categorical_logit(gamma);
  target += log_mix(lambda, log_sum_exp(x) * alpha, log10(sigma));
  target += sd(x) * beta + dot_self(gamma);
}`,
    { N: n, x, y: counts, label: h },
    7,
  );

  add(
    "cov_builders",
    `data { int<lower=0> M; int<lower=0> D; array[M] vector[D] y; vector[D] mu; array[D] real t; }
parameters { cholesky_factor_corr[D] L; vector<lower=0>[D] tau; real<lower=0> rho; }
model {
  L ~ lkj_corr_cholesky(2); tau ~ exponential(1); rho ~ exponential(1);
  matrix[D, D] S = diag_pre_multiply(tau, L);
  for (m in 1:M) y[m] ~ multi_normal_cholesky(mu, S);
  target += sum(gp_exp_quad_cov(t, tau[1], rho)[1]);
  target += sum(quad_form_diag(multiply_lower_tri_self_transpose(L), tau)[2]);
  target += sum(softmax(cumulative_sum(tau))) * (min(t) + max(t));
}`,
    {
      M: rows.length, D: d, y: rows,
      mu: Array.from({ length: d }, (_, i) => 0.3 * i),
      t: Array.from({ length: d }, (_, i) => 0.3 * i),
    },
    (d * (d - 1)) / 2 + d + 1,
  );

  // The GLM form and the two densities added with it: CmdStan checks the
  // formulas, not only that they run.
  add(
    "glm",
    `data { int<lower=0> N; int<lower=0> K; matrix[N, K] xm; array[N] int<lower=0,upper=1> y; vector[N] yr; }
parameters { real alpha; vector[K] beta; real<lower=0> s; }
model {
  alpha ~ logistic(0, 1);
  beta ~ double_exponential(0, 2);
  s ~ exponential(1);
  y ~ bernoulli_logit_glm(xm, alpha, beta);
  yr ~ normal_id_glm(xm, alpha, beta, s);
}`,
    { N: n, K: 4, xm: mat(4), y: bits, yr: y },
    6,
  );

  add(
    "funnel",
    `data { int<lower=0> D; }
parameters { real v; vector[D] z; }
model {
  v ~ normal(0, 3);
  z ~ normal(0, exp(v / 2));
}`,
    { D: 9 },
    10,
  );

  return models;
}
