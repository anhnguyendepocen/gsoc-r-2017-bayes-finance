data {
  int N;                // sample size of X
  int M;                // sample size of the X_pred
  int K;                // #features
  vector[N] y;          // response
  matrix[N,K] X;        // model matrix for training
  matrix[M,K] X_pred;   // model matrix to be predicted
}
parameters {
  vector[K] beta;       // regression associate
  real sigma;           // random noise
}
transformed parameters {
  vector[N] mu;
  mu = X * beta;
}
model {
  // hyperparameters
  sigma ~ uniform(-10, 10);
  beta ~ normal(0, 10);
  // parameter
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[M] y_pred;
  y_pred = X_pred * beta;
}

// Reference: https://datascienceplus.com/bayesian-regression-with-stan-part-1-normal-regression/
