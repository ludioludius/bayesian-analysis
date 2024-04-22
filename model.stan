data {
	int<lower=0> N;
	int<lower=0> N_test;
	vector[N] age;
	vector[N] salary;
	int<lower=0, upper=1> ys[N];
	vector[N_test] x_pred_age;
	vector[N_test] x_pred_salary;
	int<lower=0, upper=1> y_test[N_test]; // Actual outcomes for the test data
}

parameters {
	real slope_age;
	real slope_salary;
	real intercept;
}

model {
	// prior
	slope_age ~ normal(0.1400688, 0.001);
	slope_salary ~ normal(0.00002153042, 0.000001);
	intercept ~ normal(-7.775312, 0.1);
	
	for (n in 1:N) {
  		ys[n] ~ bernoulli_logit(intercept + slope_age * age[n] + slope_salary * salary[n]);
	}
}

generated quantities {
  vector[N_test] y_pred;
  real test_accuracy = 0.0;
  
  for (i in 1:N_test) {
    y_pred[i] = bernoulli_logit_rng(intercept + slope_age * x_pred_age[i] + slope_salary * x_pred_salary[i]);
    // Compute test accuracy
    if (y_pred[i] == y_test[i]) {
      test_accuracy += 1.0 / N_test;
    }
  }
}