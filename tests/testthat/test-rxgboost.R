
## Not run:
library(SuperLearner)
set.seed(1)
n = 100; p = 5

x = matrix(rnorm(n*p), n, p)
y = pmax(x[,1], 0) + x[,2] + pmin(x[,3], 0) + rnorm(n)

# fit = SuperLearner(y, data.frame(x), SL.library = "r.xgboost",
                   # cvControl = list(V=3L))
# est = predict(fit, x)

fit = r.xgboost(y, data.frame(x), newX = data.frame(x),
                family = gaussian(), obsWeights = NULL)
est = fit$pred


## End(Not run)
