
## Not run:
library(SuperLearner)
set.seed(1)
n = 100; p = 10

x = matrix(rnorm(n*p), n, p)
y = pmax(x[,1], 0) + x[,2] + pmin(x[,3], 0) + rnorm(n)

fit = SuperLearner(y, data.frame(x), SL.library = "SLxgboost")

est = predict(fit, x)

plot(est$pred, y, asp=1)
abline(a=0,b=1)
## End(Not run)
