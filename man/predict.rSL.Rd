% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rSL.R
\name{predict.rSL}
\alias{predict.rSL}
\title{Predict for \code{rSL} objects}
\usage{
\method{predict}{rSL}(object, newdata)
}
\arguments{
\item{object}{An \code{rSL} objects.}

\item{newdata}{A matrix of new values for the predictors.}
}
\value{
A list with predicted values from Super Learner fit in \code{pred} and predicted values from each algorithm in its library in \code{library.predict}.
}
\description{
Predict for \code{rSL} objects
}
\examples{
library(SuperLearner)
library(rlearner)

SL.lib <- "SL.glmnet"
n = 100; p = 10

set.seed(1234)
X = matrix(runif(n*p,-3,3), n, p)
A = rbinom(n, 1, 0.5)
Y = pmax(X[,1], 0) * A + X[,2] + pmin(X[,3], 0) + rnorm(n)

rSL_fit <- rSL(X=X, A=A, Y=Y, SL_Y = SL.lib, SL_A = SL.lib, SL_CATE = SL.lib,
               cvControl = list(V=10))
rSL_pred <- predict(rSL_fit, X)$pred

rlasso_fit <- rlasso(X, A, Y, k_folds = 10)
rlasso_pred <- predict(rlasso_fit, X)

plot(Y-rSL_pred, Y-rlasso_pred, col = 2, asp=1,
     main = "R-learner residuals",
     xlab = "rSL, 'SL.glmnet'", ylab = "rlasso")
abline(a = 0, b = 1)
legend(x = "bottomright", legend = c("Residual", "y = x line"), pch = c(1, NA),
       col = 2:1, lty = c(NA, 1))
}
