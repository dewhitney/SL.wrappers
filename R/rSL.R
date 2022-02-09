
#' R-learner, implemented via SuperLearner
#'
#' @param X The predictor variables in the training data set, usually a data.frame.
#' @param A The exposure in the training data set. Must be a numeric vector.
#' @param Y The outcome in the training data set. Must be a numeric vector.
#' @param Y_family Either gaussian (default) or binomial.
#' @param A_family Either binomial (default) or gaussian.
#' @param SL_Y Either a character vector of prediction algorithms or a list containing character vectors. A list of functions included in the SuperLearner package can be found with listWrappers().
#' @param SL_A As above, but for \code{A} (the exposure) conditional on \code{X} (the predictors).
#' @param SL_CATE As above, but for the conditional average treatment effect (CATE).
#' @param cvControl Optional arguments. See \code{?Superlearner::SuperLearner} for details.
#' @param control Optional arguments. See \code{?Superlearner::SuperLearner} for details.
#'
#' @return An object of type \code{rSL}.
#' @export
#'
#' @references{
#'   \insertRef{nie2021quasi}{SL.wrappers}
#' }
#'
#' @importFrom Rdpack reprompt
#'
#' @examples
#' library(SuperLearner)
#' SL.lib <- "SL.glmnet"
#' n = 100; p = 10
#' set.seed(1234)
#' X = matrix(runif(n*p,-3,3), n, p)
#' A = rbinom(n, 1, 0.5)
#' Y = pmax(X[,1], 0) * A + X[,2] + pmin(X[,3], 0) + rnorm(n)
#'
#' rSL_fit <- rSL(X=X, A=A, Y=Y, SL_Y = SL.lib, SL_A = SL.lib, SL_CATE = SL.lib,
#'                cvControl = list(V=10))
rSL <- function(X, A, Y, Y_family = gaussian(), A_family = binomial(),
                SL_Y, SL_A, SL_CATE, cvControl = list(), control = list()) {
  X <- data.frame(X)

  Y_fit <- SuperLearner::SuperLearner(Y, X, family = Y_family,
                                      SL.library = SL_Y, control = control,
                                      cvControl = cvControl)
  m_hat <- Y_fit$SL.predict
  # print(m_hat)

  A_fit <- SuperLearner::SuperLearner(A, X, family = A_family,
                                      SL.library = SL_A, control = control,
                                      cvControl = cvControl)
  p_hat <- A_fit$SL.predict

  Y_tilde <- Y - m_hat
  A_tilde <- A - p_hat
  pseudo_outcome <- Y_tilde/A_tilde

  weights <- A_tilde^2

  tau_fit <- SuperLearner::SuperLearner(pseudo_outcome, X, family = gaussian(),
                                        SL.library = SL_CATE,
                                        obsWeights = weights,
                                        control = control,
                                        cvControl = cvControl)

  ret <- list(tau_fit = tau_fit,
              pseudo_outcome = pseudo_outcome,
              weights = weights,
              A_fit = A_fit,
              Y_fit = Y_fit,
              p_hat = p_hat,
              m_hat = m_hat,
              X = X)
  class(ret) <- "rSL"
  ret
}

#' Predict for \code{rSL} objects
#'
#' @param object An \code{rSL} objects.
#' @param newdata A matrix of new values for the predictors.
#'
#' @return A list with predicted values from Super Learner fit in \code{pred} and predicted values from each algorithm in its library in \code{library.predict}.
#' @export predict.rSL
#' @export
#'
#' @examples
#' library(SuperLearner)
#' library(rlearner)
#'
#' SL.lib <- "SL.glmnet"
#' n = 100; p = 10
#'
#' set.seed(1234)
#' X = matrix(runif(n*p,-3,3), n, p)
#' A = rbinom(n, 1, 0.5)
#' Y = pmax(X[,1], 0) * A + X[,2] + pmin(X[,3], 0) + rnorm(n)
#'
#' rSL_fit <- rSL(X=X, A=A, Y=Y, SL_Y = SL.lib, SL_A = SL.lib, SL_CATE = SL.lib,
#'                cvControl = list(V=10))
#' rSL_pred <- predict(rSL_fit, X)$pred
#'
#' rlasso_fit <- rlasso(X, A, Y, k_folds = 10)
#' rlasso_pred <- predict(rlasso_fit, X)
#'
#' plot(Y-rSL_pred, Y-rlasso_pred, col = 2, asp=1,
#'      main = "R-learner residuals",
#'      xlab = "rSL, 'SL.glmnet'", ylab = "rlasso")
#' abline(a = 0, b = 1)
#' legend(x = "bottomright", legend = c("Residual", "y = x line"), pch = c(1, NA),
#'        col = 2:1, lty = c(NA, 1))
predict.rSL <- function(object, newdata) {
  newX <- data.frame(newdata)
  predict(object$tau_fit, newdata = newX,
          X = object$X, Y = object$pseudo_outcome,
          onlySL = FALSE)
}
