#' DR-Learner, implemented with SuperLearner
#'
#' @param X A data.frame of predictors.
#' @param A Numeric vector of treatment assignments (0, 1 only).
#' @param Y Numeric vector of outcome variable.
#' @param SL_OR A character vector containing the SuperLearner library for ensemble estimation of the first-stage outcome regression.
#' @param SL_PS A character vector containing the SuperLearner library for ensemble estimation of the first-stage propensity score.
#' @param SL_CATE A character vector containing the SuperLearner library for ensemble estimation of the second-stage regression of the double robust pseudo outcomes.
#' @param cvControl A list containing additional parameters for the SuperLearner cross-validation (see \code{help(SuperLearner)} for more).
#'
#' @references
#' \insertRef{kennedy2020optimal}{SL.wrappers}
#'
#' @return
#' @export
#'
#' @examples
#' library(SuperLearner)
#' SL.lib <- "SL.glmnet"
#' n = 100; p = 5
#' set.seed(1234)
#' X = data.frame(matrix(runif(n*p,-3,3), n, p))
#' A = rbinom(n, 1, 0.5)
#' Y = pmax(X[,1], 0) * A + X[,2] + pmin(X[,3], 0) + rnorm(n)
#'
#' fit <- DRlearner(X=X, A=A, Y=Y, SL_OR = SL.lib, SL_PS = SL.lib, SL_CATE = SL.lib,
#'                  cvControl = list(V=5))
DRlearner <- function(X, A, Y, SL_OR = "SL.mean", SL_PS = "SL.mean",
                      SL_CATE = "SL.mean", cvControl = list(V=3L)){
  # Set-up folds named 1a, 1b, 2 for sample-split/cross-fit steps
  n <- length(Y)
  folds <- c("1a","1b","2")
  fold_id <- sample(rep_len(folds, length.out = n))
  split_assignments <- list(c("1a","1b","2"),
                            c("1b","2","1a"),
                            c("2","1a","1b"))
  CATE_folds <- data.frame(CATE1=rep(NA,n),
                           CATE2=rep(NA,n),
                           CATE3=rep(NA,n))
  CATE_fit <- vector("list", 3L)
  i <- 0
  for( fold in split_assignments ){
    i <- i + 1

    # Training data for Outcome Regressions
    X_OR <- subset(X, fold_id == fold[1])
    A_OR <- A[fold_id == fold[1]]
    Y_OR <- Y[fold_id == fold[1]]

    # Training data for Propensity Score
    X_PS <- subset(X, fold_id == fold[2])
    A_PS <- A[fold_id == fold[2]]

    # Test data for second-stage regression
    X_test <- subset(X, fold_id == fold[3])
    A_test <- A[fold_id == fold[3]]
    Y_test <- Y[fold_id == fold[3]]

    # First stage estimates of OR0, OR1, and PS
    OR0_fit <- SuperLearner(Y_OR[A_OR==0], subset(X_OR, A_OR == 0),
                            newX = X_test, SL.library = SL_OR,
                            cvControl = cvControl)
    OR1_fit <- SuperLearner(Y_OR[A_OR==1], subset(X_OR, A_OR == 1),
                            newX = X_test, SL.library = SL_OR,
                            cvControl = cvControl)
    PS_fit <- SuperLearner(A_PS, X_PS, newX = X_test,
                           family = binomial(), SL.library = SL_PS,
                           cvControl = cvControl)

    # Construction of doubly-robust pseudo-outcomes
    OR0 <- c(OR0_fit$SL.predict)
    OR1 <- c(OR1_fit$SL.predict)
    PS <- c(PS_fit$SL.predict)

    ORa <- OR0 + A_test*(OR1 - OR0)
    pseudo <- (A_test - PS)/(PS*(1 - PS))*(Y_test - ORa) + OR1 - OR0

    # Estimate Conditional Average Treatment Effect
    CATE_fit[[i]] <- SuperLearner(pseudo, X_test, newX = X,
                                  SL.library = SL_CATE, cvControl = cvControl)

    CATE_folds[,i] <- CATE_fit[[i]]$SL.predict
  }
  class(CATE_fit) <- "DRlearner"
  return(CATE_fit)
}

#' Predict for \code{DRlearner} objects
#'
#' @param object A \code{Drlearner} object.
#' @param newdata A data.frame of new values for the predictors.
#'
#' @return A vector of predictions
#' @export
#'
#' @examples
#' library(SuperLearner)
#' SL.lib <- "SL.glmnet"
#' n = 100; p = 5
#' set.seed(1234)
#' X = data.frame(matrix(runif(n*p,-3,3), n, p))
#' A = rbinom(n, 1, 0.5)
#' Y = pmax(X[,1], 0) * A + X[,2] + pmin(X[,3], 0) + rnorm(n)
#'
#' fit <- DRlearner(X=X, A=A, Y=Y, SL_OR = SL.lib, SL_PS = SL.lib, SL_CATE = SL.lib,
#'                  cvControl = list(V=5))
#' predict(fit, X)
predict.DRlearner <- function(object, newdata){
  cate1 <- predict(object[[1]], newdata = newdata)$pred
  cate2 <- predict(object[[2]], newdata = newdata)$pred
  cate3 <- predict(object[[3]], newdata = newdata)$pred
  cateX <- (cate1+cate2+cate3)/3
  c(cateX)
}
