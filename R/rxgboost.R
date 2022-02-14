
#' Wrapper function for xgboost based on \code{rlearner::rboost}
#'
#' This function automates hyperparameter tuning for prediction using \code{xgboost}. Its input/output follows the template required for wrapper functions in the \code{SuperLearner} package.
#'
#' @param Y The outcome in the training data set. Must be a numeric vector.
#' @param X The predictor variables in the training data set, usually a data.frame.
#' @param newX The predictor variables in the validation data set. The structure should match X. Defaults to X.
#' @param family Currently allows gaussian or binomial to describe the error distribution. Link function information will be ignored and should be contained in the method argument below.
#' @param obsWeights Optional observation weights variable.
#' @param id Optional cluster identification variable. Not currently supported.
#' @param k_folds Number of folds used in cross validation (CV). Defaults to 5-fold CV.
#' @param ntrees_max The maximum number of trees to grow for xgboost.
#' @param num_search_rounds The number of random sampling of hyperparameter combinations for cross validating on xgboost trees.
#' @param early_stopping_rounds The number of rounds the test error stops decreasing by which the cross validation in finding the optimal number of trees stops.
#' @param nthread The number of threads to use.
#' @param verbose Boolean; whether to print statistic.
#' @param print_every_n The number of iterations (in each iteration, a tree is grown) by which the code prints out information.
#' @param ... Additional SuperLearner arguments. Not currently supported.
#'
#' @return A list containing a numeric vector \code{pred} of predictions at \code{newX} and a list \code{fit} containing an \code{r.xgboost} object based on output of the \code{rlearner::cvboost} function. Run the command \code{?rlearner::cvboost} for additional details about individual elements of \code{fit}.
#' @export
#'
#' @examples
#' set.seed(1)
#' n = 100; p = 5
#'
#' x = matrix(rnorm(n*p), n, p)
#' y = pmax(x[,1], 0) + x[,2] + pmin(x[,3], 0) + rnorm(n)
#' fit = SL.cvboost(y, data.frame(x), newX = data.frame(x),
#' family = gaussian(), obsWeights = NULL)
#' est = fit$pred
SL.cvboost <- function (Y, X, newX = X, family = gaussian(), obsWeights = NULL,
                       id = NULL, k_folds = 5L, ntrees_max = 1000, num_search_rounds = 10,
                       early_stopping_rounds = 10, nthread = 1, verbose = FALSE,
                       print_every_n = 100, ...) {
  # if (!is.matrix(X)) { X = model.matrix(~. - 1, X) }
  X = model.matrix(~. - 1, X)
  if (family$family == "gaussian") {
    if (utils::packageVersion("xgboost") >= "1.1.1.1") {
      objective <- "reg:squarederror"
    }
    else {
      objective <- "reg:linear"
    }
    model <- rlearner::cvboost(x = X, y = Y, weights = obsWeights, ntrees_max = ntrees_max,
                    num_search_rounds = num_search_rounds,
                    early_stopping_rounds = early_stopping_rounds,
                    nthread = nthread, verbose = verbose,
                    print_every_n = print_every_n)
  }
  if (family$family == "binomial") {
    model <- rlearner::cvboost(x = X, y = Y, weights = obsWeights, ntrees_max = ntrees_max,
                     num_search_rounds = num_search_rounds,
                     early_stopping_rounds = early_stopping_rounds,
                     nthread = nthread, verbose = verbose,
                     print_every_n = print_every_n)
  }
  if (family$family == "multinomial") {
    warning("'multinomial' is not a supported family at this time")
  }
  if (!is.matrix(newX)) {
    newX <- stats::model.matrix(~. - 1, newX)
  }

  # Debugging prediction
  dtest <- xgboost::xgb.DMatrix(data=newX)
  pred <- predict(model$xgb_fit, newdata=dtest)

  fit <- list(object = model)
  class(fit) <- c("SL.cvboost")
  out <- list(pred = pred, fit = fit)
  return(out)
}


#' Prediction for wrapper SL.cvboost
#'
#' @param object SL.cvboost object
#' @param newdata Dataframe to generate predictions
#' @param ... Unused additional arguments
#'
#' @return
#' @export
#'
#' @examples
#' #' set.seed(1)
#' n = 100; p = 5
#'
#' x = matrix(rnorm(n*p), n, p)
#' y = pmax(x[,1], 0) + x[,2] + pmin(x[,3], 0) + rnorm(n)
#' fit = SL.cvboost(y, data.frame(x), newX = data.frame(x),
#' family = gaussian(), obsWeights = NULL)
#' est = predict(fit$fit)
predict.SL.cvboost <- function(object, newdata=NULL, ...) {
  if (is.null(newdata)) {
    return(object$object$best_xgb_cvfit$pred)
  }
  else{
    newdata <- model.matrix(~. - 1, newdata)
    dtest <- xgboost::xgb.DMatrix(data=newdata)
    return(predict(object$object$xgb_fit, newdata=dtest))
  }
}

