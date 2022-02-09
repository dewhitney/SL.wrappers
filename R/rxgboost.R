
r.xgboost <- function (Y, X, newX, family, obsWeights, id,
                       ntrees_max = 1000, num_search_rounds = 10,
                       early_stopping_rounds = 10, nthread = 1, verbose = FALSE,
                       print_every_n = 100, ...) {
  require("xgboost")
  require("rlearner")
  if (packageVersion("xgboost") < 0.6)
    stop("SL.xgboost requires xgboost version >= 0.6, try help('SL.xgboost') for details")
  if (!is.matrix(X)) {
    X = model.matrix(~. - 1, X)
  }
  if (family$family == "gaussian") {
    if (packageVersion("xgboost") >= "1.1.1.1") {
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
    newX <- model.matrix(~. - 1, newX)
  }
  pred <- predict(model, newX)
  fit <- list(object = model)
  class(fit) <- c("SL.xgboost")
  out <- list(pred = pred, fit = fit)
  return(out)
}
