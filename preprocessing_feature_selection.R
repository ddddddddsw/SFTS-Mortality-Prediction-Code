# SFTS preprocessing and feature selection
# Copyright (c) 2026 the study authors. All rights reserved.
# See LICENSE.txt. Revised workflow; rerun before reporting revised results.

required <- c("caret", "mice", "glmnet", "Boruta")
missing_packages <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("Install required packages: ", paste(missing_packages, collapse = ", "))

input_file <- "origindata.csv"
input_encoding <- "GBK"
output_dir <- "analysis_output"
seed <- 123L
missingness_cutoff <- 0.10
imputation_number <- 5L
completed_dataset <- 2L

raw <- read.csv(input_file, header = TRUE, fileEncoding = input_encoding,
                check.names = TRUE, stringsAsFactors = FALSE)
stopifnot("Outcome" %in% names(raw), !anyDuplicated(names(raw)))
if (anyNA(raw$Outcome) || !all(as.character(raw$Outcome) %in% c("0", "1"))) {
  stop("Outcome must be complete and encoded 0 = survival, 1 = in-hospital death.")
}
raw$Outcome <- factor(as.character(raw$Outcome), levels = c("0", "1"))
# Original script factorized columns 1:10. Verify these are the intended categories.
# Only supply candidate predictors: remove identifiers before running this script.
categorical_vars <- setdiff(names(raw)[seq_len(min(10L, ncol(raw)))], "Outcome")

# 1. Split the original, incomplete dataset before imputation or selection.
set.seed(seed)
train_index <- as.integer(caret::createDataPartition(raw$Outcome, p = 0.7, list = FALSE))
train_raw <- raw[train_index, , drop = FALSE]
validation_raw <- raw[-train_index, , drop = FALSE]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(list(training = train_index, validation = setdiff(seq_len(nrow(raw)), train_index)),
        file.path(output_dir, "split_indices.rds"))

predictor_names <- setdiff(names(raw), "Outcome")
missing_count <- colSums(is.na(train_raw[predictor_names]))
missing_rate <- missing_count / nrow(train_raw)
missing_table <- data.frame(
  Variable = predictor_names, Missing_Count = as.integer(missing_count),
  Missing_Percentage = round(100 * missing_rate, 2),
  Excluded = unname(missing_rate >= missingness_cutoff)
)
write.csv(missing_table, file.path(output_dir, "training_missingness.csv"), row.names = FALSE)
cohort_counts <- rbind(
  data.frame(Cohort = "Training", N = nrow(train_raw),
             Survivors = sum(train_raw$Outcome == "0"), Deaths = sum(train_raw$Outcome == "1")),
  data.frame(Cohort = "Internal validation", N = nrow(validation_raw),
             Survivors = sum(validation_raw$Outcome == "0"), Deaths = sum(validation_raw$Outcome == "1"))
)
write.csv(cohort_counts, file.path(output_dir, "cohort_counts.csv"), row.names = FALSE)
keep <- predictor_names[missing_rate < missingness_cutoff]
if (length(keep) < 2L) stop("Too few predictors remain after missingness screening.")
train_x <- train_raw[keep]
validation_x <- validation_raw[keep]

# Categories and initial fills are defined from training observations only.
for (nm in intersect(categorical_vars, keep)) {
  levels_train <- sort(unique(as.character(train_x[[nm]][!is.na(train_x[[nm]])])))
  unseen <- setdiff(unique(as.character(validation_x[[nm]][!is.na(validation_x[[nm]])])), levels_train)
  if (length(unseen)) stop("Validation has an unseen category in ", nm, "; specify a handling rule.")
  train_x[[nm]] <- factor(as.character(train_x[[nm]]), levels = levels_train)
  validation_x[[nm]] <- factor(as.character(validation_x[[nm]]), levels = levels_train)
}
if (any(!vapply(train_x, function(x) is.numeric(x) || is.factor(x), logical(1)))) {
  stop("Non-numeric predictors must be explicitly listed in categorical_vars.")
}
constant <- names(train_x)[vapply(train_x, function(x) length(unique(x[!is.na(x)])) < 2L, logical(1))]
if (length(constant)) stop("Constant training predictors require review: ", paste(constant, collapse = ", "))

# 2. MICE: validation rows are imputed but excluded from parameter estimation.
# Outcome is deliberately absent: validation outcomes must not inform imputations.
combined_x <- rbind(train_x, validation_x)
ignore_rows <- c(rep(FALSE, nrow(train_x)), rep(TRUE, nrow(validation_x)))
methods <- mice::make.method(train_x)
for (nm in names(methods)) {
  if (is.numeric(train_x[[nm]])) methods[nm] <- "pmm"
  else if (nlevels(train_x[[nm]]) == 2L) methods[nm] <- "logreg"
  else methods[nm] <- "polyreg"
  if (!anyNA(combined_x[[nm]])) methods[nm] <- ""
}
predictor_matrix <- mice::make.predictorMatrix(train_x)
initial <- combined_x
for (nm in names(initial)) {
  observed <- train_x[[nm]][!is.na(train_x[[nm]])]
  fill <- if (is.numeric(observed)) median(observed) else names(which.max(table(observed)))
  initial[[nm]][is.na(initial[[nm]])] <- fill
}
imp <- mice::mice(combined_x, m = imputation_number, maxit = 5,
                  method = methods, predictorMatrix = predictor_matrix,
                  ignore = ignore_rows, data.init = initial,
                  remove.constant = FALSE, remove.collinear = FALSE,
                  seed = seed, printFlag = FALSE)
if (!is.null(imp$loggedEvents)) {
  write.csv(imp$loggedEvents, file.path(output_dir, "mice_logged_events.csv"), row.names = FALSE)
  warning("MICE logged events: inspect the event file and convergence before using results.")
}
completed <- mice::complete(imp, action = completed_dataset)
if (anyNA(completed)) stop("Missing values remain after imputation; review MICE diagnostics.")
traindata <- data.frame(Outcome = train_raw$Outcome,
                       completed[seq_len(nrow(train_x)), , drop = FALSE])
valdata <- data.frame(Outcome = validation_raw$Outcome,
                     completed[nrow(train_x) + seq_len(nrow(validation_x)), , drop = FALSE])
write.csv(traindata, file.path(output_dir, "traindata.csv"), row.names = FALSE)
write.csv(valdata, file.path(output_dir, "valdata.csv"), row.names = FALSE)
saveRDS(imp, file.path(output_dir, "mice_object.rds"))
write.csv(data.frame(Variable = names(methods), Method = unname(methods)),
          file.path(output_dir, "imputation_methods.csv"), row.names = FALSE)

# This source excerpt does not contain log transformations or correlation/VIF screening.
# Do not silently add transformations to values already transformed in the input.
# Supply and execute the original training-only screening block here if it was used.

# 3. LASSO: training-only min-max scaling, followed by 10-fold CV.
# glmnet default standardize = TRUE is preserved, as in the supplied source.
lassodata <- traindata
numeric_vars <- names(lassodata)[vapply(lassodata, is.numeric, logical(1))]
scaling <- data.frame(Variable = numeric_vars,
                      Minimum = vapply(lassodata[numeric_vars], min, numeric(1)),
                      Maximum = vapply(lassodata[numeric_vars], max, numeric(1)))
for (nm in numeric_vars) {
  low <- min(lassodata[[nm]])
  high <- max(lassodata[[nm]])
  if (low == high) stop("Constant predictor during scaling: ", nm)
  lassodata[[nm]] <- (lassodata[[nm]] - low) / (high - low)
}
write.csv(scaling, file.path(output_dir, "lasso_scaling.csv"), row.names = FALSE)
# Preserve the original data.matrix coding, including factor-to-integer predictor coding.
x <- data.matrix(lassodata[setdiff(names(lassodata), "Outcome")])
y <- as.integer(as.character(lassodata$Outcome))
set.seed(seed)
lasso_cv <- glmnet::cv.glmnet(x, y, alpha = 1, nfolds = 10, family = "binomial")
lasso_coef <- as.matrix(stats::coef(lasso_cv, s = "lambda.min"))
write.csv(data.frame(Variable = rownames(lasso_coef), Coefficient = lasso_coef[, 1]),
          file.path(output_dir, "lasso_coefficients.csv"), row.names = FALSE)
saveRDS(lasso_cv, file.path(output_dir, "lasso_cv.rds"))
pdf(file.path(output_dir, "lasso_cross_validation.pdf"), width = 7, height = 6)
plot(lasso_cv)
dev.off()

# 4. Boruta: training cohort only; retain original settings and tentative resolution.
set.seed(seed)
boruta_model <- Boruta::Boruta(Outcome ~ ., data = traindata,
                               pValue = 0.01, mcAdj = TRUE, maxRuns = 100,
                               doTrace = 0, holdHistory = TRUE,
                               getImp = Boruta::getImpRfZ)
final_boruta <- Boruta::TentativeRoughFix(boruta_model)
selected <- Boruta::getSelectedAttributes(final_boruta, withTentative = FALSE)
write.csv(data.frame(Variable = selected),
          file.path(output_dir, "boruta_selected.csv"), row.names = FALSE)
write.csv(data.frame(Variable = names(boruta_model$finalDecision),
                    Initial_Decision = as.character(boruta_model$finalDecision),
                    Final_Decision = as.character(final_boruta$finalDecision)),
          file.path(output_dir, "boruta_decisions.csv"), row.names = FALSE)
saveRDS(boruta_model, file.path(output_dir, "boruta_initial.rds"))
saveRDS(final_boruta, file.path(output_dir, "boruta_final.rds"))
pdf(file.path(output_dir, "boruta_importance.pdf"), width = 12, height = 7)
par(mar = c(10, 4, 2, 1))
plot(boruta_model, sort = TRUE, whichShadow = c(TRUE, TRUE, TRUE), las = 2, xlab = "")
dev.off()
capture.output(sessionInfo(), file = file.path(output_dir, "sessionInfo.txt"))
message("Completed the supplied preprocessing/LASSO/Boruta excerpt. Review outputs locally.")
