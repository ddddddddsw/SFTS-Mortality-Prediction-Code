required <- c("VennDiagram", "rms", "ResourceSelection")
missing_packages <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("Install required packages: ", paste(missing_packages, collapse = ", "))

output_dir <- "analysis_output"
train_file <- file.path(output_dir, "traindata.csv")
validation_file <- file.path(output_dir, "valdata.csv")
seed <- 123L
bootstrap_replicates <- 1000L
calibration_bins <- 10L
observation_index <- 7L
create_regplot <- FALSE
launch_dynamic_nomogram <- FALSE

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

lasso_vars <- c("Diabetes", "Age", "BA", "PCT", "CD3.CD4", "SFTSV_RNA_load", "IL_4", "IL_6")
boruta_vars <- c("Age", "PCT", "UREA", "CREA", "APTT", "CD3.CD4", "SFTSV_RNA_load", "IL_4", "IL_6", "IFN_gamma")
rfrfe_vars <- c("PCT", "CD3.CD4", "IL_6", "APTT", "SFTSV_RNA_load")
venn_list <- list(LASSO = unique(lasso_vars), Boruta = unique(boruta_vars), `RF-RFE` = unique(rfrfe_vars))
selected_vars <- Reduce(intersect, venn_list)
write.csv(data.frame(Variable = selected_vars), file.path(output_dir, "consensus_predictors.csv"), row.names = FALSE)
selection_table <- do.call(rbind, lapply(names(venn_list), function(nm) {
  data.frame(Method = nm, Variable = venn_list[[nm]])
}))
write.csv(selection_table, file.path(output_dir, "feature_selection_sets.csv"), row.names = FALSE)

venn_plot <- VennDiagram::venn.diagram(
  x = venn_list, filename = NULL,
  fill = c("#0073C2", "#EFC000", "#CD534C"), alpha = rep(0.50, 3),
  col = rep("black", 3), lwd = rep(1.2, 3), lty = rep("solid", 3),
  cex = 1.25, fontfamily = "serif", fontface = "plain", fontcolour = "black",
  cat.cex = 1.35, cat.fontfamily = "serif", cat.fontface = "plain", cat.col = rep("black", 3),
  cat.pos = c(-20, 20, 180), cat.dist = c(0.055, 0.055, 0.035), margin = 0.08
)
open_pdf <- function(path, width, height) {
  if (capabilities("cairo")) {
    grDevices::cairo_pdf(filename = path, width = width, height = height, family = "serif")
  } else {
    grDevices::pdf(file = path, width = width, height = height, family = "Times")
  }
}
open_pdf(file.path(output_dir, "venn_feature_selection.pdf"), 8, 8)
grid::grid.newpage()
grid::grid.draw(venn_plot)
grDevices::dev.off()
if (capabilities("tiff")) {
  tiff_args <- list(filename = file.path(output_dir, "venn_feature_selection.tiff"),
                    width = 8, height = 8, units = "in", res = 600,
                    compression = "lzw", family = "serif")
  if (capabilities("cairo")) tiff_args$type <- "cairo"
  do.call(grDevices::tiff, tiff_args)
  grid::grid.newpage()
  grid::grid.draw(venn_plot)
  grDevices::dev.off()
} else {
  warning("TIFF support is unavailable; the vector PDF was saved.")
}

read_cohort <- function(path) {
  if (!file.exists(path)) stop("Input file not found: ", path)
  dat <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (anyDuplicated(names(dat))) stop("Duplicated column names in: ", path)
  needed <- c("Outcome", selected_vars)


absent <- setdiff(needed, names(dat))
if (length(absent)) stop("Missing columns in ", path, ": ", paste(absent, collapse = ", "))
dat <- dat[needed]
outcome <- as.character(dat$Outcome)
if (anyNA(outcome) || !all(outcome %in% c("0", "1"))) {
  stop("Outcome must be complete and encoded 0 = survival, 1 = in-hospital death.")
}
if (length(unique(outcome)) != 2L) stop("Both outcome classes are required in: ", path)
dat$Outcome <- as.integer(outcome)
for (nm in selected_vars) {
  if (!is.numeric(dat[[nm]]) || any(!is.finite(dat[[nm]]))) {
    stop("Predictor must contain complete finite numeric values: ", nm, " in ", path)
  }
}
dat
}
traindata <- read_cohort(train_file)
valdata <- read_cohort(validation_file)
if (any(vapply(traindata[selected_vars], function(x) length(unique(x)) < 2L, logical(1)))) {
  stop("Constant predictors in the training cohort.")
}

model_formula <- stats::reformulate(selected_vars, response = "Outcome")
dd <- rms::datadist(traindata)
previous_datadist <- getOption("datadist")
options(datadist = "dd")
fit1 <- rms::lrm(model_formula, data = traindata, x = TRUE, y = TRUE)
if (isTRUE(fit1$fail)) stop("Logistic model fitting failed.")
saveRDS(fit1, file.path(output_dir, "nomogram_model.rds"))
write.csv(data.frame(Term = names(stats::coef(fit1)), Coefficient = unname(stats::coef(fit1))),
          file.path(output_dir, "nomogram_model_coefficients.csv"), row.names = FALSE)
nom1 <- rms::nomogram(fit1, fun = stats::plogis,
                      fun.at = c(.001, .01, .05, .5, .95, .99, .999),
                      lp = TRUE, funlabel = "In-hospital mortality probability")
open_pdf(file.path(output_dir, "logistic_nomogram.pdf"), 11, 7)
plot(nom1)
grDevices::dev.off()

set.seed(seed)
cal_train <- rms::calibrate(fit1, method = "boot", B = bootstrap_replicates)
saveRDS(cal_train, file.path(output_dir, "training_bootstrap_calibration.rds"))
write.csv(as.data.frame(unclass(cal_train)), file.path(output_dir, "training_bootstrap_calibration.csv"), row.names = FALSE)
open_pdf(file.path(output_dir, "training_bootstrap_calibration.pdf"), 7, 6)
par(mar = c(5.1, 4.1, 4.1, 2.1))
plot(cal_train, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted Probability", ylab = "Observed Probability",
     cex.lab = 1.2, cex.axis = 1, cex.main = 1.2, cex.sub = 0.8,
     subtitles = FALSE, legend = FALSE)
lines(cal_train[, c("predy", "calibrated.corrected")], type = "l", lwd = 3, col = "#2166AC")
lines(cal_train[, c("predy", "calibrated.orig")], type = "l", lwd = 3, col = "tomato")
abline(0, 1, lty = 2, lwd = 2, col = "#224444")
legend("bottomright", legend = c("Ideal", "Bias-corrected", "Apparent"),
       lty = c(2, 1, 1), lwd = c(2, 3, 3), col = c("#224444", "#2166AC", "tomato"), bty = "n")
grDevices::dev.off()

p_train <- as.numeric(stats::plogis(predict(fit1, newdata = traindata, type = "lp")))
p_validation <- as.numeric(stats::plogis(predict(fit1, newdata = valdata, type = "lp")))
write.csv(data.frame(Row = seq_len(nrow(traindata)), Outcome = traindata$Outcome, Probability = p_train),
          file.path(output_dir, "logistic_training_predictions.csv"), row.names = FALSE)
write.csv(data.frame(Row = seq_len(nrow(valdata)), Outcome = valdata$Outcome, Probability = p_validation),
          file.path(output_dir, "logistic_validation_predictions.csv"), row.names = FALSE)

breaks <- unique(as.numeric(stats::quantile(p_validation, probs = seq(0, 1, length.out = calibration_bins + 1L))))
if (length(breaks) < 3L) stop("Too few distinct predicted probabilities for grouped calibration.")
groups <- cut(p_validation, breaks = breaks, include.lowest = TRUE, labels = FALSE)
cal_validation <- do.call(rbind, lapply(sort(unique(groups)), function(g) {
  idx <- groups == g
  data.frame(Group = g, N = sum(idx), Mean_Predicted = mean(p_validation[idx]),
             Observed_Proportion = mean(valdata$Outcome[idx]))
}))
write.csv(cal_validation, file.path(output_dir, "validation_grouped_calibration.csv"), row.names = FALSE)
open_pdf(file.path(output_dir, "validation_calibration.pdf"), 7, 6)
plot(cal_validation$Mean_Predicted, cal_validation$Observed_Proportion,
     type = "o", pch = 16, lwd = 3, col = "#2166AC", xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted Probability", ylab = "Observed Probability", cex.lab = 1.2)
abline(0, 1, lty = 2, lwd = 2, col = "#224444")
legend("bottomright", legend = c("Ideal", "Validation calibration"),
       lty = c(2, 1), lwd = c(2, 3), col = c("#224444", "#2166AC"), bty = "n")
grDevices::dev.off()

hl_train <- ResourceSelection::hoslem.test(traindata$Outcome, p_train, g = calibration_bins)
hl_validation <- ResourceSelection::hoslem.test(valdata$Outcome, p_validation, g = calibration_bins)
hl_table <- do.call(rbind, lapply(list(Training = hl_train, Validation = hl_validation), function(test) {
  data.frame(Chi_square = unname(test$statistic), Degrees_of_freedom = unname(test$parameter), P_value = test$p.value)
}))
hl_table <- data.frame(Cohort = rownames(hl_table), hl_table, row.names = NULL)
write.csv(hl_table, file.path(output_dir, "hosmer_lemeshow_tests.csv"), row.names = FALSE)
capture.output(hl_train, hl_validation, file = file.path(output_dir, "hosmer_lemeshow_tests.txt"))

if (create_regplot) {
  if (!requireNamespace("regplot", quietly = TRUE)) stop("Install required package: regplot")
  if (observation_index < 1L || observation_index > nrow(traindata)) stop("Invalid observation_index.")
  open_pdf(file.path(output_dir, "logistic_regplot.pdf"), 11, 8)
  regplot::regplot(fit1, plots = c("violin", "boxes"), observation = traindata[observation_index, , drop = FALSE],
                   center = TRUE, subticks = TRUE, droplines = TRUE, title = "Nomogram",
                   points = TRUE, odds = FALSE, showP = FALSE, rank = "sd",
                   interval = "confidence", clickable = FALSE)
  grDevices::dev.off()
}
options(datadist = previous_datadist)
capture.output(sessionInfo(), file = file.path(output_dir, "remaining_analysis_session_info.txt"))
message("Completed feature intersection, nomogram, calibration and Hosmer-Lemeshow analyses.")
message("Consensus predictors: ", paste(selected_vars, collapse = ", "))
if (launch_dynamic_nomogram) {
  if (!requireNamespace("DynNom", quietly = TRUE)) stop("Install required package: DynNom")
  DynNom::DynNom(fit1, data = traindata, DNtitle = "SFTS Mortality Nomogram", DNxlab = "In-hospital mortality probability")
}