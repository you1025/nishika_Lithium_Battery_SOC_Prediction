source("functions.R")

calc_pred_evals <- function(a, data) {

  data %>%

    dplyr::mutate(
#      pred_diff_SOC = Current / (36 * a) * ifelse(is.na(diff_Time), 0, diff_Time)
      pred_diff_SOC = dplyr::if_else(is.na(prev_A), Current, (prev_A + Current) / 2) / (36 * a) * dplyr::if_else(is.na(diff_Time), 0, diff_Time)
    ) %>%

    # SOC 差分の累積を算出
    dplyr::group_by(experiment_type) %>%
    dplyr::mutate(pred_eff_SOC = cumsum(pred_diff_SOC)) %>%
    dplyr::ungroup() %>%

    # SOC の予測値を算出
    dplyr::mutate(
      pred_SOC = 100 + pred_eff_SOC
    ) %>%

    # RMSE の算出
    yardstick::rmse(
      truth = SOC,
      estimate = pred_SOC
    ) %>%

    dplyr::select(-.estimator)
}
