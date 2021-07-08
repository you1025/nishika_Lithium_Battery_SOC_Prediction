source("functions.R")

calc_pred_evals <- function(model, data) {

  data %>%

    dplyr::mutate(
      pred_diff_SOC = predict(
        model,
        data %>%
          dplyr::select(-c(
            experiment_type,
            SOC,
            diff_SOC
          )) %>%
          as.matrix()
      ) %>%
        tidyr::replace_na(replace = 0)
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
