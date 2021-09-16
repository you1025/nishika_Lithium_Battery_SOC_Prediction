source("functions.R")

add_coulomb_diff_SOC <- function(a, data, flg_test = F) {

  df.tmp.data <-  data %>%

    dplyr::mutate(
      pred_coulomb_SOC = (1 + (1/a) * Ah) * 100
    )

  if(!flg_test) {
    df.tmp.data <- df.tmp.data %>%
      dplyr::mutate(
        coulomb_diff_SOC = SOC - pred_coulomb_SOC
      )
  }

  df.tmp.data
}

calc_pred_evals <- function(fit, data) {

  data %>%

    dplyr::mutate(
      pred_diff_SOC = predict(
        fit,
        data %>%
          filter_columns() %>%
          as.matrix()
      ) %>%
        tidyr::replace_na(replace = 0),
      pred_SOC = pred_coulomb_SOC + pred_diff_SOC
    ) %>%

    # RMSE の算出
    yardstick::rmse(
      truth = SOC,
      estimate = pred_SOC
    ) %>%

    dplyr::select(-.estimator)
}

filter_columns <- function(data, flg_test = F) {

  df.tmp.data <- data %>%

    dplyr::select(-c(
      # クーロン法に用いた
      Ah,
      diff_Time,
      pred_coulomb_SOC,

      experiment_type,

      # 目的変数
      coulomb_diff_SOC
    ))

  if(!flg_test) {
    df.tmp.data <- df.tmp.data %>%
      dplyr::select(-SOC)
  }

  df.tmp.data
}
