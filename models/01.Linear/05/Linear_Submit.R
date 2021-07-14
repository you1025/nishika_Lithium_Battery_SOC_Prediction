library(tidyverse)
library(tidymodels)

source("models/01.Linear/05/functions.R")

# Predict by Test Data ----------------------------------------------------

# 提出用データの作成
{
  # # 学習用データのロード
  # df.train <- load_train_data("data/01.input/train") %>%
  #   clean_data()
  # a <- calc_a(df.train)
  # df.train <- add_coulomb_diff_SOC(a, df.train)
  # 
  # # テストデータのロード
  # df.test <- load_test_data("data/01.input/test.csv.gz") %>%
  #   clean_data(flg_test = T) %>%
  #   add_coulomb_diff_SOC(a, data = ., flg_test = T)

  # 全訓練データで学習
  fit <- parsnip::linear_reg(
    mode = "regression"
  ) %>%
    parsnip::set_engine(engine = "lm") %>%

    parsnip::fit(
      coulomb_diff_SOC ~
        (
          Current
          + Ah              
        ) : flg_var_dV_10_scaled_gt_n50
      
      + diff_Temp_degC
      ,
      df.train
    )

  df.test %>%

    dplyr::mutate(
      pred_diff_SOC = predict(fit, df.test, type = "raw") %>%
        tidyr::replace_na(replace = 0),
      pred_SOC = pred_coulomb_SOC + pred_diff_SOC
    ) %>%

    dplyr::select(
      ID,
      SOC = pred_SOC
    )
} %>%

  # ファイルに出力
  {
    df.submit <- (.)

    # ファイル名
    filename <- stringr::str_c(
      "Linear",
      lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
      sep = "_"
    ) %>%
      stringr::str_c("csv", sep = ".")

    # 出力ファイルパス
    filepath <- stringr::str_c("models/01.Linear/05/data/output", filename, sep = "/")

    # 書き出し
    readr::write_csv(df.submit, filepath, col_names = T)
  }