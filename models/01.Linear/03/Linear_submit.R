library(tidyverse)
library(tidymodels)

source("models/01.Linear/03/functions.R")

# Model Definition --------------------------------------------------------

model <- parsnip::linear_reg(
  mode = "regression"
) %>%
  parsnip::set_engine(engine = "lm")


# Predict by Test Data ----------------------------------------------------

# 提出用データの作成
{
  # 学習データのロード
  df.train <- load_train_data("data/01.input/train") %>%
    clean_data()

  # テストデータのロード
  df.test <- load_test_data("data/01.input/test.csv.gz") %>%
    clean_data(flg_test = T)

  a <- calc_a(df.train)
  df.train <- add_coulomb_diff_SOC(a, df.train)
  df.test  <- add_coulomb_diff_SOC(a, df.test)
  
  # 全訓練データで学習
  fit <- parsnip::fit(
    model,
    diff_SOC ~
      coulomb_diff_SOC
#    + Current
    + dA2
#    + ma10_Battery_Temp_degC
    + Battery_Temp_degC
#    + diff_Temp_degC
    + eff_Temp_degC

#    + flg_var_dV_01_scaled_gt_n05
#    + flg_var_dA_05_scaled_gt_n50
    + flg_var_dV_10_scaled_gt_n50
#    + flg_var_dV_30_scaled_gt_n10
    ,
    df.train
  )

  df.test %>%

    # SOC 差分予測の追加
    dplyr::mutate(
      pred_diff_SOC = predict(fit, df.test, type = "raw") %>%
        tidyr::replace_na(replace = 0)
    ) %>%

    # SOC 差分の累積を算出
    dplyr::group_by(experiment_type) %>%
    dplyr::mutate(eff_SOC = cumsum(pred_diff_SOC)) %>%
    dplyr::ungroup() %>%

    # SOC (予測値)の算出
    dplyr::mutate(
      SOC = 100 + eff_SOC
    ) %>%

    dplyr::select(
      ID,
      SOC
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
    filepath <- stringr::str_c("models/01.Linear/03/data/output", filename, sep = "/")
    
    # 書き出し
    readr::write_csv(df.submit, filepath, col_names = T)
  }
