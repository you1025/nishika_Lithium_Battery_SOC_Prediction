library(tidyverse)
library(tidymodels)

source("models/03.LightGBM/03/functions.R")

# Predict by Test Data ----------------------------------------------------

# 提出用データの作成
{
  # 学習用データのロード
  df.train <- load_train_data("data/01.input/train") %>%
    clean_data() %>%
    dplyr::select(
      # 実験パターン
      experiment_type,
      
      # 目的変数
      SOC,
      
      # クーロン法に必要
      Ah,
      diff_Time,
      
      # 説明変数
      Current,
    )
  a <- calc_a(df.train)
  df.train <- add_coulomb_diff_SOC(a, df.train)

  # テストデータのロード
  df.test <- load_test_data("data/01.input/test.csv.gz") %>%
    clean_data(flg_test = T) %>%
    dplyr::select(
      # 実験パターン
      experiment_type,

      # クーロン法に必要
      Ah,
      diff_Time,

      # 説明変数
      Current,
    ) %>%
    add_coulomb_diff_SOC(a, data = ., flg_test = T)

  train_valid_split <- rsample::initial_split(df.train, prop = 14/15)
  df.train_train <- rsample::training(train_valid_split)
  df.train_test  <- rsample::testing(train_valid_split)

  # 訓練用データ
  dtrain <- lightgbm::lgb.Dataset(
    data = df.train_train %>%
      filter_columns() %>%
      as.matrix(),
    label = df.train_train %>%
      dplyr::pull(coulomb_diff_SOC)
  )

  # 検証用データ
  dtest <- lightgbm::lgb.Dataset(
    data = df.train_test %>%
      filter_columns() %>%
      as.matrix(),
    label = df.train_test %>%
      dplyr::pull(coulomb_diff_SOC)
  )

  # 全訓練データで学習
  fit <- lightgbm::lgb.train(
    params = list(
      boosting_type = "gbdt",
      objective     = "regression",
      metric        = "rmse",

      max_depth        = -1,
      num_leaves       = 41,
      min_data_in_leaf = 23,

      max_bin = 4096,
      bin_construct_sample_cnt = 250000, # たまたま見つけただけなので適当で良いw

      verbosity = 0,
      seed = 1234
    ),

    data = dtrain,
    valids = list(test = dtest),

    learning_rate         = 0.1,
    nrounds               = 1000,
    early_stopping_rounds = 10
  )

  df.test %>%

    dplyr::mutate(
      pred_diff_SOC = predict(
        fit,
        df.test %>%
          filter_columns(flg_test = T) %>%
          as.matrix()
      ) %>%
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