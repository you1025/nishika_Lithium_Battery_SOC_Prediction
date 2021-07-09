library(tidyverse)
library(tidymodels)
library(lightgbm)
library(furrr)

source("models/03.LightGBM/02/functions.R")


# Data Load & Feature Engineering ---------------------------------------------------------------

df.train <- load_train_data("data/01.input/train") %>%
  clean_data() %>%
  dplyr::mutate(
#    eff_Power_Temp_degC = eff_Power * eff_Temp_degC,
    Current_Voltage = Current * Voltage,
    d_A_V = dA * dV,
    d_A2_V2 = dA2 * dV2
  ) %>%
  dplyr::select(
    # 実験パターン
    experiment_type,

    # 目的変数
    SOC,
    diff_SOC,

    # クーロン法に必要
    Current,
    prev_A,
    Ah,
    diff_Time,

    # 説明変数
    Voltage,
    cumsum_A,
    dA,
    dA2,
    Power,

    var_dA_10_scaled,
    flg_var_dA_10_scaled_gt_n05,
    flg_var_dA_10_scaled_gt_n10,
    flg_var_dA_10_scaled_gt_n25,
    flg_var_dA_10_scaled_gt_n50,
    var_dA_30_scaled,
    flg_var_dA_30_scaled_gt_n05,
    flg_var_dA_30_scaled_gt_n10,
    flg_var_dA_30_scaled_gt_n25,
    flg_var_dA_30_scaled_gt_n50,
    var_dA_60_scaled,
    flg_var_dA_60_scaled_gt_n05,
    flg_var_dA_60_scaled_gt_n10,
    flg_var_dA_60_scaled_gt_n25,
    flg_var_dA_60_scaled_gt_n50,

    var_dV_10_scaled,
    flg_var_dV_10_scaled_gt_n05,
    flg_var_dV_10_scaled_gt_n10,
    flg_var_dV_10_scaled_gt_n25,
    flg_var_dV_10_scaled_gt_n50,
    var_dV_30_scaled,
    flg_var_dV_30_scaled_gt_n05,
    flg_var_dV_30_scaled_gt_n10,
    flg_var_dV_30_scaled_gt_n25,
    flg_var_dV_30_scaled_gt_n50,
    var_dV_60_scaled,
    flg_var_dV_60_scaled_gt_n05,
    flg_var_dV_60_scaled_gt_n10,
    flg_var_dV_60_scaled_gt_n25,
    flg_var_dV_60_scaled_gt_n50
  )

df.cv <- create_cv(df.train)


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- tibble(
  learning_rate = 0.1,
  nrounds = 1000,
  early_stopping_rounds = 10,

  max_bin = 4096,
#  num_leaves = 41,
#  min_data_in_leaf = 23,
  max_depth = -1
) %>%
  tidyr::crossing(
    num_leaves = c(42, 44, 46),
    min_data_in_leaf = c(22, 24, 26),
#    max_depth = c(7, 10, 13)
  )
df.grid.params


# Cross Validation --------------------------------------------------------

# https://github.com/rstudio/rstudio/issues/6692
#future::plan(future::multisession, workers = 8, setup_strategy = "sequential")

system.time({

  set.seed(1025)

  df.results <-

    # ハイパーパラメータをモデルに適用
    purrr::pmap_dfr(df.grid.params, function(...) {
      # パラメータ一覧
      hyper_params <- list(...)

      furrr::future_map_dfr(df.cv$splits, function(split, hyper_params) {

        # 訓練/学習 データ
        df.cv.train <- rsample::training(split)
        df.cv.test  <- rsample::testing(split)

        a <- calc_a(df.cv.train)
        df.cv.train <- add_coulomb_diff_SOC(a, df.cv.train)
        df.cv.test  <- add_coulomb_diff_SOC(a, df.cv.test)

        train_valid_split <- rsample::initial_split(df.cv.train, prop = 4/5)
        df.cv.train_train <- rsample::training(train_valid_split)
        df.cv.train_test  <- rsample::testing(train_valid_split)

        # 訓練用データ
        dtrain <- lightgbm::lgb.Dataset(
          data = df.cv.train_train %>%
            dplyr::select(-c(
              # クーロン法に用いた
              #Current,
              prev_A,
              Ah,
              diff_Time,

              experiment_type,
              SOC,
              diff_SOC
            )) %>%
            as.matrix(),
          label = df.cv.train_train %>%
            dplyr::pull(diff_SOC)
        )

        # 検証用データ
        dtest <- lightgbm::lgb.Dataset(
          data = df.cv.train_test %>%
            dplyr::select(-c(
              # クーロン法に用いた
              #Current,
              prev_A,
              Ah,
              diff_Time,
              
              experiment_type,
              SOC,
              diff_SOC
            )) %>%
            as.matrix(),
          label = df.cv.train_test %>%
            dplyr::pull(diff_SOC)
        )

        fit <- lightgbm::lgb.train(
          params = list(
            boosting_type = "gbdt",
            objective     = "regression",
            metric        = "rmse",

            max_depth        = hyper_params$max_depth,
            num_leaves       = hyper_params$num_leaves,
            min_data_in_leaf = hyper_params$min_data_in_leaf,

            max_bin = hyper_params$max_bin,
            bin_construct_sample_cnt = 250000, # たまたま見つけただけなので適当で良いw

            verbosity = 0,
            seed = 1234
          ),

          data = dtrain,
          valids = list(test = dtest),

          learning_rate         = hyper_params$learning_rate,
          nrounds               = hyper_params$nrounds,
          early_stopping_rounds = hyper_params$early_stopping_rounds
        )

        dplyr::bind_rows(
          calc_pred_evals(fit, df.cv.train) %>%
            dplyr::mutate(.metric = stringr::str_c("train", .metric, sep = "_")),
          calc_pred_evals(fit, df.cv.test) %>%
            dplyr::mutate(.metric = stringr::str_c("test",  .metric, sep = "_"))
        ) %>%
          tidyr::pivot_wider(names_from = .metric, values_from = .estimate)

      }, hyper_params = hyper_params, .options = furrr::furrr_options(seed = 5963L)) %>%

        # CV 分割全体の平均値を評価スコアとする
        dplyr::summarise_all(mean)
    }) %>%

      # 評価結果とパラメータを結合
      dplyr::bind_cols(df.grid.params) %>%

      # 評価スコアの順にソート(昇順)
      dplyr::arrange(
        test_rmse
      ) %>%

      dplyr::select(
        # parameter
        learning_rate,
        nrounds,
        early_stopping_rounds,
        max_depth,
        num_leaves,
        min_data_in_leaf,
        max_bin,

        # 評価
        train_rmse,
        test_rmse
      )
})
df.results
