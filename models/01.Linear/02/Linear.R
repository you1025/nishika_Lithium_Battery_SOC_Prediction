library(tidyverse)
library(tidymodels)
library(furrr)

source("models/01.Linear/02/functions.R")


# Data Load & Feature Engineering ---------------------------------------------------------------

df.train <- load_train_data("data/01.input/train") %>%
  clean_data()

df.cv <- create_cv(df.train)


# Model Definition --------------------------------------------------------

model <- parsnip::linear_reg(
  mode = "regression"
) %>%
  parsnip::set_engine(engine = "lm")


# Hyper Parameter ---------------------------------------------------------

# Dummy
df.grid.params <- dials::grid_regular(
  dials::penalty(c(0.0, -10)),
  dials::mixture(c(0.0, 1)),
  levels = 1
)
df.grid.params


# Cross Validation --------------------------------------------------------

# https://github.com/rstudio/rstudio/issues/6692
future::plan(future::multisession, workers = 8, setup_strategy = "sequential")

system.time(

  df.results <-

    # ハイパーパラメータをモデルに適用
    purrr::pmap(df.grid.params, function(penalty, mixture) {
      parsnip::set_args(
        model,
        penalty = penalty,
        mixture = mixture
      )
    }) %>%

    # ハイパーパラメータの組み合わせごとにループ
    purrr::map_dfr(function(model.applied) {

      # クロスバリデーションの分割ごとにループ
      furrr::future_map_dfr(df.cv$splits, function(split, model) {

        # 訓練/学習 データ
        df.cv.train <- rsample::training(split)
        df.cv.test  <- rsample::testing(split)

        a <- calc_a(df.cv.train)

        dplyr::bind_rows(
          calc_pred_evals(a, df.cv.train) %>%
            dplyr::mutate(.metric = stringr::str_c("train", .metric, sep = "_")),
          calc_pred_evals(a, df.cv.test) %>%
            dplyr::mutate(.metric = stringr::str_c("test",  .metric, sep = "_"))
        ) %>%
          tidyr::pivot_wider(names_from = .metric, values_from = .estimate)

      }, model = model.applied) %>%

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
      # 評価
      train_rmse,
      test_rmse
    )
)
df.results


# Predict by Test Data ----------------------------------------------------

# # 提出用データの作成
# {
#   # テストデータのロード
#   df.test <- load_test_data("data/01.input/test.csv") %>%
#     clean_data(flg_test = T)
# 
#   # 全訓練データで学習
#   fit <- parsnip::fit(
#     model,
#     diff_SOC ~
#       eff_Power +
#       eff_Power:eff_Temp_degC +
#       Current +
#       Current:Voltage +
#       zero_one_curr_gt_0:Battery_Temp_degC +
#       Chamber_Temp_degC +
#       ma10_dA,
#     df.train
#   )
# 
#   df.test %>%
# 
#     # SOC 差分予測の追加
#     dplyr::mutate(
#       pred_diff_SOC = predict(fit, df.test, type = "raw") %>%
#         tidyr::replace_na(replace = 0)
#     ) %>%
# 
#     # SOC 差分の累積を算出
#     dplyr::group_by(experiment_type) %>%
#     dplyr::mutate(eff_SOC = cumsum(pred_diff_SOC)) %>%
#     dplyr::ungroup() %>%
# 
#     # SOC (予測値)の算出
#     dplyr::mutate(
#       SOC = 100 + eff_SOC
#     ) %>%
# 
#     dplyr::select(
#       ID,
#       SOC
#     )
# } %>%
# 
#   # ファイルに出力
#   {
#     df.submit <- (.)
# 
#     # ファイル名
#     filename <- stringr::str_c(
#       "Linear",
#       lubridate::now(tz = "Asia/Tokyo") %>% format("%Y%m%dT%H%M%S"),
#       sep = "_"
#     ) %>%
#       stringr::str_c("csv", sep = ".")
# 
#     # 出力ファイルパス
#     filepath <- stringr::str_c("models/01.Linear/01/data/output/", filename, sep = "/")
# 
#     # 書き出し
#     readr::write_csv(df.submit, filepath, col_names = T)
#   }
