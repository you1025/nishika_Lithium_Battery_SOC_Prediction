library(tidyverse)
library(tidymodels)
#library(furrr)

source("models/01.Linear/01/functions.R")


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

df.grid.params <- dials::grid_regular(
  dials::penalty(c(0.0, -10)),
  dials::mixture(c(0.0, 1)),
  levels = 1
)
df.grid.params


# Cross Validation --------------------------------------------------------

#future::plan(future::multisession, workers = 2)

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
      purrr::map_dfr(df.cv$splits, function(split, model) {

        # 訓練/学習 データ
        df.cv.train <- rsample::training(split)
        df.cv.test  <- rsample::testing(split)

        # 学習
        fit <- parsnip::fit(
          model,
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + cumsum_A + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC,
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC, # 1.60
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current, # 0.846
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage, #0.0275
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + cumsum_A, # 0.0273
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + cumsum_A + zero_one_curr_gt_0:Battery_Temp_degC, # 0.0305
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + cumsum_A + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC, # 0.0271
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC, # 0.0260
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC + dV2, # 0.0265
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC + dA2, # 0.0277
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC + dV2:dA2, # 0.0262
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC + ma10_dV, # 0.0294
          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC + ma10_dA, # 0.0221 ホントかなw
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC + ma10_dA + ma10_Power, # 0.0224
#          diff_SOC ~ eff_Power + eff_Power:eff_Temp_degC + Current + Current:Voltage + zero_one_curr_gt_0:Battery_Temp_degC + Chamber_Temp_degC + ma10_dA + ma10_Battery_Temp_degC, # 0.0235
          df.cv.train
        )

        dplyr::bind_rows(
          calc_pred_evals(fit, df.cv.train) %>%
            dplyr::mutate(.metric = stringr::str_c("train", .metric, sep = "_")),
          calc_pred_evals(fit, df.cv.test) %>%
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
      # parameter
      penalty,
      mixture,

      # 評価
      train_rmse,
      test_rmse
    )
)
df.results


# Predict by Test Data ----------------------------------------------------

# 提出用データの作成
{
  # テストデータのロード
  df.test <- load_test_data("data/01.input/test.csv") %>%
    clean_data(flg_test = T)

  # 全訓練データで学習
  fit <- parsnip::fit(
    model,
    diff_SOC ~
      eff_Power +
      eff_Power:eff_Temp_degC +
      Current +
      Current:Voltage +
      zero_one_curr_gt_0:Battery_Temp_degC +
      Chamber_Temp_degC +
      ma10_dA,
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
    filepath <- stringr::str_c("models/01.Linear/01/data/output/", filename, sep = "/")

    # 書き出し
    readr::write_csv(df.submit, filepath, col_names = T)
  }
