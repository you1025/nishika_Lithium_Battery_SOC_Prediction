library(tidyverse)
library(tidymodels)
#library(furrr)

source("models/01.Linear/04/functions.R")


# Data Load & Feature Engineering ---------------------------------------------------------------

#df.train <- load_train_data("data/01.input/train") %>%
#  clean_data()
#df.cv <- create_cv(df.train)


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

# https://github.com/rstudio/rstudio/issues/6692
#future::plan(future::multisession, workers = 8, setup_strategy = "sequential")

system.time(

  df.results <-

    # ハイパーパラメータをモデルに適用
    purrr::pmap(df.grid.params, function(penalty, mixture) {
      model
    }) %>%

    # ハイパーパラメータの組み合わせごとにループ
    purrr::map_dfr(function(model.applied) {

      # クロスバリデーションの分割ごとにループ
      furrr::future_map_dfr(df.cv$splits, function(split, model) {

        # 訓練/学習 データ
        df.cv.train <- rsample::training(split)
        df.cv.test  <- rsample::testing(split)

        # 学習
        fit <- parsnip::fit(
          model,
          diff_SOC ~
            Current
          + Current:Voltage
          + Current:dV
          + dA
          + dA:Voltage
          + dA:dV
          + dA2:Voltage
          + Current:dA:dV
          +  eff_Power
          + Battery_Temp_degC
          
          + flg_var_dA_05_scaled_gt_n50
          ,
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
      # 評価
      train_rmse,
      test_rmse
    )
)
df.results
