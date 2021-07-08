library(tidyverse)
library(tidymodels)
library(furrr)

source("models/02.XGBoost/01/functions.R")


# Data Load & Feature Engineering ---------------------------------------------------------------

df.train <- load_train_data("data/01.input/train") %>%
  clean_data()

df.cv <- create_cv(df.train)


# Model Definition --------------------------------------------------------

model <- parsnip::boost_tree(
  mode = "regression",
  trees = 2000,
#  learn_rate = 0.1,
  tree_depth = parsnip::varying(),
  stop_iter = 10
) %>%
  parsnip::set_engine(
    engine = "xgboost",
    eval_metric = "rmse"
  )


# Hyper Parameter ---------------------------------------------------------

df.grid.params <- dials::grid_regular(
  dials::tree_depth(range = c(10, 10)),
  levels = 1
)
df.grid.params


# Cross Validation --------------------------------------------------------

# https://github.com/rstudio/rstudio/issues/6692
future::plan(future::multisession, workers = 8, setup_strategy = "sequential")

system.time(

  df.results <-

    # ハイパーパラメータをモデルに適用
    purrr::pmap(df.grid.params, function(...) {
      params <- list(...)

      parsnip::set_args(
        model,
        tree_depth = params$tree_depth,
      )
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
            Voltage +
            cumsum_V + # 不要の可能性あり(要検討)
            ma10_dV +
            dV +
            dV2 + # 不要の可能性あり(要検討)
            Current +
            cumsum_A + # 不要の可能性あり(要検討)
            dA + # 不要の可能性あり(要検討)
            dA2 + # 不要の可能性あり(要検討)
            ma10_dA +
            Power +
            dPower +
            avg_Power +
            cumsum_Power + # 不要の可能性あり(要検討)
            ma10_Power +
            eff_Power +
            Battery_Temp_degC +
            diff_Temp_degC +
            ma10_Battery_Temp_degC +
            eff_Temp_degC +
            Chamber_Temp_degC
            ,
          df.cv.train
        )
        print(fit)

        dplyr::bind_rows(
          calc_pred_evals(fit, df.cv.train) %>%
            dplyr::mutate(.metric = stringr::str_c("train", .metric, sep = "_")),
          calc_pred_evals(fit, df.cv.test) %>%
            dplyr::mutate(.metric = stringr::str_c("test",  .metric, sep = "_"))
        ) %>%
          tidyr::pivot_wider(names_from = .metric, values_from = .estimate)

      }, model = model.applied, .options = furrr::furrr_options(seed = 5963L)) %>%

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
      tree_depth,

      # 評価
      train_rmse,
      test_rmse
    )
)
df.results


