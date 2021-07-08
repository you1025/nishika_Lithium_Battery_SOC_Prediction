

# Data Load ---------------------------------------------------------------

# Load Train Data
load_train_data <- function(dirpath) {

  # Load Train File
  load_train_file <- function(filepath) {
    readr::read_csv(
      file = filepath,
      col_types = cols(
        TimeStamp = col_datetime(format = "%m/%d/%Y %I:%M:%S %p"),
        Voltage = col_double(),
        Current = col_double(),
        Ah = col_double(),
        Wh = col_double(),
        Power = col_double(),
        Battery_Temp_degC = col_double(),
        Time = col_double(),
        Chamber_Temp_degC = col_double(),
        SOC = col_double(),
        `Drive Cycle` = col_character()
      )
    ) %>%

      dplyr::rename(
        Drive_Cycle = `Drive Cycle`
      ) %>%

      # 実験タイプを識別するための変数をデータの先頭に追加
      dplyr::mutate(
        experiment_type = stringr::str_c(
          dplyr::case_when(
            Chamber_Temp_degC  < 0 ~ "n",
            Chamber_Temp_degC == 0 ~ "z",
            Chamber_Temp_degC  > 0 ~ "p"
          ),
          abs(Chamber_Temp_degC),
          Drive_Cycle,
          sep = "_"
        )
      ) %>%
      dplyr::relocate(experiment_type)
  }

  # 指定パスの csv 一覧を取得
  list.files(path = dirpath, pattern = "*.csv", full.names = F) %>%

    purrr::map_dfr(function(filename) {
      # ファイル読み込み
      df.tmp.data <- load_train_file(filepath = stringr::str_c(dirpath, filename, sep = "/"))

      # 待機期間を除去
      idx.start <- which.max(abs(cumsum(df.tmp.data$Current)) > 0)
      start_at <- df.tmp.data[idx.start,]$Time
      df.tmp.data %>%
        dplyr::filter(Time >= start_at) %>%
        dplyr::mutate(Time = Time - start_at) %>%
        dplyr::distinct()
    })
}

# Load Test Data
load_test_data <- function(path) {

  readr::read_csv(
    file = path,
    col_types = cols(
      ID = col_double(),
      TimeStamp = col_datetime(format = "%m/%d/%Y %I:%M:%S %p"),
      Voltage = col_double(),
      Current = col_double(),
      Power = col_double(),
      Battery_Temp_degC = col_double(),
      Time = col_double(),
      Chamber_Temp_degC = col_double(),
      `Drive Cycle` = col_character()
    )
  ) %>%

    dplyr::rename(
      Drive_Cycle = `Drive Cycle`
    ) %>%

    # 実験タイプを識別するための変数をデータの先頭に追加
    dplyr::mutate(
      experiment_type = stringr::str_c(
        dplyr::case_when(
          Chamber_Temp_degC  < 0 ~ "n",
          Chamber_Temp_degC == 0 ~ "z",
          Chamber_Temp_degC  > 0 ~ "p"
        ),
        abs(Chamber_Temp_degC),
        Drive_Cycle,
        sep = "_"
      )        
    ) %>%
    dplyr::relocate(experiment_type)
}


# Cleansing ---------------------------------------------------------------

clean_data <- function(data, flg_test = F) {

  # 訓練データの時のみ SOC への処理を行う
  if(!flg_test) {
    df.tmp.data <- data %>%

      # 実験タイプ毎に時系列の処理
      dplyr::group_by(experiment_type) %>%
      dplyr::mutate(
        # SOC
        prev_SOC = lag(SOC),
        diff_SOC = SOC - prev_SOC
      ) %>%
      dplyr::ungroup()
  } else {
    df.tmp.data <- data
  }

  df.tmp.data %>%

    # 実験タイプ毎に時系列の処理
    dplyr::group_by(experiment_type) %>%
    dplyr::mutate(
      # 経過時間
      prev_Time = lag(Time),
      diff_Time = Time - prev_Time,

      # Ah
      Ah = cumsum(Current * ifelse(is.na(diff_Time), 0, diff_Time)) / 3600,

      # V
      prev_V = lag(Voltage),
      diff_V = Voltage - prev_V,
      dV = ifelse(diff_Time > 0, diff_V / diff_Time, 0),
      cumsum_V = cumsum(Voltage),
      ma10_dV = slider::slide_dbl(dV, mean, na.rm = T, .before = 9),
      
      # A
      prev_A = lag(Current),
      diff_A = Current - prev_A,
      dA = ifelse(diff_Time > 0, diff_A / diff_Time, 0),
      cumsum_A = cumsum(Current),
      ma10_dA = slider::slide_dbl(dA, mean, na.rm = T, .before = 9),

      # Power
      prev_Power = lag(Power),
      diff_Power = Power - prev_Power,
      dPower = ifelse(diff_Time > 0, diff_Power / diff_Time, 0),
      avg_Power = (prev_Power + Power) / 2,
      cumsum_Power = cumsum(Power),
      ma10_Power = slider::slide_dbl(Power, mean, na.rm = T, .before = 9),

      # 温度
      ma10_Battery_Temp_degC = slider::slide_dbl(Battery_Temp_degC, mean, na.rm = T, .before = 9)
    ) %>%
    dplyr::ungroup() %>%

    dplyr::mutate(
      # V
      dV2 = dV^2,

      # A
      dA2 = dA^2,
      flg_A_gt_n25 = (Current < -0.25),

      # 仕事
      eff_Power = - avg_Power * diff_Time,

      # 温度
      diff_Temp_degC = Battery_Temp_degC - Chamber_Temp_degC,
      eff_Temp_degC = diff_Temp_degC * diff_Time,

      # 仕事の正負
      flg_curr_gt_0 = (Current > 0),
      zero_one_curr_gt_0 = ifelse(Current == 0, 0, 1),

      # # 温度
      # seg_Chamber_Temp_degC = dplyr::case_when(
      #   Chamber_Temp_degC < 0  ~ "n",
      #   Chamber_Temp_degC == 0 ~ "z",
      #   Chamber_Temp_degC > 0  ~ "p"
      # ) %>%
      #   factor(levels = c("n", "z", "p")),
      # 
      # Chamber_Temp_degC = factor(Chamber_Temp_degC, levels = c(-20, -10, 0, 10, 25)),
    )
}


# Others ------------------------------------------------------------------

# Cross Validation
# type 毎に分割する
create_cv <- function(train_data) {

  # type 一覧
  types <- unique(train_data$experiment_type)

  # type 毎に分割したリストを作成
  purrr::map(types, function(str_type) {

    lst.ids <- train_data %>%

      dplyr::mutate(
        id = dplyr::row_number(),
        flg_target = (experiment_type == str_type)
      ) %>%

      # 分割用 id のリストを作成
      {
        data <- (.)
        list(
          analysis   = dplyr::filter(data, !flg_target)$id,
          assessment = dplyr::filter(data,  flg_target)$id
        )
      }

    rsample::make_splits(lst.ids, train_data)
  }) %>%

    # tibble に変換
    rsample::manual_rset(
      ids = stringr::str_c("Sample", stringr::str_pad(1:15, 2, "left", "0"))
    )
}

calc_a <- function(data) {

  model.lm <- data %>%

    dplyr::group_by(experiment_type) %>%
    dplyr::mutate(
      Ah = cumsum(Current * ifelse(is.na(diff_Time), 0, diff_Time)) / 3600
    ) %>%
    dplyr::ungroup() %>%

    lm(data = ., formula = SOC ~ Ah)
  
  100 / model.lm$coefficients[2]
}
