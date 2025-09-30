import json
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from models import LSTMAttentionClassifier, TransformerClassifier, PatchTSTClassifier, TCNClassifier
from data_pipeline import prepare_dataloaders
from trainer import train_model
from onnx_exporter import export_model_to_onnx
from evaluate_model import evaluate_model_comprehensive
from evaluate_model import predict_proba, find_best_thresholds
from config import get_project_configs
from regime import get_regime_series, latest_regime

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_data(symbol: str, days: int, timeframe: str, no_cache: bool = False) -> pd.DataFrame:
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    
    symbol_safe = symbol.replace('/', '_').lower()
    cache_file = cache_dir / f"{symbol_safe}_{timeframe}_master.parquet"
    
    if no_cache or not cache_file.exists():
        if no_cache:
            logging.info(f"--no-cache is set. Forcing API fetch for {symbol} ({timeframe})...")
        else:
            logging.info(f"Master cache not found for {symbol} ({timeframe}). Fetching from API...")
        
        master_days_to_fetch = 365 * 5 
        df = fetch_alpaca_bars(symbol, days=master_days_to_fetch, timeframe=timeframe)
        
        if not df.empty:
            logging.info(f"Saving new master data to cache: {cache_file}")
            df.to_parquet(cache_file)
    else:
        logging.info(f"Loading master cached data from: {cache_file}")
        df = pd.read_parquet(cache_file)

    if not df.empty:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        return df[df.index >= cutoff_date]
    
    return pd.DataFrame()

def fetch_alpaca_bars(symbol: str, days: int, timeframe: str) -> pd.DataFrame:
    # key = os.getenv("ALPACA_API_KEY")
    # sec = os.getenv("ALPACA_SECRET_KEY")
    # if not key or not sec:
    #     raise ValueError("ALPACA_API_KEY/ALPACA_SECRET_KEY가 .env 파일에 설정되어야 합니다.")
    
    client = CryptoHistoricalDataClient()
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    tf_map = {"15m": TimeFrame(15, TimeFrameUnit.Minute), "1h": TimeFrame.Hour, "4h": TimeFrame(4, TimeFrameUnit.Hour), "1d": TimeFrame.Day}
    if timeframe not in tf_map:
        raise ValueError(f"지원하지 않는 timeframe: {timeframe}")

    req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=tf_map[timeframe], start=start_time, end=end_time)
    
    try:
        bars = client.get_crypto_bars(req)
        if bars.df.empty:
            logging.warning(f"No bars returned for {symbol} with timeframe {timeframe}")
            return pd.DataFrame()
        
        df = bars.df.reset_index().rename(columns={"time": "timestamp"})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logging.error(f"Alpaca API 호출 중 오류 발생: {e}")
        return pd.DataFrame()

def run_pipeline(symbol: str, timeframe: str, args: argparse.Namespace, specific_config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"\n{'#'*30} Starting Pipeline for {symbol} @ {timeframe} {'#'*30}")

    if 'AIP_MODEL_DIR' in os.environ:
        output_base_dir = Path(os.environ['AIP_MODEL_DIR'])
        logging.info(f"Running on Vertex AI. Output base directory set to GCS: {output_base_dir}")
    else:
        output_base_dir = Path("./")
        logging.info(f"Running locally. Output base directory set to: {output_base_dir}")

    all_results: list[dict] = []
    ens_val_probs: dict[str, np.ndarray] = {}
    ens_test_probs: dict[str, np.ndarray] = {}
    ens_y_true_val: np.ndarray | None = None
    ens_y_true_test: np.ndarray | None = None

    for model_name in args.model_types:
        if model_name not in specific_config:
            logging.warning(f"Config for model type '{model_name}' not found. Skipping.")
            continue
        model_config = specific_config[model_name]

        days_to_fetch = model_config.get("days", 365)
        buffer_days = 30
        logging.info(f"--- [Step 1] Fetching data for {model_name.upper()} covering last {days_to_fetch} days ---")
        df_main = get_data(symbol, days=days_to_fetch, timeframe=timeframe, no_cache=args.no_cache)
        if df_main.empty:
            logging.error(f"Main dataframe ({timeframe}) fetching failed. Skipping.")
            continue

        all_higher_tfs = {"1h": 1, "4h": 4, "1d": 24}
        main_tf_hours = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}.get(timeframe, 0)
        higher_timeframes = {
            tf: get_data(symbol, days=days_to_fetch + buffer_days, timeframe=tf, no_cache=args.no_cache)
            for tf, hours in all_higher_tfs.items() if hours > main_tf_hours
        }
        logging.info(f"Loading higher timeframes for {timeframe}: {list(higher_timeframes.keys())}")

        seq_len_to_use = model_config.get("seq_len", args.seq_len)
        horizon_to_use = model_config.get("horizon", args.horizon)
        labeling_params = model_config.get("labeling_params", {}).copy()
        labeling_params["look_forward_candles"] = horizon_to_use
        logging.info(f"--- [Step 2] Preparing DataLoaders (seq_len={seq_len_to_use}, horizon={horizon_to_use}) ---")

        data_package = prepare_dataloaders(
            df_main=df_main,
            higher_dfs=higher_timeframes,
            seq_len=seq_len_to_use,
            batch_size=args.batch_size,
            use_sampler=args.use_sampler,
            labeling_params=labeling_params
        )
        if data_package is None:
            logging.error(f"Data preparation failed for {model_name}. Skipping.")
            continue

        logging.info(f"\n{'='*25} Processing Model: {model_name.upper()} {'='*25}")
        model_class = args.model_map[model_name]
        model_args = {"input_size": data_package["input_size"], "num_classes": 3, **model_config.get("arch_params", {})}
        model = model_class(**model_args).to(device)

        out_dir = (output_base_dir / "models" / symbol.replace('/', '_').lower())
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"{model_name}_{timeframe}_model.pt"
        scaler_path = out_dir / f"{model_name}_{timeframe}_scaler.pkl"
        onnx_path = out_dir / f"{model_name}_{timeframe}_model.onnx"

        if not args.skip_train:
            logging.info(f"--- [Step 3a] Training {model_name.upper()} ---")
            y_train = data_package["train_loader"].dataset.y
            counts = np.bincount(y_train, minlength=3)
            total = counts.sum() if counts.sum() > 0 else 1
            ratios = counts / total
            auto_sampler = (ratios[0] > 0.55) or (ratios[1] < 0.20) or (ratios[2] < 0.20)
            use_sampler_flag = args.use_sampler or auto_sampler
            logging.info(f"Class ratios [H,B,S]={ratios.round(3).tolist()} | use_sampler={use_sampler_flag}")

            trained_model, best_val_score = train_model(
                model,
                data_package["train_loader"].dataset.X, data_package["train_loader"].dataset.y,
                data_package["val_loader"].dataset.X, data_package["val_loader"].dataset.y,
                epochs=model_config["epochs"], batch_size=args.batch_size,
                lr=model_config.get("lr", 3e-4),
                patience=args.patience, device=device,
                model_type=model_name, use_weighted_sampler=use_sampler_flag,
                weight_decay=model_config.get("weight_decay", 1e-4),
                scheduler_params=model_config.get("scheduler_params")
            )
            logging.info(f"Training complete for {model_name.upper()}. Best validation F1 Macro score: {best_val_score:.5f}")
            torch.save(trained_model.state_dict(), model_path)
            joblib.dump(data_package["scaler"], scaler_path)
            logging.info(f"Artifacts saved: {model_path}, {scaler_path}")

        logging.info(f"--- [Step 3b] Evaluating {model_name.upper()} ---")
        if not model_path.exists():
            logging.error(f"Model file not found for evaluation: {model_path}. Skipping.")
            continue

        eval_model = model_class(**model_args).to(device)
        eval_model.load_state_dict(torch.load(model_path, map_location=device))

        decision_thresholds = model_config.get("decision_thresholds")
        try:
            y_true_val, probs_val = predict_proba(eval_model, data_package["val_loader"], device)
            if not decision_thresholds:
                decision_thresholds, _ = find_best_thresholds(y_true_val, probs_val)
                logging.info(f"Auto-derived decision thresholds from validation: {decision_thresholds}")
            else:
                logging.info(f"Using decision thresholds for inference: {decision_thresholds}")
        except Exception as e:
            logging.warning(f"Failed to derive decision thresholds from validation: {e}")
            y_true_val, probs_val = predict_proba(eval_model, data_package["val_loader"], device)

        y_true_test, probs_test = predict_proba(eval_model, data_package["test_loader"], device)
        ens_val_probs[model_name] = probs_val
        ens_test_probs[model_name] = probs_test
        ens_y_true_val = y_true_val if ens_y_true_val is None else ens_y_true_val
        ens_y_true_test = y_true_test if ens_y_true_test is None else ens_y_true_test

        results = evaluate_model_comprehensive(eval_model, data_package["test_loader"], device, thresholds=decision_thresholds)
        results["Model"] = f"{model_name.upper()}_{timeframe}"
        all_results.append(results)

        try:
            reports_dir = output_base_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            cm_path = reports_dir / f"cm_{symbol.replace('/', '_')}_{timeframe}_{model_name}.png"
            from evaluate_model import plot_confusion_matrix
            plot_confusion_matrix(results['confusion_matrix'], class_names=['HOLD', 'BUY', 'SELL'], title=f'CM for {results["Model"]}')
            plt.savefig(cm_path)
            plt.close()
            logging.info(f"Confusion matrix saved to: {cm_path}")
        except Exception as plot_e:
            logging.error(f"@@@Could not plot or save confusion matrix for {model_name}: {plot_e}", exc_info=True)

        logging.info(f"--- [Step 3c] Exporting {model_name.upper()} to ONNX ---")
        export_model_to_onnx(eval_model, model_name, str(onnx_path), seq_len_to_use, data_package["input_size"])

    if len(ens_val_probs) >= 2:
        try:
            used_models = sorted(list(ens_val_probs.keys()))
            logging.info(f"Computing ensemble with models: {used_models}")

            val_stack = np.stack([ens_val_probs[m] for m in used_models], axis=0)
            test_stack = np.stack([ens_test_probs[m] for m in used_models], axis=0)
            avg_val = val_stack.mean(axis=0)
            avg_test = test_stack.mean(axis=0)

            ens_thresholds, _ = find_best_thresholds(ens_y_true_val, avg_val)

            def _apply_thresholds(probs: np.ndarray, thresholds: dict) -> np.ndarray:
                th_buy = float(thresholds.get("buy", 0.5))
                th_sell = float(thresholds.get("sell", 0.5))
                p_hold, p_buy, p_sell = probs[:, 0], probs[:, 1], probs[:, 2]
                preds = np.zeros(len(probs), dtype=np.int64)
                buy_mask = p_buy >= th_buy
                sell_mask = p_sell >= th_sell
                both_mask = buy_mask & sell_mask
                preds[buy_mask & ~both_mask] = 1
                preds[sell_mask & ~both_mask] = 2
                preds[both_mask] = (p_sell[both_mask] > p_buy[both_mask]).astype(np.int64) + 1
                return preds

            y_pred = _apply_thresholds(avg_test, ens_thresholds)
            precision, recall, f1, support = precision_recall_fscore_support(
                ens_y_true_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0
            )
            ens_result = {
                "macro_f1": f1_score(ens_y_true_test, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(ens_y_true_test, y_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(ens_y_true_test, y_pred, labels=[0, 1, 2]),
                "class_metrics": {
                    'Hold': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0], 'support': support[0]},
                    'Buy':  {'precision': precision[1], 'recall': recall[1], 'f1': f1[1], 'support': support[1]},
                    'Sell': {'precision': precision[2], 'recall': recall[2], 'f1': f1[2], 'support': support[2]},
                },
                "signal_accuracy": (precision[1] + precision[2]) / 2 if (support[1] + support[2]) > 0 else 0.0,
                "Model": f"ENSEMBLE_{timeframe}({','.join([m.upper() for m in used_models])})"
            }
            all_results.append(ens_result)
        except Exception as e:
            logging.error(f"Failed to compute ensemble metrics: {e}", exc_info=True)

    if all_results:
        logging.info("\n" + "="*80)
        logging.info(f"PERFORMANCE REPORT for {symbol} @ {timeframe}")
        report_df = pd.DataFrame(all_results).sort_values("macro_f1", ascending=False)
        print(report_df[['Model', 'macro_f1', 'weighted_f1', 'signal_accuracy']].to_string(index=False))
        reports_dir = output_base_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"report_{symbol.replace('/', '_')}_{timeframe}.csv"
        report_df.to_csv(report_path, index=False, float_format="%.5f")
        logging.info(f"@@@Performance report saved to: {report_path}")
        logging.info("=" * 80)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(42)
    load_dotenv()

    all_configs = get_project_configs()

    symbols_str = os.getenv("SYMBOLS", "BTC/USD")
    timeframes_str = os.getenv("TIMEFRAMES", "15m")
    
    symbols_to_train = [s.strip() for s in symbols_str.split(',')]
    timeframes_to_train = [t.strip() for t in timeframes_str.split(',')]

    parser = argparse.ArgumentParser(description="Production-Grade Crypto Prediction Pipeline")
    parser.add_argument("--model_types", type=lambda s: [item.strip() for item in s.split(',')], default=["lstm_attention", "patchtst", "tcn"])
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=4)
    # parser.add_argument("--quantile_upper", type=float, default=0.7)
    # parser.add_argument("--quantile_lower", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_sampler", action='store_true')
    parser.add_argument("--skip_train", action='store_true')
    parser.add_argument("--no-cache", action='store_true', help="Force refetching data from API, ignoring local cache.")
    args = parser.parse_args()
    
    args.model_map = {
        "lstm_attention": LSTMAttentionClassifier,
        "transformer": TransformerClassifier,
        "patchtst": PatchTSTClassifier,
        "tcn": TCNClassifier
    }
    
    if 'AIP_MODEL_DIR' in os.environ:
        input_base_dir = Path(os.environ['AIP_MODEL_DIR'])
    else:
        input_base_dir = Path("./")

    for symbol in symbols_to_train:
        for timeframe in timeframes_to_train:
            try:
                specific_config = all_configs[symbol][timeframe]
            except KeyError:
                logging.error(f"Config not found for {symbol}/{timeframe} in config.py. Skipping.")
                continue

            hyperparams_dir = input_base_dir / "hyperparams"
            best_params_path = hyperparams_dir / f"best_params_{symbol.replace('/', '_')}_{timeframe}.json"
            
            if best_params_path.exists():
                logging.info(f"Loading best parameters from {best_params_path}")
                with open(best_params_path, 'r') as f:
                    best_params = json.load(f)
                
                for model_name, params in best_params.items():
                    if model_name in specific_config:
                        specific_config[model_name].update(params)
            else:
                logging.warning(f"Best parameter file not found: {best_params_path}. Using default configs from config.py.")

            try:
                run_pipeline(symbol, timeframe, args, specific_config)
            except Exception as e:
                logging.error(f"FATAL ERROR during pipeline for {symbol} @ {timeframe}: {e}", exc_info=True)
                continue

if __name__ == "__main__":
    main()