import json
from pathlib import Path
import optuna
import logging
import argparse
import os
import torch
import numpy as np
from dotenv import load_dotenv
from data_pipeline import prepare_dataloaders, prepare_walkforward_folds
from evaluate_model import predict_proba, find_best_thresholds

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from run import get_data
from data_pipeline import prepare_dataloaders
from trainer import train_model
from models import (
    TransformerClassifier,
    LSTMAttentionClassifier,
    PatchTSTClassifier,
    TCNClassifier 
)
from config import get_project_configs

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _class_for(model_type: str):
    if model_type == "lstm_attention":
        return LSTMAttentionClassifier
    if model_type == "transformer":
        return TransformerClassifier
    if model_type == "patchtst":
        return PatchTSTClassifier
    if model_type == "tcn":
        return TCNClassifier
    raise ValueError(f"Unsupported model type: {model_type}")

def _fix_transformer_dims(d_model: int, nhead: int) -> tuple[int, int]:
    if d_model % nhead == 0:
        return d_model, nhead
    candidates = [16, 8, 4, 2, 1]
    cand = next((h for h in candidates if h <= nhead and d_model % h == 0), None)
    if cand is not None:
        return d_model, cand
    fixed = (d_model // nhead) * nhead
    fixed = max(fixed, nhead)
    return fixed, nhead


def objective(trial: optuna.Trial, args: argparse.Namespace, symbol: str, timeframe: str, specific_config: dict, model_type: str):
    try:
        model_cfg = specific_config[model_type]
        search_space = model_cfg["search_space"]

        params = {}
        for name, conf in search_space.items():
            suggest = getattr(trial, f"suggest_{conf['type']}")
            params[name] = suggest(name, *conf["args"], **conf.get("kwargs", {}))

        days_to_fetch = model_cfg.get("days", 730)
        
        df_main = get_data(symbol, days=days_to_fetch, timeframe=timeframe, no_cache=args.no_cache)
        if df_main.empty:
            raise optuna.exceptions.TrialPruned(f"Empty main data for {symbol} {timeframe}")

        buffer_days = 30
        all_higher_tfs = {"1h": 1, "4h": 4, "1d": 24}
        main_tf_hours = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24}.get(timeframe, 0)

        higher_timeframes = {}
        for tf, hours in all_higher_tfs.items():
            if hours > main_tf_hours:
                higher_timeframes[tf] = get_data(symbol, days=days_to_fetch + buffer_days, timeframe=tf, no_cache=args.no_cache)

        seq_len = int(params["seq_len"])
        horizon = int(params["horizon"])
        labeling_params = model_cfg.get("labeling_params", {}).copy()
        labeling_params["look_forward_candles"] = horizon

        if "pt" in params and "sl" in params:
            labeling_params["pt_sl"] = [float(params["pt"]), float(params["sl"])]
        if "volatility_span" in params:
            labeling_params["volatility_span"] = int(params["volatility_span"])

        use_wf = getattr(args, "wf_folds", 1) and args.wf_folds > 1

        if use_wf:
            folds = prepare_walkforward_folds(
                df_main=df_main,
                higher_dfs=higher_timeframes,
                seq_len=seq_len,
                batch_size=args.batch_size,
                use_sampler=args.use_sampler,
                labeling_params=labeling_params,
                n_folds=args.wf_folds,
                val_ratio=args.wf_val_ratio,
                min_train_ratio=0.6
            )
            if not folds:
                raise optuna.exceptions.TrialPruned("Walk-forward folds preparation failed")
        else:
            data_package = prepare_dataloaders(
                df_main=df_main,
                higher_dfs=higher_timeframes,
                seq_len=seq_len,
                batch_size=args.batch_size,
                use_sampler=args.use_sampler,
                labeling_params=labeling_params
            )
            if data_package is None:
                raise optuna.exceptions.TrialPruned("Data preparation failed")

        ModelClass = _class_for(model_type)
        arch_params = {}
        if model_type == "lstm_attention":
            arch_params = {
                "hidden_size": int(params["hidden_size"]),
                "num_layers": int(params["num_layers"]),
                "dropout": float(params["dropout"]),
            }
        elif model_type == "transformer":
            d_model = int(params["d_model"])
            nhead = int(params["nhead"])
            d_model, nhead = _fix_transformer_dims(d_model, nhead)
            trial.set_user_attr("nhead_effective", nhead)
            trial.set_user_attr("d_model_effective", d_model)
            arch_params = {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": int(params["num_layers"]),
                "dropout": float(params["dropout"]),
                "dim_feedforward": int(params["dim_feedforward"])
            }
        elif model_type == "patchtst":
            d_model = int(params["d_model"])
            nhead = int(params["nhead"])
            d_model, nhead = _fix_transformer_dims(d_model, nhead)
            trial.set_user_attr("nhead_effective", nhead)
            trial.set_user_attr("d_model_effective", d_model)
            arch_params = {
                "patch_len": int(params["patch_len"]),
                "stride": int(params["stride"]),
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": int(params["num_layers"]),
                "dropout": float(params["dropout"]),
                "dim_feedforward": int(params["dim_feedforward"])
            }
        elif model_type == "tcn":
            arch_params = {
                "channels": params["channels"],
                "kernel_size": int(params["kernel_size"]),
                "dropout": float(params["dropout"]),
            }

        model_args = {
            "input_size": data_package["input_size"],
            "num_classes": 3,
            **arch_params,
        }
        model = ModelClass(**model_args).to(args.device)

        epochs = int(model_cfg.get("epochs", args.epochs))
        scheduler_params = None
        if model_type not in ("transformer", "patchtst"):
            scheduler_params = {
                "T_0": int(params["T_0"]),
                "T_mult": int(params["T_mult"])
            }

        wf_scores = []
        wf_thresholds = []
        if use_wf:
            for k, f in enumerate(folds, start=1):
                y_train = f["train_loader"].dataset.y
                counts = np.bincount(y_train, minlength=3)
                total = counts.sum() if counts.sum() > 0 else 1
                ratios = counts / total
                auto_sampler = (ratios[0] > 0.55) or (ratios[1] < 0.20) or (ratios[2] < 0.20)
                use_sampler_flag = bool(args.use_sampler or auto_sampler)
                logging.info(f"[WF fold {k}] Class ratios [H,B,S]={ratios.round(3).tolist()} | use_sampler={use_sampler_flag}")
                trial.set_user_attr(f"wf{k}_class_ratios", ratios.tolist())
                trial.set_user_attr(f"wf{k}_use_sampler", use_sampler_flag)

                model = ModelClass(input_size=f["input_size"], num_classes=3, **arch_params).to(args.device)
                _, _ = train_model(
                    model,
                    f["train_loader"].dataset.X, f["train_loader"].dataset.y,
                    f["val_loader"].dataset.X, f["val_loader"].dataset.y,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=float(params["lr"]),
                    patience=args.patience,
                    device=args.device,
                    model_type=model_type,
                    use_weighted_sampler=use_sampler_flag,
                    weight_decay=float(params["weight_decay"]),
                    scheduler_params=scheduler_params,
                    early_metric="val_f1_macro_thr"
                )
                y_true, probs = predict_proba(model, f["val_loader"], args.device)
                best_th, best_score = find_best_thresholds(y_true, probs)
                wf_scores.append(best_score)
                wf_thresholds.append(best_th)

                trial.report(float(np.mean(wf_scores)), step=k)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned(f"Pruned at fold {k} with score {np.mean(wf_scores):.4f}")

            mean_score = float(np.mean(wf_scores))
            avg_buy = float(np.mean([t["buy"] for t in wf_thresholds]))
            avg_sell = float(np.mean([t["sell"] for t in wf_thresholds]))
            trial.set_user_attr("decision_thresholds", {"buy": avg_buy, "sell": avg_sell})
            return mean_score
        else:
            data_package = prepare_dataloaders(
                df_main=df_main,
                higher_dfs=higher_timeframes,
                seq_len=seq_len,
                batch_size=args.batch_size,
                use_sampler=args.use_sampler,
                labeling_params=labeling_params
            )
            if data_package is None:
                raise optuna.exceptions.TrialPruned("Data preparation failed")

            y_train = data_package["train_loader"].dataset.y
            counts = np.bincount(y_train, minlength=3)
            total = counts.sum() if counts.sum() > 0 else 1
            ratios = counts / total
            auto_sampler = (ratios[0] > 0.55) or (ratios[1] < 0.20) or (ratios[2] < 0.20)
            use_sampler_flag = bool(args.use_sampler or auto_sampler)
            logging.info(f"[Single split] Class ratios [H,B,S]={ratios.round(3).tolist()} | use_sampler={use_sampler_flag}")
            trial.set_user_attr("class_ratios", ratios.tolist())
            trial.set_user_attr("use_sampler", use_sampler_flag)

            model = ModelClass(input_size=data_package["input_size"], num_classes=3, **arch_params).to(args.device)
            _, _ = train_model(
                model,
                data_package["train_loader"].dataset.X, data_package["train_loader"].dataset.y,
                data_package["val_loader"].dataset.X, data_package["val_loader"].dataset.y,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=float(params["lr"]),
                patience=args.patience,
                device=args.device,
                model_type=model_type,
                use_weighted_sampler=use_sampler_flag,
                weight_decay=float(params["weight_decay"]),
                scheduler_params=scheduler_params,
                early_metric="val_f1_macro_thr"
            )
            y_true, probs = predict_proba(model, data_package["val_loader"], args.device)
            best_th, best_score = find_best_thresholds(y_true, probs)
            trial.set_user_attr("decision_thresholds", best_th)
            return best_score

    except optuna.exceptions.TrialPruned as e:
        logging.warning(f"Trial pruned: {e}")
        raise
    except Exception as e:
        logging.error(f"Trial failed with error: {e}", exc_info=True)
        return 0.0


def run_single_tuning_study(symbol: str, timeframe: str, args: argparse.Namespace, all_configs: dict):
    logging.info(f"\n{'#'*30} Tuning {symbol} @ {timeframe} {'#'*30}")

    try:
        specific_config = all_configs[symbol][timeframe]
    except KeyError:
        logging.error(f"Config not found for {symbol}/{timeframe}. Skipping.")
        return

    defined_models = list(specific_config.keys())
    models_to_tune = [m for m in args.model_types if m in defined_models]
    if not models_to_tune:
        logging.warning(f"No model types to tune for {symbol}/{timeframe} based on args and config. Skipping.")
        return

    for model_type in models_to_tune:
        logging.info(f"\n--- Starting tuning for model: {model_type.upper()} ---")

        study_name = f"tuning-{symbol.replace('/', '_')}-{timeframe}-{model_type}"
        storage_name = f"sqlite:///{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
        )

        logging.info(f"Starting Optuna study '{study_name}' with n_trials={args.n_trials}, timeout={args.timeout}s")
        study.optimize(
            lambda tr: objective(tr, args, symbol, timeframe, specific_config, model_type),
            n_trials=args.n_trials,
            timeout=args.timeout,
        )

        logging.info(f"\nTuning finished for {model_type.upper()}. Updating best parameters...")
        
        if 'AIP_MODEL_DIR' in os.environ:
            output_base_dir = os.environ['AIP_MODEL_DIR']
            logging.info(f"Running on Vertex AI. Output base directory set to GCS: {output_base_dir}")
        else:
            output_base_dir = "./"
            logging.info(f"Running locally. Output base directory set to: {output_base_dir}")

        hyperparams_dir = Path("./hyperparams")
        hyperparams_dir.mkdir(exist_ok=True)
        best_params_path = hyperparams_dir / f"best_params_{symbol.replace('/', '_')}_{timeframe}.json"

        saved = {}
        if best_params_path.exists():
            with open(best_params_path, "r") as f:
                try: saved = json.load(f)
                except json.JSONDecodeError: saved = {}
        
        trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            logging.warning(f"No completed trials found for {model_type}. Cannot save params.")
            continue

        best_t = max(trials, key=lambda t: t.value)
        logging.info(f"Best trial for {model_type.upper()}: val_f1_macro={best_t.value:.5f}")

        if model_type not in saved: saved[model_type] = {}
        saved[model_type]["lr"] = best_t.params["lr"]
        saved[model_type]["weight_decay"] = best_t.params["weight_decay"]
        if "decision_thresholds" in best_t.user_attrs:
            saved[model_type]["decision_thresholds"] = best_t.user_attrs["decision_thresholds"]

        arch_params = {}
        if model_type == "lstm_attention":
            arch_params = {
                "hidden_size": best_t.params["hidden_size"],
                "num_layers": best_t.params["num_layers"],
                "dropout": best_t.params["dropout"]
            }
        elif model_type == "transformer":
            arch_params = {
                "d_model": best_t.params["d_model"],
                "nhead": best_t.params["nhead"],
                "num_layers": best_t.params["num_layers"],
                "dropout": best_t.params["dropout"],
                "dim_feedforward": best_t.params["dim_feedforward"]
            }
        
        saved[model_type]["seq_len"] = best_t.params["seq_len"]
        saved[model_type]["horizon"] = best_t.params["horizon"]
        saved[model_type]["arch_params"] = arch_params
        if "pt" in best_t.params and "sl" in best_t.params:
            saved[model_type]["labeling_params"] = saved[model_type].get("labeling_params", {})
            saved[model_type]["labeling_params"]["pt_sl"] = [best_t.params["pt"], best_t.params["sl"]]
        if "volatility_span" in best_t.params:
            saved[model_type]["labeling_params"] = saved[model_type].get("labeling_params", {})
            saved[model_type]["labeling_params"]["volatility_span"] = best_t.params["volatility_span"]

        
        if model_type != "transformer":
            saved[model_type]["scheduler_params"] = {
                "T_0": best_t.params["T_0"],
                "T_mult": best_t.params["T_mult"]
            }

        with open(best_params_path, "w") as f:
            json.dump(saved, f, indent=4)
        logging.info(f"Saved/Updated best parameters for {model_type.upper()} -> {best_params_path}")

    logging.info("=" * 80)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(42)
    load_dotenv()

    all_configs = get_project_configs()

    symbols_str = os.getenv("SYMBOLS", "BTC/USD")
    timeframes_str = os.getenv("TIMEFRAMES", "15m")

    symbols = [s.strip() for s in symbols_str.split(",")]
    timeframes = [t.strip() for t in timeframes_str.split(",")]

    parser = argparse.ArgumentParser(description="Automated Hyperparameter Tuning")
    parser.add_argument("--tune_mode", type=str, default="quick", choices=["quick", "deep"], help="quick=빠른 대략 탐색(50trials), deep=정밀 탐색(200trials)")
    parser.add_argument("--model_types", type=lambda s: [x.strip() for x in s.split(',')], default=["lstm_attention","patchtst","tcn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--no-cache", action='store_true', help="Force refetching data from API, ignoring local cache.")
    parser.add_argument("--smoke-test", action="store_true", help="Run with n_trials=1 for quick pipeline validation.")
    parser.add_argument("--wf-folds", type=int, default=1, help="Walk-forward folds (>=2 사용 시 활성화)")
    parser.add_argument("--wf-val-ratio", type=float, default=0.12, help="각 폴드의 검증 구간 비율")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke_test:
        args.n_trials = 1
        args.timeout = 300
        logging.warning("--- SMOKE TEST MODE: Running with n_trials=1 ---")
    elif args.tune_mode == "deep":
        args.n_trials = 200
        args.timeout = 3600 * 12
    else:
        args.n_trials = 50
        args.timeout = 3600 * 2

    for sym in symbols:
        for tf in timeframes:
            try:
                run_single_tuning_study(sym, tf, args, all_configs)
            except Exception as e:
                logging.error(f"FATAL ERROR during tuning for {sym} @ {tf}: {e}", exc_info=True)
                continue


if __name__ == "__main__":
    main()