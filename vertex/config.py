def get_project_configs():
    
    common_search_space = {
        "lr": {"type": "float", "args": [1e-5, 1e-2], "kwargs": {"log": True}},
        "weight_decay": {"type": "float", "args": [1e-6, 1e-3], "kwargs": {"log": True}},
    }

    search_space_by_timeframe = {
        "15m": {
            "seq_len": {"type": "int", "args": [48, 192], "kwargs": {"step": 16}},
            "horizon": {"type": "int", "args": [4, 16], "kwargs": {"step": 4}},
        },
        "1h": {
            "seq_len": {"type": "int", "args": [24, 96], "kwargs": {"step": 8}},
            "horizon": {"type": "int", "args": [2, 8], "kwargs": {"step": 2}},
        }
    }

    common_scheduler_search_space = {
        "T_0": {"type": "int", "args": [5, 20], "kwargs": {"step": 5}},
        "T_mult": {"type": "categorical", "args": [[1, 2]]},
    }

    configs = {
        "BTC/USD": {
            "15m": {
                "lstm_attention": {
                    "days": 500, "epochs": 45,
                    "arch_params": {"hidden_size": 64, "num_layers": 2, "dropout": 0.3},
                    "labeling_params": {
                            "pt_sl": [1.5, 1.0],
                            "volatility_span": 100
                        },
                    "scheduler_params": {"T_0": 10, "T_mult": 2},
                    "search_space": {
                        **common_search_space,
                        **search_space_by_timeframe["15m"],
                        **common_scheduler_search_space,
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "hidden_size": {"type": "categorical", "args": [[32, 64, 128]]},
                        "num_layers": {"type": "int", "args": [1, 3]},
                        "dropout": {"type": "float", "args": [0.2, 0.5]},
                    }
                },
                "transformer": {
                    "days": 730, "epochs": 50,
                    "arch_params": {"d_model": 128, "nhead": 8, "num_layers": 4, "dropout": 0.15},                    
                    "labeling_params": {
                            "pt_sl": [1.5, 1.0],
                            "volatility_span": 100
                        },
                    "search_space": {
                        **common_search_space,
                        **search_space_by_timeframe["15m"],
                        
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "d_model": {"type": "categorical", "args": [[64, 128, 256]]},
                        
                        "nhead": {"type": "categorical", "args": [[4, 8, 16]]},
                        
                        "num_layers": {"type": "int", "args": [3, 8]},
                        
                        "dropout": {"type": "float", "args": [0.1, 0.4]},
                        
                        "dim_feedforward": {"type": "int", "args": [256, 1024], "kwargs": {"step": 256}}
                    }
                },
                
                "patchtst": {
                    "days": 730, "epochs": 50,
                    "arch_params": {"patch_len": 16, "stride": 8, "d_model": 128, "nhead": 8, "num_layers": 4, "dropout": 0.15, "dim_feedforward": 512},
                    "labeling_params": {"pt_sl": [1.5, 1.0], "volatility_span": 100},
                    "search_space": {
                        "lr": {"type": "float", "args": [1e-5, 1e-3], "kwargs": {"log": True}},
                        "weight_decay": {"type": "float", "args": [1e-6, 1e-3], "kwargs": {"log": True}},
                        "seq_len": {"type": "int", "args": [64, 256], "kwargs": {"step": 16}},
                        "horizon": {"type": "int", "args": [4, 16], "kwargs": {"step": 4}},
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "patch_len": {"type": "int", "args": [8, 32], "kwargs": {"step": 4}},
                        "stride": {"type": "int", "args": [4, 16], "kwargs": {"step": 4}},
                        "d_model": {"type": "categorical", "args": [[64, 128, 256]]},
                        "nhead": {"type": "categorical", "args": [[4, 8, 16]]},
                        "num_layers": {"type": "int", "args": [3, 8]},
                        "dropout": {"type": "float", "args": [0.1, 0.4]},
                        "dim_feedforward": {"type": "int", "args": [256, 1024], "kwargs": {"step": 256}}
                    }
                },
                "tcn": {
                    "days": 500, "epochs": 45,
                    "arch_params": {"channels": [64, 128, 128], "kernel_size": 3, "dropout": 0.2},
                    "labeling_params": {"pt_sl": [1.5, 1.0], "volatility_span": 100},
                    "search_space": {
                        "lr": {"type": "float", "args": [1e-5, 1e-2], "kwargs": {"log": True}},
                        "weight_decay": {"type": "float", "args": [1e-6, 1e-3], "kwargs": {"log": True}},
                        "seq_len": {"type": "int", "args": [48, 192], "kwargs": {"step": 16}},
                        "horizon": {"type": "int", "args": [4, 16], "kwargs": {"step": 4}},
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "channels": {"type": "categorical", "args": [[[64,128,128],[64,128,256],[128,128,256]]]},
                        "kernel_size": {"type": "int", "args": [3, 7], "kwargs": {"step": 2}},
                        "dropout": {"type": "float", "args": [0.1, 0.4]},
                        "T_0": {"type": "int", "args": [5, 20], "kwargs": {"step": 5}},
                        "T_mult": {"type": "categorical", "args": [[1, 2]]},
                    }
                },
            },
            "1h": {
                "lstm_attention": {
                    "days": 700, "epochs": 45,
                    "arch_params": {"hidden_size": 128, "num_layers": 2, "dropout": 0.3},
                    "labeling_params": {
                            "pt_sl": [1.5, 1.0],
                            "volatility_span": 100
                        },
                    "scheduler_params": {"T_0": 10, "T_mult": 1},
                    "search_space": {
                        **common_search_space,
                        **search_space_by_timeframe["1h"],
                        **common_scheduler_search_space,
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "hidden_size": {"type": "categorical", "args": [[64, 128, 256]]},
                        "num_layers": {"type": "int", "args": [2, 4]},
                        "dropout": {"type": "float", "args": [0.2, 0.5]},
                    }
                },
                 "transformer": {
                    "days": 1000, "epochs": 50,
                    "arch_params": {"d_model": 128, "nhead": 8, "num_layers": 4, "dropout": 0.1},
                    "labeling_params": {
                            "pt_sl": [1.5, 1.0],
                            "volatility_span": 100
                        },
                    "search_space": {
                        **common_search_space,
                        **search_space_by_timeframe["1h"],
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "d_model": {"type": "categorical", "args": [[64, 128]]},
                        "nhead": {"type": "categorical", "args": [[4, 8]]},
                        "num_layers": {"type": "int", "args": [3, 6]},
                        "dropout": {"type": "float", "args": [0.1, 0.3]},
                    }
                },
                "patchtst": {
                    "days": 1000, "epochs": 50,
                    "arch_params": {"patch_len": 16, "stride": 8, "d_model": 128, "nhead": 4, "num_layers": 3, "dropout": 0.1, "dim_feedforward": 512},
                    "labeling_params": {"pt_sl": [1.5, 1.0], "volatility_span": 100},
                    "search_space": {
                        "lr": {"type": "float", "args": [1e-5, 1e-3], "kwargs": {"log": True}},
                        "weight_decay": {"type": "float", "args": [1e-6, 1e-3], "kwargs": {"log": True}},
                        "seq_len": {"type": "int", "args": [48, 128], "kwargs": {"step": 8}},
                        "horizon": {"type": "int", "args": [2, 8], "kwargs": {"step": 2}},
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "patch_len": {"type": "int", "args": [8, 24], "kwargs": {"step": 4}},
                        "stride": {"type": "int", "args": [4, 16], "kwargs": {"step": 4}},
                        "d_model": {"type": "categorical", "args": [[64, 128]]},
                        "nhead": {"type": "categorical", "args": [[4, 8]]},
                        "num_layers": {"type": "int", "args": [2, 5]},
                        "dropout": {"type": "float", "args": [0.1, 0.3]},
                        "dim_feedforward": {"type": "int", "args": [256, 1024], "kwargs": {"step": 256}}
                    }
                },
                "tcn": {
                    "days": 700, "epochs": 45,
                    "arch_params": {"channels": [64, 128, 128], "kernel_size": 3, "dropout": 0.2},
                    "labeling_params": {"pt_sl": [1.5, 1.0], "volatility_span": 100},
                    "search_space": {
                        "lr": {"type": "float", "args": [1e-5, 1e-2], "kwargs": {"log": True}},
                        "weight_decay": {"type": "float", "args": [1e-6, 1e-3], "kwargs": {"log": True}},
                        "seq_len": {"type": "int", "args": [24, 96], "kwargs": {"step": 8}},
                        "horizon": {"type": "int", "args": [2, 8], "kwargs": {"step": 2}},
                        "pt": {"type": "float", "args": [0.5, 3.0]},
                        "sl": {"type": "float", "args": [0.5, 3.0]},
                        "volatility_span": {"type": "int", "args": [50, 300], "kwargs": {"step": 25}},
                        "channels": {"type": "categorical", "args": [[[64,128,128],[64,128,256],[128,256,256]]]},
                        "kernel_size": {"type": "int", "args": [3, 7], "kwargs": {"step": 2}},
                        "dropout": {"type": "float", "args": [0.1, 0.4]},
                        "T_0": {"type": "int", "args": [5, 20], "kwargs": {"step": 5}},
                        "T_mult": {"type": "categorical", "args": [[1, 2]]},
                    }
                }
            }
        },
    }

    configs["ETH/USD"] = configs["BTC/USD"]
    
    return configs