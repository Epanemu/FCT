
# names of datasets and paths to data
# most have numerical inaccuracies, that are being solved by rounding off data
DATASETS = {
    "categorical": {
        "compass": "data/openml/categorical/compass.pickle",
        # "covertype": "data/openml/categorical/covertype.pickle", # >50 000 data points
        "electricity": "data/openml/categorical/electricity.pickle",
        "eye_movements": "data/openml/categorical/eye_movements.pickle",
        "KDDCup09_upselling": "data/openml/categorical/KDDCup09_upselling.pickle",
        "rl": "data/openml/categorical/rl.pickle",
        # "road-safety": "data/openml/categorical/road-safety.pickle", # >50 000 data points
    },
    "numerical": {
        "bank-marketing": "data/openml/numerical/bank-marketing.pickle",
        "california": "data/openml/numerical/california.pickle",
        # "covertype": "data/openml/numerical/covertype.pickle", # >50 000 data points
        "credit": "data/openml/numerical/credit.pickle",
        # "electricity": "data/openml/numerical/electricity.pickle", # also in categorical
        # "eye_movements": "data/openml/numerical/eye_movements.pickle", # also in categorical
        # "Higgs": "data/openml/numerical/Higgs.pickle", # >50 000 data points
        "house_16H": "data/openml/numerical/house_16H.pickle",
        # "jannis": "data/openml/numerical/jannis.pickle", # >50 000 data points
        "kdd_ipums_la_97-small": "data/openml/numerical/kdd_ipums_la_97-small.pickle",
        "MagicTelescope": "data/openml/numerical/MagicTelescope.pickle",
        # "MiniBooNE": "data/openml/numerical/MiniBooNE.pickle", # >50 000 data points
        "phoneme": "data/openml/numerical/phoneme.pickle",
        "pol": "data/openml/numerical/pol.pickle",
        "wine": "data/openml/numerical/wine.pickle",
    },
}

ALL_DATASETS = {
    "categorical": {
        "compass": "data/openml/categorical/compass.pickle",
        "covertype": "data/openml/categorical/covertype.pickle", # >50 000 data points
        "electricity": "data/openml/categorical/electricity.pickle",
        "eye_movements": "data/openml/categorical/eye_movements.pickle",
        "KDDCup09_upselling": "data/openml/categorical/KDDCup09_upselling.pickle",
        "rl": "data/openml/categorical/rl.pickle",
        "road-safety": "data/openml/categorical/road-safety.pickle", # >50 000 data points
    },
    "numerical": {
        "bank-marketing": "data/openml/numerical/bank-marketing.pickle",
        "california": "data/openml/numerical/california.pickle",
        "covertype": "data/openml/numerical/covertype.pickle", # >50 000 data points
        "credit": "data/openml/numerical/credit.pickle",
        "electricity": "data/openml/numerical/electricity.pickle", # also in categorical
        "eye_movements": "data/openml/numerical/eye_movements.pickle", # also in categorical
        "Higgs": "data/openml/numerical/Higgs.pickle", # >50 000 data points
        "house_16H": "data/openml/numerical/house_16H.pickle",
        "jannis": "data/openml/numerical/jannis.pickle", # >50 000 data points
        "kdd_ipums_la_97-small": "data/openml/numerical/kdd_ipums_la_97-small.pickle",
        "MagicTelescope": "data/openml/numerical/MagicTelescope.pickle",
        "MiniBooNE": "data/openml/numerical/MiniBooNE.pickle", # >50 000 data points
        "phoneme": "data/openml/numerical/phoneme.pickle",
        "pol": "data/openml/numerical/pol.pickle",
        "wine": "data/openml/numerical/wine.pickle",
    },
}

# difficulty distinction based on best found models by sklearn
# least epsilon is the least difference in feature values, if they are all normalized to 0-1 with only linear scaling
DATASET_INFO = {
    "categorical": {
        "compass": {
            "n_points": 16644,
            "n_features": 17,
            "n_classes": 2,
            "least_eps": 0.000105429625724823,
            "difficulty": "HARD",
            "path": "data/openml/categorical/compass.pickle",
            "pure_xgb_benchmark_estim_acc": 0.772,
        },
        "covertype": {
            "n_points": 423680,
            "n_features": 54,
            "n_classes": 2,
            "least_eps": 0.000139431121026212,
            "difficulty": "MEDIUM",
            "path": "data/openml/categorical/covertype.pickle",
            "pure_xgb_benchmark_estim_acc": 0.863,
        },
        "electricity": {
            "n_points": 38474,
            "n_features": 8,
            "n_classes": 2,
            "least_eps": 0.00000199999999999853,
            "difficulty": "MEDIUM",
            "path": "data/openml/categorical/electricity.pickle",
            "pure_xgb_benchmark_estim_acc": 0.877,
        },
        "eye_movements": {
            "n_points": 7608,
            "n_features": 23,
            "n_classes": 2,
            "least_eps": 0.0000002078856537191,
            "difficulty": "HARD",
            "path": "data/openml/categorical/eye_movements.pickle",
            "pure_xgb_benchmark_estim_acc": 0.642,
        },
        "KDDCup09_upselling": {
            "n_points": 5128,
            "n_features": 49,
            "n_classes": 2,
            "least_eps": 0.00000000738310695914,
            "difficulty": "EASY",
            "path": "data/openml/categorical/KDDCup09_upselling.pickle",
            "pure_xgb_benchmark_estim_acc": 0.802,
        },
        "rl": {
            "n_points": 4970,
            "n_features": 12,
            "n_classes": 2,
            "least_eps": 0.00071326676176886,
            "difficulty": "HARD",
            "path": "data/openml/categorical/rl.pickle",
            "pure_xgb_benchmark_estim_acc": 0.778,
        },
        "road-safety": {
            "n_points": 111762,
            "n_features": 32,
            "n_classes": 2,
            "least_eps": 0.00000009759691815825,
            "difficulty": "MEDIUM",
            "path": "data/openml/categorical/road-safety.pickle",
            "pure_xgb_benchmark_estim_acc": 0.765,
        },
    },
    "numerical": {
        "bank-marketing": {
            "n_points": 10578,
            "n_features": 7,
            "n_classes": 2,
            "least_eps": 0.0000118236852061915,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/bank-marketing.pickle",
            "pure_xgb_benchmark_estim_acc": 0.812,
        },
        "california": {
            "n_points": 20634,
            "n_features": 8,
            "n_classes": 2,
            "least_eps": 0.00000000057462132969,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/california.pickle",
            "pure_xgb_benchmark_estim_acc": 0.907,
        },
        "covertype": {
            "n_points": 566602,
            "n_features": 32,
            "n_classes": 2,
            "least_eps": 0.000139348468883305,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/covertype.pickle",
            "pure_xgb_benchmark_estim_acc": 0.818,
        },
        "credit": {
            "n_points": 16714,
            "n_features": 10,
            "n_classes": 2,
            "least_eps": 0.00000000000018181818,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/credit.pickle",
            "pure_xgb_benchmark_estim_acc": 0.765,
        },
        "electricity": {
            "n_points": 38474,
            "n_features": 7,
            "n_classes": 2,
            "least_eps": 0.00000199999999999853,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/electricity.pickle",
            "pure_xgb_benchmark_estim_acc": 0.876,
        },
        "eye_movements": {
            "n_points": 7608,
            "n_features": 20,
            "n_classes": 2,
            "least_eps": 0.0000002078856537191,
            "difficulty": "HARD",
            "path": "data/openml/numerical/eye_movements.pickle",
            "pure_xgb_benchmark_estim_acc": 0.662,
        },
        "Higgs": {
            "n_points": 940160,
            "n_features": 24,
            "n_classes": 2,
            "least_eps": 0.00000000079703732503,
            "difficulty": "HARD",
            "path": "data/openml/numerical/Higgs.pickle",
            "pure_xgb_benchmark_estim_acc": 0.718,
        },
        "house_16H": {
            "n_points": 13488,
            "n_features": 16,
            "n_classes": 2,
            "least_eps": 0.00000013656426167859,
            "difficulty": "EASY",
            "path": "data/openml/numerical/house_16H.pickle",
            "pure_xgb_benchmark_estim_acc": 0.884,
        },
        "jannis": {
            "n_points": 57580,
            "n_features": 54,
            "n_classes": 2,
            "least_eps": 3.91071689517844e-19,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/jannis.pickle",
            "pure_xgb_benchmark_estim_acc": 0.777,
        },
        "kdd_ipums_la_97-small": {
            "n_points": 5188,
            "n_features": 20,
            "n_classes": 2,
            "least_eps": 0.00000099019802970307,
            "difficulty": "EASY",
            "path": "data/openml/numerical/kdd_ipums_la_97-small.pickle",
            "pure_xgb_benchmark_estim_acc": 0.884,
        },
        "MagicTelescope": {
            "n_points": 13376,
            "n_features": 10,
            "n_classes": 2,
            "least_eps": 0.0000000967907290983,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/MagicTelescope.pickle",
            "pure_xgb_benchmark_estim_acc": 0.858,
        },
        "MiniBooNE": {
            "n_points": 72998,
            "n_features": 50,
            "n_classes": 2,
            "least_eps": 6.24457083264751e-15,
            "difficulty": "EASY",
            "path": "data/openml/numerical/MiniBooNE.pickle",
            "pure_xgb_benchmark_estim_acc": 0.939,
        },
        "phoneme": {
            "n_points": 3172,
            "n_features": 5,
            "n_classes": 2,
            "least_eps": 0.00000014787828667151,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/phoneme.pickle",
            "pure_xgb_benchmark_estim_acc": 0.881,
        },
        "pol": {
            "n_points": 10082,
            "n_features": 26,
            "n_classes": 2,
            "least_eps": 0.00534759358288761,
            "difficulty": "EASY",
            "path": "data/openml/numerical/pol.pickle",
            "pure_xgb_benchmark_estim_acc": 0.980,
        },
        "wine": {
            "n_points": 2554,
            "n_features": 11,
            "n_classes": 2,
            "least_eps": 0.000621504039766574,
            "difficulty": "MEDIUM",
            "path": "data/openml/numerical/wine.pickle",
            "pure_xgb_benchmark_estim_acc": 0.810,
        },
    },
}
