
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