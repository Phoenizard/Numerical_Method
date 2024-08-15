# Config File

config = {
    # Dataset name
    "dataset_name": "example_3_20D",

    # Model Hyperparameters
    "m": 10000,
    "batch_size": 256,
    "epochs": 10,
    "lr": 1,

    # SAV related parameters
    "C": 100,
    "_lambda": 4,
    "ratio_n": 0.99,

    # Adaptive related parameters
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-8,

    # SPM related parameters
    "J": 10,
    "h": 0.000001,

    # Add more parameters here if needed
    "recording": False
}
