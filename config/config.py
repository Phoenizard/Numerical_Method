# Config File

config = {
    # Dataset name
    "dataset_name": "example_2_40D",

    # Model Hyperparameters
    "m": 100,
    "batch_size": 64,
    "epochs": 10000,
    "lr": 0.4,

    # SAV related parameters
    "C": 1,
    "_lambda": 4,
    "ratio_n": 0.99,

    # Adaptive related parameters
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-8,

    # SPM related parameters
    "J": 10,
    "h": 0.0001,

    # Add more parameters here if needed
    "recording": True
}
