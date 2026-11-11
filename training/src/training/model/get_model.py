

from training.models.rnn import RNNForecaster
from training.models.lstm import LSTMForecaster


MODEL_DEFAULTS = {
    "rnn": {
        "input_size": 6,
        "hidden_size": 32,
        "num_layers": 1,
        "output_size": 1,
        "dropout": 0.0,
    },
    "lstm": {
        "input_size": 6,
        "hidden_size": 32,
        "num_layers": 1,
        "output_size": 1,
        "dropout": 0.0,
    },
}


MODEL_REGISTRY = {
    "rnn": RNNForecaster,
    "lstm": LSTMForecaster,
}


def get_model(name: str, **params):
    name = name.lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    model_params = MODEL_DEFAULTS[name].copy()
    model_params.update(params)

    return MODEL_REGISTRY[name](**model_params)

