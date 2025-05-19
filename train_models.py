from models.predictor import LSTMPredictor
from models.predictor import GRUPredictor

LSTMTrainer = LSTMPredictor(
    data_pkl="data/traffic_model_ready.pkl",
    models_dir="lstm_saved_models"
)
LSTMTrainer.train_all()

GRUTrainer = GRUPredictor(
    data_pkl="data/traffic_model_ready.pkl",
    models_dir="gru_saved_models"
)
GRUTrainer.train_all()