from models.lstm_predictor import LSTMPredictor

predictor = LSTMPredictor(
    data_pkl="data/traffic_model_ready.pkl",
    models_dir="saved_models"
)
predictor.train_all()