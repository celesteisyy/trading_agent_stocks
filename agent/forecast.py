import pickle
import pandas as pd
import numpy as np
from analysis import AnalysisAgent


class ForecastAgent:
    """
    ForecastAgent runs AnalysisAgent to compute indicators,
    then loads a model (sklearn or GRU) to produce n‑step forecasts
    and simple buy/sell signals.
    """
    def __init__(
        self,
        df_raw: pd.DataFrame,
        model_path: str,
        model_type: str = 'sklearn',
        seq_len: int = 60,
        device: str = 'cpu'
    ):
        # 1) compute technical indicators via AnalysisAgent
        ana = AnalysisAgent(df_raw)
        ana.compute_all_indicators()
        # merge raw prices + indicators into one DataFrame
        self.df = pd.concat([ana.df, ana.indicators], axis=1)

        # 2) store parameters
        self.seq_len    = seq_len
        self.device     = device
        self.model_type = model_type.lower()
        # 3) load the forecasting model
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        if self.model_type == 'sklearn':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model

        elif self.model_type == 'gru':
            import torch
            model = torch.load(model_path, map_location=self.device)
            model.to(self.device).eval()
            return model

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def forecast(self, n_steps: int = 1):
        """
        Generate n_steps forecasts.
        - For sklearn: use an autoregressive loop on the 'Close' price.
        - For GRU: expect model(seq) -> tensor of shape [1, n_steps].
        Returns a list of length n_steps.
        """
        if self.model_type == 'sklearn':
            last_val = float(self.df['Close'].iloc[-1])
            preds = []
            for _ in range(n_steps):
                pred = self.model.predict([[last_val]])[0]
                preds.append(pred)
                last_val = pred
            return preds

        elif self.model_type == 'gru':
            import torch
            seq_vals = self.df['Close'].values[-self.seq_len:]
            seq = (
                torch.tensor(seq_vals, dtype=torch.float32)
                     .unsqueeze(0)      # batch dim
                     .unsqueeze(-1)     # feature dim
                     .to(self.device)
            )
            with torch.no_grad():
                out = self.model(seq)  # assume shape [1, n_steps]
            return out.squeeze(0).cpu().tolist()

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def generate_signals(self, n_steps: int = 1):
        """
        Simple rule: if forecast > last known price => 'buy', else 'sell'.
        Returns list of length n_steps.
        """
        preds      = self.forecast(n_steps)
        last_close = float(self.df['Close'].iloc[-1])
        return ['buy' if p > last_close else 'sell' for p in preds]


if __name__ == '__main__':
    # smoke test with randomly generated data

    # generate a random walk for Close prices
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    close = np.cumsum(np.random.randn(100)) + 100

    df_raw = pd.DataFrame({
        'Open':   close + np.random.randn(100) * 0.5,
        'High':   close + np.random.rand(100),
        'Low':    close - np.random.rand(100),
        'Close':  close,
        'Volume': np.random.randint(1000, 2000, size=100),
    }, index=dates)

    # create a dummy sklearn model: predict last_val + noise
    class RandomWalkModel:
        def predict(self, X):
            # add small random noise to last_val
            return [x[0] + np.random.randn() * 0.1 for x in X]

    model_path = 'agent/dummy_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(RandomWalkModel(), f)

    # instantiate and run
    agent = ForecastAgent(
        df_raw=df_raw,
        model_path=model_path,
        model_type='sklearn',
        seq_len=10
    )
    forecasts = agent.forecast(n_steps=5)
    signals   = agent.generate_signals(n_steps=5)

    print("Random-walk Forecasts:", np.round(forecasts, 3).tolist())
    print("Generated Signals:    ", signals)
