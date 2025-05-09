"""
GRU Model for time series prediction in trading strategies.
This module handles model definition, training, and prediction.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional, List, Union

# Configure logger
logger = logging.getLogger(__name__)

class GRUModel(nn.Module):
    """
    GRU model for time series prediction with technical indicators and sentiment.
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Prediction layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU forward
        out, _ = self.gru(x, h0)
        
        # Get prediction using the last output
        out = self.fc(out[:, -1, :])
        
        return out

class GRUModelTrainer:
    """
    Handles training and prediction for GRU models.
    """
    def __init__(self, seq_len=60, device=None):
        """
        Initialize the GRU model trainer.
        
        Args:
            seq_len: Sequence length for time series prediction
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.seq_len = seq_len
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"GRU trainer initialized with device: {self.device}")
        
        # Normalization parameters
        self.X_mean = None
        self.X_std = None
        self.model = None
    
    def prepare_data(self, price_df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame] = None, 
                    tech_indicators: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for GRU model training.
        
        Args:
            price_df: DataFrame with price data (needs 'Close' column)
            sentiment_df: Optional DataFrame with sentiment data
            tech_indicators: Optional DataFrame with technical indicators
            
        Returns:
            Tuple of (X_sequences, y_values) for training
        """
        # 1. Select features for training
        X_data = price_df[['Close']].values
        
        # 2. Add technical indicators if available
        if tech_indicators is not None:
            tech_features = ['rsi_14', 'macd', 'bb_upper']
            tech_values = []
            
            for feature in tech_features:
                if feature in tech_indicators.columns:
                    tech_values.append(tech_indicators[feature].values.reshape(-1, 1))
            
            if tech_values:
                tech_array = np.column_stack(tech_values)
                X_data = np.column_stack((X_data, tech_array))
        
        # 3. Add sentiment if available
        if sentiment_df is not None and not sentiment_df.empty:
            # Align sentiment data with price dates
            aligned_sentiment = pd.DataFrame(index=price_df.index)
            aligned_sentiment['sentiment'] = np.nan
            
            sentiment_dates = sentiment_df.index
            for date in aligned_sentiment.index:
                closest_date = sentiment_dates[sentiment_dates <= date]
                if not closest_date.empty:
                    closest_date = closest_date[-1]
                    aligned_sentiment.loc[date, 'sentiment'] = sentiment_df.loc[closest_date, 'avg_compound']
            
            # Fill missing values
            aligned_sentiment = aligned_sentiment.fillna(method='ffill').fillna(0)
            
            # Add sentiment as a feature
            X_with_sentiment = np.column_stack((X_data, aligned_sentiment['sentiment'].values))
            X_data = X_with_sentiment
        
        # 4. Normalize data (important for GRU performance)
        self.X_mean = np.mean(X_data, axis=0)
        self.X_std = np.std(X_data, axis=0)
        self.X_std[self.X_std == 0] = 1  # Avoid division by zero
        X_normalized = (X_data - self.X_mean) / self.X_std
        
        # 5. Create sequences for time series prediction
        X_sequences = []
        y_values = []
        
        for i in range(len(X_normalized) - self.seq_len):
            X_sequences.append(X_normalized[i:i+self.seq_len])
            # Target is the next closing price
            y_values.append(X_data[i+self.seq_len, 0])  # 0 is Close price
        
        return np.array(X_sequences), np.array(y_values)
    
    def train(self, price_df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame] = None, 
             tech_indicators: Optional[pd.DataFrame] = None, epochs: int = 50, 
             batch_size: int = 32, learning_rate: float = 0.001) -> GRUModel:
        """
        Train GRU model on historical price and technical data.
        
        Args:
            price_df: DataFrame with price data (needs 'Close' column)
            sentiment_df: Optional DataFrame with sentiment data
            tech_indicators: Optional DataFrame with technical indicators
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            
        Returns:
            Trained GRU model
        """
        logger.info("Training GRU model on historical data...")
        
        # 1. Prepare data
        X_sequences, y_values = self.prepare_data(price_df, sentiment_df, tech_indicators)
        
        # 2. Convert to tensors
        X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(1)
        
        # 3. Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 4. Initialize model
        input_dim = X_sequences.shape[2]  # Number of features
        logger.info(f"Creating GRU model with {input_dim} input features")
        self.model = GRUModel(input_dim=input_dim)
        self.model.to(self.device)
        
        # 5. Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 6. Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # 7. Set to evaluation mode
        self.model.eval()
        logger.info("GRU model training completed")
        
        return self.model
    
    def save_model(self, model_path: str) -> bool:
        """
        Save trained model and normalization parameters.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Create a dictionary with model and normalization parameters
            save_dict = {
                'model': self.model,
                'X_mean': self.X_mean,
                'X_std': self.X_std,
                'input_dim': next(self.model.parameters()).shape[1]
            }
            
            # Save to file
            torch.save(save_dict, model_path)
            logger.info(f"Saved GRU model and parameters to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Load model and normalization parameters.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Load from file
            saved_dict = torch.load(model_path, map_location=self.device)
            
            # Check if it's a dictionary with parameters or just a model
            if isinstance(saved_dict, dict) and 'model' in saved_dict:
                self.model = saved_dict['model']
                self.X_mean = saved_dict['X_mean']
                self.X_std = saved_dict['X_std']
            else:
                # Old format - just the model
                self.model = saved_dict
                logger.warning("Loaded model without normalization parameters")
            
            # Set model to evaluation mode
            self.model.to(self.device).eval()
            logger.info(f"Loaded GRU model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> float:
        """
        Make prediction with GRU model.
        
        Args:
            features: Array of shape [seq_len, n_features]
            
        Returns:
            Predicted value
        """
        if self.model is None:
            logger.error("No model available for prediction")
            return 0.0
        
        try:
            # Normalize features
            if self.X_mean is not None and self.X_std is not None:
                features = (features - self.X_mean) / self.X_std
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                pred = self.model(features_tensor)
            
            return float(pred.item())
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0
    
    def predict_signal(self, current_price: float, features: np.ndarray, threshold: float = 0.01) -> int:
        """
        Generate trading signal based on price prediction.
        
        Args:
            current_price: Current price
            features: Feature array for prediction
            threshold: Price change threshold for generating signals
            
        Returns:
            Signal: 1 (buy), -1 (sell), or 0 (hold)
        """
        predicted_price = self.predict(features)
        
        # Calculate percentage difference
        pct_diff = (predicted_price - current_price) / current_price
        
        # Generate signal based on threshold
        if pct_diff > threshold:
            return 1  # Buy signal
        elif pct_diff < -threshold:
            return -1  # Sell signal
        else:
            return 0  # Hold signal


# Example usage if run as script
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # Simple test
    test_data = np.random.rand(100, 5)  # 100 days, 5 features
    test_df = pd.DataFrame(test_data, columns=['Close', 'Feature1', 'Feature2', 'Feature3', 'Feature4'])
    
    trainer = GRUModelTrainer(seq_len=10)
    print("Training simple test model...")
    trainer.train(test_df[['Close']], epochs=10, batch_size=4)
    
    # Test prediction
    test_features = np.random.rand(10, 5)  # 10 time steps, 5 features
    prediction = trainer.predict(test_features)
    print(f"Test prediction: {prediction}")
    
    signal = trainer.predict_signal(1.0, test_features)
    print(f"Trading signal: {signal}")