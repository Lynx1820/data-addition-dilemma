from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.metrics import brier_score_loss
from functools import partial
import numpy as np
import pandas as pd
import os 
import argparse
import sys
from pathlib import Path
import torch 
import torch.nn as nn
import crypten 
from crypten import CrypTensor
from jaxtyping import Int, Float
from dataclasses import dataclass
from typing import Tuple, Optional, Union, List
import optuna

crypten.init()
torch.manual_seed(42)
torch.set_num_threads(1)

@dataclass
class ModelConfig:
    """Configuration for model hyperparameters"""
    eta0: float = 0.06
    max_iter: int = 1000
    patience: int = 5
    tol: float = 0.0001
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0

@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters"""
    data_path: str
    output_dir: str
    n_samples: int = 3000
    hospital_file: str = '../YAIB-cohorts/data/mortality24/eicu/above2000.txt'
    n_hospitals: int = 12
    train_split: float = 0.9
    numerical_eps: float = 1e-7
    clip_min: float = 0.01
    clip_max: float = 0.99

class BaseLogisticRegression(nn.Module):
    """Base class for logistic regression with common functionality"""
    
    def __init__(self, feature_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(feature_dim, 1, dtype=torch.float64)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sigmoid()
    
    def shuffle_data(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        permutation = torch.randperm(X.shape[0])
        return X[permutation], y[permutation]
    
    def adjust_learning_rate(self, lr: float, prev_loss: torch.Tensor, curr_loss: torch.Tensor, 
                           factor: int = 5, patience: int = 5) -> Tuple[float, int]:
        has_plateaued = self.loss_has_plateaued(prev_loss, curr_loss)
        if patience == 1 and has_plateaued: 
            lr = lr / factor 
            patience = self.config.patience
        else: 
            patience -= 1
        return lr, patience

    def loss_has_plateaued(self, previous_loss: torch.Tensor, current_loss: torch.Tensor) -> bool:
        return current_loss > previous_loss

class LogisticRegression(BaseLogisticRegression):
    """Plain PyTorch logistic regression implementation"""
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        criterion = torch.nn.BCELoss()
        prev_loss = torch.Tensor([float('inf')])
        patience, learning_rate = self.config.patience, self.config.eta0
        
        optimizer = torch.optim.SGD(
            self.parameters(), 
            learning_rate, 
            momentum=self.config.momentum, 
            dampening=self.config.dampening, 
            weight_decay=self.config.weight_decay
        )
        
        train_samples = int(X.shape[0] * 0.9)
        X_train, X_val = X[:train_samples], X[train_samples:]
        y_train, y_val = y[:train_samples], y[train_samples:]
        
        for epoch in range(self.config.max_iter):
            optimizer.zero_grad()
            output = self.forward(X_train)
            loss = criterion(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                output = self.forward(X_val)
                val_loss = criterion(output.squeeze(dim=1), y_val)
                
                if abs(prev_loss - val_loss) < self.config.tol or learning_rate < 1e-6:
                    break
                    
                learning_rate, patience = self.adjust_learning_rate(
                    learning_rate, prev_loss, val_loss, patience=patience
                )
                prev_loss = val_loss
                
                if epoch != 0 and epoch % 500 == 0:
                    print(f"epoch {epoch} loss: {val_loss}")
                    
            X, y = self.shuffle_data(X, y)
            X_train, X_val = X[:train_samples], X[train_samples:]
            y_train, y_val = y[:train_samples], y[train_samples:]

class EncryptedLogisticRegression(crypten.nn.Module):
    """Encrypted logistic regression using CrypTen"""
    
    def __init__(self, feature_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.linear = crypten.nn.Linear(feature_dim, 1)
        
    def forward(self, x: CrypTensor) -> CrypTensor:
        return self.linear(x).sigmoid()
    
    def shuffle_data(self, X: CrypTensor, y: CrypTensor) -> Tuple[CrypTensor, CrypTensor]:
        permutation = torch.randperm(X.shape[0])
        X_shuffled = X.index_select(0, permutation)
        y_shuffled = y.index_select(0, permutation)
        return X_shuffled, y_shuffled
    
    def fit(self, X: CrypTensor, y: CrypTensor) -> None:
        criterion = crypten.nn.BCELoss()
        prev_loss = crypten.cryptensor(torch.Tensor([float('inf')]))
        patience, learning_rate = self.config.patience, self.config.eta0
        
        optimizer = crypten.optim.SGD(
            self.parameters(), 
            learning_rate, 
            momentum=self.config.momentum, 
            dampening=self.config.dampening, 
            weight_decay=self.config.weight_decay
        )
        
        train_samples = int(X.shape[0] * 0.9)
        X_train, X_val = X[:train_samples], X[train_samples:]
        y_train, y_val = y[:train_samples], y[train_samples:]
        
        for epoch in range(self.config.max_iter):
            optimizer.zero_grad()
            output = self.forward(X_train)
            loss = criterion(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                output = self.forward(X_val)
                val_loss = criterion(output.squeeze(), y_val)
                
                if (prev_loss.sub(val_loss).abs().get_plain_text() < self.config.tol 
                    or learning_rate < 1e-6):
                    break
                    
                learning_rate, patience = self.adjust_learning_rate(
                    learning_rate, prev_loss, val_loss, patience=patience
                )
                prev_loss = val_loss
                
                if epoch != 0 and epoch % 500 == 0:
                    print(f"epoch {epoch} loss: {val_loss.get_plain_text()}")
                    
            X, y = self.shuffle_data(X, y)
            X_train, X_val = X[:train_samples], X[train_samples:]
            y_train, y_val = y[:train_samples], y[train_samples:]
    
    def adjust_learning_rate(self, lr: float, prev_loss: CrypTensor, curr_loss: CrypTensor, 
                           factor: int = 5, patience: int = 5) -> Tuple[float, int]:
        has_plateaued = self.loss_has_plateaued(prev_loss, curr_loss)
        if patience == 1:
            lr = crypten.where(has_plateaued, lr/factor, lr).get_plain_text()
            patience = crypten.where(has_plateaued, self.config.patience, patience).get_plain_text()
        else:
            patience = crypten.where(has_plateaued, patience-1, patience).get_plain_text()
        return lr, patience

    def loss_has_plateaued(self, previous_loss: CrypTensor, current_loss: CrypTensor) -> CrypTensor:
        return current_loss.gt(previous_loss)

class StandardScaler:
    """Custom standard scaler for PyTorch tensors"""
    
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.mean = None
        self.std = None
    
    def fit(self, x: torch.Tensor) -> None:
        n_samples = x.shape[0]
        self.mean = x.mean(dim=0)
        var = (x.sub(self.mean)).pow(2).sum(dim=0).div(n_samples)
        self.std = var.sqrt()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.sub(self.mean)).div(self.std + self.eps)

class EncryptedStandardScaler:
    """Encrypted standard scaler for CrypTen tensors"""
    
    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.mean = None
        self.std = None
    
    def fit(self, x: CrypTensor) -> None:
        n_samples = x.shape[0]
        self.mean = x.mean(dim=0)
        var = (x.sub(self.mean)).pow(2).sum(dim=0).div(n_samples)
        self.std = var.sqrt()

    def transform(self, x: CrypTensor) -> CrypTensor:
        return (x.sub(self.mean)).div(self.std.add(self.eps))

class DataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def get_hospital_data(self, hospital_id: int, split: str = 'train', 
                         max_samples: Optional[int] = None, 
                         sample_ratio: float = 1.0, 
                         rand_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load hospital data with optional sampling"""
        file_path = f'{self.config.data_path}/train{hospital_id}-test{hospital_id}/data.npz'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        hospital_data = np.load(file_path, allow_pickle=True)
        x_temp = hospital_data[split].item()['features']
        
        # Convert categorical to one-hot
        last_col = x_temp[:, -1].astype(int)
        one_hot = np.eye(4)[last_col] 
        x = np.concatenate([x_temp[:, :-1], one_hot], axis=1)
        y = hospital_data[split].item()['labels']
        xy = np.concatenate((x, y.reshape(-1, 1)), axis=1)
        
        # Apply sampling
        if sample_ratio < 1:
            x, y, xy = self._sample_data(x, y, xy, sample_ratio, rand_seed)
        elif max_samples is not None and len(x) > max_samples:
            x, y, xy = self._sample_data(x, y, xy, max_samples/len(x), rand_seed)
            
        return x, y, xy
    
    def _sample_data(self, x: np.ndarray, y: np.ndarray, xy: np.ndarray, 
                    ratio_or_count: float, rand_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample data randomly"""
        rng = np.random.default_rng(rand_seed)
        if ratio_or_count < 1:
            sample_size = int(len(x) * ratio_or_count)
        else:
            sample_size = int(ratio_or_count)
            
        indices = rng.choice(len(x), size=sample_size, replace=False)
        return x[indices], y[indices], xy[indices]
    
    def get_hospital_ids(self) -> List[int]:
        """Get list of hospital IDs from file"""
        if not os.path.exists(self.config.hospital_file):
            raise FileNotFoundError(f"Hospital file not found: {self.config.hospital_file}")
            
        df = pd.read_csv(self.config.hospital_file, header=None)
        return df[0].values[:self.config.n_hospitals].tolist()

class ModelFactory:
    """Factory for creating models and scalers"""
    
    @staticmethod
    def create_model(feature_dim: int, config: ModelConfig, encrypted: bool = False):
        """Create either encrypted or plain model"""
        if encrypted:
            return EncryptedLogisticRegression(feature_dim, config).encrypt()
        else:
            return LogisticRegression(feature_dim, config)
    
    @staticmethod
    def create_scaler(encrypted: bool = False, eps: float = 1e-7):
        """Create either encrypted or plain scaler"""
        if encrypted:
            return EncryptedStandardScaler(eps)
        else:
            return StandardScaler(eps)

class ModelTrainer:
    """Handles model training and prediction"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def train_model(self, model, scaler, features1: np.ndarray, features2: np.ndarray, 
                   encrypted: bool = False):
        """Train a model with two sets of features"""
        if encrypted:
            return self._train_encrypted_model(model, scaler, features1, features2)
        else:
            return self._train_plain_model(model, scaler, features1, features2)
    
    def _train_plain_model(self, model, scaler, features1: np.ndarray, features2: np.ndarray):
        """Train plain PyTorch model"""
        fea1_tensor = torch.tensor(features1)
        fea2_tensor = torch.tensor(features2)
        
        X_train = torch.cat([fea1_tensor, fea2_tensor], dim=0)
        Y_train = torch.from_numpy(
            np.concatenate((np.ones(len(features1)), np.zeros(len(features2))), axis=0)
        ).double()
        
        permutation = torch.randperm(len(X_train))
        X_train, Y_train = X_train[permutation], Y_train[permutation]
        
        scaler.fit(X_train)
        scaled_X_train = scaler.transform(X_train)
        model.fit(scaled_X_train, Y_train)
        
        return scaler, model
    
    def _train_encrypted_model(self, model, scaler, features1: np.ndarray, features2: np.ndarray):
        """Train encrypted CrypTen model"""
        encrypted_fea1 = crypten.cryptensor(torch.tensor(features1))
        encrypted_fea2 = crypten.cryptensor(torch.tensor(features2))
        
        enc_X_train = crypten.cat([encrypted_fea1, encrypted_fea2], dim=0)
        enc_Y_train = crypten.cryptensor(
            torch.Tensor(np.concatenate((np.ones(len(features1)), np.zeros(len(features2))), axis=0))
        )
        
        scaler.fit(enc_X_train)
        enc_scaled_X_train = scaler.transform(enc_X_train)
        model.fit(enc_scaled_X_train, enc_Y_train)
        
        return scaler, model
    
    def predict(self, model, scaler, features: np.ndarray, encrypted: bool = False) -> np.ndarray:
        """Make predictions with trained model"""
        if encrypted:
            enc_features = crypten.cryptensor(torch.tensor(features))
            scaled_features = scaler.transform(enc_features)
            predictions = model(scaled_features).get_plain_text()
        else:
            features_tensor = torch.from_numpy(features).double()
            scaled_features = scaler.transform(features_tensor)
            predictions = model(scaled_features).detach().numpy()
        
        return np.clip(predictions, 0.0, 1.0)

class ScoreComputer:
    """Computes pairwise scores between hospitals"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.trainer = ModelTrainer(config)
    
    def compute_pairwise_scores(self, hospital_ids: List[int], model_config: ModelConfig, 
                              encrypted: bool = False, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pairwise scores between hospitals"""
        n_hospitals = len(hospital_ids)
        results_x = np.zeros((n_hospitals, n_hospitals))
        results_xy = np.zeros((n_hospitals, n_hospitals))
        
        if debug:
            hospital_ids = hospital_ids[:2]
            
        for test_i, test_hos in enumerate(hospital_ids):
            for i, hos in enumerate(hospital_ids):
                if hos != test_hos:
                    # Load training data
                    x, _, xy = self.data_loader.get_hospital_data(
                        hos, 'train', max_samples=self.config.n_samples
                    )
                    x2, _, xy2 = self.data_loader.get_hospital_data(
                        test_hos, 'train', max_samples=self.config.n_samples
                    )
                    
                    # Load test data
                    x_val, _, xy_val = self.data_loader.get_hospital_data(hos, 'test')
                    
                    # Train and evaluate X model
                    x_model = ModelFactory.create_model(x.shape[1], model_config, encrypted)
                    x_scaler = ModelFactory.create_scaler(encrypted)
                    x_scaler, x_model = self.trainer.train_model(x_model, x_scaler, x, x2, encrypted)
                    x_predictions = self.trainer.predict(x_model, x_scaler, x_val, encrypted)
                    results_x[i, test_i] = x_predictions.mean()
                    
                    # Train and evaluate XY model
                    xy_model = ModelFactory.create_model(xy.shape[1], model_config, encrypted)
                    xy_scaler = ModelFactory.create_scaler(encrypted)
                    xy_scaler, xy_model = self.trainer.train_model(xy_model, xy_scaler, xy, xy2, encrypted)
                    xy_predictions = self.trainer.predict(xy_model, xy_scaler, xy_val, encrypted)
                    results_xy[i, test_i] = xy_predictions.mean()
        
        return results_x, results_xy
    
    def save_results(self, results_x: np.ndarray, results_xy: np.ndarray, 
                    model_config: ModelConfig, encrypted: bool = False) -> None:
        """Save results to files"""
        max_iter = 50 if hasattr(self, '_debug') and self._debug else model_config.max_iter
        
        path = (f"{self.config.output_dir}/max_it{max_iter}_eta0{model_config.eta0}_"
                f"alpha{model_config.weight_decay}_tol{model_config.tol}_"
                f"pat{model_config.patience}_mom{model_config.momentum}_"
                f"damp{model_config.dampening}_n{self.config.n_samples}")
        
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        
        prefix = "encrypted-" if encrypted else ""
        
        with open(save_dir / f'{prefix}score-x.npy', 'wb') as f:
            np.save(f, results_x)
        with open(save_dir / f'{prefix}score-xy.npy', 'wb') as f:
            np.save(f, results_xy)

class HyperparameterOptimizer:
    """Handles hyperparameter optimization using Optuna"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.trainer = ModelTrainer(config)
    
    def optimize(self, hospital_ids: List[int], n_trials: int = 100, 
                encrypted: bool = False) -> ModelConfig:
        """Optimize hyperparameters using Optuna"""
        study = optuna.create_study(direction='minimize')
        
        if encrypted:
            objective_fn = partial(self._objective_encrypted, 
                                 data_path=self.config.data_path, ids=hospital_ids)
        else:
            objective_fn = partial(self._objective_plaintext, 
                                 data_path=self.config.data_path, ids=hospital_ids)
        
        study.optimize(objective_fn, n_trials=n_trials)
        best_params = study.best_params
        
        return ModelConfig(
            eta0=best_params['eta0'],
            max_iter=best_params['max_iter'],
            patience=best_params['patience'],
            tol=best_params['tol'],
            momentum=best_params['momentum'],
            weight_decay=best_params['weight_decay'],
            dampening=best_params.get('damp', 0.0)
        )
    
    def _objective_encrypted(self, trial, data_path: str, ids: List[int]) -> float:
        """Objective function for encrypted model optimization"""
        return self._objective_common(trial, ids, encrypted=True)
    
    def _objective_plaintext(self, trial, data_path: str, ids: List[int]) -> float:
        """Objective function for plaintext model optimization"""
        return self._objective_common(trial, ids, encrypted=False)
    
    def _objective_common(self, trial, ids: List[int], encrypted: bool) -> float:
        """Common objective function logic"""
        # Sample hyperparameters
        config = ModelConfig(
            eta0=trial.suggest_float('eta0', 1e-5, 1e-1),
            max_iter=trial.suggest_int('max_iter', 100, 2000),
            patience=trial.suggest_int('patience', 1, 5),
            tol=trial.suggest_float('tol', 0.0001, 0.01, log=True),
            momentum=trial.suggest_float('momentum', 0.0, 0.99),
            weight_decay=trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True),
            dampening=trial.suggest_float('damp', 0, 0.1)
        )
        
        scores = []
        
        for test_hospital in ids:
            for other_hospital in ids:
                if other_hospital != test_hospital:
                    # Load data
                    xy1, _, _ = self.data_loader.get_hospital_data(
                        test_hospital, 'train', max_samples=1500
                    )
                    xy2, _, _ = self.data_loader.get_hospital_data(
                        other_hospital, 'train', max_samples=1500
                    )
                    
                    # Load test data
                    _, _, xy_val_test = self.data_loader.get_hospital_data(test_hospital, 'test')
                    _, _, xy_val_other = self.data_loader.get_hospital_data(other_hospital, 'test')
                    
                    # Prepare test set
                    X_test = np.concatenate((xy_val_test, xy_val_other), axis=0)
                    Y_test = np.concatenate((
                        np.ones(len(xy_val_test)), 
                        np.zeros(len(xy_val_other))
                    ), axis=0)
                    
                    # Train model
                    model = ModelFactory.create_model(xy1.shape[1], config, encrypted)
                    scaler = ModelFactory.create_scaler(encrypted)
                    scaler, model = self.trainer.train_model(model, scaler, xy1, xy2, encrypted)
                    
                    # Make predictions and compute score
                    predictions = self.trainer.predict(model, scaler, X_test, encrypted)
                    score = brier_score_loss(Y_test, predictions)
                    scores.append(score)
        
        return np.mean(scores)

def main(my_args=tuple(sys.argv[1:])):
    """Main function with improved argument parsing and workflow"""
    parser = argparse.ArgumentParser(description="Run KL Check with refactored code")
    
    # Experiment configuration
    parser.add_argument('--n_samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('--output_dir', type=str, default='../YAIB/results/distances/')
    parser.add_argument('--data_path', type=str, default='../yaib_logs/eicu/Mortality24/LogisticRegression')
    
    # Execution modes
    parser.add_argument('--score', action='store_true', default=False, help='Compute scores')
    parser.add_argument('--kl', action='store_true', default=False, help='Compute KL scores')
    parser.add_argument('--hp_search', action='store_true', default=False, help='Run hyperparameter search')
    parser.add_argument('--encrypted', action='store_true', default=False, help='Use encrypted computation')
    parser.add_argument('--plaintext', action='store_true', default=False, help='Use plaintext computation')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    
    # Model hyperparameters
    parser.add_argument('--eta0', type=float, default=0.06, help='Learning rate')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--tol', type=float, default=0.0001, help='Tolerance for convergence')
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--damp', type=float, default=0.0, help='Dampening')
    
    args, _ = parser.parse_known_args(my_args)
    
    # Create configurations
    exp_config = ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples
    )
    
    model_config = ModelConfig(
        eta0=args.eta0,
        max_iter=args.max_iter,
        patience=args.patience,
        tol=args.tol,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        dampening=args.damp
    )
    
    # Initialize components
    data_loader = DataLoader(exp_config)
    hospital_ids = data_loader.get_hospital_ids()
    
    print(f"Working with {len(hospital_ids)} hospitals")
    print(f"Using {'encrypted' if args.encrypted else 'plaintext'} computation")
    
    # Hyperparameter optimization
    if args.hp_search:
        optimizer = HyperparameterOptimizer(exp_config)
        n_trials = 100 if args.encrypted else 10
        model_config = optimizer.optimize(hospital_ids, n_trials, args.encrypted)
        print(f"Best hyperparameters found: {model_config}")
    
    # Score computation
    if args.score:
        score_computer = ScoreComputer(exp_config)
        score_computer._debug = args.debug  # Set debug flag
        
        results_x, results_xy = score_computer.compute_pairwise_scores(
            hospital_ids, model_config, args.encrypted, args.debug
        )
        
        score_computer.save_results(results_x, results_xy, model_config, args.encrypted)
        print("Score computation completed and saved")
    

if __name__ == "__main__":
    main()