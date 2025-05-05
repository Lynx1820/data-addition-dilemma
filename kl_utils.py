from sklearn.preprocessing import StandardScaler
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
crypten.init()
torch.manual_seed(42)
torch.set_num_threads(1)
import optuna
import timeit
import time 

class LogisticRegression(nn.Module):    
    def __init__(self, feature_dim, tol=0.0001, max_iter=400,  eta0=0.01, momentum=0,dampening=0, weight_decay=0, patience = 5):
        super().__init__()
        self.linear = torch.nn.Linear(feature_dim, 1, dtype=torch.float64)
        self.tol = tol
        self.learning_rate = eta0
        self.patience= patience
        self.max_iter = max_iter
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
    def forward(self, x):
        return self.linear(x).sigmoid()
    
    def shuffle_data(self, X, y): 
        permutation = torch.randperm(X.shape[0]) 
        X, y = X[permutation], y[permutation]
        return X, y 

    def fit(self, X, y): 
        criterion = torch.nn.BCELoss()
        prev_loss = torch.Tensor([float('inf')])
        patience, learning_rate = self.patience, self.learning_rate
        optimizer = torch.optim.SGD(self.parameters(), learning_rate, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay)
        train_samples =int(X.shape[0] * .9 )
        X_train, X_val, y_train, y_val = X[:train_samples], X[train_samples:], y[:train_samples], y[train_samples:]
        for epoch in range(self.max_iter): 
            optimizer.zero_grad()
            output = self.forward(X_train)
            loss = criterion(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            with torch.no_grad(): 
                output = self.forward(X_val)
                val_loss = criterion(output.squeeze(dim=1), y_val)
                if (abs(prev_loss - val_loss) < self.tol) or (learning_rate < 1e-6):
                    break
                learning_rate, patience = self.adjust_learning_rate(learning_rate, prev_loss, val_loss, patience=patience)
                prev_loss = val_loss
                if epoch != 0 and epoch % 10 == 0: 
                    print(f"epoch {epoch} loss: {val_loss}")
            X, y = self.shuffle_data(X, y)
            X_train, X_val, y_train, y_val = X[:train_samples], X[train_samples:], y[:train_samples], y[train_samples:]
    def adjust_learning_rate(self, lr, prev_loss, curr_loss, factor=5, patience=5):
        # Decrease learning rate by `factor` if loss doesn't improve after `patience` epochs
        has_plateaued = self.loss_has_plateaued(prev_loss, curr_loss)
        if patience == 1 and has_plateaued: 
            lr = lr/factor 
            patience = self.patience
        else: 
            patience-=1
        return lr, patience

    def loss_has_plateaued(self, previous_loss, current_loss):
        return current_loss > previous_loss
    
class EncryptedLogisticRegression(crypten.nn.Module):
    
    def __init__(self, feature_dim, tol=0.0001, max_iter=400,  eta0=0.0001, momentum=0,dampening=0, weight_decay=0, patience = 5):
        super().__init__()
        self.linear = crypten.nn.Linear(feature_dim, 1)
        self.tol = tol
        self.learning_rate = eta0
        self.patience= patience
        self.max_iter = max_iter
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        
    def forward(self, x):
        return self.linear(x).sigmoid()
    
    def shuffle_data(self, X, y): 
        permutation = torch.randperm(X.shape[0]) 
        X, y = X[permutation], y[permutation]
        return X, y 
    
    def fit(self, X, y): 
        criterion = crypten.nn.BCELoss()
        prev_loss = crypten.cryptensor(torch.Tensor([float('inf')]))
        patience, learning_rate = self.patience, self.learning_rate
        optimizer = crypten.optim.SGD(self.parameters(), learning_rate, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay, grad_threshold=1)
        train_samples =int(X.shape[0] * .9 )
        X_train, X_val, y_train, y_val = X[:train_samples], X[train_samples:], y[:train_samples], y[train_samples:]
        for epoch in range(self.max_iter): 
            optimizer.zero_grad()
            output = self.forward(X_train)
            loss = criterion(output.squeeze(), y_train)
            loss.backward()
            optimizer.step()
            with torch.no_grad(): 
                output = self.forward(X_val)
                val_loss = criterion(output.squeeze(), y_val)
                if (prev_loss.sub(val_loss).abs().get_plain_text() < self.tol) or (learning_rate < 1e-6):
                    break
                learning_rate, patience = self.adjust_learning_rate(learning_rate, prev_loss, val_loss, patience=patience)
                prev_loss = val_loss
                if val_loss.get_plain_text() < 0 or val_loss.get_plain_text() > 10: 
                        print("below zero")
                        print(val_loss)
                if epoch != 0 and epoch % 10 == 0:
                    print(f"epoch {epoch} loss: {val_loss.get_plain_text()}")
            X, y = self.shuffle_data(X, y)
            X_train, X_val, y_train, y_val = X[:train_samples], X[train_samples:], y[:train_samples], y[train_samples:]

    def adjust_learning_rate(self, lr, prev_loss, curr_loss, factor=5, patience=5):
        # Decrease learning rate by `factor` if loss doesn't improve after `patience` epochs
        has_plateaued = self.loss_has_plateaued(prev_loss, curr_loss)
        if patience == 1:
            lr = crypten.where(has_plateaued, lr/factor, lr).get_plain_text()
            patience= crypten.where(has_plateaued, self.patience, patience).get_plain_text()
            
        else: 
            patience= crypten.where(has_plateaued, patience-1, patience).get_plain_text()
        return lr, patience

    def loss_has_plateaued(self, previous_loss, current_loss):
        return current_loss.gt(previous_loss)


class EncryptedStandardScaler(): 
    ## (x- mean)/std 
    def fit(self, x):
        n_samples = x.shape[0]
        self.mean = x.mean(dim=0) ## samples, features
        var = (x.sub(self.mean)).pow(2).sum(dim=0).div(n_samples)
        self.std = var.sqrt()

    def transform(self, x):
        x_p = x.get_plain_text()
        # NOT WOKRING
        return crypten.cryptensor(torch.Tensor((x_p.sub(self.mean.get_plain_text())).div(self.std.get_plain_text() + 1e-7)))
        #return (x.sub(self.mean)).div(self.std.add(1e-7))

class StandardScaler(): 
    ## (x- mean)/std 
    def fit(self, x):
        n_samples = x.shape[0]
        self.mean = x.mean(dim=0) ## samples, features
        var = (x.sub(self.mean)).pow(2).sum(dim=0).div(n_samples)
        self.std = var.sqrt()
    def transform(self, x):
        return (x.sub(self.mean)).div(self.std + 1e-7)

def subsample_data(x, y, xy, sample_ratio=1, max_samples=None, rand_seed=42):
    if sample_ratio < 1: 
        rng = np.random.default_rng(rand_seed)
        ind = rng.choice(len(x), size=int(len(x)*sample_ratio), replace=False)
        return x[ind], y[ind], xy[ind]
    elif max_samples is not None: 
        if len(x) > max_samples: 
            rng = np.random.default_rng(rand_seed)
            ind = rng.choice(len(x), size=int(max_samples), replace=False)
            return x[ind], y[ind], xy[ind]
        else: 
            return x, y, xy
    else: 
        return x, y, xy
def get_hospital(hid, data_path, split='train', max_samples=None, sample_ratio=1, rand_seed=42): 
    file_name = f'{data_path}/train{hid}-test{hid}/data.npz'
    hos = np.load(os.path.join(file_name), allow_pickle=True)
    x = hos[split].item()['features']
    y = hos[split].item()['labels']
    xy = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    return subsample_data(x, y, xy, sample_ratio, max_samples, rand_seed)
 
def get_breast_data(id, data_path, split='train',max_samples=None, sample_ratio=1, rand_seed=42):
    file_name = f'{data_path}/bcwd.npz'
    data = np.load(os.path.join(file_name), allow_pickle=True)
    x = data[split].item()['features']
    y = data[split].item()['labels']
    if split == 'train': 
        feat_label = "perturbed" if id != 'orig' else id
        x = x[feat_label]
        y = y[id]
    xy = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    return subsample_data(x, y, xy, sample_ratio, max_samples, rand_seed)

def run_model(scaler, model, fea1 , fea2, perm): 
    X_train = torch.cat([fea1, fea2], dim=0)
    Y_train = torch.from_numpy(np.concatenate((np.ones(len(fea1)), np.zeros(len(fea2))), axis=0)).double()
    X_train, Y_train = X_train[perm], Y_train[perm]

    scaler.fit(X_train)
    enc_scaled_X_train = scaler.transform(X_train)
    model.fit(enc_scaled_X_train, Y_train)
    return scaler, model

def run_encrypted_model(scaler, model, encrypted_fea1 : Float[CrypTensor, "n_samples features"], encrypted_fea2 :  Float[CrypTensor, "n_samples features"], perm): 
    enc_X_train = crypten.cat([encrypted_fea1, encrypted_fea2], dim=0)
    enc_Y_train = crypten.cryptensor(torch.Tensor(np.concatenate((np.ones(len(encrypted_fea1)), np.zeros(len(encrypted_fea2))), axis=0)))
    enc_X_train, enc_Y_train = enc_X_train[perm], enc_Y_train[perm]

    scaler.fit(enc_X_train.get_plain_text())
    enc_scaled_X_train = crypten.cryptensor(torch.Tensor(scaler.transform(enc_X_train.get_plain_text())))
    model.fit(enc_scaled_X_train, enc_Y_train)
    return scaler, model

def get_prediction(model, scaler, features): 
    enc_scaled_X_test = scaler.transform(features)
    return model(enc_scaled_X_test)

def get_encrypted_prediction(model, scaler, encrypted_features): 
    enc_scaled_X_test = scaler.transform(encrypted_features)
    return model(enc_scaled_X_test)


def run_pipeline(scaler, model, x1, x2, permutation):
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)
    return run_model(scaler, model, x1, x2, permutation)
    
def run_encrypted_pipeline(scaler, model, x1, x2, permutation):
    encrypted_x1 = crypten.cryptensor(torch.tensor(x1))
    encrypted_x2 = crypten.cryptensor(torch.tensor(x2))
    return run_encrypted_model(scaler, model, encrypted_x1, encrypted_x2, permutation)

def get_data(dataset, id, data_path, split='train', max_samples=1500):
    if dataset == 'bcwd': 
        x, y, xy = get_breast_data(id, data_path, split, max_samples)
    else: 
        x, y, xy = get_hospital(id, data_path, split, max_samples)
    return x, y, xy

def get_ids(dataset): 
    if dataset == 'eicu': 
        hospital_file = '../YAIB-cohorts/data/mortality24/eicu/above2000.txt'
        if not os.path.exists(hospital_file):
            raise ValueError(f"The file {hospital_file} does not exist. Clone the our YAIB-cohorts repo (see readme)")

        df = pd.read_csv(hospital_file, header=None)
        n = 12
        hospital_ids = df[0].values[:n]   
        return hospital_ids
    elif dataset == 'bcwd':
        RUNS = 5
        p_perturbations = [0, .1, .2, .3, .4, .5, .6, .7]
        ids = ['orig']
        for i in p_perturbations:
            for run in range(RUNS):
                ids.append(f"perturbed_{i}_{run}")
        return ids

def compute_score(dataset: str, data_path: str, save_dir: str, hps: dict, num_samples: int=1000, compute_encrypted=True, debug = False, only_source=None):
    ids = get_ids(dataset) 
    results_x = np.zeros((len(ids), len(ids)))
    results_xy = np.zeros((len(ids), len(ids)))
    encrypted_results_x = np.zeros((len(ids), len(ids)))
    encrypted_results_xy = np.zeros((len(ids), len(ids)))
    print("computing pairwise score distance function")
    max_iter= 50 if debug else hps['max_iter']
    if debug: 
        ids = ids[:2]
        print(f"number of samples: {num_samples}")
    for test_i, test_hos in enumerate(ids):
        for i, hos in enumerate(ids):
            if only_source and only_source != hos: continue
            if hos != test_hos:
                x, _, xy = get_data(dataset, hos, data_path, 'train', max_samples=num_samples)
                x2, _, xy2 = get_data(dataset, test_hos, data_path, 'train', max_samples=num_samples)

                permutation = torch.randperm(x.shape[0] + x2.shape[0]) 
                x_val, _, xy_val = get_data(dataset,hos, data_path, 'test')
                scaler = StandardScaler()
                if not compute_encrypted:
                    model = LogisticRegression(x.shape[1], max_iter=max_iter, eta0= hps['eta0'], weight_decay=hps['weight_decay'], 
                                                        momentum=hps['momentum'], dampening=hps['damp'],
                                                        patience=hps['patience'], tol=hps['tol'])
                    scaler, model = run_pipeline(scaler, model, x, x2, permutation)
                    predictions = get_prediction(model, scaler, torch.from_numpy(x_val).double())
                    results_x[i, test_i] = predictions.mean()
                    model = LogisticRegression(xy.shape[1], max_iter=max_iter, eta0= hps['eta0'], weight_decay=hps['weight_decay'], 
                                    momentum=hps['momentum'], dampening=hps['damp'],
                                    patience=hps['patience'], tol=hps['tol'])
                    scaler, model = run_pipeline(scaler, model, xy, xy2, permutation)
                    predictions = get_prediction(model, scaler, torch.from_numpy(xy_val).double())
                    results_xy[i, test_i] = predictions.mean()
                else:
                    model = EncryptedLogisticRegression(x.shape[1], max_iter=max_iter, eta0= hps['eta0'], weight_decay=hps['weight_decay'], 
                                                        momentum=hps['momentum'], dampening=hps['damp'],
                                                        patience=hps['patience'], tol=hps['tol']).encrypt()
                    scaler, model = run_encrypted_pipeline(scaler, model, x, x2, permutation)
                    predictions = get_encrypted_prediction(model, scaler, crypten.cryptensor(torch.tensor(x_val)))
                    encrypted_results_x[i, test_i] = predictions.mean().get_plain_text()
                    model = EncryptedLogisticRegression(xy.shape[1], max_iter=max_iter, eta0= hps['eta0'], weight_decay=hps['weight_decay'], 
                                                        momentum=hps['momentum'], dampening=hps['damp'],
                                                        patience=hps['patience'], tol=hps['tol']).encrypt()
                    scaler, model = run_encrypted_pipeline(scaler, model, xy, xy2, permutation)
                    predictions = get_encrypted_prediction(model, scaler, crypten.cryptensor(torch.tensor(xy_val)))
                    encrypted_results_xy[i, test_i] = predictions.mean().get_plain_text()
    path = f"{save_dir}/{dataset}/max_it{max_iter}_eta0{hps['eta0']}_alpha{hps['weight_decay']}_tol{hps['tol']}_pat{hps['patience']}_mom{hps['momentum']}_damp{hps['damp']}_n{num_samples}"
    save_dir = Path(path)
    print(f"Saving in {save_dir}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not compute_encrypted: 
        with open(save_dir / 'score-xy.npy', 'wb') as f:
            np.save(f, results_xy)
        with open(save_dir / 'score-x.npy', 'wb') as f:
            np.save(f, results_x)
    if compute_encrypted: 
        with open(save_dir / 'encrypted-score-xy.npy', 'wb') as f:
            np.save(f, encrypted_results_xy)
        with open(save_dir / 'encrypted-score-x.npy', 'wb') as f:
            np.save(f, encrypted_results_x)

def compute_kl_score(dataset: str, data_path: str, save_dir: str, hps: dict, num_samples: int=1000, encrypt=False):
 
    ids = get_ids(dataset)
    results_x = np.zeros((len(ids), len(ids)))
    results_xy = np.zeros((len(ids), len(ids)))
    encrypted_results_x = np.zeros((len(ids), len(ids)))
    encrypted_results_xy = np.zeros((len(ids), len(ids)))
    print(f"using {num_samples}")
    for test_i, test_h in enumerate(ids):
        for i, h in enumerate(ids):
            hos = test_h
            if h != hos:
                x, y, xy = get_hospital(h, data_path, 'train', max_samples=num_samples)
                x2, y2, xy2 = get_hospital(hos, data_path, 'train', max_samples=num_samples)
                permutation = torch.randperm(x.shape[0] + x2.shape[0]) 
                scaler = StandardScaler()
                model = LogisticRegression(x.shape[1], max_iter=hps['max_iter'], eta0= hps['eta0'], weight_decay=hps['weight_decay'], 
                                                    momentum=hps['momentum'], dampening=hps['damp'],
                                                    patience=hps['patience'], tol=hps['tol'])
                scaler, model = run_pipeline(scaler, model, x, x2, permutation)
                predictions = get_prediction(model, scaler, torch.from_numpy(x_val).double())
                results_x[i, test_i] = predictions.mean()

                model = LogisticRegression(xy.shape[1], max_iter=hps['max_iter'], eta0= hps['eta0'], weight_decay=hps['weight_decay'], 
                                momentum=hps['momentum'], dampening=hps['damp'],
                                patience=hps['patience'], tol=hps['tol'])
                scaler, model = run_pipeline(scaler, model, xy, xy2, permutation)
                predictions = get_prediction(model, scaler, torch.from_numpy(xy_val).double())
                results_xy[i, test_i] = predictions.mean()

                X_train = np.concatenate((x, x2), axis=0)
                Y_train = np.concatenate((np.ones(len(x)), np.zeros(len(x2))), axis=0)

                x_val, _, _ = get_hospital(h, data_path, 'test')
                r = pipe.predict_proba(x_val)[:, 1]
                r = np.clip(r, 0.01, 0.99)
                s = r / (1 - r)
                results_x[i, test_i] = np.log2(s).mean()    

                X_train = np.concatenate((xy, xy2), axis=0)
                Y_train = np.concatenate((np.ones(len(xy)), np.zeros(len(xy2))), axis=0)

                
                _, _, xy_val = get_hospital(h, data_path,  'test')
                r = pipe.predict_proba(xy_val)[:, 1]
                r = np.clip(r, 0.01, 0.99)
                results_xy[i, test_i] = (np.log2(s)).mean()

                ## Encrypted Pipeline
                encrypted_x1 = crypten.cryptensor(torch.tensor(x))
                encrypted_x2 = crypten.cryptensor(torch.tensor(x2))
                scaler, model = run_encrypted_model(encrypted_x1, encrypted_x2)
                predictions = get_encrypted_prediction(model, scaler, x_val).get_plain_text()
                s_p =  crypten.cryptensor(1).sub(predictions)
                r= s.div(s_p) 
                encrypted_results_x[i, test_i] = r.log().mean().get_plain_text()

                encrypted_xy1 = crypten.cryptensor(torch.tensor(xy))
                encrypted_xy2 = crypten.cryptensor(torch.tensor(xy2))
                scaler, model = run_encrypted_model(encrypted_xy1, encrypted_xy2)
                predictions = get_encrypted_prediction(model, scaler, xy_val)
                s = predictions.clamp(min=.01, max=.99)
                s_p =  crypten.cryptensor(1).sub(predictions)
                r= s.div(s_p) 
                encrypted_results_xy[i, test_i] = r.log().mean().get_plain_text()
                # log_p = crypten.cryptensor(t_r.log(), ptype=crypten.mpc.arithmetic)
                # log_q = crypten.cryptensor((1-t_r).log(), ptype=crypten.mpc.arithmetic)
    save_dir = Path(save_dir)
    with open(f"{save_dir}/{dataset}/max_iKL-ratio-xy-true.npy", 'wb') as f:
        np.save(f, results_xy)
    with open(save_dir / 'KL-ratio-x-true.npy', 'wb') as f:
        np.save(f, results_x)
    with open(save_dir / 'encrypted-KL-ratio-x-true.npy', 'wb') as f:
        np.save(f, encrypted_results_x)
    with open(save_dir / 'encrypted-KL-ratio-xy-true.npy', 'wb') as f:
        np.save(f, encrypted_results_xy)

def objective_hps(trial, data_path, dataset, encrypted=False, only_source=None):
    scores = []
    ids = get_ids(dataset)
    for test_hospital in ids: 
        for other_hospital in ids:
            if only_source and only_source != other_hospital: continue
            if other_hospital != test_hospital:
                eta0 = trial.suggest_float('eta0', 1e-5, 1e-1) # lr 
                max_iter = trial.suggest_int('max_iter', 1, 2000) 
                patience = trial.suggest_int('patience', 1, 5)
                tol = trial.suggest_float('tol', .0001, .01, log=True)
                momentum = trial.suggest_float('momentum', 0.0, 0.99)  
                weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True)  # L2 regularization (weight decay)
                dampening = trial.suggest_float('damp', 0, .1) 

                x1, _, xy1 = get_data(dataset,test_hospital, data_path, 'train', max_samples=182)
                x1, _, xy2 = get_data(dataset,other_hospital, data_path, 'train', max_samples=182)

                scaler = StandardScaler()
                lr = EncryptedLogisticRegression if encrypted else LogisticRegression

                model = lr(xy1.shape[1], tol=tol, max_iter=max_iter,
                                                        dampening=dampening, momentum=momentum,
                                                        weight_decay=weight_decay, eta0=eta0, patience=patience)
                if encrypted: model.encrypt()
                permutation = torch.randperm(xy1.shape[0]+ xy2.shape[0]) 
                x_val, _, xy_val_test = get_data(dataset, test_hospital, data_path, 'test')
                x_val, _, xy_val_other= get_data(dataset, other_hospital, data_path, 'test')
                X_test = np.concatenate((xy_val_test, xy_val_other), axis=0)
                y_test = np.concatenate((np.ones(len(xy_val_test)), np.zeros(len(xy_val_other))), axis=0)
                if encrypted:
                    scaler, model = run_encrypted_pipeline(scaler, model, xy1, xy2, permutation)
                    predictions = get_encrypted_prediction(model, scaler, crypten.cryptensor(torch.tensor(X_test))).get_plain_text()
                else: 
                    scaler, model = run_pipeline(scaler, model, xy1, xy2, permutation)
                    predictions = get_prediction(model, scaler, torch.from_numpy(X_test).double()).detach().numpy()
                score = brier_score_loss(y_test, predictions)
                scores.append(score)
    return np.mean(scores)

def main(my_args=tuple(sys.argv[1:])):
    parser = argparse.ArgumentParser(description="Run KL Check")

    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Number of samples.')
    parser.add_argument('--output_dir', type=str, default='../YAIB/results/distances/')
    parser.add_argument('--dataset', type=str, default='bcwd')
    parser.add_argument('--data_path', type=str, default='../yaib_logs/eicu/Mortality24/LogisticRegression')
    parser.add_argument('--score', action='store_true', default=False)
    parser.add_argument('--kl', action='store_true', default=False)
    parser.add_argument('--hp_search',action='store_true',default=False)
    parser.add_argument('--encrypted',action='store_true',default=False)
    parser.add_argument('--plaintext',action='store_true', default=False)
    parser.add_argument('--debug',action='store_true', default=False)
    parser.add_argument('--eta0',type=float, default=0.003)
    parser.add_argument('--max_iter',type=int, default=1000)
    parser.add_argument('--patience',type=int, default=5)
    parser.add_argument('--tol',type=float, default=0.0001)
    parser.add_argument('--momentum',type=float, default=0.0)
    parser.add_argument('--weight_decay',type=float, default=0.0)
    parser.add_argument('--damp',type=float, default=0.0)
    # Parse the arguments
    args, _ = parser.parse_known_args(my_args)
    only_source = 'orig' if args.dataset == 'bcwd' else None
    encrypted = True if args.encrypted else False
    if args.hp_search:
        study = optuna.create_study(direction='minimize')
        objective = partial(objective_hps, data_path=args.data_path, dataset=args.dataset, encrypted=encrypted, only_source=only_source) 
        study.optimize(objective, n_trials=10)
        hps = study.best_params
    else: 
        hps = {'eta0': args.eta0, 'max_iter': args.max_iter, 'patience': args.patience, 'tol': args.tol, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'damp': args.damp } 
    
    if encrypted: 
        print("Computing scores on encrypted data")
    else: 
        print("Computing scores on plaintext data")
    if args.score:
        start_time = time.time()
        compute_score(dataset=args.dataset,
                      data_path=args.data_path,
                      save_dir=args.output_dir,
                      hps=hps,
                      num_samples=args.n_samples,
                      compute_encrypted=args.encrypted, 
                      debug=args.debug, only_source=only_source)
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time} secs")
        # execution_time = timeit.timeit(lambda: compute_score(ids=hospital_ids,
        #               data_path=args.data_path,
        #               save_dir=args.output_dir,
        #               hps=hps,
        #               num_samples=args.n_samples,
        #               compute_encrypted=args.encrypted, 
        #               debug=args.debug), number=100)
        #print(f"Execution Time for 100 executions: {execution_time} seconds")

    # if args.kl:
    #     compute_kl_score(hospital_ids=hospital_ids,
    #                 data_path=args.data_path,
    #                 save_dir=args.output_dir,
    #                 num_samples=args.n_samples)

if __name__ == "__main__":
    main()
