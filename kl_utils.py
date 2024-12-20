# KL Distribution Check
# python kl_utils.py --n_samples 300 --score

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
#from concrete.ml.sklearn import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os 
import argparse
import sys
from pathlib import Path
#from pytorch_lightning import seed_everything
import torch 
import crypten 
from crypten import CrypTensor
from jaxtyping import Int, Float
crypten.init()
torch.manual_seed(42)
torch.set_num_threads(1)
from folktables_exp import metrics as mt
DATA_PATH ='../yaib_logs/eicu/Mortality24/LogisticRegression'
# CONSTS
HOSPITAL_IDS = [73, 264, 420, 243, 338, 443, 199, 458, 300, 188, 252, 167]

class EncryptedLogisticRegression(crypten.nn.Module):
    
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = crypten.nn.Linear(feature_dim, 1)
        
    def forward(self, x):
        return self.linear(x).sigmoid()

def train_model(model, X, y, tol, max_iter=10, learning_rate=0.06):
    criterion = crypten.nn.BCELoss()
    prev = None
    for epoch in range(max_iter): 
        model.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        if epoch % 10 == 0: 
            print(f"epoch {epoch} loss: {loss.get_plain_text()}")
        loss.backward()
        model.update_parameters(learning_rate)
        if prev and abs(prev - loss) < tol:
            print(f"Converged after {epoch} iterations")
            break
    return model

class EncryptedStandardScaler(): 
    ## (x- mean)/std 
    def fit(self, x):
         n_samples = x.shape[0]
         self.mean = x.mean(dim=0) ## samples, features
         var = (x.sub(self.mean)).pow(2).sum(dim=0).div(n_samples)
        
        # numerical instability 
         constain_zero = var.eq(torch.tensor(0))
         self.std = var.add(constain_zero).sqrt()

    def transform(self, x):
        return (x.sub(self.mean)).div(self.std)
    
def get_hospital(hid, split='train', max_samples=None, sample_ratio=1, rand_seed=42): 
    log_dir = f'distance_data'
    ## change: testing on my data splits 
    file_name = f'{DATA_PATH}/train{hid}-test{hid}/data.npz' # f'{hid}/data.npz'
    hos = np.load(os.path.join(log_dir, file_name), allow_pickle=True)
    x = hos[split].item()['features']
    y = hos[split].item()['labels']
    xy = np.concatenate((x, y.reshape(-1, 1)), axis=1)
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

def fit_general_density(hids, split='train', max_samples=5000, n_components=3):
    # fit stratified sample density
    num_hospitals = len(hids)
    samples_per_hos = int(max_samples / num_hospitals)

    x_all = []
    xy_all = []

    for h in hids:
        x, y, xy = get_hospital(h, split=split)

        # Sample from the hospital data
        random_indices = np.random.choice(len(x), size=samples_per_hos, replace=False)
        x_sampled = x[random_indices]
        xy_sampled = xy[random_indices]

        # Append the sampled data to the overall arrays
        x_all.append(x_sampled)
        xy_all.append(xy_sampled)

    # Concatenate the sampled data from all hospitals
    x_all = np.concatenate(x_all, axis=0)
    xy_all = np.concatenate(xy_all, axis=0)
    print(f"fitting overall density function with {len(x_all)} samples from {len(hids)}")
    cx, _ = mt.init_density_scale(x_all, n_components=n_components)
    cxy, _ = mt.init_density_scale(xy_all, n_components=n_components)
    return cx, cxy


def run_hospital_kl_density(n_runs=5, n_samples=2000, n_components=3):
    '''

    '''
    KL_x = np.zeros((n_runs, len(hospital_ids), len(hospital_ids)))
    KL_xy = np.zeros((n_runs, len(hospital_ids), len(hospital_ids)))
    results = {} 
    for run in range(n_runs):
        # cx, cxy = fit_general_density(hospital_ids,
        #                       max_samples=10000,
        #                       n_components=n_components)
        print(f"iter {run}")
        for i, h1 in enumerate(hospital_ids): 
            x, y, xy = get_hospital(h1, sample_ratio=0.9, rand_seed=run)
         
            for j, h2 in enumerate(hospital_ids): 
                if i != j: 
                    print(f"computing {h1} {h2}") 
                    x2, y2, xy2 = get_hospital(h2, sample_ratio=0.9, rand_seed=run)
                    cx, _ = mt.init_density_scale(np.concatenate((x, x2), axis=0), n_components=n_components)
                    cxy, _ = mt.init_density_scale(np.concatenate((xy, xy2), axis=0), n_components=n_components)
                    x2, y2, xy2 = x2[:n_samples], y2[:n_samples], xy2[:n_samples]
                    # train set
                    # already shuffled
                    x, y, xy = x[:n_samples], y[:n_samples], xy[:n_samples]
                    pkdex = mt.init_density(x, cx) 
                    pkdexy = mt.init_density(xy, cxy)    
            
                    # test set
                    qkdex = mt.init_density(x2, cx)
                    qkdexy = mt.init_density(xy2, cxy)

                    KL_x[run, i, j] = mt.entropy_input(x, pkdex, qkdex, cx)
                    KL_xy[run, i, j] = mt.entropy_input(xy, pkdexy, qkdexy, cxy)
                    print(f"computing {h1} {h2}, kl_xy{KL_xy[run, i, j]}") 
        results['KL_x'] = KL_x
        results['KL_xy'] = KL_xy
        np.savez(f"../YAIB/results/distances/KL-n{n_samples}-c{n_components}-pair.npz", **results)
    return 

def run_encrypted_model(encrypted_fea1 : Float[CrypTensor, "n_samples features"], encrypted_fea2 :  Float[CrypTensor, "n_samples features"]): 
    features_dim = encrypted_fea1.shape[1]
    enc_X_train = crypten.cat([encrypted_fea1, encrypted_fea2], dim=0)
    
    enc_Y_train = crypten.cryptensor(np.concatenate((np.ones(len(encrypted_fea1)), np.zeros(len(encrypted_fea2))), axis=0))
    scaler = EncryptedStandardScaler()
    model = EncryptedLogisticRegression(features_dim).encrypt()
    scaler.fit(enc_X_train)
    enc_scaled_X_train = scaler.transform(enc_X_train)
    model = train_model(model, enc_scaled_X_train, enc_Y_train, tol=.1, max_iter=10, learning_rate=.05)
    return scaler, model

def get_encrypted_prediction(model, scaler, encrypted_features): 
    enc_scaled_X_test = scaler.transform(encrypted_features)
    return model(enc_scaled_X_test)

#@mpc.run_multiprocess(world_size=2)
def compute_score(hospital_ids: list, save_dir: str, num_samples: int=1000):

    results_x = np.zeros((len(hospital_ids), len(hospital_ids)))
    results_xy = np.zeros((len(hospital_ids), len(hospital_ids)))
    encrypted_results_x = np.zeros((len(hospital_ids), len(hospital_ids)))
    encrypted_results_xy = np.zeros((len(hospital_ids), len(hospital_ids)))
    print("computing pairwise score distance function")
    for test_i, test_h in enumerate(hospital_ids):
        for i, h in enumerate(hospital_ids):
            hos = test_h
            if h != hos:
                x, y, xy = get_hospital(h, 'train', max_samples=num_samples)
                x2, y2, xy2 = get_hospital(hos, 'train', max_samples=num_samples)
                #if test_i != 11 or i != 8: continue
                scaler = StandardScaler()
                logistic = LogisticRegression(max_iter=100, tol=0.1)
                pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

                X_train = np.concatenate((x, x2), axis=0)
                Y_train = np.concatenate((np.ones(len(x)), np.zeros(len(x2))), axis=0)

                pipe.fit(X_train, Y_train)

                x_val, _, _ = get_hospital(h, 'test')
                results_x[i, test_i] = pipe.predict_proba(x_val)[:, 1].mean()

                X_train = np.concatenate((xy, xy2), axis=0)
                Y_train = np.concatenate((np.ones(len(xy)), np.zeros(len(xy2))), axis=0)

                pipe.fit(X_train, Y_train)
                
                _, _, xy_val = get_hospital(h, 'test')
                results_xy[i, test_i] = pipe.predict_proba(xy_val)[:, 1].mean()
                
                ## Encrypted Pipeline
                encrypted_x1 = crypten.cryptensor(torch.tensor(x))
                encrypted_x2 = crypten.cryptensor(torch.tensor(x2))
                scaler, model = run_encrypted_model(encrypted_x1, encrypted_x2)
                predictions = get_encrypted_prediction(model, scaler, crypten.cryptensor(torch.tensor(x_val))).get_plain_text()
                encrypted_results_x[i, test_i] = predictions.mean()

                encrypted_xy1 = crypten.cryptensor(torch.tensor(xy))
                encrypted_xy2 = crypten.cryptensor(torch.tensor(xy2))
                scaler, model = run_encrypted_model(encrypted_xy1, encrypted_xy2)
                predictions = get_encrypted_prediction(model, scaler, crypten.cryptensor(torch.tensor(xy_val))).get_plain_text()
                encrypted_results_xy[i, test_i] = predictions.mean()

    save_dir = Path(save_dir)
    with open(save_dir / 'score-xy.npy', 'wb') as f:
        np.save(f, results_xy)

    with open(save_dir / 'score-x.npy', 'wb') as f:
        np.save(f, results_x)

    with open(save_dir / 'encrypted-score-xy.npy', 'wb') as f:
        np.save(f, encrypted_results_xy)

    with open(save_dir / 'encrypted-score-x.npy', 'wb') as f:
        np.save(f, encrypted_results_x)

def compute_kl_score(hospital_ids: list, save_dir: str, num_samples: int=1000, encrypt=False):
    results_x = np.zeros((len(hospital_ids), len(hospital_ids)))
    results_xy = np.zeros((len(hospital_ids), len(hospital_ids)))
    encrypted_results_x = np.zeros((len(hospital_ids), len(hospital_ids)))
    encrypted_results_xy = np.zeros((len(hospital_ids), len(hospital_ids)))
    for test_i, test_h in enumerate(hospital_ids):
        for i, h in enumerate(hospital_ids):
            hos = test_h
            if h != hos:
                x, y, xy = get_hospital(h, 'train', max_samples=num_samples)
                x2, y2, xy2 = get_hospital(hos, 'train', max_samples=num_samples)
                scaler = StandardScaler()
                logistic = LogisticRegression(max_iter=10000, tol=0.1)
                pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

                X_train = np.concatenate((x, x2), axis=0)
                Y_train = np.concatenate((np.ones(len(x)), np.zeros(len(x2))), axis=0)

                pipe.fit(X_train, Y_train)

                x_val, _, _ = get_hospital(h, 'test')
                r = pipe.predict_proba(x_val)[:, 1]
                r = np.clip(r, 0.01, 0.99)
                s = r / (1 - r)
                results_x[i, test_i] = np.log2(s).mean()

                X_train = np.concatenate((xy, xy2), axis=0)
                Y_train = np.concatenate((np.ones(len(xy)), np.zeros(len(xy2))), axis=0)

                pipe.fit(X_train, Y_train)
                
                _, _, xy_val = get_hospital(h, 'test')
                r = pipe.predict_proba(xy_val)[:, 1]
                r = np.clip(r, 0.01, 0.99)
                results_xy[i, test_i] = (np.log2(s)).mean()

                ## Encrypted Pipeline
                encrypted_x1 = crypten.cryptensor(torch.tensor(x))
                encrypted_x2 = crypten.cryptensor(torch.tensor(x2))
                scaler, model = run_encrypted_model(encrypted_x1, encrypted_x2)
                predictions = get_encrypted_prediction(model, scaler, x_val).get_plain_text()
                predictions = np.clip(predictions, 0.01, 0.99)
                s = predictions / (1 - predictions)
                encrypted_results_x[i, test_i] = (np.log2(s)).mean()

                encrypted_xy1 = crypten.cryptensor(torch.tensor(xy))
                encrypted_xy2 = crypten.cryptensor(torch.tensor(xy2))
                scaler, model = run_encrypted_model(encrypted_xy1, encrypted_xy2)
                predictions = get_encrypted_prediction(model, scaler, xy_val).get_plain_text()
                predictions = np.clip(predictions, 0.01, 0.99)
                # log_p = crypten.cryptensor(t_r.log(), ptype=crypten.mpc.arithmetic)
                # log_q = crypten.cryptensor((1-t_r).log(), ptype=crypten.mpc.arithmetic)
                s = predictions / (1 - predictions)
                encrypted_results_xy[i, test_i] = (np.log2(s)).mean()

    save_dir = Path(save_dir)
    with open(save_dir / 'KL-ratio-xy-true.npy', 'wb') as f:
        np.save(f, results_xy)
    with open(save_dir / 'KL-ratio-x-true.npy', 'wb') as f:
        np.save(f, results_x)
    with open(save_dir / 'encrypted-KL-ratio-x-true.npy', 'wb') as f:
        np.save(f, encrypted_results_x)
    with open(save_dir / 'encrypted-KL-ratio-xy-true.npy', 'wb') as f:
        np.save(f, encrypted_results_xy)

def compute_addition_score(hospital_ids: list, save_dir: str, num_samples: int=1000):
    results_x = np.zeros((len(hospital_ids), len(hospital_ids)))
    results_xy = np.zeros((len(hospital_ids), len(hospital_ids)))
    for test_i, test_h in enumerate(hospital_ids):
        for i, h in enumerate(hospital_ids):
            hos = test_h
            if h != hos:
                x, y, xy = get_hospital(h, 'train', max_samples=num_samples)
                x2, y2, xy2 = get_hospital(hos, 'train', max_samples=num_samples)

                xhalf1, xhalf2 = partition_array(x2)
                x = np.concatenate((x, xhalf1), axis=0)

                scaler = StandardScaler()
                logistic = LogisticRegression(max_iter=10000, tol=0.1)
                pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logistic)])

                # fit x
                X_train = np.concatenate((x, xhalf2), axis=0)
                Y_train = np.concatenate((np.ones(len(x)), np.zeros(len(xhalf2))), axis=0)

                pipe.fit(X_train, Y_train)

                x_val, _, _ = get_hospital(h, 'test')
                results_x[i, test_i] = (pipe.predict_log_proba(x_val)[:, 1].mean())

                # fit xy
                xyhalf1, xyhalf2 = partition_array(xy2)
                xy = np.concatenate((xy, xyhalf1), axis=0)

                X_train = np.concatenate((xy, xyhalf2), axis=0)
                Y_train = np.concatenate((np.ones(len(xy)), np.zeros(len(xyhalf2))), axis=0)


                pipe.fit(X_train, Y_train)
                _, _, xy_val = get_hospital(h, 'test')
                results_xy[i, test_i] = (pipe.predict_log_proba(xy_val)[:, 1].mean())

    save_dir = Path(save_dir)
    with open(save_dir / 'addition-score-xy.npy', 'wb') as f:
        np.save(f, results_xy)

    with open(save_dir / 'addition-score-x.npy', 'wb') as f:
        np.save(f, results_x)

def main(my_args=tuple(sys.argv[1:])):
    crypten.init()
    parser = argparse.ArgumentParser(description="Run KL Check")

    parser.add_argument('--n_samples', type=int, default=3000,
                        help='Number of samples.')
    parser.add_argument('--output_dir', type=str, default='YAIB/results/distances/')
    parser.add_argument('--score', action='store_true', default=False)
    parser.add_argument('--kl', action='store_true', default=False)
    parser.add_argument('--addition-score', action='store_true', default=False)
    parser.add_argument('--seed', default=2024, type=int)
    # Parse the arguments
    args, _ = parser.parse_known_args(my_args)

    hospital_file = '../YAIB-cohorts/data/mortality24/eicu/above2000.txt'
    if not os.path.exists(hospital_file):
        raise ValueError(f"The file {hospital_file} does not exist. Clone the our YAIB-cohorts repo (see readme)")

    df = pd.read_csv(hospital_file, header=None)
    n = 12
    hospital_ids = df[0].values[:n]

    # Call the function using parsed arguments
    if args.score:
        compute_score(hospital_ids=hospital_ids,
                      save_dir=args.output_dir,
                      num_samples=args.n_samples,)

    if args.kl:
        compute_kl_score(hospital_ids=hospital_ids,
                      save_dir=args.output_dir,
                      num_samples=args.n_samples,)
    if args.addition_score:
        compute_addition_score(hospital_ids=hospital_ids,
                      save_dir=args.output_dir,
                      num_samples=args.n_samples,)


def partition_array(arr):
    # Shuffle the array using the Fisher-Yates shuffle algorithm
    for i in range(len(arr) - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

    # Partition the array into two halves
    mid = len(arr) // 2
    first_half = arr[:mid]
    second_half = arr[mid:]

    return first_half, second_half

if __name__ == "__main__":
    main()
