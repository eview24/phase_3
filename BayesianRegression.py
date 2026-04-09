""""
Install dependencies:
    pip install pymc
    pip install odf
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def main():
    ### 1. loading and combining all sheets from excel ###
    df_dict = pd.read_excel('dataset.ods', sheet_name=None, engine='odf')
    
    dfs = []
    for sheet_name, df_temp in df_dict.items():
        df_temp['League'] = sheet_name
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)

    ### 2. cleaning numeric columns ###
    def clean_data(col):
        '''
        Returns NaN on empty cells and strings.
        Reformats notation of k and m to correct corresponding values.
        '''
        def convert_value(x):
            # missing or empty strings get converting to NaN
            if pd.isna(x) or str(x).strip() == '':
                return np.nan
            try:
                # removing non-numeric characters except digits, dot, k, m, minus
                # converting to lowercase
                x = str(x).lower()
                x = re.sub(r'[^0-9km\.-]', '', x)
                # converting notation to corresponding values
                if x.endswith('k'):
                    return float(x[:-1]) * 1_000
                elif x.endswith('m'):
                    return float(x[:-1]) * 1_000_000
                else:
                    return float(x)
            except ValueError:
                return np.nan
        return col.apply(convert_value)

    features = ['Expenditure', 'Income', 'Squad Value', 'Avg. Squad Age',
                'Goals For', 'Goals Against']
    target_var = 'Points per game'

    # cleaning the data in the feature columns
    for f in features:
        if df[f].dtype == object:
            df[f] = clean_data(df[f])

    ### 3. handling missing feature values ###
    df = df.dropna(subset=[target_var])
    df_scaled = df.copy()
    # ensuring numeric
    df_scaled[features] = df_scaled[features].apply(pd.to_numeric, errors='coerce')
    # imputing missing values
    imputer = SimpleImputer(strategy="mean")
    df_scaled[features] = imputer.fit_transform(df_scaled[features])

    ### 4. standardising features ###
    scaler = StandardScaler()
    df_scaled[features] = scaler.fit_transform(df_scaled[features])

    ### 5. group indexing for partial pooling ###
    group_idx, groups = pd.factorize(df_scaled['League'])
    n_groups = len(groups)
    n_features = len(features)

    ### 6. preparing predictor matrix and target vector ###
    X = df_scaled[features].values
    y = df_scaled[target_var].values

    ### 7. building Bayesian hierarchical model ###
    with pm.Model(coords={"group": groups, "predictor": features}) as model:
        # population-level hyperpriors
        mu_alpha = pm.Normal("mu_alpha", 0, 1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1)

        mu_beta = pm.Normal("mu_beta", 0, 1, dims="predictor")
        sigma_beta = pm.HalfNormal("sigma_beta", 1, dims="predictor")

        # group-level non-centered parameters
        z_alpha = pm.Normal("z_alpha", 0, 1, dims="group")
        alpha_j = pm.Deterministic("alpha_j", mu_alpha + sigma_alpha * z_alpha, dims="group")

        z_beta = pm.Normal("z_beta", 0, 1, dims=("group", "predictor"))
        beta_j = pm.Deterministic("beta_j", mu_beta + sigma_beta * z_beta, dims=("group", "predictor"))

        # observation error
        sigma_eps = pm.HalfNormal("sigma_eps", 1)

        # predicted mean per observation
        mu_obs = alpha_j[group_idx] + (beta_j[group_idx, :] * X).sum(axis=1)

        # likelihood
        pm.Normal("y_obs", mu=mu_obs, sigma=sigma_eps, observed=y)

    ### 8. sampling ###
    with model:
        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            random_seed=42,
            return_inferencedata=True
        )

    ### 9. summarising results ###
    print("\n=== Population-level summary ===")
    pop_vars = ["mu_alpha", "sigma_alpha", "mu_beta", "sigma_beta", "sigma_eps"]
    print(az.summary(idata, var_names=pop_vars, round_to=3))

if __name__ == "__main__":
    main()

