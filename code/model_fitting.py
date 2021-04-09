import pymc3 as pm
import theano.tensor as T
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from theano import scan
import theano
from itertools import product
import argparse
import sys
sys.path.insert(0, '.')
from .models import base_model, model_4, rw_update, rw_update_dual, bmt_update_dual


def generate_choices2(pa):

    """
    Simulates choices based on choice probabilities
    """

    return (np.random.random(pa.shape) < pa).astype(int)

def beta_response_transform_t(responses):
    
    """
    Transforms things to make sure there are no zero or one values
    """

    responses = (responses * (T.prod(responses.shape) - 1) + 0.5) / T.prod(responses.shape)

    return responses

def split_subjects(df, columns, n_trials=138):
    
    """
    Splits columns of a dataframe based on subject ID
    """  
    arrays = []
    for sub in df['subjectID'].unique():
        temp_df = df[df['subjectID'] == sub]
        if len(temp_df) == n_trials:
            vals = temp_df[columns].T.values
            arrays.append(vals)
    arrays = np.dstack(arrays)
    return arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=int)  
    args = parser.parse_args()

    # Get model combination ID
    model_id = int(args.model_id)

    # Get learning and decision model IDs
    decision_models = list(range(1, 6))
    learning_models = list(range(0, 10))

    combinations = list(product(decision_models, learning_models))

    decision_model_id, learning_model_id = combinations[model_id - 1]

    print(decision_model_id, learning_model_id)

    # Load data
    gamble_data = pd.read_csv('../data/gamble_data_anon.csv')
    gamble_data = gamble_data.sort_values(['subjectID', 'trial_index'])

    # Exclude subjects who don't have all trials 
    # For some reason a couple of subject have 137 trials instead of 138

    included_subs = []
    for sub in gamble_data['subjectID'].unique():
        temp_df = gamble_data[gamble_data['subjectID'] == sub]
        if len(temp_df) == 138 and \
        len(temp_df[(temp_df['amb_chosen'] == True) & (temp_df['points_received'] > 0)]) and \
        len(temp_df[(temp_df['amb_chosen'] == True) & (temp_df['points_received'] < 0)]):
            included_subs.append(sub)
            
    gamble_data = gamble_data[gamble_data['subjectID'].isin(included_subs)]

    # Ambiguous trial type, 0 = ambiguous not chosen, -1 = negative ambiguous, +1 = positive ambiguous
    gamble_data['trial_type'] = 0
    gamble_data.loc[(gamble_data['amb_chosen']) & (gamble_data['points_received'] > 0), 'trial_type'] = 1
    gamble_data.loc[(gamble_data['amb_chosen']) & (gamble_data['points_received'] < 0), 'trial_type'] = -1

    # Ambiguous values are currently NaN, this confuses theano and we need to indicate whether ambiguous trials are positive or negative
    gamble_data.loc[gamble_data['condition'].isin([2, 8]), ['number1', 'number2', 'number3', 'condition']] = gamble_data.loc[gamble_data['condition'].isin([2, 8]), ['number1', 'number2', 'number3', 'condition']].replace(np.nan, -999)
    gamble_data.loc[gamble_data['condition'].isin([3, 5, 6, 7]), ['number1', 'number2', 'number3', 'condition']] = gamble_data.loc[gamble_data['condition'].isin([3, 5, 6, 7]), ['number1', 'number2', 'number3', 'condition']].replace(np.nan, 999)


    gamble_idx = split_subjects(gamble_data, ['gamble_1', 'gamble_2', 'gamble_3'])
    exclude = split_subjects(gamble_data, ['exclude_1', 'exclude_2', 'exclude_3'])
    A_v = split_subjects(gamble_data, ['number1']) 
    B_v = split_subjects(gamble_data, ['number2']) 
    C_v = split_subjects(gamble_data, ['number3']) 
    amb_A = split_subjects(gamble_data, ['amb_1'])
    amb_B = split_subjects(gamble_data, ['amb_2'])
    amb_C = split_subjects(gamble_data, ['amb_3'])
    amb_gain_mean = gamble_data[(gamble_data['amb_chosen'] == True) & (gamble_data['points_received'] > 0)].groupby('subjectID').mean()['points_received'].values / 10
    amb_loss_mean = gamble_data[(gamble_data['amb_chosen'] == True) & (gamble_data['points_received'] < 0)].groupby('subjectID').mean()['points_received'].values / 10
    outcome = split_subjects(gamble_data, ['points_received']) / 10
    trial_type = split_subjects(gamble_data, ['trial_type'])


    choices = split_subjects(gamble_data, ['gamble_chosen'])
    N_SUBS = gamble_idx.shape[-1]

    # SET UP PYMC3 MODEL

    with pm.Model() as model:

        ##########################################
        # Group level parameters - means and SDs #
        ##########################################
        
        ## DECISION MODEL PARAMETERS ##
        
        # Rho/lambda/gamma common to all models
        rho_mean = pm.Normal('rho_mean', 1, 3)
        rho_sd = pm.HalfCauchy('rho_sd', 10)
        lambda_mean = pm.Normal('lambda_mean', 1, 3)
        lambda_sd = pm.HalfCauchy('lambda_sd', 10)
        gamma_mean = pm.Normal('gamma_mean', 1, 3)
        gamma_sd = pm.HalfCauchy('gamma_sd', 10)
        
        # Alpha depends on the model   
        if decision_model_id == 2:
            alpha_mean = pm.Normal('alpha_mean', 1, 3)
            alpha_sd = pm.HalfCauchy('alpha_sd', 10)
            
        elif decision_model_id == 3:
            alpha_g_mean = pm.Normal('alpha_g_mean', 1, 3)
            alpha_g_sd = pm.HalfCauchy('alpha_g_sd', 10)
            alpha_l_mean = pm.Normal('alpha_l_mean', 1, 3)
            alpha_l_sd = pm.HalfCauchy('alpha_l_sd', 10)
            
        elif decision_model_id == 4:
            alpha_noloss_mean = pm.Normal('alpha_noloss_mean', 1, 3)
            alpha_noloss_sd = pm.HalfCauchy('alpha_noloss_sd', 10)
            alpha_loss_mean = pm.Normal('alpha_loss_mean', 1, 3)
            alpha_loss_sd = pm.HalfCauchy('alpha_loss_sd', 10)
        
        if decision_model_id == 5:
            alpha_rg_mean = pm.Normal('alpha_rg_mean', 1, 3)
            alpha_rg_sd = pm.HalfCauchy('alpha_rg_sd', 10)
            alpha_rl_mean = pm.Normal('alpha_rl_mean', 1, 3)
            alpha_rl_sd = pm.HalfCauchy('alpha_rl_sd', 10)
            alpha_sg_mean = pm.Normal('alpha_sg_mean', 1, 3)
            alpha_sg_sd = pm.HalfCauchy('alpha_sg_sd', 10)
            alpha_sl_mean = pm.Normal('alpha_sl_mean', 1, 3)
            alpha_sl_sd = pm.HalfCauchy('alpha_sl_sd', 10)
            
        ## LEARNING MODEL PARAMETERS ##
            
        # Learning rate for RW models
        if learning_model_id in [1, 2, 3]:
            learning_rate_mean = pm.Normal('learning_rate_mean', 0.5, 3)
            learning_rate_sd = pm.HalfCauchy('learning_rate_sd', 10)
            
        elif learning_model_id in [4, 5, 6]:
            learning_rate_g_mean = pm.Normal('learning_rate_g_mean', 0.5, 3)
            learning_rate_g_sd = pm.HalfCauchy('learning_rate_g_sd', 10)
            learning_rate_l_mean = pm.Normal('learning_rate_l_mean', 0.5, 3)
            learning_rate_l_sd = pm.HalfCauchy('learning_rate_l_sd', 10)
            
        # Theta for BMT models
        if learning_model_id in [7, 8, 9]:
            theta_mean = pm.Normal('theta_mean', 1, 3)
            theta_sd = pm.HalfCauchy('theta_sd', 10)
            
        # Initial value
        if learning_model_id in [2, 5, 8]:  # Single initial value
            initial_value_mean = pm.Normal('initial_value_mean', 5, 5)
            initial_value_sd = pm.HalfCauchy('initial_value_sd', 5)
            
        elif learning_model_id in [3, 6, 9]:  # Different initial values for gain/loss
            initial_value_g_mean = pm.Normal('initial_value_g_mean', 5, 5)
            initial_value_g_sd = pm.HalfCauchy('initial_value_g_sd', 5)
            initial_value_l_mean = pm.Normal('initial_value_l_mean', 5, 5)
            initial_value_l_sd = pm.HalfCauchy('initial_value_l_sd', 5)
            

        ############################
        # Subject level parameters #
        ############################
        
        # Used for bounding parameters
        BoundedNormal = pm.Bound(pm.Normal, lower=0.01, upper=5)
        BoundedNormal2 = pm.Bound(pm.Normal, lower=0, upper=1)
        BoundedNormal3 = pm.Bound(pm.Normal, lower=0, upper=100)
        
        ## DECISION MODEL PARAMETERS ##
        
        # Rho, lambda, gamma common to all models
        rho = BoundedNormal('rho', mu=rho_mean, sd=rho_sd, shape=N_SUBS)
        lambda_param = BoundedNormal('lambda', mu=lambda_mean, sd=lambda_sd, shape=N_SUBS)
        gamma = BoundedNormal('gamma', mu=gamma_mean, sd=gamma_sd, shape=N_SUBS)
        
        # Alpha depends on the model
        if decision_model_id == 1:
            alpha = 1
        
        elif decision_model_id == 2:
            alpha = BoundedNormal('alpha', mu=alpha_mean, sd=alpha_sd, shape=N_SUBS)
        
        elif decision_model_id == 3:
            alpha_g = BoundedNormal('alpha_g', mu=alpha_g_mean, sd=alpha_g_sd, shape=N_SUBS)
            alpha_l = BoundedNormal('alpha_l', mu=alpha_l_mean, sd=alpha_l_sd, shape=N_SUBS)
        
        elif decision_model_id == 4:
            alpha_noloss_context = BoundedNormal('alpha_noloss_context', mu=alpha_noloss_mean, sd=alpha_noloss_sd, shape=N_SUBS)
            alpha_loss_context = BoundedNormal('alpha_loss_context', mu=alpha_loss_mean, sd=alpha_loss_sd, shape=N_SUBS)
        
        if decision_model_id == 5:
            alpha_rg = BoundedNormal('alpha_rg', mu=alpha_rg_mean, sd=alpha_rg_sd, shape=N_SUBS)
            alpha_rl = BoundedNormal('alpha_rl', mu=alpha_rl_mean, sd=alpha_rl_sd, shape=N_SUBS)
            alpha_sg = BoundedNormal('alpha_sg', mu=alpha_sg_mean, sd=alpha_sg_sd, shape=N_SUBS)
            alpha_sl = BoundedNormal('alpha_sl', mu=alpha_sl_mean, sd=alpha_sl_sd, shape=N_SUBS)

        ## LEARNING MODEL PARAMETERS ##
        
        # Learning rate for RW models
        if learning_model_id in [1, 2, 3]:
            learning_rate = BoundedNormal2('learning_rate', mu=learning_rate_mean, sd=learning_rate_sd, shape=N_SUBS)
            
        elif learning_model_id in [4, 5, 6]:
            learning_rate_g = BoundedNormal2('learning_rate_g', mu=learning_rate_g_mean, sd=learning_rate_g_sd, shape=N_SUBS)
            learning_rate_l = BoundedNormal2('learning_rate_l', mu=learning_rate_l_mean, sd=learning_rate_l_sd, shape=N_SUBS)
            
        # Theta for BMT models
        if learning_model_id in [7, 8, 9]:
            theta = BoundedNormal3('theta', mu=theta_mean, sd=theta_sd, shape=N_SUBS)
            
        # Initial value
        if learning_model_id in [2, 5, 8]:  # Single initial value
            initial_value = BoundedNormal3('initial_value', mu=initial_value_mean, sd=initial_value_sd, shape=N_SUBS)
            
        elif learning_model_id in [3, 6, 9]:  # Different initial values for gain/loss
            initial_value_g = BoundedNormal3('initial_value_g', mu=initial_value_g_mean, sd=initial_value_g_sd, shape=N_SUBS)
            initial_value_l = BoundedNormal3('initial_value_l', mu=initial_value_l_mean, sd=initial_value_l_sd, shape=N_SUBS)
            
        
        ####################################
        # USE LEARNING MODEL TO GET VALUES #
        ####################################
        
        # These have an extra zero value added at the start as we need to use t-1 for the learning model
        outcome_T = T.as_tensor_variable(np.vstack([np.zeros((1, outcome.shape[-1])), np.abs(outcome.squeeze())]))
        trial_type_T = T.as_tensor_variable(np.vstack([np.zeros((1, trial_type.shape[-1])), np.abs(trial_type.squeeze())]))

        ## PARAMETERS ##
        
        # No learning model
        if learning_model_id == 0:
            amb_gain_est = np.abs(amb_gain_mean)
            amb_loss_est = np.abs(amb_loss_mean)
        
        else:

            # Initial values
            if learning_model_id in [1, 4, 7]:
                V_loss_est = np.ones((N_SUBS)) * 5
                V_gain_est = np.ones((N_SUBS)) * 5

            elif learning_model_id in [2, 5, 8]:
                V_loss_est = np.ones((N_SUBS)) * initial_value
                V_gain_est = np.ones((N_SUBS)) * initial_value

            elif learning_model_id in [3, 6, 9]:
                V_loss_est = np.ones((N_SUBS)) * initial_value_l
                V_gain_est = np.ones((N_SUBS)) * initial_value_g

            # Variance and Kalman gain starting values for BMT models
            if learning_model_id in [7, 8, 9]:
                var_loss_est = np.ones((N_SUBS)) * 5
                var_gain_est = np.ones((N_SUBS)) * 5
                kGain_loss_est = np.ones((N_SUBS)) * 0.5
                kGain_gain_est = np.ones((N_SUBS)) * 0.5

            ## RUN MODELS ##
            # RW model, single learning rate
            if learning_model_id in [1, 2, 3]:
                output, updates = scan(fn=rw_update,
                                sequences=[dict(input=outcome_T, taps=[-1]), dict(input=trial_type_T, taps=[-1])],
                                outputs_info=[V_loss_est, V_gain_est],
                                non_sequences=[learning_rate])

            # RW model, dual learning rate
            elif learning_model_id in [4, 5, 6]:
                output, updates = scan(fn=rw_update_dual,
                        sequences=[dict(input=outcome_T, taps=[-1]), dict(input=trial_type_T, taps=[-1])],
                        outputs_info=[V_loss_est, V_gain_est],
                        non_sequences=[learning_rate_g, learning_rate_l])

            # BMT model
            elif learning_model_id in [7, 8, 9]:
                    output, updates = scan(fn=bmt_update_dual,
                            sequences=[dict(input=outcome_T, taps=[-1]), dict(input=trial_type_T, taps=[-1])],
                            outputs_info=[V_loss_est, V_gain_est, var_loss_est, var_gain_est, kGain_loss_est, kGain_gain_est],
                            non_sequences=[theta])

                    # These variables are returned when sampling from the posterior only for BMT
                    estimated_amb_gain_var = pm.Deterministic("estimated_amb_gain_var", output[3])
                    estimated_amb_loss_var = pm.Deterministic("estimated_amb_loss_var", output[2])
                    estimated_amb_gain_LR = pm.Deterministic("estimated_amb_gain_LR", output[5])
                    estimated_amb_loss_LR = pm.Deterministic("estimated_amb_loss_LR", output[4])

            # Get output
            amb_loss_est = output[0]
            amb_gain_est = output[1]

            # These variables are returned when sampling from the posterior for every model
            estimated_amb_gain = pm.Deterministic("estimated_amb_gain", amb_gain_est)
            estimated_amb_loss = pm.Deterministic("estimated_amb_loss", amb_loss_est)

        
        ###################################
        # PLUG VALUES INTO DECISION MODEL #
        ###################################
        
        if decision_model_id in [1, 2]:
            p = base_model(gamble_idx, exclude, A_v, B_v, C_v, amb_A, amb_B, amb_C, rho, 
                lambda_param, alpha, alpha, alpha, alpha, gamma, amb_gain_est, amb_loss_est)
            
        elif decision_model_id == 3:
            p = base_model(gamble_idx, exclude, A_v, B_v, C_v, amb_A, amb_B, amb_C, rho, 
            lambda_param, alpha_g, alpha_l, alpha_g, alpha_l, gamma, amb_gain_est, amb_loss_est)
        
        elif decision_model_id == 4:
            p = model_4(gamble_idx, exclude, A_v, B_v, C_v, amb_A, amb_B, amb_C, rho, 
            lambda_param, alpha_noloss_context, alpha_loss_context, gamma, amb_gain_est, amb_loss_est)
            
        elif decision_model_id == 5:
            p = base_model(gamble_idx, exclude, A_v, B_v, C_v, amb_A, amb_B, amb_C, rho, 
            lambda_param, alpha_rg, alpha_rl, alpha_sg, alpha_sl, gamma, amb_gain_est, amb_loss_est)

        # Make sure we don't have zeros or ones
        p = beta_response_transform_t(p) # remove zeros and ones

        # Likelihood
        likelihood = pm.Bernoulli('likelihood', p=p, observed=choices)

        
        # FIT MODEL USING ADVI
        with model:
            approx = pm.fit(method='advi', n=60000)
        trace = approx.sample(4000)

        # Get WAIC
        waic = pm.waic(trace, model, scale='deviance')

        # Sample from posterior
        ppc = pm.sample_posterior_predictive(trace, samples=2000, model=model, var_names=[i.name for i in model.deterministics if 'estimated' in i.name])

        # Extract parameters etc
        fitting_results = pm.summary(trace);

        fitting_results = fitting_results[~(fitting_results.index.str.contains('_mean')) & ~(fitting_results.index.str.contains('_sd'))]
        fitting_results['subjectID'] = np.tile(gamble_data['subjectID'].unique(), int(len(fitting_results) / N_SUBS))
        fitting_results['parameter'] = fitting_results.index.str.replace('(\[[0-9]+\])', '')
        fitting_results['WAIC'] = waic.waic
        fitting_results['WAIC_SE'] = waic.waic_se
        fitting_results['decision_model_id'] = decision_model_id
        fitting_results['learning_model_id'] = learning_model_id

        # Add PPC stuff
        ppc_df = {}
        for k, v in ppc.items():
            ppc_df[k] = v.mean(axis=1).mean(axis=0)
            ppc_df[k + '_sd'] = v.mean(axis=1).std(axis=0)
            
        ppc_df = pd.DataFrame(ppc_df)
        ppc_df['subjectID'] = gamble_data['subjectID'].unique()

        fitting_results = pd.merge(fitting_results, ppc_df, on='subjectID')

        fitting_results.to_csv('../data/gamble_fit_results/fitting_results__decision_model_id{0}__learning_model_id{1}.csv'.format(decision_model_id, learning_model_id))