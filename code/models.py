import pymc3 as pm
import theano.tensor as T
import numpy as np

"""
Code for behavioural models.
"""

def inv_logit(x):
    """
    Calculate the inverse logit value of a given number
    """
    return 1 / (1 + T.exp(-x))


def base_model(gamble, exclude, A_v, B_v, C_v, amb_A, amb_B, amb_C, rho, lambda_param, alpha_rg, 
               alpha_rl, alpha_sg, alpha_sl, gamma, amb_gain_value, amb_loss_value):
    
    # When using non-centered paramaterisation parameters can go below zero - need to stop this
    rho = T.switch(T.lt(rho, 0.00001), 0.00001, rho)
    lambda_param = T.switch(T.lt(lambda_param, 0.00001), 0.00001, lambda_param)
    gamma = T.switch(T.lt(gamma, 0.00001), 0.00001, gamma)
    alpha_rg = T.switch(T.lt(alpha_rg, 0.00001), 0.00001, alpha_rg)
    alpha_rl = T.switch(T.lt(alpha_rl, 0.00001), 0.00001, alpha_rl)
    alpha_sg = T.switch(T.lt(alpha_sg, 0.00001), 0.00001, alpha_sg)
    alpha_sl = T.switch(T.lt(alpha_sl, 0.00001), 0.00001, alpha_sl)

    # Calculate values for the 3 options (one of these may not be an option, in which case its value ends up being zero)
    n_exclusions = exclude.sum(axis=0)
    u_A = T.switch(T.gt(A_v, 0), # If the value of this option is greater than zero, use gain things, if not use loss things
                   ((1 - amb_A) * T.power(A_v, rho)) +  # If not ambiguous, use value ^ rho
                            amb_A * (gamble[0] * (alpha_rg * T.power(amb_gain_value, rho)) +  # If ambiguous - if it's a gamble, use the risky gain parameter
                            ((1 - gamble[0]) * (alpha_sg * T.power(amb_gain_value, rho)))),   # If it's safe use the safe gain parameter
                   -((1 - amb_A) * lambda_param * T.power(T.abs_(A_v), rho) + 
                             amb_A * (gamble[0] * (alpha_rl * T.power(amb_loss_value, rho)) + 
                              ((1 - gamble[0]) * (alpha_sl * T.power(amb_loss_value, rho))))))
                   
    u_B = T.switch(T.gt(B_v, 0), 
                   ((1 - amb_B) * T.power(B_v, rho)) + 
                            amb_B * (gamble[1] * (alpha_rg * T.power(amb_gain_value, rho)) + 
                            ((1 - gamble[1]) * (alpha_sg * T.power(amb_gain_value, rho)))), 
                   -((1 - amb_B) * lambda_param * T.power(T.abs_(B_v), rho) + 
                            amb_B * (gamble[1] * (alpha_rl * T.power(amb_loss_value, rho)) + 
                              ((1 - gamble[1]) * (alpha_sl * T.power(amb_loss_value, rho))))))
                   
    u_C = T.switch(T.gt(C_v, 0), 
                   ((1 - amb_C) * T.power(C_v, rho)) + 
                           amb_C * (gamble[2] * (alpha_rg * T.power(amb_gain_value, rho)) + 
                            ((1 - gamble[2]) * (alpha_sg * T.power(amb_gain_value, rho)))),
                   -((1 - amb_C) * lambda_param * T.power(T.abs_(C_v), rho) + 
                             amb_C * (gamble[2] * (alpha_rl * T.power(amb_loss_value, rho)) + 
                              ((1 - gamble[2]) * (alpha_sl * T.power(amb_loss_value, rho))))))
    
    
    # If we have only two choices (i.e. no gamble), the ambiguous option should be labelled as a gamble
    gamble = T.switch(T.eq(exclude.sum(axis=0), 1), T.stack([amb_A, amb_B, amb_C]).squeeze(), gamble)
    
    # Get value of gamble option
    gamble_weighting = gamble / gamble.sum(axis=0)
    u_gamble = gamble_weighting[0] * (u_A * (1 - exclude[0])) + \
               gamble_weighting[1] * (u_B * (1 - exclude[1])) + \
               gamble_weighting[2] * (u_C * (1 - exclude[2]))
    
    # Get value of sure option
    sure_weighting = (1 - gamble) / ((1 - gamble).sum(axis=0) - exclude.sum(axis=0))
    u_sure = sure_weighting[0] * (u_A * (1 - exclude[0])) + \
             sure_weighting[1] * (u_B * (1 - exclude[1])) + \
             sure_weighting[2] * (u_C * (1 - exclude[2]))
    
    # Calculate choice probability
    p = inv_logit(gamma * (u_gamble - u_sure))

    return p

def model_4(gamble, exclude, A_v, B_v, C_v, amb_A, amb_B, amb_C, rho, lambda_param, alpha_noloss_context, 
           alpha_loss_context, gamma, amb_gain_value, amb_loss_value):
    
    # When using non-centered paramaterisation parameters can go below zero - need to stop this
    rho = T.switch(T.lt(rho, 0.01), 0.01, rho)
    lambda_param = T.switch(T.lt(lambda_param, 0.01), 0.01, lambda_param)
    gamma = T.switch(T.lt(gamma, 0.01), 0.01, gamma)
    alpha_noloss_context = T.switch(T.lt(alpha_noloss_context, 0.01), 0.01, alpha_noloss_context)
    alpha_loss_context = T.switch(T.lt(alpha_loss_context, 0.01), 0.01, alpha_loss_context)
    
    alpha = T.switch(T.any(T.stack([A_v, B_v, C_v]).squeeze() < 0, axis=0), alpha_loss_context, alpha_noloss_context)
    
    # Calculate values for the 3 options (one of these may not be an option, in which case its value ends up being zero)
    u_A = T.switch(T.gt(A_v, 0), 
                   ((1 - amb_A) * T.power(A_v, rho)) + 
                            amb_A * (gamble[0] * (alpha * T.power(amb_gain_value, rho)) + 
                            ((1 - gamble[0]) * (alpha * T.power(amb_gain_value, rho)))), 
                   -((1 - amb_A) * lambda_param * T.power(T.abs_(A_v), rho) + 
                             amb_A * (gamble[0] * (alpha * T.power(amb_loss_value, rho)) + 
                              ((1 - gamble[0]) * (alpha * T.power(amb_loss_value, rho))))))
                   
    u_B = T.switch(T.gt(B_v, 0), 
                   ((1 - amb_B) * T.power(B_v, rho)) + 
                            amb_B * (gamble[1] * (alpha * T.power(amb_gain_value, rho)) + 
                            ((1 - gamble[1]) * (alpha * T.power(amb_gain_value, rho)))), 
                   -((1 - amb_B) * lambda_param * T.power(T.abs_(B_v), rho) + 
                            amb_B * (gamble[1] * (alpha * T.power(amb_loss_value, rho)) + 
                              ((1 - gamble[1]) * (alpha * T.power(amb_loss_value, rho))))))
                   
    u_C = T.switch(T.gt(C_v, 0), 
                   ((1 - amb_C) * T.power(C_v, rho)) + 
                           amb_C * (gamble[2] * (alpha * T.power(amb_gain_value, rho)) + 
                            ((1 - gamble[2]) * (alpha * T.power(amb_gain_value, rho)))),
                   -((1 - amb_C) * lambda_param * T.power(T.abs_(C_v), rho) + 
                             amb_C * (gamble[2] * (alpha * T.power(amb_loss_value, rho)) + 
                              ((1 - gamble[2]) * (alpha * T.power(amb_loss_value, rho))))))

    # If we have only two choices (i.e. no gamble), the ambiguous option should be labelled as a gamble
    gamble = T.switch(T.eq(exclude.sum(axis=0), 1), T.stack([amb_A, amb_B, amb_C]).squeeze(), gamble)
    
    # Get value of gamble option
    gamble_weighting = gamble / gamble.sum(axis=0)
    u_gamble = gamble_weighting[0] * (u_A * (1 - exclude[0])) + \
               gamble_weighting[1] * (u_B * (1 - exclude[1])) + \
               gamble_weighting[2] * (u_C * (1 - exclude[2]))
    
    # Get value of sure option
    sure_weighting = (1 - gamble) / ((1 - gamble).sum(axis=0) - exclude.sum(axis=0))
    u_sure = sure_weighting[0] * (u_A * (1 - exclude[0])) + \
             sure_weighting[1] * (u_B * (1 - exclude[1])) + \
             sure_weighting[2] * (u_C * (1 - exclude[2]))
    
    # Calculate choice probability
    p = inv_logit(gamma * (u_gamble - u_sure))
    
    return p

def rw_update(outcome, trial_type, V_loss, V_gain, alpha):
    
    V_loss = T.switch(T.eq(trial_type, -1), V_loss + alpha * (outcome - V_loss), V_loss)
    V_gain = T.switch(T.eq(trial_type, 1), V_gain + alpha * (outcome - V_gain), V_gain)

    return V_loss, V_gain

def rw_update_dual(outcome, trial_type, V_loss, V_gain, alpha_g, alpha_l):
    
    V_loss = T.switch(T.eq(trial_type, -1), V_loss + alpha_l * (outcome - V_loss), V_loss)
    V_gain = T.switch(T.eq(trial_type, 1), V_gain + alpha_g * (outcome - V_gain), V_gain)

    return V_loss, V_gain


def bmt_update_dual(outcome, trial_type, V_loss, V_gain, var_loss, var_gain, kGain_loss, kGain_gain, theta):
    
    """With help from https://github.com/charleywu/cognitivemaps/blob/6570746510f0b27043bc97a01af65da2d3f88c44/models.R"""
    
    kGain_loss = T.switch(T.eq(trial_type, -1), var_loss / (var_loss + T.power(theta, 2)), kGain_loss)
    kGain_gain = T.switch(T.eq(trial_type, 1), var_gain / (var_gain + T.power(theta, 2)), kGain_gain)
    
    V_loss = T.switch(T.eq(trial_type, -1), V_loss + kGain_loss * (outcome - V_loss), V_loss)
    V_gain = T.switch(T.eq(trial_type, 1), V_gain + kGain_gain * (outcome - V_gain), V_gain)
    
    var_loss = T.switch(T.eq(trial_type, -1), var_loss * (1 - kGain_loss), var_loss)
    var_gain = T.switch(T.eq(trial_type, 1), var_gain * (1 - kGain_gain), var_gain)

    return V_loss, V_gain, var_loss, var_gain, kGain_loss, kGain_gain