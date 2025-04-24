import os
import argparse
import warnings
import numpy as np

import pandas as pd
from leaspy import Leaspy, Data, AlgorithmSettings, IndividualParameters
from brlp import const
import matplotlib.pyplot as plt
from leaspy.api import Leaspy
from leaspy.io.logs.visualization.plotter import Plotter
from leaspy.io.logs.visualization.plotting import Plotting
from leaspy.io.data.individual_data import IndividualData



def prepare_dcm_data(df):
    """
    Convert dataframe to leaspy.Data according to Leaspy formatting rules, which are:
    i)   name `subject_id` as `ID` and `age` as `TIME` and use this pair as row unique index.
    ii)  all the biomarkers should be normalized between [0,1].
    iii) all the biomarkers should be increasing with time, invert the trend if not.  
    """
    df = df.copy()
    
    # age should be unnormalized.
    # df['age'] = df['age'] * 100 
    df['age'] = df['age'] * 6 

    # if the patient has 2 scans at the same age, we will
    # have more than one row with the same index in our dataframe.
    # Use something as the months to the screening visits to distinguish
    # different visits at the same age. 
    if 'months_to_screening' in df.columns:
        df['age'] += df['months_to_screening'] / 1000

    # only the lateral ventricle volume increases over time among
    # the five selected regions. Invert the trend where needed.
    # for region in const.CONDITIONING_REGIONS:
    #     if region == 'lateral_ventricle': continue    
    #     df[region] = 1 - df[region]

    # set the age and subject id as indices.
    df = df.rename({'subject_id': 'ID', 'age': 'TIME'}, axis=1) \
           .set_index(["ID","TIME"], verify_integrity=False) \
           .sort_index()

    # select only the regions to model.
    df = df[const.CONDITIONING_REGIONS]
    return Data.from_dataframe(df)
    

def train_leaspy(data, name, logs_path):
    """
    Train a DCM logistic model with provided data.
    """
    model_logs_path = os.path.join(logs_path, f'leaspy_logs_{name}')
    os.makedirs(model_logs_path, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        leaspy = Leaspy("logistic", source_dimension=4)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=5000, seed=64)
        algo_settings.set_logs(
            path=model_logs_path,          
            plot_periodicity=50,  
            save_periodicity=10,  
            console_print_periodicity=None,  
            overwrite_logs_folder=True
        )
        leaspy.fit(data, settings=algo_settings)
        return leaspy

def personalize_model(data, model_path):
    settings_personalization = AlgorithmSettings('scipy_minimize', progress_bar=True, use_jacobian=True)
    leaspy_model = Leaspy.load(model_path)
    ip = leaspy_model.personalize(data, settings_personalization)
    return ip

def plot_trajectory(timepoints, reconstruction, model_path, save_path, observations=None):

    if observations is not None:
        ages = observations.index.values

    leaspy_model = Leaspy.load(model_path)
    
    plt.figure(figsize=(14, 6))
    plt.grid()
    plt.ylim(0, .75)
    plt.ylabel('Biomarker normalized value')
    plt.xlim(1, 7)
    plt.xlabel('Patient age')
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600', '#3c8f91']
    
    for c, name, val in zip(colors, leaspy_model.model.features, reconstruction.T):
        plt.plot(timepoints, val, label=name, c=c, linewidth=3)
        if observations is not None:
            plt.plot(ages, observations[name], c=c, marker='o', markersize=12)
    
    plt.legend()
    plt.savefig(f'{save_path}/estimations_vs_obs.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    train_df = df[ df.split == 'train' ]
    test_df = df[ df.split == 'test']

    logs_path = os.path.join(args.output_path, 'logs')
    os.makedirs(logs_path, exist_ok=True)

    print('>\033[1m Training DCM on cognitively normal subjects \033[0m')
    cn_df       = train_df
    cn_data     = prepare_dcm_data(cn_df)
    # cn_leaspy   = train_leaspy(cn_data, 'cn', logs_path)
    # cn_leaspy.save(os.path.join(args.output_path, 'dcm_cn.json'))

    # Load model
    model_path = os.path.join(args.output_path, 'dcm_cn.json')
    leaspy_model = Leaspy.load(model_path)

    # Define save path
    # save_path = os.path.join(args.output_path, 'avg_trajectory.png')

    # # Create plotter with output path
    # plotter = Plotter(output_path=args.output_path)

    # # Plot and save the figure
    # plotter.plot_mean_trajectory(leaspy_model.model, save_as=save_path)

    
    # Create the plotting object (deprecated class, still usable)
    plotting = Plotting(leaspy_model.model, output_path=args.output_path)

    # Save filename
    save_name = 'obs_train.png'

    # Plot all patient observations and save the figure
    plotting.patient_observations(
        data=cn_data,
        patients_idx='all',
        save_as=save_name  # this will be joined with output_path inside the class
    )
    save_name = f'avg_traject.png'

    # Plot all patient observations and save the figure
    plotting.average_trajectory(
        alpha = 1,
        save_as=save_name  # this will be joined with output_path inside the class
    )

    # PERSONALIZING ON INDIVIDUAL DATA
    cn_df       = test_df
    cn_data     = prepare_dcm_data(cn_df)
    ip = personalize_model(cn_data, model_path)

    save_name = f'reparametrized_age_obs.png'
    # Plot all patient observations and save the figure
    plotting.patient_observations_reparametrized(
        data = cn_data,
        individual_parameters=ip,
        save_as = save_name
    )
    save_name = f'non_reparametrized_age_obs.png'
    # Plot all patient observations and save the figure
    plotting.patient_observations(
        data = cn_data,
        save_as = save_name
    )

    # Estimating trajectories on test data
    print(cn_data.to_dataframe())

    observations = test_df.loc[test_df["subject_id"] == 'sub-10089']
    print(f'Observations: {observations}')
    print(f'Seen ages: {observations.index.values}')
    print("Individual Parameters : ", ip['sub-10089'])

    timepoints = np.linspace(1, 7, 100)
    reconstruction = leaspy_model.estimate({'sub-10089': timepoints}, ip)
    save_path = os.path.join(args.output_path)
    plot_trajectory(timepoints, reconstruction['sub-10089'], model_path, save_path, observations)
    save_name = 'individual_trajec_sub_10089_148796.png'
    plotting.patient_trajectories(cn_data, ip,
                                          patients_idx=['sub-10089','sub-148796'],
                                          #reparametrized_ages=True, # check sources effect
                                          
                                          # plot kwargs
                                          #color=['#003f5c', '#7a5195', '#ef5675', '#ffa600'],
                                          alpha=1, linestyle='-', linewidth=2,
                                          #marker=None,
                                          markersize=8, obs_alpha=.5, #obs_ls=':', 
                                          figsize=(16, 6),
                                          factor_past=.5,
                                          factor_future=5, # future extrapolation
                                          save_as = save_name
                                          )
    # Save filename 
    # features = [0, 1, 2, 3]
    # for feature in features:
    #     save_name = f'avg_traject_feat{feature}.png'

    #     # Plot all patient observations and save the figure
    #     plotting.average_trajectory(
    #         alpha = 1,
    #         features = [feature],
    #         save_as=save_name  # this will be joined with output_path inside the class
    #     )



    # train DCM on cognitively normal subjects.
    # print('>\033[1m Training DCM on cognitively normal subjects \033[0m')
    # cn_df       = train_df[train_df.last_diagnosis == 0.]
    # cn_data     = prepare_dcm_data(cn_df)
    # cn_leaspy   = train_leaspy(cn_data, 'cn', logs_path)
    # cn_leaspy.save(os.path.join(args.output_path, 'dcm_cn.json'))

    # # train DCM on subjects with mild cognitive impairment.
    # print('>\033[1m Training DCM on subjects with Mild Cognitive Impairment \033[0m')
    # mci_df       = train_df[train_df.last_diagnosis == 0.5]
    # mci_data     = prepare_dcm_data(mci_df)
    # mci_leaspy   = train_leaspy(mci_data, 'mci', logs_path)
    # mci_leaspy.save(os.path.join(args.output_path, 'dcm_mci.json'))

    # # train DCM on subjects with Alzheimer's disease.
    # print('>\033[1m Training DCM on subjects with Alzheimer\'s disease \033[0m')
    # ad_df       = train_df[train_df.last_diagnosis == 1.]
    # ad_data     = prepare_dcm_data(ad_df)
    # ad_leaspy   = train_leaspy(ad_data, 'ad', logs_path)
    # ad_leaspy.save(os.path.join(args.output_path, 'dcm_ad.json'))