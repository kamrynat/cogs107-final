"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
MODIFIED FOR FINAL ASSIGNMENT - COGS 107

This file has been modified to:
1. Quantify effects of Stimulus Type and Trial Difficulty on SDT parameters
2. Check model convergence
3. Display posterior distributions
4. Compare Trial Difficulty vs Stimulus Type effects
5. Generate delta plots for diffusion model analysis
"""
#Used Claude to troubleshoot any errors/problems I had.

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
from scipy import stats

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data


def apply_hierarchical_sdt_model(data):
    """
    Apply enhanced hierarchical Signal Detection Theory model with condition effects.
    
    This function implements a Bayesian hierarchical model that quantifies:
    - Effects of Stimulus Type (Simple vs Complex) on d-prime and criterion
    - Effects of Trial Difficulty (Easy vs Hard) on d-prime and criterion
    - Interaction effects between Stimulus Type and Trial Difficulty
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object with condition effects
    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    print(f"Modeling {P} participants across {C} conditions")
    print("Model includes effects of Stimulus Type and Trial Difficulty")
    
    with pm.Model() as enhanced_sdt_model:
        # Group-level baseline parameters
        baseline_d_prime = pm.Normal('baseline_d_prime', mu=1.0, sigma=1.0)
        baseline_criterion = pm.Normal('baseline_criterion', mu=0.0, sigma=1.0)
        
        # Effect of Stimulus Type on d-prime and criterion
        stimulus_effect_d_prime = pm.Normal('stimulus_effect_d_prime', mu=0.0, sigma=0.5)
        stimulus_effect_criterion = pm.Normal('stimulus_effect_criterion', mu=0.0, sigma=0.5)
        
        # Effect of Trial Difficulty on d-prime and criterion  
        difficulty_effect_d_prime = pm.Normal('difficulty_effect_d_prime', mu=0.0, sigma=0.5)
        difficulty_effect_criterion = pm.Normal('difficulty_effect_criterion', mu=0.0, sigma=0.5)
        
        # Interaction effects
        interaction_d_prime = pm.Normal('interaction_d_prime', mu=0.0, sigma=0.25)
        interaction_criterion = pm.Normal('interaction_criterion', mu=0.0, sigma=0.25)
        
        # Individual-level variance
        sigma_d_prime = pm.HalfNormal('sigma_d_prime', sigma=0.5)
        sigma_criterion = pm.HalfNormal('sigma_criterion', sigma=0.5)
        
        # Create design matrix for conditions
        # Condition coding: 0=Easy Simple, 1=Easy Complex, 2=Hard Simple, 3=Hard Complex
        stimulus_type = np.array([0, 1, 0, 1])  # 0=Simple, 1=Complex
        difficulty = np.array([0, 0, 1, 1])     # 0=Easy, 1=Hard
        
        # Calculate condition-specific means
        mean_d_prime = (baseline_d_prime + 
                       stimulus_effect_d_prime * stimulus_type +
                       difficulty_effect_d_prime * difficulty +
                       interaction_d_prime * stimulus_type * difficulty)
        
        mean_criterion = (baseline_criterion + 
                         stimulus_effect_criterion * stimulus_type +
                         difficulty_effect_criterion * difficulty +
                         interaction_criterion * stimulus_type * difficulty)
        
        # Individual-level parameters (participants x conditions)
        d_prime = pm.Normal('d_prime', mu=mean_d_prime, sigma=sigma_d_prime, shape=(P, C))
        criterion = pm.Normal('criterion', mu=mean_criterion, sigma=sigma_criterion, shape=(P, C))
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime/2 - criterion)
        false_alarm_rate = pm.math.invlogit(-d_prime/2 - criterion)
        
        # Likelihood for observations
        # Note: adjusting for 0-indexed participants in model vs 1-indexed in data
        pm.Binomial('hit_obs', 
                   n=data['nSignal'], 
                   p=hit_rate[data['pnum']-1, data['condition']], 
                   observed=data['hits'])
        
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'], 
                   p=false_alarm_rate[data['pnum']-1, data['condition']], 
                   observed=data['false_alarms'])
    
    return enhanced_sdt_model

def check_convergence(trace):
    """Check MCMC convergence using multiple diagnostics"""
    print("\n" + "="*50)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*50)
    
    # R-hat statistic
    rhat = az.rhat(trace)
    print(f"R-hat summary (should be < 1.01):")
    for var in rhat.data_vars:
        rhat_values = rhat[var].values.flatten()
        max_rhat = np.max(rhat_values)
        print(f"  {var}: mean={np.mean(rhat_values):.3f}, max={max_rhat:.3f}", end="")
        if max_rhat > 1.01:
            print(" ⚠️  WARNING: Poor convergence!")
        else:
            print(" ✅")
    
    # Effective sample size
    ess = az.ess(trace)
    print(f"\nEffective Sample Size summary (should be > 400):")
    for var in ess.data_vars:
        ess_values = ess[var].values.flatten()
        min_ess = np.min(ess_values)
        print(f"  {var}: mean={np.mean(ess_values):.0f}, min={min_ess:.0f}", end="")
        if min_ess < 400:
            print(" ⚠️  WARNING: Low ESS!")
        else:
            print(" ✅")
    
    # Check for divergences
    divergent = trace.sample_stats.diverging.sum().item() if hasattr(trace.sample_stats, 'diverging') else 0
    print(f"\nDivergent transitions: {divergent}", end="")
    if divergent > 0:
        print(" ⚠️  WARNING: Divergent transitions detected!")
    else:
        print(" ✅")
    
    return rhat, ess

def plot_posterior_distributions(trace):
    """Create comprehensive posterior distribution plots"""
    print("\nCreating posterior distribution plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Main effects on d-prime
    az.plot_posterior(trace, var_names=['stimulus_effect_d_prime'], ax=axes[0,0])
    axes[0,0].set_title('Stimulus Type Effect on d-prime\n(Complex - Simple)')
    axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['difficulty_effect_d_prime'], ax=axes[0,1])
    axes[0,1].set_title('Trial Difficulty Effect on d-prime\n(Hard - Easy)')
    axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['interaction_d_prime'], ax=axes[0,2])
    axes[0,2].set_title('Interaction Effect on d-prime')
    axes[0,2].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Main effects on criterion
    az.plot_posterior(trace, var_names=['stimulus_effect_criterion'], ax=axes[1,0])
    axes[1,0].set_title('Stimulus Type Effect on Criterion\n(Complex - Simple)')
    axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['difficulty_effect_criterion'], ax=axes[1,1])
    axes[1,1].set_title('Trial Difficulty Effect on Criterion\n(Hard - Easy)')
    axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['interaction_criterion'], ax=axes[1,2])
    axes[1,2].set_title('Interaction Effect on Criterion')
    axes[1,2].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Create output directory
    OUTPUT_DIR = Path('output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_effect_summary_table(trace):
    """Create a summary table comparing effects"""
    print("\n" + "="*50)
    print("EFFECT SIZE SUMMARY")
    print("="*50)
    
    # Extract posterior samples for effects
    effects = {}
    for var in ['stimulus_effect_d_prime', 'difficulty_effect_d_prime', 
                'stimulus_effect_criterion', 'difficulty_effect_criterion',
                'interaction_d_prime', 'interaction_criterion']:
        samples = trace.posterior[var].values.flatten()
        effects[var] = {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'hdi_2.5': np.percentile(samples, 2.5),
            'hdi_97.5': np.percentile(samples, 97.5),
            'prob_positive': np.mean(samples > 0)
        }
    
    # Create formatted table
    df_effects = pd.DataFrame(effects).T
    df_effects.columns = ['Mean', 'SD', 'HDI 2.5%', 'HDI 97.5%', 'P(>0)']
    
    print("\nPosterior Effect Estimates:")
    print(df_effects.round(3))
    
    # Save to file
    OUTPUT_DIR = Path('output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_effects.to_csv(OUTPUT_DIR / 'effects_summary.csv')
    print(f"Effect summary saved to: {OUTPUT_DIR / 'effects_summary.csv'}")
    
    return df_effects

def compare_main_effects(trace):
    """Compare the relative magnitude of Stimulus Type vs Trial Difficulty effects"""
    print("\n" + "="*50)
    print("COMPARING STIMULUS TYPE vs TRIAL DIFFICULTY EFFECTS")
    print("="*50)
    
    # Extract effect estimates
    stimulus_d_effect = np.mean(trace.posterior['stimulus_effect_d_prime'].values)
    difficulty_d_effect = np.mean(trace.posterior['difficulty_effect_d_prime'].values)
    
    stimulus_c_effect = np.mean(trace.posterior['stimulus_effect_criterion'].values)
    difficulty_c_effect = np.mean(trace.posterior['difficulty_effect_criterion'].values)
    
    print(f"Effects on d-prime (sensitivity):")
    print(f"  Stimulus Type (Complex - Simple): {stimulus_d_effect:.3f}")
    print(f"  Trial Difficulty (Hard - Easy):   {difficulty_d_effect:.3f}")
    
    if abs(stimulus_d_effect) > abs(difficulty_d_effect):
        print(f"  → Stimulus Type has larger effect on sensitivity (|{stimulus_d_effect:.3f}| > |{difficulty_d_effect:.3f}|)")
    else:
        print(f"  → Trial Difficulty has larger effect on sensitivity (|{difficulty_d_effect:.3f}| > |{stimulus_d_effect:.3f}|)")
    
    print(f"\nEffects on criterion (response bias):")
    print(f"  Stimulus Type (Complex - Simple): {stimulus_c_effect:.3f}")
    print(f"  Trial Difficulty (Hard - Easy):   {difficulty_c_effect:.3f}")
    
    if abs(stimulus_c_effect) > abs(difficulty_c_effect):
        print(f"  → Stimulus Type has larger effect on bias (|{stimulus_c_effect:.3f}| > |{difficulty_c_effect:.3f}|)")
    else:
        print(f"  → Trial Difficulty has larger effect on bias (|{difficulty_c_effect:.3f}| > |{stimulus_c_effect:.3f}|)")

def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    OUTPUT_DIR = Path('output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')
    plt.show()
    
    return fig

def generate_delta_plots_analysis(max_participants=5):
    """Generate delta plots for multiple participants and provide interpretation"""
    print("\n" + "="*50)
    print("DELTA PLOT ANALYSIS (Diffusion Model Approach)")
    print("="*50)
    
    # Load data for delta plots
    print("Loading data for delta plot analysis...")
    delta_data = read_data('../data/data.csv', prepare_for='delta plots', display=False)
    
    # Get unique participants
    participants = sorted(delta_data['pnum'].unique())
    
    print(f"Generating delta plots for {min(len(participants), max_participants)} participants...")
    
    for i, pnum in enumerate(participants[:max_participants]):
        print(f"  Creating delta plot for participant {pnum}")
        draw_delta_plots(delta_data, pnum)
    
    print(f"\nDelta Plot Interpretation Guide:")
    print(f"  - Increasing delta functions → Drift rate differences")
    print(f"  - Positive intercepts → Boundary separation differences") 
    print(f"  - Flat delta functions → Non-decision time differences")
    print(f"  - Error vs Accurate differences → Response bias effects")
    
    return delta_data

# Main execution function for the assignment
def run_complete_analysis():
    """Run the complete analysis required for the final assignment"""
    print("="*60)
    print("COGS 107 FINAL ASSIGNMENT - SDT AND DELTA PLOT ANALYSIS")
    print("="*60)
    
    # Step 1: Load and prepare SDT data
    print("\n1. LOADING AND PREPARING DATA")
    print("-" * 40)
    sdt_data = read_data('../data/data.csv', prepare_for='sdt', display=True)
    
    if sdt_data.empty:
        print("ERROR: No data loaded. Please check the data file path.")
        return None, None, None
    
    # Step 2: Fit enhanced SDT model with condition effects
    print("\n2. FITTING ENHANCED HIERARCHICAL SDT MODEL")
    print("-" * 40)
    model = apply_hierarchical_sdt_model(sdt_data)
    
    with model:
        print("Running MCMC sampling...")
        print("(This may take a few minutes...)")
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, 
                         target_accept=0.95, random_seed=42)
    
    # Step 3: Check convergence
    print("\n3. CHECKING MODEL CONVERGENCE")
    print("-" * 40)
    rhat, ess = check_convergence(trace)
    
    # Step 4: Display posterior distributions
    print("\n4. DISPLAYING POSTERIOR DISTRIBUTIONS")
    print("-" * 40)
    posterior_fig = plot_posterior_distributions(trace)
    
    # Step 5: Create effect summary table
    print("\n5. CREATING EFFECT SUMMARY TABLE")
    print("-" * 40)
    effects_table = create_effect_summary_table(trace)
    
    # Step 6: Compare main effects
    print("\n6. COMPARING MAIN EFFECTS")
    print("-" * 40)
    compare_main_effects(trace)
    
    # Step 7: Generate delta plots
    print("\n7. GENERATING DELTA PLOTS")
    print("-" * 40)
    delta_data = generate_delta_plots_analysis()
    
    # Step 8: Save trace for further analysis
    print("\n8. SAVING RESULTS")
    print("-" * 40)
    OUTPUT_DIR = Path('output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save trace
    trace.to_netcdf(OUTPUT_DIR / 'sdt_trace.nc')
    print(f"Model trace saved to: {OUTPUT_DIR / 'sdt_trace.nc'}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nFiles created in output/ folder:")
    print("  - posterior_distributions.png (Effect visualizations)")
    print("  - effects_summary.csv (Numerical effect estimates)")
    print("  - delta_plots_*.png (RT distribution analysis)")
    print("  - sdt_trace.nc (Complete MCMC trace)")
    
    print("\nKey Results Summary:")
    stimulus_d_effect = np.mean(trace.posterior['stimulus_effect_d_prime'].values)
    difficulty_d_effect = np.mean(trace.posterior['difficulty_effect_d_prime'].values)
    
    if abs(stimulus_d_effect) > abs(difficulty_d_effect):
        print(f"  → Stimulus Type has larger effect on sensitivity")
        print(f"    (|{stimulus_d_effect:.3f}| > |{difficulty_d_effect:.3f}|)")
    else:
        print(f"  → Trial Difficulty has larger effect on sensitivity")
        print(f"    (|{difficulty_d_effect:.3f}| > |{stimulus_d_effect:.3f}|)")
    
    print("\n✅ Ready for assignment interpretation and testing!")
    
    return trace, sdt_data, effects_table

# Main execution
if __name__ == "__main__":
    # Run the complete analysis when the file is executed
    trace, data, effects = run_complete_analysis()