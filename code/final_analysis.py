"""
Final Assignment: Signal Detection Theory and Delta Plot Analysis
Complete analysis script for COGS 107 Final Assignment

This script:
1. Reads and processes the experimental data
2. Applies hierarchical SDT modeling with condition effects
3. Generates delta plots for diffusion model analysis
4. Compares effects of Trial Difficulty vs Stimulus Type
5. Includes convergence diagnostics and posterior visualization
"""
#Utilized Claude throughout to troubleshoot any problems/errors I ran into

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
from scipy import stats

# Import the existing functions from your provided code
from sdt_ddm import read_data, apply_hierarchical_sdt_model, draw_delta_plots, CONDITION_NAMES, PERCENTILES

def enhanced_sdt_model_with_effects(data):
    """
    Enhanced hierarchical SDT model that quantifies effects of 
    Stimulus Type and Trial Difficulty on d-prime and criterion.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object with condition effects
    """
    # Get unique participants and conditions
    P = len(data['pnum'].unique())
    C = len(data['condition'].unique())
    
    print(f"Modeling {P} participants across {C} conditions")
    
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
        
        # Calculate hit and false alarm rates
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
    print("=== Convergence Diagnostics ===")
    
    # R-hat statistic
    rhat = az.rhat(trace)
    print(f"R-hat summary:")
    for var in rhat.data_vars:
        rhat_values = rhat[var].values.flatten()
        print(f"  {var}: mean={np.mean(rhat_values):.3f}, max={np.max(rhat_values):.3f}")
    
    # Effective sample size
    ess = az.ess(trace)
    print(f"\nEffective Sample Size summary:")
    for var in ess.data_vars:
        ess_values = ess[var].values.flatten()
        print(f"  {var}: mean={np.mean(ess_values):.0f}, min={np.min(ess_values):.0f}")
    
    # Check for divergences
    divergent = trace.sample_stats.diverging.sum().item() if hasattr(trace.sample_stats, 'diverging') else 0
    print(f"\nDivergent transitions: {divergent}")
    
    return rhat, ess

def plot_posterior_distributions(trace):
    """Create comprehensive posterior distribution plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Main effects on d-prime
    az.plot_posterior(trace, var_names=['stimulus_effect_d_prime'], ax=axes[0,0])
    axes[0,0].set_title('Stimulus Type Effect on d-prime')
    axes[0,0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['difficulty_effect_d_prime'], ax=axes[0,1])
    axes[0,1].set_title('Trial Difficulty Effect on d-prime')
    axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['interaction_d_prime'], ax=axes[0,2])
    axes[0,2].set_title('Interaction Effect on d-prime')
    axes[0,2].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    # Main effects on criterion
    az.plot_posterior(trace, var_names=['stimulus_effect_criterion'], ax=axes[1,0])
    axes[1,0].set_title('Stimulus Type Effect on Criterion')
    axes[1,0].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['difficulty_effect_criterion'], ax=axes[1,1])
    axes[1,1].set_title('Trial Difficulty Effect on Criterion')
    axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    az.plot_posterior(trace, var_names=['interaction_criterion'], ax=axes[1,2])
    axes[1,2].set_title('Interaction Effect on Criterion')
    axes[1,2].axvline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Create output directory
    OUTPUT_DIR = Path('../output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_condition_comparison_table(trace):
    """Create a summary table comparing effects"""
    print("=== Effect Size Summary ===")
    
    # Extract posterior samples for effects
    effects = {}
    for var in ['stimulus_effect_d_prime', 'difficulty_effect_d_prime', 
                'stimulus_effect_criterion', 'difficulty_effect_criterion']:
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
    
    return df_effects

def analyze_individual_differences(trace, data):
    """Analyze individual participant parameters"""
    print("=== Individual Differences Analysis ===")
    
    # Extract individual d-prime and criterion estimates
    d_prime_samples = trace.posterior['d_prime'].values  # chains x draws x participants x conditions
    criterion_samples = trace.posterior['criterion'].values
    
    # Calculate means across chains and draws
    d_prime_mean = np.mean(d_prime_samples, axis=(0,1))  # participants x conditions
    criterion_mean = np.mean(criterion_samples, axis=(0,1))
    
    # Create participant summary
    participants = sorted(data['pnum'].unique())
    
    print(f"\nIndividual d-prime estimates (mean across conditions):")
    for i, pnum in enumerate(participants[:10]):  # Show first 10 participants
        mean_d_prime = np.mean(d_prime_mean[i, :])
        print(f"  Participant {pnum}: d-prime = {mean_d_prime:.3f}")
    
    return d_prime_mean, criterion_mean

def create_delta_plots_for_all_participants(data, max_participants=5):
    """Generate delta plots for multiple participants"""
    print("=== Generating Delta Plots ===")
    
    # Prepare data for delta plots
    delta_data = read_data('../data/data.csv', prepare_for='delta plots', display=False)
    
    # Get unique participants
    participants = sorted(delta_data['pnum'].unique())
    
    print(f"Creating delta plots for {min(len(participants), max_participants)} participants...")
    
    for i, pnum in enumerate(participants[:max_participants]):
        print(f"  Generating delta plot for participant {pnum}")
        draw_delta_plots(delta_data, pnum)
    
    return delta_data

def compare_sdt_and_delta_plot_insights(trace, delta_data):
    """Compare insights from SDT model and delta plots"""
    print("=== Comparing SDT and Delta Plot Insights ===")
    
    # Extract effect estimates from SDT model
    stimulus_d_effect = np.mean(trace.posterior['stimulus_effect_d_prime'].values)
    difficulty_d_effect = np.mean(trace.posterior['difficulty_effect_d_prime'].values)
    
    print(f"\nSDT Model Insights:")
    print(f"  Stimulus Type effect on d-prime: {stimulus_d_effect:.3f}")
    print(f"  Trial Difficulty effect on d-prime: {difficulty_d_effect:.3f}")
    
    if abs(stimulus_d_effect) > abs(difficulty_d_effect):
        print(f"  → Stimulus Type has larger effect on sensitivity")
    else:
        print(f"  → Trial Difficulty has larger effect on sensitivity")
    
    print(f"\nDelta Plot Insights:")
    print(f"  Delta plots show RT distribution differences between conditions")
    print(f"  - Drift rate differences appear as increasing delta functions")
    print(f"  - Boundary separation differences appear as positive intercepts")
    print(f"  - Non-decision time differences appear as flat delta functions")
    
    print(f"\nIntegrated Interpretation:")
    print(f"  The SDT analysis quantifies how experimental manipulations affect")
    print(f"  sensitivity (d-prime) and bias (criterion), while delta plots reveal")
    print(f"  the underlying cognitive processes through RT distributions.")

def main():
    """Main analysis pipeline"""
    print("Starting Final Assignment Analysis")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    sdt_data = read_data('../data/data.csv', prepare_for='sdt', display=True)
    
    if sdt_data.empty:
        print("ERROR: No data loaded. Please check the data file.")
        return
    
    # Step 2: Fit enhanced SDT model
    print("\n2. Fitting enhanced hierarchical SDT model...")
    model = enhanced_sdt_model_with_effects(sdt_data)
    
    with model:
        # Sample from posterior
        print("   Running MCMC sampling...")
        trace = pm.sample(2000, tune=1000, chains=4, cores=4, 
                         target_accept=0.95, random_seed=42)
    
    # Step 3: Check convergence
    print("\n3. Checking model convergence...")
    rhat, ess = check_convergence(trace)
    
    # Step 4: Analyze results
    print("\n4. Analyzing posterior distributions...")
    plot_posterior_distributions(trace)
    
    # Step 5: Create effect comparison table
    print("\n5. Creating effect comparison table...")
    effects_table = create_condition_comparison_table(trace)
    
    # Step 6: Analyze individual differences
    print("\n6. Analyzing individual differences...")
    d_prime_individual, criterion_individual = analyze_individual_differences(trace, sdt_data)
    
    # Step 7: Generate delta plots
    print("\n7. Generating delta plots...")
    delta_data = create_delta_plots_for_all_participants(sdt_data)
    
    # Step 8: Compare insights
    print("\n8. Comparing SDT and delta plot insights...")
    compare_sdt_and_delta_plot_insights(trace, delta_data)
    
    # Step 9: Save results
    print("\n9. Saving results...")
    OUTPUT_DIR = Path('../output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save trace
    trace.to_netcdf(OUTPUT_DIR / 'sdt_trace.nc')
    
    # Save effects table
    effects_table.to_csv(OUTPUT_DIR / 'effects_summary.csv')
    
    print(f"\n✅ Analysis complete! Results saved to {OUTPUT_DIR}/")
    print("\nKey Findings:")
    print("- Check posterior_distributions.png for effect visualizations")
    print("- Check effects_summary.csv for numerical summaries")
    print("- Check delta_plots_*.png for RT distribution analysis")
    print("- Model trace saved for further analysis")
    
    return trace, sdt_data, effects_table

if __name__ == "__main__":
    # Run the complete analysis
    trace, data, effects = main()