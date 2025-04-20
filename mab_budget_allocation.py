# mab_budget_allocation.py
"""
Multi-Armed Bandit (MAB) Budget Allocation System with Recommendations
Streamlit frontend for interactive campaign optimization and actionable insights.
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import math
import plotly.express as px
import plotly.graph_objects as go

# --- Data Loading ---
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if df['account_name'].isnull().any():
        mode_val = df['account_name'].mode()[0]
        df['account_name'].fillna(mode_val, inplace=True)
    df['CTR'] = 0.0
    non_zero_imps_mask = df['imps'] != 0
    df.loc[non_zero_imps_mask, 'CTR'] = (
        df.loc[non_zero_imps_mask, 'clicks'] / df.loc[non_zero_imps_mask, 'imps']
    ) * 100

    # Add conversion rate and ROAS metrics
    df['conversion_rate'] = 0.0
    non_zero_clicks = df['clicks'] != 0
    df.loc[non_zero_clicks, 'conversion_rate'] = (df.loc[non_zero_clicks, 'purchase_total'] / df.loc[non_zero_clicks, 'clicks']) * 100

    df['ROAS'] = 0.0
    non_zero_spend = df['spend'] != 0
    df.loc[non_zero_spend, 'ROAS'] = df.loc[non_zero_spend, 'purchase_total_value'] / df.loc[non_zero_spend, 'spend']
    
    return df

# --- MAB Algorithms ---
class EpsilonGreedy:
    def __init__(self, arms, epsilon=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.name = f"Epsilon-Greedy (ε={epsilon})"
    def select_arm(self):
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        else:
            return max(self.values.items(), key=lambda x: x[1])[0] if self.values else random.choice(self.arms)
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

class UCB:
    def __init__(self, arms, alpha=2.0):
        self.arms = arms
        self.alpha = alpha
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.total_count = 0
        self.name = f"UCB (α={alpha})"
    def select_arm(self):
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm
        ucb_values = {}
        for arm in self.arms:
            bonus = math.sqrt((self.alpha * math.log(self.total_count)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_count += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

class ThompsonSampling:
    def __init__(self, arms):
        self.arms = arms
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)
        self.name = "Thompson Sampling"
    def select_arm(self):
        samples = {arm: np.random.beta(self.alpha[arm], self.beta[arm]) for arm in self.arms}
        return max(samples.items(), key=lambda x: x[1])[0]
    def update(self, chosen_arm, reward):
        # Normalize reward to prevent negative beta values
        # For metrics like CTR and conversion rate (0-100 scale)
        if reward <= 100:
            success = reward / 100
            failure = 1 - success
        # For metrics like ROAS that can exceed 100
        else:
            # For high values, use a dampening function to keep within reasonable bounds
            success = min(0.99, 1 - (1 / (1 + reward/100)))
            failure = 1 - success
            
        # Update beta distribution parameters
        self.alpha[chosen_arm] += success
        self.beta[chosen_arm] += failure

# --- Simulation and Recommendation Logic ---
def run_simulation(algorithms, campaign_data, metric_name='CTR', n_simulations=1000):
    all_rewards = {algo.name: [] for algo in algorithms}
    cumulative_rewards = {algo.name: [] for algo in algorithms}
    chosen_arms = {algo.name: [] for algo in algorithms}
    
    for _ in range(n_simulations):
        for algo in algorithms:
            chosen_arm = algo.select_arm()
            chosen_arms[algo.name].append(chosen_arm)
            
            reward = np.random.choice(campaign_data[chosen_arm]) if campaign_data[chosen_arm].size > 0 else 0
            algo.update(chosen_arm, reward)
            all_rewards[algo.name].append(reward)
            
            if not cumulative_rewards[algo.name]:
                cumulative_rewards[algo.name].append(reward)
            else:
                cumulative_rewards[algo.name].append(cumulative_rewards[algo.name][-1] + reward)
                
    return all_rewards, cumulative_rewards, chosen_arms

def calculate_regret(chosen_arms, campaign_stats, metric_name='CTR'):
    mean_key = f'mean_{metric_name.lower()}'
    optimal_arm = max(campaign_stats.items(), key=lambda x: x[1][mean_key])[0]
    optimal_reward = campaign_stats[optimal_arm][mean_key]
    
    regret = []
    cumulative_regret = []
    total_regret = 0
    
    for arm in chosen_arms:
        instant_regret = optimal_reward - campaign_stats[arm][mean_key]
        regret.append(instant_regret)
        total_regret += instant_regret
        cumulative_regret.append(total_regret)
        
    return regret, cumulative_regret, optimal_arm

def get_recommendations(campaign_stats, metric_name='CTR', top_n=5, low_n=5):
    mean_key = f'mean_{metric_name.lower()}'
    sorted_campaigns = sorted(campaign_stats.items(), key=lambda x: x[1][mean_key], reverse=True)
    
    # Get top performers
    top = sorted_campaigns[:top_n]
    
    # Get bottom performers, ensuring no overlap with top performers
    top_names = [item[0] for item in top]
    low = []
    
    # Start from the end of the list (worst performers)
    for campaign in reversed(sorted_campaigns):
        # Skip if this campaign is already in the top performers list
        if campaign[0] not in top_names:
            low.append(campaign)
            # Stop once we have enough bottom performers
            if len(low) >= low_n:
                break
    
    return top, low

def get_dos_and_donts(top, low, metric_name='CTR'):
    mean_key = f'mean_{metric_name.lower()}'
    dos = [f"Allocate more budget to: {c[0]} (Mean {metric_name}: {c[1][mean_key]:.2f})" for c in top]
    donts = [f"Reduce/avoid budget for: {c[0]} (Mean {metric_name}: {c[1][mean_key]:.2f})" for c in low]
    return dos, donts

def plot_cumulative_rewards(cumulative_rewards, algo_name, n_simulations, future_iterations=0):
    fig = go.Figure()
    iterations = list(range(len(cumulative_rewards)))
    fig.add_trace(go.Scatter(
        x=iterations, 
        y=cumulative_rewards,
        mode='lines',
        name=algo_name
    ))
    if future_iterations > 0:
        fig.add_vline(
            x=n_simulations, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Future Iterations Start", 
            annotation_position="top right"
        )
    fig.update_layout(
        title=f'Cumulative Rewards Over Time',
        xaxis_title='Iteration',
        yaxis_title='Cumulative Reward',
        height=400
    )
    return fig

def plot_moving_average(all_rewards, algo_name, n_simulations, window=50, future_iterations=0):
    if len(all_rewards) < window:
        window = max(10, len(all_rewards) // 5)
    moving_avg = [np.mean(all_rewards[max(0, i-window):i+1]) for i in range(len(all_rewards))]
    fig = go.Figure()
    iterations = list(range(len(moving_avg)))
    fig.add_trace(go.Scatter(
        x=iterations, 
        y=moving_avg,
        mode='lines',
        name=f'Moving Avg (Window={window})'
    ))
    if future_iterations > 0:
        fig.add_vline(
            x=n_simulations, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Future Iterations Start", 
            annotation_position="top right"
        )
    fig.update_layout(
        title=f'Moving Average Reward (Window={window})',
        xaxis_title='Iteration',
        yaxis_title='Moving Avg Reward',
        height=400
    )
    return fig

def plot_regret(cumulative_regret, n_simulations, future_iterations=0):
    fig = go.Figure()
    iterations = list(range(len(cumulative_regret)))
    fig.add_trace(go.Scatter(
        x=iterations, 
        y=cumulative_regret,
        mode='lines',
        name='Cumulative Regret'
    ))
    if future_iterations > 0:
        fig.add_vline(
            x=n_simulations, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Future Iterations Start", 
            annotation_position="top right"
        )
    fig.update_layout(
        title='Cumulative Regret Over Time',
        xaxis_title='Iteration',
        yaxis_title='Cumulative Regret',
        height=400
    )
    return fig

def plot_arm_selection_frequency(chosen_arms, campaign_stats, metric_name='CTR', top_n=5):
    mean_key = f'mean_{metric_name.lower()}'
    top_arms = sorted(campaign_stats.items(), key=lambda x: x[1][mean_key], reverse=True)[:top_n]
    top_arm_names = [arm[0] for arm in top_arms]
    all_arms = set(chosen_arms)
    arm_counts = {arm: chosen_arms.count(arm) for arm in all_arms}
    sorted_arms = sorted(arm_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    arms = [a[0] for a in sorted_arms]
    counts = [a[1] for a in sorted_arms]
    is_top = [arm in top_arm_names for arm in arms]
    colors = ['#1f77b4' if top else '#ff7f0e' for top in is_top]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts,
        y=arms,
        orientation='h',
        marker_color=colors,
        text=[f"{count} selections" for count in counts],
        textposition='auto'
    ))
    fig.update_layout(
        title='Campaign Selection Frequency',
        xaxis_title='Number of Selections',
        yaxis_title='Campaign',
        height=max(400, len(arms) * 30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.add_trace(go.Bar(
        x=[None],
        y=[None],
        marker_color='#1f77b4',
        name='Top Performing Campaign',
        showlegend=True
    ))
    fig.add_trace(go.Bar(
        x=[None],
        y=[None],
        marker_color='#ff7f0e',
        name='Other Campaign',
        showlegend=True
    ))
    return fig

def plot_exploration_exploitation_ratio(chosen_arms, campaign_stats, metric_name='CTR', top_n=5):
    mean_key = f'mean_{metric_name.lower()}'
    top_arms = sorted(campaign_stats.items(), key=lambda x: x[1][mean_key], reverse=True)[:top_n]
    top_arm_names = [arm[0] for arm in top_arms]
    top_selections = sum(1 for arm in chosen_arms if arm in top_arm_names)
    other_selections = len(chosen_arms) - top_selections
    top_percentage = (top_selections / len(chosen_arms)) * 100
    other_percentage = 100 - top_percentage
    fig = go.Figure(go.Pie(
        labels=['Exploitation (Top Campaigns)', 'Exploration (Other Campaigns)'],
        values=[top_percentage, other_percentage],
        hole=.4,
        marker_colors=['#2ca02c', '#9467bd']
    ))
    fig.update_layout(
        title='Exploration vs Exploitation Balance',
        annotations=[dict(text=f'{top_percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=400
    )
    return fig

# --- Streamlit Frontend ---
def main():
    st.set_page_config(page_title="MAB Budget Allocation System", layout="wide", page_icon="💡")
    st.title("💡 Multi-Armed Bandit Budget Allocation System")
    st.markdown("""
    Optimize your campaign budget allocation using advanced MAB algorithms. Get actionable recommendations, see do's and don'ts, and simulate future performance.
    """)
    
    # Load data
    df = load_data("sample2_encrypted.csv")
    
    # Add arms selector
    st.sidebar.header("Arms Configuration")
    arm_column = st.sidebar.selectbox(
        "Select Arms (Granularity)",
        options=["campaign_name", "account_name", "ad_set_name"] if "ad_set_name" in df.columns else ["campaign_name", "account_name"],
        index=0,
        help="Choose which entity to optimize - campaigns, accounts, or ad sets"
    )
    
    # Add metric selector
    st.sidebar.header("Metrics")
    metric_choice = st.sidebar.selectbox(
        "Optimization Metric", 
        ["CTR", "conversion_rate", "ROAS"],
        help="Choose the metric to optimize for"
    )
    
    # Get arms based on selected column
    arms = df[arm_column].unique()
    campaign_data = {arm: df[df[arm_column] == arm][metric_choice].values for arm in arms}
    
    # Calculate statistics for each arm
    mean_key = f'mean_{metric_choice.lower()}'
    campaign_stats = {}
    for arm, values in campaign_data.items():
        campaign_stats[arm] = {
            mean_key: np.mean(values) if len(values) > 0 else 0,
            f'std_{metric_choice.lower()}': np.std(values) if len(values) > 0 else 0,
            'count': len(values)
        }
    
    # Simulation settings
    st.sidebar.header("Simulation Settings")
    n_simulations = st.sidebar.slider("Number of MAB Iterations", min_value=100, max_value=10000, value=1000, step=100)
    future_iterations = st.sidebar.slider("Simulate Future Iterations", min_value=0, max_value=5000, value=500, step=100)
    algo_choice = st.sidebar.selectbox("Algorithm", ["UCB", "Epsilon-Greedy", "Thompson Sampling"])
    
    # Algorithm-specific parameters
    if algo_choice == "Epsilon-Greedy":
        epsilon = st.sidebar.slider("Epsilon (exploration rate)", 0.01, 0.5, 0.1, 0.01)
        algorithms = [EpsilonGreedy(arms, epsilon=epsilon)]
    elif algo_choice == "UCB":
        alpha = st.sidebar.slider("Alpha (exploration parameter)", 0.1, 5.0, 2.0, 0.1)
        algorithms = [UCB(arms, alpha=alpha)]
    else:
        algorithms = [ThompsonSampling(arms)]
    
    # Run simulation button
    if st.sidebar.button("Run Simulation", use_container_width=True):
        with st.spinner(f"Running MAB simulation on {len(arms)} {arm_column}s..."):
            total_iterations = n_simulations + future_iterations
            all_rewards, cumulative_rewards, chosen_arms = run_simulation(
                algorithms, 
                campaign_data, 
                metric_name=metric_choice,
                n_simulations=total_iterations
            )
            
            # Get first (and only) algorithm results
            algo_name = algorithms[0].name
            all_rewards_algo = all_rewards[algo_name]
            cumulative_rewards_algo = cumulative_rewards[algo_name]
            chosen_arms_algo = chosen_arms[algo_name]
            
            # Calculate regret
            regret, cumulative_regret, optimal_arm = calculate_regret(
                chosen_arms_algo, 
                campaign_stats, 
                metric_name=metric_choice
            )
            
            # Get recommendations
            top, low = get_recommendations(campaign_stats, metric_name=metric_choice)
            dos, donts = get_dos_and_donts(top, low, metric_name=metric_choice)
            
            # Display simulation info
            st.info(f"""
            **Simulation Info**
            - Algorithm: {algo_name}
            - Arms: {arm_column} ({len(arms)} arms)
            - Metric: {metric_choice}
            - MAB Iterations: {n_simulations}
            - Future Iterations: {future_iterations}
            - Optimal {arm_column}: {optimal_arm} (Mean {metric_choice}: {campaign_stats[optimal_arm][mean_key]:.2f})
            """)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Recommendations", 
                "Performance Metrics",
                "Algorithm Analysis",
                "Implementation Plan",
                "Marketing Action Plan",
                "Advanced Analytics Opportunities"  # New tab for additional analytics
            ])
            
            with tab1:
                st.subheader(f"Budget Allocation Recommendations for {arm_column}s")
                st.success("\n".join(dos))
                st.error("\n".join(donts))
                
                col1, col2 = st.columns(2)
                with col1:
                    # Campaign performance comparison
                    st.subheader(f"{arm_column.replace('_', ' ').title()} Performance (Mean {metric_choice})")
                    sorted_stats = sorted(campaign_stats.items(), key=lambda x: x[1][mean_key], reverse=True)
                    names = [c[0][:20] + '...' if len(c[0]) > 20 else c[0] for c in sorted_stats]
                    values = [c[1][mean_key] for c in sorted_stats]
                    
                    fig = px.bar(
                        x=values,
                        y=names,
                        orientation='h',
                        labels={'x': f'Mean {metric_choice}', 'y': arm_column.replace('_', ' ').title()},
                        title=f'{arm_column.replace("_", " ").title()}s by Mean {metric_choice}'
                    )
                    fig.update_layout(height=max(400, len(names) * 25))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Arm selection distribution
                    st.subheader(f"{arm_column.replace('_', ' ').title()} Selection Distribution")
                    arm_selection_fig = plot_arm_selection_frequency(
                        chosen_arms_algo, 
                        campaign_stats, 
                        metric_name=metric_choice
                    )
                    st.plotly_chart(arm_selection_fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cumulative rewards visualization
                    st.subheader("Cumulative Rewards")
                    cumulative_fig = plot_cumulative_rewards(
                        cumulative_rewards_algo, 
                        algo_name, 
                        n_simulations, 
                        future_iterations
                    )
                    st.plotly_chart(cumulative_fig, use_container_width=True)
                
                with col2:
                    # Moving average visualization
                    st.subheader("Moving Average Reward")
                    window = min(100, len(all_rewards_algo) // 10)  # 10% of iterations or max 100
                    moving_avg_fig = plot_moving_average(
                        all_rewards_algo, 
                        algo_name, 
                        n_simulations, 
                        window, 
                        future_iterations
                    )
                    st.plotly_chart(moving_avg_fig, use_container_width=True)
                
                # Regret analysis
                st.subheader("Regret Analysis")
                regret_fig = plot_regret(cumulative_regret, n_simulations, future_iterations)
                st.plotly_chart(regret_fig, use_container_width=True)
                
                st.info("""
                **Understanding Regret**: 
                Regret measures the difference between the reward obtained by always selecting the optimal campaign 
                and the reward obtained by the algorithm's selections. Lower regret means better algorithm performance.
                
                A good algorithm should have:
                1. Sublinear regret growth (curve flattens over time)
                2. Quick initial learning (steep rise followed by flattening)
                """)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Exploration vs Exploitation
                    st.subheader("Exploration vs Exploitation Balance")
                    exploration_fig = plot_exploration_exploitation_ratio(
                        chosen_arms_algo, 
                        campaign_stats, 
                        metric_name=metric_choice
                    )
                    st.plotly_chart(exploration_fig, use_container_width=True)
                    
                    # Add explanation
                    st.markdown("""
                    This chart shows how often the algorithm selected top-performing campaigns (exploitation) 
                    versus other campaigns (exploration).
                    
                    A good balance is essential for:
                    - Finding the best campaigns (exploration)
                    - Maximizing returns from known good campaigns (exploitation)
                    """)
                
                with col2:
                    # Algorithm convergence rate
                    st.subheader("Algorithm Convergence")
                    
                    # Count transitions between exploration and exploitation
                    transitions = 0
                    in_top = None
                    for arm in chosen_arms_algo:
                        is_in_top = arm in [a[0] for a in top]
                        if in_top is not None and is_in_top != in_top:
                            transitions += 1
                        in_top = is_in_top
                    
                    # Metrics about algorithm behavior
                    st.metric("Exploration-Exploitation Transitions", transitions)
                    
                    # Selection of optimal arm
                    optimal_selections = chosen_arms_algo.count(optimal_arm)
                    optimal_percentage = (optimal_selections / len(chosen_arms_algo)) * 100
                    
                    st.metric(
                        "Optimal Campaign Selection Rate", 
                        f"{optimal_percentage:.1f}%",
                        help=f"How often the algorithm selected the best campaign ({optimal_arm})"
                    )
                    
                    # Learning speed (time to converge to top campaigns)
                    window_size = min(50, n_simulations // 10)
                    windows = [chosen_arms_algo[i:i+window_size] for i in range(0, len(chosen_arms_algo), window_size)]
                    
                    if windows:
                        top_in_windows = []
                        for i, window in enumerate(windows):
                            if window:
                                top_count = sum(1 for arm in window if arm in [a[0] for a in top])
                                top_percentage = (top_count / len(window)) * 100
                                top_in_windows.append(top_percentage)
                        
                        # Plot convergence
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(len(top_in_windows))),
                            y=top_in_windows,
                            mode='lines+markers',
                            name='Top Campaign Selection %'
                        ))
                        
                        fig.update_layout(
                            title='Algorithm Convergence Over Time',
                            xaxis_title=f'Time Window (each = {window_size} iterations)',
                            yaxis_title='% Selection of Top Campaigns',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("📊 Budget Allocation Strategy")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Top Performers", value="70%", help="Allocate 70% to proven high-performance campaigns")
                    st.markdown("**Invest in campaigns listed in the DO section**")
                
                with col2:
                    st.metric(label="Medium Performers", value="20%", help="Allocate 20% to medium-performance campaigns")
                    st.markdown("**Continue testing campaigns with moderate performance**")
                
                with col3:
                    st.metric(label="Exploration", value="10%", help="Allocate 10% to new campaign ideas")
                    st.markdown("**Always explore new creative approaches and audiences**")
                
                # Add a visual chart
                fig = go.Figure(go.Pie(
                    labels=["Top Performers", "Medium Performers", "Exploration"],
                    values=[70, 20, 10],
                    hole=.4,
                    marker_colors=['#50C878', '#ADD8E6', '#FFD700']
                ))
                
                fig.update_layout(
                    title="Recommended Budget Allocation",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🛠️ Implementation Timeline")
                
                # Implementation timeline
                timeline_data = [
                    {"Phase": "Phase 1", "Task": "Budget Reallocation", "Duration": "Week 1", 
                     "Description": "Gradually adjust budgets based on MAB recommendations"},
                    {"Phase": "Phase 2", "Task": "Performance Monitoring", "Duration": "Weeks 2-3", 
                     "Description": "Monitor performance of reallocated budget"},
                    {"Phase": "Phase 3", "Task": "Algorithm Refinement", "Duration": "Week 4", 
                     "Description": "Run new MAB simulation with fresh data"},
                    {"Phase": "Continuous", "Task": "Ongoing Optimization", "Duration": "Monthly", 
                     "Description": "Repeat this cycle monthly or when significant performance changes occur"}
                ]
                
                st.table(pd.DataFrame(timeline_data))
            
            with tab5:
                st.header("📊 Marketing Action Plan")
                st.subheader("What This Means For Your Marketing Budget")
                
                # Simple explanation
                st.markdown(f"""
                Our AI-driven analysis has identified the best performing {arm_column.replace('_', ' ')}s 
                based on {metric_choice}. Here's what you should do next:
                """)
                
                # Create three columns layout
                action_col1, action_col2 = st.columns([3, 2])
                
                with action_col1:
                    # Winners and losers in simple terms
                    st.subheader("🏆 Top Performers - Increase Budget")
                    
                    # Create a dataframe for better formatting
                    top_df = pd.DataFrame([
                        {
                            f"{arm_column.replace('_', ' ').title()}": c[0],
                            f"{metric_choice}": f"{c[1][mean_key]:.2f}",
                            "Recommended Action": "➕ Increase budget"
                        } for c in top[:3]  # Show top 3 for simplicity
                    ])
                    
                    st.dataframe(
                        top_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    st.subheader("⚠️ Underperformers - Reduce Budget")
                    
                    # Create a dataframe for better formatting
                    bottom_df = pd.DataFrame([
                        {
                            f"{arm_column.replace('_', ' ').title()}": c[0],
                            f"{metric_choice}": f"{c[1][mean_key]:.2f}",
                            "Recommended Action": "➖ Reduce budget"
                        } for c in low[:3]  # Show bottom 3 for simplicity
                    ])
                    
                    st.dataframe(
                        bottom_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    # Simple next steps
                    st.subheader("📝 Next Steps")
                    st.markdown("""
                    1. **Reallocate Budget**: Move budget from underperformers to top performers
                    2. **Monitor Weekly**: Check if performance continues to improve
                    3. **Run This Tool Again**: After collecting new data (2-4 weeks)
                    """)
                
                with action_col2:
                    # Simple budget reallocation pie chart
                    st.subheader("Recommended Budget Split")
                    
                    # Get the names for the first few top and bottom performers
                    top_names = [c[0][:15] + '...' if len(c[0]) > 15 else c[0] for c in top[:3]]
                    other_names = ["Other " + arm_column.replace('_', ' ') + "s"]
                    
                    # Calculate suggested budget percentages (simplified)
                    total_performers = len(arms)
                    top_percentage = min(70, 100 * (3 / total_performers) * 2)  # Roughly doubling the fair share
                    other_percentage = 100 - top_percentage
                    
                    # Each top performer gets an equal share of the top_percentage
                    top_individual = [top_percentage / len(top_names) for _ in top_names]
                    
                    # Create labels and values for pie chart
                    budget_labels = top_names + other_names
                    budget_values = top_individual + [other_percentage]
                    
                    # Create a colorful pie chart
                    colors = ['#50C878', '#6CC4A4', '#88DFD1', '#AAAAAA'][:len(budget_labels)]
                    
                    fig = go.Figure(go.Pie(
                        labels=budget_labels,
                        values=budget_values,
                        hole=.4,
                        marker_colors=colors,
                        textinfo='percent+label'
                    ))
                    
                    fig.update_layout(
                        title="Suggested Budget Allocation",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Expected results callout
                    st.success(f"""
                    ### Expected Outcome
                    
                    By implementing this plan, you can expect to see:
                    
                    • Improved overall {metric_choice}
                    • Better return on ad spend
                    • Clearer insights on what works
                    """)
                
                # Horizontal comparison before/after
                st.subheader("Before vs After: Expected Performance Impact")
                
                # Calculate current average performance
                current_avg = np.mean([stats[mean_key] for stats in campaign_stats.values()])
                
                # Estimate improved performance (simplified model - assumes 30% improvement from reallocation)
                improved_est = current_avg * 1.3
                
                before_after_data = pd.DataFrame({
                    "Scenario": ["Current Performance", "After Optimization"],
                    f"Average {metric_choice}": [current_avg, improved_est]
                })
                
                fig = px.bar(
                    before_after_data, 
                    x="Scenario", 
                    y=f"Average {metric_choice}",
                    color="Scenario",
                    color_discrete_map={
                        "Current Performance": "#AAAAAA", 
                        "After Optimization": "#50C878"
                    },
                    text_auto='.2f'
                )
                
                fig.update_layout(
                    title=f"Projected {metric_choice} Improvement",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Print a simple PDF-style executive summary
                with st.expander("📑 Executive Summary (Click to Expand)"):
                    st.markdown(f"""
                    ## Budget Optimization Executive Summary
                    
                    **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}
                    
                    **Metric Analyzed**: {metric_choice}
                    
                    **Key Findings**:
                    
                    1. Your **top performing {arm_column.replace('_', ' ')}** is **{top[0][0]}** with {metric_choice} of {top[0][1][mean_key]:.2f}
                    
                    2. Your **lowest performing {arm_column.replace('_', ' ')}** is **{low[0][0]}** with {metric_choice} of {low[0][1][mean_key]:.2f}
                    
                    3. By reallocating budget to top performers, we estimate a **{(improved_est/current_avg - 1)*100:.1f}%** improvement in overall {metric_choice}
                    
                    **Budget Reallocation Suggestion**:
                    
                    * Increase budget for {", ".join([c[0] for c in top[:3]])}
                    * Decrease budget for {", ".join([c[0] for c in low[:3]])}
                    * Maintain current budget for other {arm_column.replace('_', ' ')}s
                    
                    **Timeline**: Implement changes within 1 week and monitor for 3-4 weeks before next optimization.
                    """)
                    
                    if st.button("Download Executive Summary (PDF)", type="primary"):
                        st.info("This would generate a downloadable PDF report in a production environment.")
                
            # New tab for advanced analytics opportunities
            with tab6:
                st.header("🚀 Advanced Analytics Opportunities")
                
                st.markdown("""
                ## Taking Your Marketing Analytics to the Next Level
                
                While Multi-Armed Bandit optimization provides powerful budget allocation recommendations, 
                there are many other advanced analytics approaches that can further enhance your marketing performance.
                
                Explore the possibilities below to see how AI and machine learning can transform your marketing analytics.
                """)
                
                # Organize the advanced analytics into categories
                st.subheader("Explore Advanced Analytics Capabilities")
                
                # Create category tabs for different types of advanced analytics
                analytics_tabs = st.tabs([
                    "Predictive Analytics", 
                    "Audience Insights", 
                    "Creative Optimization",
                    "Advanced Optimization"
                ])
                
                # Tab 1: Predictive Analytics
                with analytics_tabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 Predictive Performance Modeling")
                        st.markdown("""
                        **What it does**: Builds a "crystal ball" that estimates how well each ad will convert 
                        (e.g. lead, sale) before you spend money.
                        
                        **How it helps**: You can prioritize budget on ads likely to perform best, reducing wasted spend.
                        """)
                        
                        # Illustrative image or icon would go here in a production app
                        st.info("**Implementation difficulty**: Medium - Requires historical conversion data and machine learning expertise")
                    
                    with col2:
                        st.markdown("### 🔮 Time-Series Forecasting")
                        st.markdown("""
                        **What it does**: Projects future trends in cost-per-click, conversion rates, or overall spend.
                        
                        **How it helps**: Plan budgets weeks or months ahead, smoothing out seasonality dips and peaks.
                        """)
                        
                        st.info("**Implementation difficulty**: Medium - Requires minimum 1 year of historical data for seasonal patterns")
                    
                    st.markdown("### 🚨 Anomaly Detection")
                    st.markdown("""
                    **What it does**: Monitors your campaigns to flag sudden spikes or drops in spend, clicks, or conversions.
                    
                    **How it helps**: Early alerts mean you can pause a broken campaign or double-down on a surprise winner immediately.
                    """)
                    
                    st.info("**Implementation difficulty**: Low to Medium - Can start with simple statistical rules and grow in sophistication")
                
                # Tab 2: Audience Insights
                with analytics_tabs[1]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 👥 Customer Segmentation")
                        st.markdown("""
                        **What it does**: Automatically groups your audience into clusters (e.g. bargain-hunters vs. brand-loyalists) 
                        based on past behavior.
                        
                        **How it helps**: Tailor ad creatives and bids to each group, improving relevance and boosting engagement.
                        """)
                        
                        st.info("**Implementation difficulty**: Medium - Requires customer behavioral data and clustering algorithms")
                    
                    with col2:
                        st.markdown("### ⚖️ Uplift (Incrementality) Modeling")
                        st.markdown("""
                        **What it does**: Estimates the true "lift" your ads provide—i.e. sales you wouldn't have otherwise gotten.
                        
                        **How it helps**: Prevents you from attributing natural demand to your ads, so you only pay for genuine incremental impact.
                        """)
                        
                        st.info("**Implementation difficulty**: High - Requires controlled experiments or advanced causal inference methods")
                    
                    st.markdown("### 💬 Sentiment & Keyword Analysis")
                    st.markdown("""
                    **What it does**: Reads comments and ad text to flag positive vs. negative reactions and high-value keywords.
                    
                    **How it helps**: You can refine messaging in real time—emphasizing words that resonate, dropping those that don't.
                    """)
                    
                    st.info("**Implementation difficulty**: Medium - Requires natural language processing techniques and social media data")
                
                # Tab 3: Creative Optimization
                with analytics_tabs[2]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### ✍️ Automated Ad Copy & Creative Optimization")
                        st.markdown("""
                        **What it does**: Uses AI (e.g. a simple neural network) to test thousands of headlines/images 
                        and learn which combinations drive the best results.
                        
                        **How it helps**: Quickly surfaces high-performing ad variations without manual A/B testing.
                        """)
                        
                        st.info("**Implementation difficulty**: Medium to High - Requires dynamic ad creation capabilities and ML models")
                    
                    with col2:
                        st.markdown("### 👁️ Image Recognition for Creative Scoring")
                        st.markdown("""
                        **What it does**: Analyzes visual elements (faces, colors, objects) in your ad images to predict their effectiveness.
                        
                        **How it helps**: Guides your design team to focus on the visual styles and elements that drive clicks.
                        """)
                        
                        st.info("**Implementation difficulty**: High - Requires computer vision models and substantial image performance data")
                    
                    # Demo visualization for creative optimization
                    st.subheader("Sample: Creative Element Performance Heatmap")
                    
                    # Create sample data for the heatmap
                    creative_elements = ["People/Faces", "Product Close-up", "Text Overlay", "Bright Colors", "Brand Logo Size"]
                    metrics = ["CTR", "Conversion Rate", "Engagement", "Brand Recall"]
                    
                    # Generate random correlation data for the heatmap
                    np.random.seed(42)  # For reproducibility
                    correlation_data = np.random.uniform(low=-0.2, high=0.8, size=(len(creative_elements), len(metrics)))
                    
                    # Create a heatmap using Plotly
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_data,
                        x=metrics,
                        y=creative_elements,
                        colorscale='RdBu_r',
                        zmid=0,
                    ))
                    
                    fig.update_layout(
                        title="Impact of Creative Elements on Performance Metrics",
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=30)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("This heatmap shows how different creative elements correlate with performance metrics. Red indicates positive correlation, blue indicates negative correlation.")
                
                # Tab 4: Advanced Optimization
                with analytics_tabs[3]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🤖 Reinforcement Learning for Automated Bidding")
                        st.markdown("""
                        **What it does**: Extends your bandit setup into a continuous learning agent that adjusts bids 
                        in real time across all placements.
                        
                        **How it helps**: Maintains optimal bidding as market conditions shift minute-to-minute, maximizing ROI.
                        """)
                        
                        st.info("**Implementation difficulty**: High - Requires reinforcement learning expertise and real-time bid API integration")
                    
                    with col2:
                        st.markdown("### 💰 ROI-Driven Budget Optimization")
                        st.markdown("""
                        **What it does**: Takes all the above signals (predictions, segments, lifts, forecasts) and uses 
                        optimization to allocate your total monthly budget.
                        
                        **How it helps**: Ensures every dollar is channeled to the combination of ads, audiences, and times 
                        that deliver the highest overall return.
                        """)
                        
                        st.info("**Implementation difficulty**: Medium to High - Combines multiple models into a unified optimization framework")
                
                # Roadmap section
                st.subheader("Implementation Roadmap")
                
                # Create columns for the roadmap phases
                phase1, phase2, phase3 = st.columns(3)
                
                with phase1:
                    st.markdown("### Phase 1: Foundation")
                    st.markdown("""
                    1. **Anomaly Detection**
                    2. **Time-Series Forecasting**
                    3. **Customer Segmentation**
                    
                    
                    Focus on establishing the data pipeline and basic analytics that provide immediate value.
                    """)
                
                with phase2:
                    st.markdown("### Phase 2: Advancement")
                    st.markdown("""
                    1. **Predictive Performance Modeling**
                    2. **Sentiment & Keyword Analysis**
                    3. **Automated Ad Copy Optimization**
                    
                    
                    Build on the foundation with more sophisticated models that optimize specific aspects of campaigns.
                    """)
                
                with phase3:
                    st.markdown("### Phase 3: Mastery")
                    st.markdown("""
                    1. **Uplift Modeling**
                    2. **Image Recognition**
                    3. **Reinforcement Learning**
                    4. **Unified ROI Optimization**
                    
                    
                    Implement the most advanced techniques that integrate all data sources for maximum impact.
                    """)
                
                # Call to action
                st.markdown("---")
                col1, col2 = st.columns([2,1])
                
                with col1:
                    st.markdown("""
                    ## Ready to take your marketing analytics to the next level?
                    
                    Start with the Implementation Roadmap and gradually build your analytics capabilities.
                    Each step adds value and prepares you for more advanced techniques.
                    """)
                
                with col2:
                    # This would be a button that redirects to a contact form or documentation in a real application
                    st.button("Explore Implementation Options", type="primary", use_container_width=True)
                    st.button("Download Analytics Roadmap PDF", use_container_width=True)

    else:
        st.info("👈 Adjust the arms, metrics, and simulation parameters in the sidebar and click 'Run Simulation' to get started.")

if __name__ == "__main__":
    main()
