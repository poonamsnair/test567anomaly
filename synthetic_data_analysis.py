#!/usr/bin/env python3
"""
Comprehensive Synthetic Data Analysis and Visualization

This script creates detailed visualizations of synthetic trajectory data characteristics,
comparing normal vs anomalous trajectories across multiple dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class SyntheticDataAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.charts_dir = self.results_dir / "charts"
        self.data_dir = self.results_dir / "data"
        
        # Load data
        self.features_df = pd.read_csv(self.data_dir / "extracted_features.csv")
        self.normal_trajectories = self._load_pickle(self.data_dir / "normal_trajectories.pkl")
        self.anomalous_trajectories = self._load_pickle(self.data_dir / "anomalous_trajectories.pkl")
        
        # Separate normal and anomalous data
        self.normal_data = self.features_df[~self.features_df['is_anomalous']]
        self.anomalous_data = self.features_df[self.features_df['is_anomalous']]
        
        print(f"Loaded {len(self.normal_data)} normal and {len(self.anomalous_data)} anomalous trajectories")
    
    def _load_pickle(self, filepath: Path):
        """Load pickle file safely"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return []
    
    def create_comprehensive_analysis(self):
        """Create all synthetic data analysis visualizations"""
        print("Creating comprehensive synthetic data analysis...")
        
        # Create subplots for different analysis categories
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Comprehensive Synthetic Data Analysis: Normal vs Anomalous Trajectories', 
                     fontsize=16, fontweight='bold')
        
        # 1. Agent Type Distribution
        self._plot_agent_type_distribution(axes[0, 0])
        
        # 2. Tool Usage Patterns
        self._plot_tool_usage_patterns(axes[0, 1])
        
        # 3. Node Type Distribution
        self._plot_node_type_distribution(axes[0, 2])
        
        # 4. Trajectory Length and Duration
        self._plot_trajectory_metrics(axes[1, 0])
        
        # 5. Handoff Patterns
        self._plot_handoff_patterns(axes[1, 1])
        
        # 6. LLM Call Patterns
        self._plot_llm_call_patterns(axes[1, 2])
        
        # 7. Planning and User Enquiry Patterns
        self._plot_planning_patterns(axes[2, 0])
        
        # 8. Error Patterns
        self._plot_error_patterns(axes[2, 1])
        
        # 9. Success Rate and Completion
        self._plot_success_metrics(axes[2, 2])
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "comprehensive_synthetic_data_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed individual plots
        self._create_detailed_agent_analysis()
        self._create_detailed_tool_analysis()
        self._create_detailed_workflow_analysis()
        self._create_statistical_summary()
        
        print("Comprehensive synthetic data analysis completed!")
    
    def _plot_agent_type_distribution(self, ax):
        """Plot agent type distribution comparison"""
        agent_cols = [col for col in self.features_df.columns if 'agent_type_' in col and 'count' in col]
        
        normal_agent_data = []
        anomalous_agent_data = []
        agent_names = []
        
        for col in agent_cols:
            agent_name = col.replace('agent_type_', '').replace('_count', '')
            agent_names.append(agent_name)
            
            normal_agent_data.append(self.normal_data[col].mean())
            anomalous_agent_data.append(self.anomalous_data[col].mean())
        
        x = np.arange(len(agent_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_agent_data, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_agent_data, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Agent Types')
        ax.set_ylabel('Average Count per Trajectory')
        ax.set_title('Agent Type Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_tool_usage_patterns(self, ax):
        """Plot tool usage patterns comparison"""
        tool_cols = [col for col in self.features_df.columns if 'tool_type_' in col and 'count' in col]
        
        normal_tool_data = []
        anomalous_tool_data = []
        tool_names = []
        
        for col in tool_cols:
            tool_name = col.replace('tool_type_', '').replace('_count', '')
            tool_names.append(tool_name)
            
            normal_tool_data.append(self.normal_data[col].mean())
            anomalous_tool_data.append(self.anomalous_data[col].mean())
        
        # Create horizontal bar chart
        y = np.arange(len(tool_names))
        width = 0.35
        
        bars1 = ax.barh(y - width/2, normal_tool_data, width, label='Normal', alpha=0.8)
        bars2 = ax.barh(y + width/2, anomalous_tool_data, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Average Count per Trajectory')
        ax.set_ylabel('Tool Types')
        ax.set_title('Tool Usage Patterns')
        ax.set_yticks(y)
        ax.set_yticklabels(tool_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            width_val = bar.get_width()
            ax.text(width_val + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{width_val:.2f}', ha='left', va='center', fontsize=8)
        
        for bar in bars2:
            width_val = bar.get_width()
            ax.text(width_val + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{width_val:.2f}', ha='left', va='center', fontsize=8)
    
    def _plot_node_type_distribution(self, ax):
        """Plot node type distribution comparison"""
        node_cols = [col for col in self.features_df.columns if 'node_type_' in col and 'count' in col]
        
        normal_node_data = []
        anomalous_node_data = []
        node_names = []
        
        for col in node_cols:
            node_name = col.replace('node_type_', '').replace('_count', '')
            node_names.append(node_name)
            
            normal_node_data.append(self.normal_data[col].mean())
            anomalous_node_data.append(self.anomalous_data[col].mean())
        
        # Create stacked bar chart
        x = np.arange(len(node_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_node_data, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_node_data, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Node Types')
        ax.set_ylabel('Average Count per Trajectory')
        ax.set_title('Node Type Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(node_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_trajectory_metrics(self, ax):
        """Plot trajectory length and duration metrics"""
        metrics = ['num_nodes', 'total_duration']
        metric_names = ['Number of Nodes', 'Total Duration (s)']
        
        normal_metrics = []
        anomalous_metrics = []
        
        for metric in metrics:
            normal_metrics.append(self.normal_data[metric].mean())
            anomalous_metrics.append(self.anomalous_data[metric].mean())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_metrics, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_metrics, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Average Value')
        ax.set_title('Trajectory Size and Duration')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_handoff_patterns(self, ax):
        """Plot handoff patterns"""
        handoff_metrics = ['handoff_count', 'handoff_frequency']
        metric_names = ['Handoff Count', 'Handoff Frequency']
        
        normal_handoffs = []
        anomalous_handoffs = []
        
        for metric in handoff_metrics:
            normal_handoffs.append(self.normal_data[metric].mean())
            anomalous_handoffs.append(self.anomalous_data[metric].mean())
        
        x = np.arange(len(handoff_metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_handoffs, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_handoffs, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Handoff Metrics')
        ax.set_ylabel('Average Value')
        ax.set_title('Agent Handoff Patterns')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_llm_call_patterns(self, ax):
        """Plot LLM call patterns"""
        llm_metrics = ['node_type_llm_call_count', 'node_type_llm_call_ratio']
        metric_names = ['LLM Call Count', 'LLM Call Ratio']
        
        normal_llm = []
        anomalous_llm = []
        
        for metric in llm_metrics:
            normal_llm.append(self.normal_data[metric].mean())
            anomalous_llm.append(self.anomalous_data[metric].mean())
        
        x = np.arange(len(llm_metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_llm, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_llm, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('LLM Call Metrics')
        ax.set_ylabel('Average Value')
        ax.set_title('LLM Call Patterns')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_planning_patterns(self, ax):
        """Plot planning and user enquiry patterns"""
        planning_metrics = ['node_type_planning_count', 'node_type_user_enquiry_count']
        metric_names = ['Planning Count', 'User Enquiry Count']
        
        normal_planning = []
        anomalous_planning = []
        
        for metric in planning_metrics:
            normal_planning.append(self.normal_data[metric].mean())
            anomalous_planning.append(self.anomalous_data[metric].mean())
        
        x = np.arange(len(planning_metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_planning, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_planning, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Planning Metrics')
        ax.set_ylabel('Average Count per Trajectory')
        ax.set_title('Planning and User Enquiry Patterns')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_error_patterns(self, ax):
        """Plot error patterns"""
        error_metrics = ['error_count', 'error_frequency']
        metric_names = ['Error Count', 'Error Frequency']
        
        normal_errors = []
        anomalous_errors = []
        
        for metric in error_metrics:
            normal_errors.append(self.normal_data[metric].mean())
            anomalous_errors.append(self.anomalous_data[metric].mean())
        
        x = np.arange(len(error_metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_errors, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_errors, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Error Metrics')
        ax.set_ylabel('Average Value')
        ax.set_title('Error Patterns')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_success_metrics(self, ax):
        """Plot success rate and completion metrics"""
        success_metrics = ['success', 'completion_rate']
        metric_names = ['Success Rate', 'Completion Rate']
        
        normal_success = []
        anomalous_success = []
        
        for metric in success_metrics:
            normal_success.append(self.normal_data[metric].mean())
            anomalous_success.append(self.anomalous_data[metric].mean())
        
        x = np.arange(len(success_metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_success, width, label='Normal', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomalous_success, width, label='Anomalous', alpha=0.8)
        
        ax.set_xlabel('Success Metrics')
        ax.set_ylabel('Average Rate')
        ax.set_title('Success and Completion Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _create_detailed_agent_analysis(self):
        """Create detailed agent analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Agent Analysis', fontsize=16, fontweight='bold')
        
        # Agent type ratios
        agent_ratio_cols = [col for col in self.features_df.columns if 'agent_type_' in col and 'ratio' in col]
        agent_names = [col.replace('agent_type_', '').replace('_ratio', '') for col in agent_ratio_cols]
        
        normal_ratios = [self.normal_data[col].mean() for col in agent_ratio_cols]
        anomalous_ratios = [self.anomalous_data[col].mean() for col in agent_ratio_cols]
        
        # Pie chart for normal trajectories
        axes[0, 0].pie(normal_ratios, labels=agent_names, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Normal Trajectories: Agent Type Distribution')
        
        # Pie chart for anomalous trajectories
        axes[0, 1].pie(anomalous_ratios, labels=agent_names, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Anomalous Trajectories: Agent Type Distribution')
        
        # Agent type counts comparison
        agent_count_cols = [col for col in self.features_df.columns if 'agent_type_' in col and 'count' in col]
        agent_names = [col.replace('agent_type_', '').replace('_count', '') for col in agent_count_cols]
        
        normal_counts = [self.normal_data[col].mean() for col in agent_count_cols]
        anomalous_counts = [self.anomalous_data[col].mean() for col in agent_count_cols]
        
        x = np.arange(len(agent_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, normal_counts, width, label='Normal', alpha=0.8)
        axes[1, 0].bar(x + width/2, anomalous_counts, width, label='Anomalous', alpha=0.8)
        axes[1, 0].set_xlabel('Agent Types')
        axes[1, 0].set_ylabel('Average Count')
        axes[1, 0].set_title('Agent Type Counts Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(agent_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Agent type correlation heatmap
        agent_data = self.features_df[agent_count_cols + ['is_anomalous']]
        correlation_matrix = agent_data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
        axes[1, 1].set_title('Agent Type Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "detailed_agent_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_tool_analysis(self):
        """Create detailed tool analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Tool Usage Analysis', fontsize=16, fontweight='bold')
        
        # Tool type ratios
        tool_ratio_cols = [col for col in self.features_df.columns if 'tool_type_' in col and 'ratio' in col]
        tool_names = [col.replace('tool_type_', '').replace('_ratio', '') for col in tool_ratio_cols]
        
        normal_ratios = [self.normal_data[col].mean() for col in tool_ratio_cols]
        anomalous_ratios = [self.anomalous_data[col].mean() for col in tool_ratio_cols]
        
        # Horizontal bar chart for tool ratios
        y = np.arange(len(tool_names))
        width = 0.35
        
        axes[0, 0].barh(y - width/2, normal_ratios, width, label='Normal', alpha=0.8)
        axes[0, 0].barh(y + width/2, anomalous_ratios, width, label='Anomalous', alpha=0.8)
        axes[0, 0].set_xlabel('Average Ratio')
        axes[0, 0].set_ylabel('Tool Types')
        axes[0, 0].set_title('Tool Usage Ratios')
        axes[0, 0].set_yticks(y)
        axes[0, 0].set_yticklabels(tool_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Tool type counts
        tool_count_cols = [col for col in self.features_df.columns if 'tool_type_' in col and 'count' in col]
        tool_names = [col.replace('tool_type_', '').replace('_count', '') for col in tool_count_cols]
        
        normal_counts = [self.normal_data[col].mean() for col in tool_count_cols]
        anomalous_counts = [self.anomalous_data[col].mean() for col in tool_count_cols]
        
        x = np.arange(len(tool_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, normal_counts, width, label='Normal', alpha=0.8)
        axes[0, 1].bar(x + width/2, anomalous_counts, width, label='Anomalous', alpha=0.8)
        axes[0, 1].set_xlabel('Tool Types')
        axes[0, 1].set_ylabel('Average Count')
        axes[0, 1].set_title('Tool Usage Counts')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(tool_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tool correlation heatmap
        tool_data = self.features_df[tool_count_cols + ['is_anomalous']]
        correlation_matrix = tool_data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
        axes[1, 0].set_title('Tool Usage Correlation Matrix')
        
        # Tool usage patterns over trajectory length
        axes[1, 1].scatter(self.normal_data['num_nodes'], self.normal_data['total_tool_calls'], 
                          alpha=0.6, label='Normal', s=50)
        axes[1, 1].scatter(self.anomalous_data['num_nodes'], self.anomalous_data['total_tool_calls'], 
                          alpha=0.6, label='Anomalous', s=50)
        axes[1, 1].set_xlabel('Number of Nodes')
        axes[1, 1].set_ylabel('Total Tool Calls')
        axes[1, 1].set_title('Tool Usage vs Trajectory Length')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "detailed_tool_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detailed_workflow_analysis(self):
        """Create detailed workflow analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Workflow Analysis', fontsize=16, fontweight='bold')
        
        # Workflow metrics comparison
        workflow_metrics = ['handoff_count', 'node_type_llm_call_count', 
                           'node_type_planning_count', 'node_type_user_enquiry_count']
        metric_names = ['Handoffs', 'LLM Calls', 'Planning', 'User Enquiries']
        
        normal_workflow = [self.normal_data[metric].mean() for metric in workflow_metrics]
        anomalous_workflow = [self.anomalous_data[metric].mean() for metric in workflow_metrics]
        
        x = np.arange(len(workflow_metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, normal_workflow, width, label='Normal', alpha=0.8)
        axes[0, 0].bar(x + width/2, anomalous_workflow, width, label='Anomalous', alpha=0.8)
        axes[0, 0].set_xlabel('Workflow Components')
        axes[0, 0].set_ylabel('Average Count')
        axes[0, 0].set_title('Workflow Component Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metric_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Duration vs complexity
        axes[0, 1].scatter(self.normal_data['num_nodes'], self.normal_data['total_duration'], 
                          alpha=0.6, label='Normal', s=50)
        axes[0, 1].scatter(self.anomalous_data['num_nodes'], self.anomalous_data['total_duration'], 
                          alpha=0.6, label='Anomalous', s=50)
        axes[0, 1].set_xlabel('Number of Nodes')
        axes[0, 1].set_ylabel('Total Duration (s)')
        axes[0, 1].set_title('Duration vs Complexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate analysis
        success_metrics = ['success', 'completion_rate']
        metric_names = ['Success Rate', 'Completion Rate']
        
        normal_success = [self.normal_data[metric].mean() for metric in success_metrics]
        anomalous_success = [self.anomalous_data[metric].mean() for metric in success_metrics]
        
        x = np.arange(len(success_metrics))
        
        axes[1, 0].bar(x - width/2, normal_success, width, label='Normal', alpha=0.8)
        axes[1, 0].bar(x + width/2, anomalous_success, width, label='Anomalous', alpha=0.8)
        axes[1, 0].set_xlabel('Success Metrics')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].set_title('Success Rate Analysis')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metric_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error analysis
        error_metrics = ['error_count', 'error_frequency']
        metric_names = ['Error Count', 'Error Frequency']
        
        normal_errors = [self.normal_data[metric].mean() for metric in error_metrics]
        anomalous_errors = [self.anomalous_data[metric].mean() for metric in error_metrics]
        
        x = np.arange(len(error_metrics))
        
        axes[1, 1].bar(x - width/2, normal_errors, width, label='Normal', alpha=0.8)
        axes[1, 1].bar(x + width/2, anomalous_errors, width, label='Anomalous', alpha=0.8)
        axes[1, 1].set_xlabel('Error Metrics')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].set_title('Error Analysis')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metric_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "detailed_workflow_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_summary(self):
        """Create statistical summary report"""
        summary_data = []
        
        # Key metrics to summarize
        key_metrics = [
            'num_nodes', 'total_duration', 'handoff_count', 'total_tool_calls',
            'node_type_llm_call_count', 'node_type_planning_count', 
            'node_type_user_enquiry_count', 'error_count', 'success', 'completion_rate'
        ]
        
        metric_names = [
            'Number of Nodes', 'Total Duration (s)', 'Handoff Count', 'Total Tool Calls',
            'LLM Call Count', 'Planning Count', 'User Enquiry Count', 
            'Error Count', 'Success Rate', 'Completion Rate'
        ]
        
        for metric, name in zip(key_metrics, metric_names):
            normal_stats = self.normal_data[metric].describe()
            anomalous_stats = self.anomalous_data[metric].describe()
            
            summary_data.append({
                'Metric': name,
                'Normal_Mean': normal_stats['mean'],
                'Normal_Std': normal_stats['std'],
                'Normal_Min': normal_stats['min'],
                'Normal_Max': normal_stats['max'],
                'Anomalous_Mean': anomalous_stats['mean'],
                'Anomalous_Std': anomalous_stats['std'],
                'Anomalous_Min': anomalous_stats['min'],
                'Anomalous_Max': anomalous_stats['max'],
                'Difference': anomalous_stats['mean'] - normal_stats['mean'],
                'Difference_Percent': ((anomalous_stats['mean'] - normal_stats['mean']) / normal_stats['mean']) * 100
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        summary_df.to_csv(self.charts_dir / "synthetic_data_statistical_summary.csv", index=False)
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Statistical Summary: Normal vs Anomalous Trajectories', fontsize=16, fontweight='bold')
        
        # Mean comparison
        x = np.arange(len(metric_names))
        width = 0.35
        
        normal_means = summary_df['Normal_Mean'].values
        anomalous_means = summary_df['Anomalous_Mean'].values
        
        bars1 = axes[0].bar(x - width/2, normal_means, width, label='Normal', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, anomalous_means, width, label='Anomalous', alpha=0.8)
        
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Mean Value')
        axes[0].set_title('Mean Values Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Percentage difference
        percent_diff = summary_df['Difference_Percent'].values
        
        colors = ['red' if x > 0 else 'blue' for x in percent_diff]
        bars3 = axes[1].bar(x, percent_diff, color=colors, alpha=0.8)
        
        axes[1].set_xlabel('Metrics')
        axes[1].set_ylabel('Percentage Difference (%)')
        axes[1].set_title('Percentage Difference (Anomalous - Normal)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                        f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / "statistical_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical summary saved to {self.charts_dir}")
        return summary_df

def main():
    """Main function to run the synthetic data analysis"""
    analyzer = SyntheticDataAnalyzer()
    analyzer.create_comprehensive_analysis()
    
    print("\nSynthetic data analysis completed!")
    print("Generated visualizations:")
    print("- comprehensive_synthetic_data_analysis.png")
    print("- detailed_agent_analysis.png")
    print("- detailed_tool_analysis.png")
    print("- detailed_workflow_analysis.png")
    print("- statistical_summary.png")
    print("- synthetic_data_statistical_summary.csv")

if __name__ == "__main__":
    main() 