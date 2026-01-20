"""
Utility functions for model and dataset comparison notebooks.

This module provides reusable functions for:
- Fine-tuning models
- Computing metrics
- Comparing model performances
- Visualizing results
- Meta-features analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding


def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dictionary with computed metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    # For multi-class, use weighted average
    average = 'binary' if len(np.unique(labels)) == 2 else 'weighted'
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def fine_tune_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    num_epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    seed: int = 42
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Fine-tune a model and return trainer with results.
    
    Args:
        model: Model to fine-tune
        tokenizer: Associated tokenizer
        train_dataset: Training dataset (tokenized)
        val_dataset: Validation dataset (tokenized)
        output_dir: Directory to save checkpoints
        model_name: Name for logging
        dataset_name: Dataset name for logging
        num_epochs: Number of training epochs
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        learning_rate: Learning rate
        seed: Random seed
    
    Returns:
        Tuple of (trainer, results_dict)
    """
    print(f"\n{'='*80}")
    print(f"🏋️ Fine-tuning {model_name} on {dataset_name}")
    print(f"{'='*80}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",
        save_total_limit=1,
        seed=seed,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    train_result = trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    print(f"\n✅ Training completed!")
    print(f"   Training loss: {train_result.training_loss:.4f}")
    print(f"   Validation accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   Validation F1: {eval_results['eval_f1']:.4f}")
    
    return trainer, {
        'train_loss': train_result.training_loss,
        'eval_accuracy': eval_results['eval_accuracy'],
        'eval_precision': eval_results['eval_precision'],
        'eval_recall': eval_results['eval_recall'],
        'eval_f1': eval_results['eval_f1'],
        'eval_loss': eval_results['eval_loss'],
    }


def compare_models_performance(
    results_dict: Dict[str, Dict[str, float]],
    dataset_name: str,
    save_path: Optional[str] = None
):
    """
    Create a bar chart comparing model performances.
    
    Args:
        results_dict: Dict of {model_name: results_dict}
        dataset_name: Name of dataset for title
        save_path: Optional path to save figure
    """
    metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    models = list(results_dict.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, model_name in enumerate(models):
        values = [results_dict[model_name][m] for m in metrics]
        offset = width * (i - len(models)/2 + 0.5)
        ax.bar(x + offset, values, width, label=model_name, 
               color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for j, v in enumerate(values):
            ax.text(x[j] + offset, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Comparison on {dataset_name}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    trainers_dict: Dict[str, Trainer],
    dataset_name: str,
    save_path: Optional[str] = None
):
    """
    Plot training loss curves for multiple models.
    
    Args:
        trainers_dict: Dict of {model_name: trainer}
        dataset_name: Name of dataset for title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (model_name, trainer) in enumerate(trainers_dict.items()):
        log_history = trainer.state.log_history
        
        # Extract training loss
        train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
        eval_logs = [log for log in log_history if 'eval_loss' in log]
        
        color = colors[idx % len(colors)]
        
        if train_logs:
            train_steps = [log['step'] for log in train_logs if 'loss' in log]
            train_loss = [log['loss'] for log in train_logs if 'loss' in log]
            axes[0].plot(train_steps, train_loss, label=model_name, 
                        marker='o', color=color, linewidth=2)
        
        if eval_logs:
            eval_steps = [log['step'] for log in eval_logs]
            eval_loss = [log['eval_loss'] for log in eval_logs]
            axes[1].plot(eval_steps, eval_loss, label=model_name,
                        marker='s', color=color, linewidth=2)
    
    axes[0].set_xlabel('Steps', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Steps', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1].set_title('Validation Loss', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    fig.suptitle(f'Training History - {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_metafeatures(
    meta_dfs_dict: Dict[str, pd.DataFrame],
    dataset_name: str,
    top_n: int = 15,
    save_path: Optional[str] = None
) -> list:
    """
    Compare meta-features across models using box plots.
    
    Args:
        meta_dfs_dict: Dict of {model_name: meta_features_df}
        dataset_name: Name of dataset for title
        top_n: Number of top features to show
        save_path: Optional path to save figure
    
    Returns:
        List of top feature names
    """
    # Combine all meta-features
    for model_name, df in meta_dfs_dict.items():
        df['model'] = model_name
    
    combined_df = pd.concat(meta_dfs_dict.values(), ignore_index=True)
    
    # Select top N features by variance
    feature_variance = combined_df.groupby('feature')['value'].var().sort_values(ascending=False)
    top_features = feature_variance.head(top_n).index.tolist()
    
    df_filtered = combined_df[combined_df['feature'].isin(top_features)]
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for seaborn
    plot_data = df_filtered[['feature', 'value', 'model']]
    
    sns.boxplot(data=plot_data, x='feature', y='value', hue='model', ax=ax)
    
    ax.set_xlabel('Meta-feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Meta-features Comparison - {dataset_name}', 
                fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.legend(title='Model', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return top_features


def visualize_feature_space(
    features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    dataset_name: str,
    method: str = 'pca',
    save_path: Optional[str] = None
):
    """
    Visualize feature space using PCA or t-SNE.
    
    Args:
        features_dict: Dict of {model_name: features_array}
        labels_dict: Dict of {model_name: labels_array}
        dataset_name: Name of dataset for title
        method: 'pca' or 'tsne'
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, len(features_dict), figsize=(7*len(features_dict), 6))
    if len(features_dict) == 1:
        axes = [axes]
    
    for idx, (model_name, features) in enumerate(features_dict.items()):
        labels = labels_dict[model_name]
        
        # Ensure 2D
        if features.ndim == 3:
            features = features.reshape(features.shape[0], -1)
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(features)
            var_exp = reducer.explained_variance_ratio_
            subtitle = f'PCA (var: {var_exp[0]:.2%}, {var_exp[1]:.2%})'
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            reduced = reducer.fit_transform(features)
            subtitle = 't-SNE'
        
        # Plot
        scatter = axes[idx].scatter(reduced[:, 0], reduced[:, 1], 
                                   c=labels, cmap='viridis', 
                                   alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[idx].set_xlabel('Component 1', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Component 2', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{model_name}\n{subtitle}', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx], label='Class')
    
    fig.suptitle(f'Feature Space Visualization - {dataset_name}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_summary_table(all_results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Create a summary table of all results.
    
    Args:
        all_results: Nested dict of {dataset: {model: results}}
    
    Returns:
        DataFrame with summary of all results
    """
    rows = []
    for dataset_name, models_results in all_results.items():
        for model_name, results in models_results.items():
            row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': results['eval_accuracy'],
                'Precision': results['eval_precision'],
                'Recall': results['eval_recall'],
                'F1': results['eval_f1'],
                'Loss': results['eval_loss']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_summary_heatmap(summary_df: pd.DataFrame, metric: str = 'Accuracy', save_path: Optional[str] = None):
    """
    Plot heatmap of results across datasets and models.
    
    Args:
        summary_df: Summary DataFrame from create_summary_table
        metric: Metric to visualize
        save_path: Optional path to save figure
    """
    pivot_df = summary_df.pivot(index='Dataset', columns='Model', values=metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': metric}, ax=ax, linewidths=0.5)
    
    ax.set_title(f'{metric} Comparison Across Datasets and Models', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
