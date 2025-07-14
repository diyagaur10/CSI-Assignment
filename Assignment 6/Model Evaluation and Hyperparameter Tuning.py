Objective : Model Evaluation and Hyperparameter Tuning
Train multiple machine learning models and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score. Implement hyperparameter tuning techniques like GridSearchCV and RandomizedSearchCV to optimize model parameters. Analyze the results to select the best-performing model.
Resources :
https://www.kdnuggets.com/hyperparameter-tuning-gridsearchcv-and-randomizedsearchcv-explained

code :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluationTuning:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initialize the model evaluation and tuning class
        
        Parameters:
        X: Features
        y: Target variable
        test_size: Test set size
        random_state: Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=random_state),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'SVM': SVC(random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        # Store results
        self.results = {}
        self.best_models = {}
        
    def evaluate_models(self):
        """
        Evaluate all models with default parameters
        """
        print("=" * 60)
        print("MODEL EVALUATION WITH DEFAULT PARAMETERS")
        print("=" * 60)
        
        evaluation_results = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Fit the model
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Store results
            result = {
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            evaluation_results.append(result)
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        
        # Convert to DataFrame for better visualization
        self.results['default'] = pd.DataFrame(evaluation_results)
        
        # Display summary table
        print("\n" + "="*60)
        print("SUMMARY OF MODEL PERFORMANCE (DEFAULT PARAMETERS)")
        print("="*60)
        print(self.results['default'].round(4))
        
        return self.results['default']
    
    def hyperparameter_tuning_grid_search(self):
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
        print("="*60)
        
        # Define parameter grids for each model
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6]
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        grid_search_results = []
        
        for name, model in self.models.items():
            if name in param_grids and name != 'Naive Bayes':  # Skip Naive Bayes for grid search
                print(f"\nTuning {name}...")
                
                # Choose scaled or unscaled data
                if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Perform grid search
                grid_search = GridSearchCV(
                    model, 
                    param_grids[name], 
                    cv=5, 
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_use, self.y_train)
                
                # Get best model and predictions
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_use)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Store results
                result = {
                    'Model': name,
                    'Best Parameters': grid_search.best_params_,
                    'Best CV Score': grid_search.best_score_,
                    'Test Accuracy': accuracy,
                    'Test Precision': precision,
                    'Test Recall': recall,
                    'Test F1-Score': f1
                }
                grid_search_results.append(result)
                
                # Store best model
                self.best_models[f'{name}_grid'] = best_model
                
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV score: {grid_search.best_score_:.4f}")
                print(f"Test accuracy: {accuracy:.4f}")
                print(f"Test F1-score: {f1:.4f}")
        
        self.results['grid_search'] = grid_search_results
        return grid_search_results
    
    def hyperparameter_tuning_random_search(self):
        """
        Perform hyperparameter tuning using RandomizedSearchCV
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV")
        print("="*60)
        
        # Define parameter distributions for random search
        param_distributions = {
            'Logistic Regression': {
                'C': np.logspace(-3, 2, 100),
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            },
            'Random Forest': {
                'n_estimators': np.arange(50, 301, 10),
                'max_depth': [None] + list(np.arange(10, 51, 5)),
                'min_samples_split': np.arange(2, 21),
                'min_samples_leaf': np.arange(1, 11),
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'SVM': {
                'C': np.logspace(-1, 2, 50),
                'kernel': ['rbf', 'linear'],
                'gamma': np.logspace(-4, 0, 50)
            },
            'Gradient Boosting': {
                'n_estimators': np.arange(50, 301, 10),
                'learning_rate': np.logspace(-3, 0, 50),
                'max_depth': np.arange(3, 11),
                'subsample': np.arange(0.8, 1.01, 0.05)
            }
        }
        
        random_search_results = []
        
        for name, model in self.models.items():
            if name in param_distributions:
                print(f"\nTuning {name} with Random Search...")
                
                # Choose scaled or unscaled data
                if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                    X_train_use = self.X_train_scaled
                    X_test_use = self.X_test_scaled
                else:
                    X_train_use = self.X_train
                    X_test_use = self.X_test
                
                # Perform random search
                random_search = RandomizedSearchCV(
                    model, 
                    param_distributions[name], 
                    n_iter=50,  # Number of parameter settings sampled
                    cv=5, 
                    scoring='f1_weighted',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0
                )
                
                random_search.fit(X_train_use, self.y_train)
                
                # Get best model and predictions
                best_model = random_search.best_estimator_
                y_pred = best_model.predict(X_test_use)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Store results
                result = {
                    'Model': name,
                    'Best Parameters': random_search.best_params_,
                    'Best CV Score': random_search.best_score_,
                    'Test Accuracy': accuracy,
                    'Test Precision': precision,
                    'Test Recall': recall,
                    'Test F1-Score': f1
                }
                random_search_results.append(result)
                
                # Store best model
                self.best_models[f'{name}_random'] = best_model
                
                print(f"Best parameters: {random_search.best_params_}")
                print(f"Best CV score: {random_search.best_score_:.4f}")
                print(f"Test accuracy: {accuracy:.4f}")
                print(f"Test F1-score: {f1:.4f}")
        
        self.results['random_search'] = random_search_results
        return random_search_results
    
    def compare_results(self):
        """
        Compare results from different approaches
        """
        print("\n" + "="*60)
        print("COMPARISON OF RESULTS")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Add default results
        for _, row in self.results['default'].iterrows():
            comparison_data.append({
                'Model': row['Model'],
                'Method': 'Default',
                'Accuracy': row['Accuracy'],
                'Precision': row['Precision'],
                'Recall': row['Recall'],
                'F1-Score': row['F1-Score']
            })
        
        # Add grid search results
        if 'grid_search' in self.results:
            for result in self.results['grid_search']:
                comparison_data.append({
                    'Model': result['Model'],
                    'Method': 'Grid Search',
                    'Accuracy': result['Test Accuracy'],
                    'Precision': result['Test Precision'],
                    'Recall': result['Test Recall'],
                    'F1-Score': result['Test F1-Score']
                })
        
        # Add random search results
        if 'random_search' in self.results:
            for result in self.results['random_search']:
                comparison_data.append({
                    'Model': result['Model'],
                    'Method': 'Random Search',
                    'Accuracy': result['Test Accuracy'],
                    'Precision': result['Test Precision'],
                    'Recall': result['Test Recall'],
                    'F1-Score': result['Test F1-Score']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\nDetailed Comparison:")
        print(comparison_df.round(4))
        
        # Find best performing model overall
        best_model_idx = comparison_df['F1-Score'].idxmax()
        best_model_info = comparison_df.loc[best_model_idx]
        
        print(f"\nBEST PERFORMING MODEL:")
        print(f"Model: {best_model_info['Model']}")
        print(f"Method: {best_model_info['Method']}")
        print(f"F1-Score: {best_model_info['F1-Score']:.4f}")
        print(f"Accuracy: {best_model_info['Accuracy']:.4f}")
        
        return comparison_df
    
    def plot_results(self):
        """
        Create visualizations of the results
        """
        if not hasattr(self, 'results') or 'default' not in self.results:
            print("No results to plot. Run evaluate_models() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Model performance comparison (default parameters)
        default_results = self.results['default']
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(default_results))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i * width, default_results[metric], width, label=metric)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison (Default Parameters)')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(default_results['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1-Score comparison across methods
        ax2 = axes[0, 1]
        comparison_data = []
        
        # Collect F1-scores for comparison
        for _, row in self.results['default'].iterrows():
            comparison_data.append({
                'Model': row['Model'],
                'Default': row['F1-Score'],
                'Grid Search': None,
                'Random Search': None
            })
        
        # Add grid search results
        if 'grid_search' in self.results:
            for result in self.results['grid_search']:
                for item in comparison_data:
                    if item['Model'] == result['Model']:
                        item['Grid Search'] = result['Test F1-Score']
        
        # Add random search results
        if 'random_search' in self.results:
            for result in self.results['random_search']:
                for item in comparison_data:
                    if item['Model'] == result['Model']:
                        item['Random Search'] = result['Test F1-Score']
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('Model').plot(kind='bar', ax=ax2)
        ax2.set_title('F1-Score Comparison Across Methods')
        ax2.set_ylabel('F1-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Best model confusion matrix (if available)
        ax3 = axes[1, 0]
        if self.best_models:
            # Get the best model
            best_model_name = max(self.best_models.keys(), 
                                key=lambda x: self.get_model_f1_score(x))
            best_model = self.best_models[best_model_name]
            
            # Make predictions
            if 'SVM' in best_model_name or 'Logistic' in best_model_name or 'Neighbors' in best_model_name:
                y_pred = best_model.predict(self.X_test_scaled)
            else:
                y_pred = best_model.predict(self.X_test)
            
            # Create confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_title(f'Confusion Matrix - {best_model_name}')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
        else:
            ax3.text(0.5, 0.5, 'No tuned models available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Confusion Matrix')
        
        # Plot 4: Feature importance (if available)
        ax4 = axes[1, 1]
        if self.best_models:
            # Try to get feature importance from the best model
            try:
                best_model_name = max(self.best_models.keys(), 
                                    key=lambda x: self.get_model_f1_score(x))
                best_model = self.best_models[best_model_name]
                
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feature_names = [f'Feature {i+1}' for i in range(len(importances))]
                    
                    # Sort by importance
                    sorted_idx = np.argsort(importances)[::-1]
                    top_10_idx = sorted_idx[:10]  # Top 10 features
                    
                    ax4.bar(range(len(top_10_idx)), importances[top_10_idx])
                    ax4.set_title(f'Top 10 Feature Importances - {best_model_name}')
                    ax4.set_xlabel('Features')
                    ax4.set_ylabel('Importance')
                    ax4.set_xticks(range(len(top_10_idx)))
                    ax4.set_xticklabels([feature_names[i] for i in top_10_idx], rotation=45)
                else:
                    ax4.text(0.5, 0.5, 'Feature importance not available for this model', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Feature Importance')
            except:
                ax4.text(0.5, 0.5, 'Error getting feature importance', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Feature Importance')
        else:
            ax4.text(0.5, 0.5, 'No models available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    
    def get_model_f1_score(self, model_name):
        """
        Helper method to get F1-score for a model
        """
        if 'grid' in model_name:
            for result in self.results.get('grid_search', []):
                if result['Model'] in model_name:
                    return result['Test F1-Score']
        elif 'random' in model_name:
            for result in self.results.get('random_search', []):
                if result['Model'] in model_name:
                    return result['Test F1-Score']
        return 0
    
    def get_detailed_report(self, model_name=None):
        """
        Get detailed classification report for a specific model
        """
        if not self.best_models:
            print("No tuned models available. Run hyperparameter tuning first.")
            return
        
        if model_name is None:
            model_name = max(self.best_models.keys(), 
                           key=lambda x: self.get_model_f1_score(x))
        
        if model_name not in self.best_models:
            print(f"Model {model_name} not found in best models.")
            return
        
        best_model = self.best_models[model_name]
        
        # Make predictions
        if 'SVM' in model_name or 'Logistic' in model_name or 'Neighbors' in model_name:
            y_pred = best_model.predict(self.X_test_scaled)
        else:
            y_pred = best_model.predict(self.X_test)
        
        print(f"\nDETAILED CLASSIFICATION REPORT - {model_name}")
        print("=" * 60)
        print(classification_report(self.y_test, y_pred))
        
        return classification_report(self.y_test, y_pred, output_dict=True)


def main():
    """
    Main function to demonstrate the model evaluation and tuning process
    """
    print("MACHINE LEARNING MODEL EVALUATION AND HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Load sample dataset (you can replace this with your own dataset)
    print("Loading dataset...")
    
    # Option 1: Use breast cancer dataset (built-in)
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Option 2: Create synthetic dataset (uncomment to use)
    # X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
    #                           n_redundant=10, n_clusters_per_class=1, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Initialize the evaluation system
    evaluator = ModelEvaluationTuning(X, y)
    
    # Step 1: Evaluate models with default parameters
    default_results = evaluator.evaluate_models()
    
    # Step 2: Perform hyperparameter tuning with GridSearchCV
    grid_results = evaluator.hyperparameter_tuning_grid_search()
    
    # Step 3: Perform hyperparameter tuning with RandomizedSearchCV
    random_results = evaluator.hyperparameter_tuning_random_search()
    
    # Step 4: Compare all results
    comparison = evaluator.compare_results()
    
    # Step 5: Get detailed report for best model
    evaluator.get_detailed_report()
    
    # Step 6: Create visualizations
    evaluator.plot_results()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nKey Insights:")
    print("1. Compare default vs. tuned model performance")
    print("2. GridSearchCV vs. RandomizedSearchCV trade-offs")
    print("3. Best performing model and its parameters")
    print("4. Feature importance (if available)")
    print("5. Detailed classification metrics")


if __name__ == "__main__":
    main()
