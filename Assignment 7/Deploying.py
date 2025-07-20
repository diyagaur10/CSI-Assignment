Objective: Develop a web application using Streamlit to deploy a trained machine learning model. The app should allow users to input data, receive predictions, and understand model outputs through visualizations. This task will help you learn how to make your models accessible and interactive.
Resources :
https://docs.streamlit.io/https://machinelearningmastery.com/how-to-quickly-deploy-machine-learning-models-streamlit/

code:
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, r2_score
import pickle
import io

st.set_page_config(page_title="ML Model Deployment Hub", page_icon="@", layout="wide")

@st.cache_data
def load_sample_data(dataset_name):
    if dataset_name == "Iris Classification":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names
    elif dataset_name == "Custom Classification":
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                                 n_informative=5, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df, ['Class 0', 'Class 1', 'Class 2']
    else:  # Boston Housing
        np.random.seed(42)
        n_samples = 506
        data = {
            'CRIM': np.random.exponential(3, n_samples),
            'ZN': np.random.uniform(0, 100, n_samples),
            'INDUS': np.random.uniform(0, 30, n_samples),
            'NOX': np.random.uniform(0.3, 0.9, n_samples),
            'RM': np.random.normal(6, 1, n_samples),
            'AGE': np.random.uniform(0, 100, n_samples),
            'DIS': np.random.uniform(1, 15, n_samples),
            'TAX': np.random.uniform(150, 800, n_samples),
            'PTRATIO': np.random.uniform(12, 22, n_samples),
            'LSTAT': np.random.uniform(1, 40, n_samples)
        }
        df = pd.DataFrame(data)
        df['target'] = (df['RM'] * 5 + df['DIS'] * 2 - df['LSTAT'] * 0.5 + 
                       np.random.normal(0, 3, n_samples))
        return df, None

@st.cache_resource
def train_model(df, model_type, problem_type):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if problem_type == "Classification":
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:  # SVM
            model = SVC(probability=True, random_state=42)
    else:  # Regression
        if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # Linear Regression
            model = LinearRegression()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred

def create_feature_importance_plot(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_names[indices],
            y=importances[indices],
            marker_color='lightblue'
        ))
        fig.update_layout(title="Feature Importance", height=400)
        return fig
    return None

def create_prediction_plot(y_true, y_pred, problem_type):
    if problem_type == "Regression":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions'))
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                               mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
        fig.update_layout(title="Actual vs Predicted Values", height=400)
        return fig
    else:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        return fig

def main():
    st.title("ML Model Deployment Hub")
    st.markdown("Deploy, interact with, and visualize machine learning models with ease!")
    
    st.sidebar.title("🔧 Configuration")
    
    dataset_options = ["Iris Classification", "Custom Classification", "Boston Housing"]
    selected_dataset = st.sidebar.selectbox("Choose Dataset", dataset_options)
    
    problem_type = "Classification" if "Classification" in selected_dataset else "Regression"
    
    if problem_type == "Classification":
        model_options = ["Random Forest", "Logistic Regression", "SVM"]
    else:
        model_options = ["Random Forest", "Linear Regression"]
    
    selected_model = st.sidebar.selectbox("Choose Model", model_options)
    
    with st.spinner("Loading data and training model..."):
        df, class_names = load_sample_data(selected_dataset)
        model, X_train, X_test, y_train, y_test, y_pred = train_model(df, selected_model, problem_type)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Model Overview", "Make Predictions", "Model Analysis", "Model Management"])
    
    with tab1:
        st.subheader("Model Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset", selected_dataset)
            st.metric("Problem Type", problem_type)
        with col2:
            st.metric("Model Type", selected_model)
            st.metric("Training Samples", len(X_train))
        with col3:
            if problem_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.3f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                st.metric("MSE", f"{mse:.3f}")
            st.metric("Test Samples", len(X_test))
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        if problem_type == "Classification" and class_names:
            fig = px.histogram(df, x='target', title="Target Distribution")
        else:
            fig = px.histogram(df, x='target', title="Target Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Make Predictions")
        
        input_method = st.radio("Choose input method:", ["Manual Input", "Upload CSV", "Random Sample"])
        
        if input_method == "Manual Input":
            st.markdown("### Enter Feature Values")
            feature_cols = st.columns(2)
            input_data = {}
            
            feature_names = df.drop('target', axis=1).columns
            for i, feature in enumerate(feature_names):
                col = feature_cols[i % 2]
                with col:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    input_data[feature] = st.number_input(feature, min_value=min_val, max_value=max_val, value=mean_val)
            
            if st.button("Make Prediction", type="primary"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                
                if problem_type == "Classification":
                    probabilities = model.predict_proba(input_df)[0]
                    st.success(f"Predicted Class: {class_names[int(prediction)] if class_names else int(prediction)}")
                    
                    prob_data = pd.DataFrame({
                        'Class': class_names if class_names else [f"Class {i}" for i in range(len(probabilities))],
                        'Probability': probabilities
                    })
                    
                    fig = px.bar(prob_data, x='Class', y='Probability', title="Class Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success(f"Predicted Value: {prediction:.3f}")
        
        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                upload_df = pd.read_csv(uploaded_file)
                st.dataframe(upload_df.head())
                
                if st.button("Make Predictions", type="primary"):
                    predictions = model.predict(upload_df)
                    
                    if problem_type == "Classification":
                        probabilities = model.predict_proba(upload_df)
                        results_df = upload_df.copy()
                        results_df['Prediction'] = [class_names[int(p)] if class_names else int(p) for p in predictions]
                        
                        for i, class_name in enumerate(class_names if class_names else [f"Class_{i}" for i in range(len(probabilities[0]))]):
                            results_df[f'Prob_{class_name}'] = probabilities[:, i]
                    else:
                        results_df = upload_df.copy()
                        results_df['Prediction'] = predictions
                    
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False)
                    st.download_button(" Download Results", data=csv, file_name="predictions.csv", mime="text/csv")
        
        else:  # Random Sample
            if st.button("Generate Random Sample & Predict", type="primary"):
                feature_names = df.drop('target', axis=1).columns
                random_data = {feature: np.random.uniform(df[feature].min(), df[feature].max()) for feature in feature_names}
                
                input_df = pd.DataFrame([random_data])
                prediction = model.predict(input_df)[0]
                
                st.markdown("**Random Input:**")
                st.dataframe(input_df)
                
                if problem_type == "Classification":
                    probabilities = model.predict_proba(input_df)[0]
                    st.success(f"Predicted Class: {class_names[int(prediction)] if class_names else int(prediction)}")
                    
                    prob_data = pd.DataFrame({
                        'Class': class_names if class_names else [f"Class {i}" for i in range(len(probabilities))],
                        'Probability': probabilities
                    })
                    fig = px.bar(prob_data, x='Class', y='Probability', title="Class Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success(f"Predicted Value: {prediction:.3f}").predict(input_df)[0]
                
                # Display input
                st.markdown("**Random Input:**")
                st.dataframe(input_df)
                
                # Display prediction
                if problem_type == "Classification":
                    probabilities = model.predict_proba(input_df)[0]
                    st.success(f"Predicted Class: {class_names[int(prediction)] if class_names else int(prediction)}")
                    
                    prob_data = pd.DataFrame({
                        'Class': class_names if class_names else [f"Class {i}" for i in range(len(probabilities))],
                        'Probability': probabilities
                    })
                    
                    fig = px.bar(prob_data, x='Class', y='Probability', title="Class Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success(f"Predicted Value: {prediction:.3f}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Model Analysis</h2>', unsafe_allow_html=True)
        
        # Model performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Performance")
            if problem_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Test Accuracy", f"{accuracy:.3f}")
                
                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3))
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                st.metric("Test MSE", f"{mse:.3f}")
                st.metric("Test RMSE", f"{rmse:.3f}")
                
                # R² score
                from sklearn.metrics import r2_score
                r2 = r2_score(y_test, y_pred)
                st.metric("R² Score", f"{r2:.3f}")
        
        with col2:
            # Prediction plot
            pred_fig = create_prediction_plot(y_test, y_pred, problem_type)
            if pred_fig:
                st.plotly_chart(pred_fig, use_container_width=True)
        
        # Feature importance
        importance_fig = create_feature_importance_plot(model, df.drop('target', axis=1).columns)
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Model parameters
        st.markdown("### Model Parameters")
        params = model.get_params()
        param_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        st.dataframe(param_df, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Model Management</h2>', unsafe_allow_html=True)
        
        # Save model
        st.markdown("### Save Model")
        model_name = st.text_input("Model Name", value=f"{selected_model}_{selected_dataset}")
        
        if st.button("Save Model", type="primary"):
            # Serialize model
            model_data = {
                'model': model,
                'feature_names': list(df.drop('target', axis=1).columns),
                'model_type': selected_model,
                'problem_type': problem_type,
                'dataset': selected_dataset
            }
            
            # Create download link
            buffer = io.BytesIO()
            pickle.dump(model_data, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="Download Model",
                data=buffer.getvalue(),
                file_name=f"{model_name}.pkl",
                mime="application/octet-stream"
            )
            
            st.success("Model ready for download!")
        
        # Load model
        st.markdown("### Load Model")
        uploaded_model = st.file_uploader("Upload Model File", type="pkl")
        
        if uploaded_model is not None:
            try:
                loaded_data = pickle.load(uploaded_model)
                st.success("Model loaded successfully!")
                
                st.markdown("**Model Information:**")
                st.write(f"- **Model Type:** {loaded_data['model_type']}")
                st.write(f"- **Problem Type:** {loaded_data['problem_type']}")
                st.write(f"- **Dataset:** {loaded_data['dataset']}")
                st.write(f"- **Features:** {', '.join(loaded_data['feature_names'])}")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
        
        # Model comparison
        st.markdown("### Model Comparison")
        st.info("Train multiple models to compare performance")
        
        if st.button("Compare All Models"):
            comparison_results = []
            
            if problem_type == "Classification":
                models_to_compare = ["Random Forest", "Logistic Regression", "SVM"]
            else:
                models_to_compare = ["Random Forest", "Linear Regression"]
            
            for model_type in models_to_compare:
                temp_model, _, temp_X_test, _, temp_y_test, temp_y_pred = train_model(df, model_type, problem_type)
                
                if problem_type == "Classification":
                    score = accuracy_score(temp_y_test, temp_y_pred)
                    metric = "Accuracy"
                else:
                    score = mean_squared_error(temp_y_test, temp_y_pred)
                    metric = "MSE"
                
                comparison_results.append({
                    'Model': model_type,
                    metric: score
                })
            
            comparison_df = pd.DataFrame(comparison_results)
            
            fig = px.bar(
                comparison_df, x='Model', y=metric,
                title=f"Model Comparison - {metric}",
                color=metric,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()
