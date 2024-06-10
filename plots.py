import matplotlib.pyplot as plt
import plotly.express as px

def plot_distribution(housing):
    fig, ax = plt.subplots()
    housing['median_house_value'].hist(bins=50, ax=ax, figsize=(10, 5))
    ax.set_xlabel('Median House Value')
    ax.set_ylabel('Frequency')
    return fig

def plot_actual_vs_predicted(y_test, y_pred):
    fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual Median House Value', 'y':'Predicted Median House Value'})
    fig.update_layout(title='Actual vs Predicted Median House Values (Random Forest) with Capped Indicator')
    return fig

def plot_residuals(y_test, residuals):
    fig = px.scatter(x=y_test, y=residuals, labels={'x':'Actual Median House Value', 'y':'Residuals'})
    fig.update_layout(title='Residuals vs Actual Median House Values (Random Forest) with Capped Indicator')
    return fig
