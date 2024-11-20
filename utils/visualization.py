import plotly.express as px


def plot_age_distribution(data):
    """
    Create a histogram to show the distribution of ages in the dataset.
    """
    fig = px.histogram(
        data,
        x='Age',
        nbins=20,
        title='Age Distribution',
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(bargap=0.1)
    return fig


def plot_diagnosis_pie(data):
    """
    Create a pie chart to show the distribution of diagnoses in the dataset.
    """
    diagnosis_counts = data['Diagnosis'].value_counts().reset_index()
    diagnosis_counts.columns = ['Diagnosis', 'Count']
    fig = px.pie(
        diagnosis_counts,
        names='Diagnosis',
        values='Count',
        title='Diagnosis Distribution',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    return fig


def plot_mmse_distribution(data):
    """
    Create a histogram to show the distribution of MMSE scores.
    """
    fig = px.histogram(
        data,
        x='MMSE',
        nbins=20,
        title='MMSE Score Distribution',
        color_discrete_sequence=['#00CC96']
    )
    fig.update_layout(bargap=0.1)
    return fig


def plot_correlation(data, x_var, y_var):
    """
    Create a scatter plot to visualize the correlation between two variables.
    Includes a trendline.
    """
    fig = px.scatter(
        data,
        x=x_var,
        y=y_var,
        title=f'{x_var} vs {y_var}',
        trendline='ols',
        color_discrete_sequence=['#AB63FA']
    )
    return fig
