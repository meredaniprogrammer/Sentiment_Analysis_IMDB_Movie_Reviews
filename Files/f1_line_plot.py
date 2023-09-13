# f1_line_plot.py
import plotly.graph_objects as go

def plot_f1_line(models, f1s):
    """
    Plots F1 scores in an interactive line graph using Plotly.

    :param models: A list of model names.
    :param f1s: A list of F1 scores corresponding to each model.
    """
    trace = go.Scatter(
        x=models,
        y=f1s,
        mode='lines+markers',
        text=f1s,  # this will display the F1 score when hovering
        hoverinfo='x+text',
        marker=dict(
            size=10,
            color='rgb(255, 65, 54)',
            line=dict(
                color='rgb(0,0,0)',
                width=2
            )
        ),
        line=dict(
            color='rgb(255, 65, 54)'
        )
    )

    layout = go.Layout(
        title='F1 Scores of Different Models',
        xaxis=dict(title='Models'),
        yaxis=dict(title='F1 Score'),
        hovermode='closest'
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

