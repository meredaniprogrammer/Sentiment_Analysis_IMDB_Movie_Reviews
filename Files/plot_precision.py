# import plotly.graph_objs as go
#
# def plot_precision(models, precs):
#     """
#     Plot precision for all the models using an interactive line chart.
#
#     Parameters:
#     - models: List of model names
#     - precs: List of precision scores corresponding to each model
#     """
#
#     # Create a trace
#     trace = go.Scatter(
#         x=models,
#         y=precs,
#         mode='lines+markers+text',
#         text=precs,
#         textposition="top center"
#     )
#
#     layout = go.Layout(
#         title="Precision of Various Models",
#         xaxis=dict(title="Models"),
#         yaxis=dict(title="Precision"),
#         showlegend=False
#     )
#
#     fig = go.Figure(data=[trace], layout=layout)
#     fig.show()

# import plotly.graph_objs as go
#
# def plot_precision(models, precs):
#     """
#     Plot precision for all the models using a bar chart.
#
#     Parameters:
#     - models: List of model names
#     - precs: List of precision scores corresponding to each model
#     """
#
#     # Create a bar trace
#     trace = go.Bar(
#         x=models,
#         y=precs,
#         text=precs,
#         textposition="outside",
#         marker_color='blue'  # You can change the color as desired
#     )
#
#     layout = go.Layout(
#         title="Precision of Various Models",
#         xaxis=dict(title="Models"),
#         yaxis=dict(title="Precision"),
#         showlegend=False
#     )
#
#     fig = go.Figure(data=[trace], layout=layout)
#     fig.show()

# import plotly.graph_objs as go
# from Tools.scripts.dutree import display
#
#
# def plot_precision(models, precs):
#     """
#     Plot precision for all the models using a bar chart.
#
#     Parameters:
#     - models: List of model names
#     - precs: List of precision scores corresponding to each model
#     """
#
#     # Define custom colors for bars
#     colors = ['blue', 'green', 'red', 'purple', 'orange']
#
#     # Create a bar trace
#     trace = go.Bar(
#         x=models,
#         y=precs,
#         text=[f'{prec:.2f}' for prec in precs],  # Format precision to two decimal places
#         textposition="outside",
#         marker_color=colors,  # Assign different colors to bars
#         width=0.3  # Further reduce the width of the bars
#     )
#
#     layout = go.Layout(
#         title="Precision of Various Models",
#         xaxis=dict(title="Models"),
#         yaxis=dict(title="Precision"),
#         showlegend=False
#     )
#
#     fig = go.Figure(data=[trace], layout=layout)
#
#     # Use the display method to show the plot in a separate window
#     display(fig)

#------------------------------------------------------------------
import matplotlib.pyplot as plt

def plot_precision(models, precs):
    """
    Function to plot F1-Score of various models.

    Parameters:
    - models: List of names of the models.
    - f1_scores: List of F1-Score values corresponding to each model.
    """

    # Define custom colors for bars
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.figure(figsize=(10, 5))

    # Create a bar chart with tiny bars and different colors
    plt.bar(models, precs, color=colors, width=0.2)

    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.title('Precision of Various Models')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display F1-Score value on top of each bar with two decimal places
    for i, txt in enumerate(precs):
        plt.text(models[i], txt, f"{txt:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

