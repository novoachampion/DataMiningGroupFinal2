# visualizations.py
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
import os

def generate_scatter_plots(X_test, features, output_file_name):
    output_path = output_file_name
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file(output_path)

    plots = []
    source = ColumnDataSource(X_test)
    for i, x in enumerate(features):
        for j, y in enumerate(features):
            if i < j:
                p = figure(width=300, height=300, title=f"{x} vs {y}", x_axis_label=x, y_axis_label=y)
                p.circle(x=x, y=y, source=source, size=5, alpha=0.6)
                plots.append(p)
    grid = gridplot(plots, ncols=3)
    save(grid)
    print(f"Scatter plots saved to {output_path}")
