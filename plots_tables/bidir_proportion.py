import plotnine as p9
import numpy as np
import pandas as pd

def proportion(k):
    return (k - 1) / (k + 1)

if __name__ == "__main__":

    k = np.linspace(0, 9, 10) + 1
    p_values = proportion(k)

    # Create a dictionary for plotnine (it will convert to DataFrame internally)
    data = {'k': k - 1, 'p': p_values * 100}
    data = pd.DataFrame(data)
    # Create the plot
    plot = (
        p9.ggplot(data, p9.aes(x='k', y='p'))
        + p9.geom_line(color='darkmagenta', size=1.5)
        + p9.geom_point(color='darkmagenta', size=3, shape='o')
        + p9.geom_hline(yintercept=100, linetype='dashed', color='black')
        + p9.scale_x_continuous(breaks=k - 1)
        + p9.scale_y_continuous(breaks=[0, 50, 100])
        + p9.theme_minimal()
        + p9.theme(
            panel_border=p9.element_blank(),
            panel_grid=p9.element_blank(),
            axis_line=p9.element_line(color='black'),
            text=p9.element_text(family='Times New Roman', size=10),
            axis_text=p9.element_text(family='Times New Roman', size=11, color='black'),
            axis_title=p9.element_text(family='Times New Roman', size=12, color='black')
        )
        + p9.labs(y='%Bidir. blocks', x='$r$')
    )

    # Save or display the plot
    plot.save('figures/bidir_proportion.png', dpi=500, width=2.5, height=1.5)
    print(plot)
    