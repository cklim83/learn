### Matplotlib

Tutorial Links https://matplotlib.org/tutorials/index.html

#### Import and Options
- import matplotlib.pyplot as plt  # library import
- %matplotlib inline  # plot inline in jupyter notebook

#### Line Plots
- plt.plot(X, Y)  # line chart
- plt.xlabels("X-axis title")
- plt.ylabels("Y-axis title")
- plt.title("Plot title")
- plt.xticks(rotation=90)  # rotate axis ticks
- plt.show()  # show the plot

#### Multiple Plots Using Subplots
- fig = plt.figure()  # create figure, a container for all plots
- ax1 = fig.add_subplot(2, 1, 1)  # Subplot position 1 of subplots with 2 rows, 1 column
- ax2 = fig.add_subplot(2, 1, 2)  # Subplot position 2 of subplots with 2 rows, 1 column
- ax1.plot(df['X'], df['Y'])  # Line plot in axes 1
- fig = plt.figure(figsize=(12, 5))  # Set figure size to width, height
- plt.plot(df['X'], df['Y'], c='red')  # Draw line in red color
- plt.legend(loc="upper left")  # Create legend to upper left of plot
- ax.set_title("Title string")  # Set title for specific axis

Note: plt.plot() will create a figure by default for simple plots which can be
visualized upon show command.


#### Bar Plots and Scatter Plots
- plt.bar(bar_positions, bar_heights, width)  OR
- Axes.bar(bar_positions, bar_heights, width)  # Both generate vertical bar plot
- bar_positions = arange(5) + 0.75  # use arange to generate evenly separated values
- ax.set_ticks([1, 2, 3, 4, 5])  # set ticks at locations in list provided
- ax.set_xticklabels(['label1', 'label2', 'label3', 'label4', 'label5'])  # set x labels at tick locations
- ax.set_xticklabels(['label1', 'label2', 'label3', 'label4', 'label5'], rotation=90)  # rotate xtick labels
- ax.scatter(X, Y)  # create a scatter plot

*Notes*:
<br>Bar plots are useful to visualize smallest, largest values and can be horizontal or vertical.
<br>Scatter plot useful to show correlation between 2 variables


#### Histograms and Box Plots
- df['Category_Column'].value_counts()  # count the occurrence of each category
- ax.hist(df['Category_Column'])  # create histogram
- ax.hist(df['Category_Column'], range(0, 5))  # Specify lower and upper range of bins within histogram
- ax.set_ylim(0,50)  # Set y limits
- ax.hist(norm_reviews['Numeric_Column'], bins = 20)  # Set number of bins to divide numeric data
- ax.boxplot(norm_reviews["Numeric_Column"])  # plot box and whisker (plots 25-75 percentile and outliers)


#### Text Annotations and Visual Optimisation
- ax.plot(X, Y, c=cb_dark_blue, label='Women', linewidth=3)  # use linewidth to alter line thickness for better focus
- ax.text(X_coordinate, Y_Coordinate, "String to Print")  # For annotating objects in visual plots
###### Turn off ticks on axis
- plt.tick_params(axis='x',          # changes apply to the x-axis
                  which='both',      # both major and minor ticks are affected
                  bottom=False,      # ticks along the bottom edge are off
                  top=False,         # ticks along the top edge are off
                  labelbottom=False) # labels along the bottom edge are off
- ax.spines['right'].set_visible(False)  # Remove spines for right axis
- for key, spine in ax.spines.items():
    spine.set_visible(False)  # Remove spines for all axes


#### Formatting Axis Displays
- Format y-axis strings (e.g. print in millions)
  - millions = lambda value, pos: "$%1.1fM" % (value*1e-6) # pos is location on axis
  - formatter = FuncFormatter(millions)
  - ax.yaxis.set_major_formatter(formatter)
  - Source: (https://matplotlib.org/examples/pylab_examples/custom_ticker1.html)

- Format date type to print in "mon-YYYY" strings in x-axis
  - import matplotlib.dates as mdates
  - ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

- Add vertical span to x-axis to demarcate event period
  - ax.axvspan(start_x, end_x, facecolor='grey', alpha=0.5)

- Log scale
  - ax.semilogy(t, np.exp(-t/5.0))  # only y-axis is log scale
  - ax.set_yscale("log", nonposy='clip') # clip non=positive values
  - ax.loglog(t, 20*np.exp(-t/10.0), basex=2) # for log log scales
  - Source: (https://matplotlib.org/examples/pylab_examples/log_demo.html)

- Dual axis
