This code is used for cleaning up and plotting data. 
The main goal is to display how well the data can be fitted by a simply sine and cosine expression. Then to extract the phase from it.

Class Inputs:
The code has been premade to deal with files consisting of two 250 length lists.
It can also take the two lists and deal with them directly.
There is an optional input of parameters which is useful if not using the given data sets.
This is all implemented by one mandatory input either a list and a file then an optional second input for the secondary list if that path is chosen. The third option is the list of parameters, length of 5

Functions in the class:
The first function is the __init__ function which sets the main variables for the rest of the functions
This consists of the two lists and/or the input file

The next function is the the input_func that only operates if there is a file. It will read it in and structure the lists appropriately

The third function is Curvefit. This uses the two lists and models a best fit for the guess parameters based off of the model
a*np.sin(2*x) + b*np.cos(2*x) + c*np.sin(4*x) + d*np.cos(4*x) + g, unless a different function is given.
The inputs here consist of a True/False flag that will plot if True and skip plotting if False. 
The model input will default to the given function above if left to default input. It can also take in "cos(x)", "sin(x)", and "x**2"
If plot==True then it will plot the data raw data along with the best fit model.
It returns the parameters of the best fit model and the model's output based on the parameters and the x values.

The fourth function is MCMC. Here is a more indepth fitting method compared to the previous function.
This uses the two lists and moedls a best fit based off of the guess parameters based off of the model
a*np.sin(2*x) + b*np.cos(2*x) + c*np.sin(4*x) + d*np.cos(4*x) + g, unless a different function is given.
The inputs here consist of a True/False flag that will plot if True and skip plotting if False. 
The model input will default to the given function above if left to default input. It can also take in "cos(x)", "sin(x)", and "x**2"
If plot==True then it will plot of the raw data along with all possible models, the raw data along with the best fit model, and a corner plot.
It returns the parameters of the best fit model and the model's output based on the parameters and the x values.

The next function error_bars_SD creates error envelopes where there is a line +/- one standard deviation from the mean.
This has no inputs other than self.
It will return the upper envelope and lower envelope arrays.

The next function is fancy_plot_and phase.
This has no inputs other than self.
It returns nothing. 
However it will plot and show both envelopes, the best MCMC fit, the best Curvefit, and the raw data all on one graph.
It will also compute and print the average phase based off of the phase of the waves from the MCMC best fit and the Curvefit best fit.
