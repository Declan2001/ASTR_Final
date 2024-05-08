# imports at the class level
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner

class Fitting:
    
    def __init__(self, input1, input2=None, parameters=None):
        self.parameters = parameters
        if isinstance(input1, list) and isinstance(input2, list):
            self.filename = None
            self.X = input1
            self.Y = input2
        elif isinstance(input1,str):
            self.filename = input1    
            self.X, self.Y = self.input_func()
        else: 
            print("Wrong input types, either 2 lists or a string as a filename")
        return

    def input_func(self):
        # pulling data and putting it into two lists
        with open(self.filename, "r") as f1:
            lines = f1.readlines()
        x_data, y_data = [], []
        x_str, y_str = lines[:250], lines[251:]
        for i in range(len(x_str)):
            x_str[i], y_str[i] = int(float(x_str[i].strip('\n'))), int(float(y_str[i].strip('\n')))
            x_data.append(x_str[i])
            y_data.append(y_str[i])
        x_data, y_data = np.array(x_data), np.array(y_data)

        # removing wrapping by sorting the paired values
        coord = []
        for i in range(len(x_data)):
            coord.append((x_data[i],y_data[i]))
        coord = sorted(coord)
 
        # put into two lists
        x_data = [i[0] for i in coord]
        y_data = [i[1] for i in coord]
        
        # change domain to from [0,2300) to [0,2pi)
        x_data = [i*2*np.pi/2299 for i in np.array(x_data)]

        return x_data, y_data

    def Curvefit(self, plot=True, model=""):
        from scipy.optimize import curve_fit
        
        # input value and make sure np.array
        x_data = np.array(self.X)
        y_data = np.array(self.Y)
        
        # define the function
        if model == "":
            f = lambda x, a, b, c, d, g: a*np.sin(2*x) + b*np.cos(2*x) + c*np.sin(4*x) + d*np.cos(4*x) + g
        elif model == "sin(x)":
            f = lambda x, a, b, c, d, g: a*np.sin(b*x+c)+d + g
        elif model == "cos(x)":
            f = lambda x, a, b, c, d, g: a * np.cos(b * x + c) + d + g
        elif model == "x**2":
            f = lambda x, a, b, c, d, g: a * (b * x + c)**2 + d + g
        else:
            print("Error")
            return
            
        # define the paramters for the plot
        plt.figure(figsize=[10, 6])
        x_min, x_max = 0, 6.3
        npoints = 250
        if self.parameters == None:
            a, b, c, d, g = 112.624, -0.03, 198.048,  0.011, -449.627
        else:
            a, b, c, d, g = self.parameters
        
        # fit with curve_fit
        params, pcov = curve_fit(f, x_data, y_data, p0=[a,b,c,d,g])

        if plot:
            # plot
            plt.plot(x_data, y_data, 'o', label='data')
            plt.plot(x_data, f(x_data,params[0],params[1],params[2],params[3],params[4]), label='curve_fit')
            plt.xlabel('Motor angle in Radians [0:2pi]', fontsize=15)
            plt.ylabel('Averaged Pixel Value [electron count]', fontsize=15)
            plt.title('Curvefit Plot', fontsize=18)
            plt.xlim([x_min, x_max])
            plt.legend(fontsize=12)
            plt.show()
            print('Curvefit:')
            print('original coefficients: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f' %(a,b,c,d,g))
            print('curve_fit coefficients: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f' %(params[0], params[1], params[2], params[3], params[4]))
            # save plot and save old + new parameters to a file
        return params[0], params[1], params[2], params[3], params[4], f(x_data,params[0],params[1],params[2],params[3],params[4])

    def MCMC(self, plot=True, model=None): 
        # input value and make sure np.array
        x = np.array(self.X)
        y = np.array(self.Y)

        if model ==  None:
            def model(theta, x):
                a, b, c, d, g = theta
                return a*np.sin(2*x) + b*np.cos(2*x) + c*np.sin(4*x) + d*np.cos(4*x) + g
        elif model == "sin(x)":
            def model(theta, x):
                a, b, c, d, g = theta
                g = 0
                return a*np.sin(b*x+c)+d + g
        elif model == "cos(x)":
            def model(theta, x):
                a, b, c, d, g = theta
                g = 0
                return a*np.cos(b*x+c)+d + g
        elif model == "x**2":
            def model(theta, x):
                a, b, c, d, g = theta
                g = 0
                return a*(b*x+c)**2+d + g

        
        def lnlike(theta,x,y,yerr):
            return (-0.5*((y-model(theta,x))/(yerr))**2).sum()

        if self.filename == "output_25":
            def lnprior(theta):
                a, b, c, d, g = theta
                if (a < -400 or a > 400):
                    return 0
                if (b < -0.1 or b > 0.1):
                    return 0
                if (c < -400 or c > 400):
                    return 0
                if (d < -0.1 or d > 0.1):
                    return 0
                if (g < -550 or g > -400):
                    return 0
                return 1

        elif self.filename == "output_26":
            def lnprior(theta):
                a, b, c, d, g = theta
                if (a < -200 or a > 0):
                    return 0
                if (b < -300 or b > -100):
                    return 0
                if (c < -6500 or c > -4500):
                    return 0
                if (d < 1400 or d > 3000):
                    return 0
                if (g < 5500 or g > 7000):
                    return 0
                return 1

        else: 
            def lnprior(theta):
                a, b, c, d, g = theta
                if (a < a*1.3 or a > 0):
                    return -np.inf
                if (b < b*1.3 or b > 0.7*b):
                    return -np.inf
                if (c < c*1.3 or c > 0.7*c):
                    return -np.inf
                if (d < d*1.3 or d > 0.7*d):
                    return -np.inf
                if (g < g*1.3 or g > 0.7*g):
                    return -np.inf
                return 1

        def lnprob(theta, x, y, yerr):
            lp = lnprior(theta)
            if lp == -np.inf:
                return -np.inf
            return lp + lnlike(theta,x,y,yerr)

        def main(p0,nwalkers,niter,ndim,lnprob,data):
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
        
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 100)
            sampler.reset()
        
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter)
        
            return sampler, pos, prob, state

        def plotter(sampler,x_data,y_data):
            plt.ion()
            plt.plot(x_data,y_data,label='Raw')
            samples = sampler.flatchain
            counter = 0
            for theta in samples[np.random.randint(len(samples), size=100)]:
                plt.plot(x_data, model(theta, x_data), color="r", alpha=0.1)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.title("MCMC All Possible 'Best' Fits", fontsize=18)
            plt.xlabel('Motor Angle [radians 0:2pi]', fontsize=15)
            plt.ylabel('Averaged Pixel Value [electron count]', fontsize=15)
            plt.legend()
            plt.show()

        # make all these input parameters
        y_error = [i*.05 for i in y]
        data = (x, y, y_error)
        nwalkers = 500
        niter = 100
        if self.parameters == None:
            initial = np.array([-106.375, -218.360, -5495.884, 2795.745, 6140.249])
        else:
            initial = self.parameters
        ndim = len(initial)
        p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

        sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)

        if plot:
            plotter(sampler, x, y)

        samples = sampler.flatchain
        samples[np.argmax(sampler.flatlnprobability)]

        samples = sampler.flatchain

        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        best_fit_model = model(theta_max, x)
        if plot:
            plt.plot(x,y,label='Raw Data')
            plt.plot(x,best_fit_model,label='Highest Likelihood Model')
            plt.xlabel('Motor Angle [radians 0:2pi]', fontsize=15)
            plt.ylabel('Averaged Pixel Value [electron count]', fontsize=15)
            plt.title("MCMC Best Fit", fontsize=18)
            plt.legend()
            plt.show()
            print('Theta max: ',theta_max)
    
            labels = ['a','b','c','d','g']
            fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
        return theta_max, best_fit_model

    def error_bars_SD(self):
        y = np.array(self.Y)
        upper_envelope = np.array(y + np.std(y))
        lower_envelope = np.array(y - np.std(y))
        return upper_envelope, lower_envelope

    
    def fancy_plot_and_phase(self):
        x = np.array(self.X)
        y = np.array(self.Y)
        upper_envelope, lower_envelope = self.error_bars_SD()
        a1, b1, c1, d1, g1, C_model = self.Curvefit(plot=False)
        theta_max, MCMC_model = self.MCMC(plot=False) 
        
        ### plotting ###
        plt.figure(figsize=(20,15))
        plt.xlabel('Motor Angle [radians 0:2pi]', fontsize=18)
        plt.ylabel('Averaged Pixel Value [electron count]', fontsize=18)
        plt.scatter(x,y,label='Raw Data',color='purple')
        plt.plot(x, upper_envelope, color='red', linestyle='--', label='Upper Envelope (Mean + 1 SD)')
        plt.plot(x, lower_envelope, color='red', linestyle='--', label='Lower Envelope (Mean - 1 SD)')   
        plt.plot(x, MCMC_model, label='MCMC Model', color='black')
        plt.plot(x, C_model, label='Curvefit Model', color='blue')
        plt.legend(fontsize=15)
        plt.show()

        ### phase angle ###   
        # Curvefit
        # Calculate the amplitudes and phase shifts
        R1 = np.sqrt(a1**2 + b1**2)
        phi1 = np.arctan2(b1, a1)  # Using atan2 to handle the correct quadrant
        S1 = np.sqrt(c1**2 + d1**2)
        psi1 = np.arctan2(d1, c1)  # Using atan2 to handle the correct quadrant

        # MCMC
        a2, b2, c2, d2, g2 = theta_max
        # Calculate the amplitudes and phase shifts
        R2 = np.sqrt(a2**2 + b2**2)
        phi2 = np.arctan2(b2, a2)  # Using atan2 to handle the correct quadrant
        S2 = np.sqrt(c2**2 + d2**2)
        psi2 = np.arctan2(d2, c2)  # Using atan2 to handle the correct quadrant

        if phi1 < 0:
            phi1 += 2* np.pi
        if phi2 < 0:
            phi2 += 2* np.pi
        if psi1 < 0:
            psi1 += 2* np.pi
        if psi2 < 0:
            psi2 += 2* np.pi

        #print(R1, S1, R2, S2)
        print("Curvefit phases: ", phi1, psi1, "MCMC phases: ", phi2, psi2)
        print("Curvefit phase: ", (phi1+psi1)/2, "+/-", np.abs(phi1-(phi1+psi1)/2))
        print("MCMC phase: ", (phi2+psi2)/2, "+/-", np.abs(phi2-(phi2+psi2)/2))
        return
