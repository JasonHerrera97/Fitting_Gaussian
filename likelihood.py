import numpy as np

def likelihood(x, mu, sigma, n=50):
    
    """
    This function accepts a trial mean and std value and constructs 
    a gaussian function. It also ingests the data, histograms it,
    and compares the data distribution with the trial gaussian. 
    It then computes the mean sqr deviation between the two, inverse
    the result and takes the ln and returns the value.
    
    INPUT:
    ------
    x = data
    mu = trial mean
    sigma = trial sigma
    n = number of bins (default = 50)
    
    RETURNS:
    --------
    likelihood = ln of inverse of mean sqr deviation 
    """
    
    y_values, bins = np.histogram(x,n,density=True)
    bins_center = (bins[1:] + bins[:-1])*.5
    N = 1/(sigma*np.sqrt(2*np.pi))
    trial_gaussian = N*(np.exp(-0.5*((bins_center-mu)/sigma)**2))
    error = np.sum((trial_gaussian - y_values)**2)/bins_center.size
    return np.log(1/error)
    

