import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import batman
from astropy.io import ascii
from astropy.table import Table
import astropy
import astropy.units as u
from scipy import stats
from scipy.special import kn
from contextlib import closing
from astropy.time import Time
import argparse
from astropy.io import ascii
from astropy.table import Table
import scipy.optimize

#Read in data
tabAXIS = ascii.read('tabPlanetsScaleHeightMassRadiusFinal.csv') #table of planet names and parameters, w/ AXIS count rates
tabNewATHENA = ascii.read('tabPlanetsNewAthena.csv') #table of planet names and parameters, w/ NewAthena count rates

#Add command-line arguments
parser = argparse.ArgumentParser(description='Input planet name and number of observations, output 500s observation period AXIS transit light curve. Ex: "AU Mic b" 1') 

parser.add_argument(dest='planetname', help='(str) Name of planet, must be in csv "pl_name" column') #Planet name
parser.add_argument(dest='nob', help='(int) Number of observations', type=int) #Number of observations to be stacked
parser.add_argument('--fit', action='store_true', help='Fit simulated transit data')
parser.add_argument('--radchange', help = '(float) Multiplier for planet radius. Optional - default 1', default=1.0, type=float) #Radius multiplier
parser.add_argument('--savefig', action='store_true', help='Save figures')
parser.add_argument('--scope', choices = ['AXIS', 'NEWATHENA'], help='(AXIS or NEWATHENA) Which telescope to do simulate obs for. Optional - default AXIS', default='AXIS')
parser.add_argument('--LEO', action='store_true', help='Account for AXIS Low-Earth orbit')
parser.add_argument('--flarerate', help='(float) Average flare rate per 500 sec. Optional - default 0', default=0, type=float)
parser.add_argument('--flareduration', help='(float) Average flare duration in Ks. Optional - default 0', default=0, type=float)
args = parser.parse_args() 

# Set the axes title font size
plt.rc('axes', titlesize=12) #Set the title font size
plt.rc('axes', labelsize=12) #Set the font size for axis labels
plt.rc('xtick', labelsize=12) #Set the font size for x ticks
plt.rc('ytick', labelsize=12) #Set the font size for y ticks
plt.rc('legend', fontsize=12) #Set the legend font size
	
#Setting args to variables
print("planet name: %s" % args.planetname)
pl_name = args.planetname
numstack = args.nob
fitbool = args.fit
radchange = args.radchange
figsave = args.savefig
scope = args.scope
AXIS_LEO = args.LEO
flarerate = args.flarerate
flaredur = args.flareduration

#selecting correct dataset
if scope=='AXIS':
    tabPlanets = tabAXIS
if scope=='NEWATHENA':
    tabPlanets = tabNewATHENA

#Number of coronal scale heights to include
Nscale = 6

#Functions
def LEO(data, startrange = 2.5, obsdur = 55, noobsdur = 43):
    """Function that cuts out repeating parts of data, to simulate data loss due to low-earth orbit.
    Inputs: data, range of times to start the observation (ks from default start), duration of observation periods (min), duration of non-observation periods (min)
    Outputs: LEO-simulated data
    """
    for lc in data: #For each light curve in the dataset:
        obsstart = float(np.random.uniform(0, 2.5, 1)[0]) #Select where the observation starts

        obsdur_ks = obsdur * 60 / 1000 #Put observation duration in ks
        noobsdur_ks = noobsdur * 60 / 1000 #Put non-observation duration in ks
        
        lc[ :int(np.ceil(obsstart/0.5)) + 1] = np.nan #Remove datapoints outside the observation
        
        num_nonnan = len(lc[~np.isnan(lc)]) #select real datapoints
        num_orbs = int( np.ceil(num_nonnan*0.5 / (obsdur_ks + noobsdur_ks) ) ) #calculate the number of orbits with the number of datapoints
        
        for val in range(num_orbs + 1): #For each orbit:
        
            #Remove datapoints in non-observation windows
            if val != num_orbs: 
                lc[int(np.ceil((obsstart + val*(obsdur_ks + noobsdur_ks) + obsdur_ks)/0.5))
                : int(np.ceil((obsstart + (val+1)*(obsdur_ks+noobsdur_ks))/0.5))] = np.nan 
                
            if val == num_orbs: 
                lc[int(np.ceil((obsstart + val*(obsdur_ks + noobsdur_ks) + obsdur_ks)/0.5)): ] = np.nan
            
     
def flare(dataset, flare_chance, flare_duration):
    """Function that cuts out data during simulated flares.
    Inputs: data, flare rate per 500 seconds, flare duration in ks
    Outputs: data with flares removed
    """
    for lc in dataset:#For each light curve in the dataset:
    
        #Generate random flare locations and durations
        flarechance = np.random.uniform(0,1,size=len(lc)*2) 
        will_flare = (flarechance <= flare_chance)
        flare_indexes = np.where(will_flare)[0]
        n_flares_obs = np.count_nonzero(will_flare)
        flare_durations = np.ceil(np.random.poisson(lam=flare_duration, size=n_flares_obs) * (1000/obsp) )
        
        #Create a blank dataset longer than given data, so that the flares can impact the start and end of the observation
        blank_lc = np.ones(2*len(lc))
        for flare_event in range(n_flares_obs):
            #Remove data during flares
            blank_lc[flare_indexes[flare_event] : 
               int(flare_indexes[flare_event]+flare_durations[flare_event]+1.)] = np.nan
        #Remove data from original dataset during flares
        isnanarr = np.isnan(blank_lc)[int(len(lc)*0.5) : int(len(lc) + len(lc)*0.5)]
        lc[isnanarr] = np.nan
    
def get_percentiles(fLtL0data, LtL0noisearray):
    """Function that returns percentile and p value of Lt/L0data with respect to the Lt/L0noise distribution.
    Inputs: data Lt/L0 value (fLtL0data), array of noise Lt/L0s (LtL0noisearray)
    Outputs: Percentile of data Lt/L0, p value of data Lt/L0 
    """
    
    percentgrid = np.logspace(-3, 2, 10000) #Logspace for all percentile values
    noisePercentilevals = np.percentile(LtL0noisearray, percentgrid) #Gives array of values from LtL0noisearray where percentile is pgrid
    func_interp = scipy.interpolate.interp1d(noisePercentilevals, percentgrid) #Interpolate percentile values and percentiles
    percentile = func_interp(fLtL0data) #Percentile of Lt/L0data with respect to Lt/L0noise distribution
    
    fpval = percentile / 100
    return percentile, fpval

def get_chisq(model, fdata, ferror):
    """Function that returns chi squared for model and data with error.
    Inputs: model, data (fdata), error (ferror)
    Outputs: chi-squared value (float)
    """
    nan_mask = np.isnan(fdata)
    modeldiff = fdata[~nan_mask]-model[~nan_mask]
    return np.nansum(modeldiff**2 / ferror[~nan_mask]**2)
    
def get_Lt_noise(fnoise, tab, nstack, cts):
    """Function that returns Lt for a noise array.
    Inputs: noise (fnoise), table with transit model (tab), number of stacked transits (nstack), counts (cts)
    Outputs: Chi-squared (Lt) between noise set and transit model (float)
    """
    #Calculate error of noise
    noiseerror = np.sqrt(fnoise) / (np.sqrt(nstack) * np.sqrt(cts))
    noiseerror[noiseerror==0] = 1.0
    
    #Calculate Lt (chi-squared)
    Lt = get_chisq(fnoise, tab['model2'], noiseerror)
    return Lt

def get_L0_noise(fnoise, nstack, cts):
    """Function that returns L0 for a noise array.
    Inputs: noise (fnoise), number of stacked transits (nstack), counts (cts)
    Outputs: Chi-squared (L0) between noise set and constant count rate set at 1 (float)
    """
    #Calculate error of noise
    noiseerror = np.sqrt(fnoise) / (np.sqrt(nstack) * np.sqrt(cts))
    noiseerror[noiseerror==0] = 1.0
    
    #Calculate L0 (chi-squared)
    linfunc = np.ones(len(fnoise))
    L0 = get_chisq(fnoise, linfunc, noiseerror)
    return L0

def get_h(Nscale, T, st_mass, st_rad):
    """Function that calculates the coronal scale height in batman units.
    Inputs: Number of coronal scale heights (Nscale), coronal temperature (T), stellar mass (st_mass), stellar radius (st_rad)
    Outputs: Batman coronal scale height (float)
    """
    kcgs = 1.381e-16 #erg/K
    mu = 1.27
    mH = 1.67e-24 #g
    G = 6.674e-8 #cm^3/gs^2
    Msun = 1.989e33 #g
    Rsun = 6.957e10 #cm
    
    m = st_mass * Msun #g
    r = st_rad * Rsun #cm
    
    gacc = (G * m) / r**2 #grav acceleration at surface
    h = (kcgs * T) / (mu * mH * gacc) #scale height in cgs units
    hr = h / r #Scale height in r* units
    hb = (1-(1/(1+Nscale*hr)))/Nscale #Coronal scale height in batman units (for limb darkening calc)
    return hb
    
def get_chisq_fit_const(b, fdata, ferror, tab, outoftransitonly=True):
    """Function that returns chi squared for data&constant, used for fitting constant count rate. If outoftransitonly is set to True, only out-of-transit data points are used.
    Inputs: Constant count rate being fitted (b), data (fdata), error (ferror), model (model), bool to select out of transit points only (outoftransitonly)
    Outputs: Chi-squared between constant count rate and data (float)
    """
    if outoftransitonly:
        selection = tab['model'] == 1.0
    else:
        selection = np.ones_like(tab['model'], dtype=bool) #creating an array of truth values in the shape of tab['model']
    pts = np.ones(len(tab[selection])) * b[0]
    return get_chisq(pts, fdata[selection], ferror[selection])
   
   
def get_chisq_fit(pars, fixedpars, Nscale, nBins, tab, fdata, ferror):
    """Function that returns chi squared for model&fit - FOR FITTING X-RAY TRANSIT MODEL.
    Inputs: parameter(s) to fit (pars), fixed parameters (fixedpars), number of coronal scale heights in model (Nscale), number of bins/data pts (60), table with model (tab), data (fdata), error (ferror) 
    Outputs: Chi-squared between data and batman transit model with parameters from fixedpars and varying pars (float)
    """
    #Define new batman model using fitting parameters
    rp = pars
    paramarray = fixedpars
    
    param = batman.TransitParams()
    param.t0 = paramarray[0]
    param.per = paramarray[1]
    param.rp = rp
    param.a = paramarray[2]
    param.inc = paramarray[3]
    param.ecc = paramarray[4]
    param.w = paramarray[5]
    param.limb_dark = "custom"
    param.u = [0]*6
    
    hfit = get_h(Nscale, paramarray[6], paramarray[7], paramarray[8]) #Getting scale height
    
    param.u[0] = get_intVals(hfit, Nscale)   #get limb darkening vals (Nscale is defined already as 6)
    
    modelfit = mod.light_curve(param) #fluxes
    
    modelfit = modelfit.reshape(nBins,21) #Binning like original model
    modelfit = modelfit.mean(axis=1)

    tab['modelfit'] = modelfit #Putting new model in table
   
    return get_chisq(tab['modelfit'], fdata, ferror)
    
def pop_noise_data(arr, cts, nBins):
    """Function that generates an array of Poissonian noise normalized at 1.
    Inputs: Array to fill with noise (arr), counts (cts), number of bins (nBins)
    Outputs: Array of noise with dimensions of nBins
    """
    arr = np.random.poisson(cts, nBins) / cts #Poissonian noise
    arr[arr<0] = 0.0
    return arr
    
def get_noise_data(nstack, cts, model, nBins, doFlare, LEOparams = (2.5, 55, 43), doLEO=args.LEO, flareparams=(flarerate, flaredur)):
    """Function that creates noise and data arrays.
    Inputs: Number of stacked transits (nstack), counts (cts), transit model (model), number of bins (nBins)
    Outputs: Array of all noise (multidimensional) and all data (multidimensional)
    """
    allnoise = np.empty([nstack,nBins]) #Empty array for all noises - to be avged later
    allnoises = np.apply_along_axis(arr=allnoise, func1d=pop_noise_data, axis=1, cts=cts, nBins=nBins) #Pop_noise_data for each row
    alldata = allnoises*model #Data = noise*model

    #Remove data to account for low earth orbit if specified:
    if doLEO:
        LEO(alldata, startrange = LEOparams[0], obsdur = LEOparams[1], noobsdur = LEOparams[2])
        
    #Remove data to account for flares if specified:
    if doFlare:
        flare(alldata, flareparams[0], flareparams[1])
        
    return allnoises, alldata

def integrand(z, x, tau):
    """Function defining the integrand for the integration"""
    return np.exp(  -tau * np.hypot(z, x)  )

@np.vectorize
def integration(y, Rx, h): #xIn is y
    return quad(integrand, np.sqrt(Rx*Rx - y*y), np.inf, args=(y, 1./h) )[0]

def get_intVals(h, Nscale):
    """Function to generate a new model for the intensity profile of the coronal emission"""
    #define Rx (disc edge in x coordinates) from h and Nscale
    Rx = 1. - ( Nscale * h )
    #define number of steps for the integration
    Nsteps = 10001
    #set up x array for 1001 steps from x=0 to x=1
    x = np.linspace(0, 1, Nsteps)
    #two sections of the x arrays
    xIn, xOut =  x[x<=Rx], x[x>Rx]
    #calculate the integral for the current step inside photospheric disc
    wIdisc = integration(xIn, Rx, h)
    #secondly for x > Rx
    noDisc = 2*xOut*kn(1, xOut/h)
    #combine the two
    vals = np.append(wIdisc, noDisc)

    ##add in extra point at discontinuity into outside of disc arrays##
    #extra value in x array
    xOutX = np.insert(xOut, 0, Rx)
    #insert 2 times last value in wIdisc array - value for outside of disc at discontinuity
    noDiscX = np.insert( noDisc, 0, (wIdisc[-1] * 2) ) 
    #get new vals array where the boundary discontinuity value is replaced with midpoint of the two values for that point (given by the two calculations one can do at that point)
    vals[ len(xIn)-1 ] = (noDiscX[0] + wIdisc[-1]) / 2.

    ##Find the normalization constant by integrating over each section and summing##
    #within disc#
    #get the difference between each x array value
    wIdx = np.diff(xIn)
    #area under curve using sum of trapezium areas
    areaIn = np.trapz(wIdisc*xIn * (2*np.pi), x=xIn)    #np.sum( 0.5 * ( wIdisc[1:] + wIdisc[:-1] ) * wIdx )
    #outside disc#
    #diff between each x value
    oTdx = np.diff(xOutX)
    #area under curve using sum of trapezium areas
    areaOut = np.trapz(noDiscX*xOutX * (2*np.pi), x=xOutX)    #np.sum( 0.5 * ( noDiscX[1:] + noDiscX[:-1] ) * oTdx )
    #total area under curve
    totArea = areaIn + areaOut
    
    #get the normalized values
    norm = vals / totArea
    #return the normalized values
    return norm

#Indexing by planet name
tabPlanets.add_index('pl_name')

#define system parameters based on input planet 
Rp = tabPlanets.loc[pl_name]['pl_rade'] * u.earthRad.to(u.cm) * radchange    #planet radius in optical
Rs = tabPlanets.loc[pl_name]['st_rad'] * u.Rsun.to(u.cm)          #stellar radius
a = tabPlanets.loc[pl_name]['pl_orbsmax'] * u.au.to(u.cm)   #semi-major axis 
h = tabPlanets.loc[pl_name]['hb']                 #initial scale height
t0 = 1              #initial central phase

#intial photosphere edge in x coordinates
Rx = 1 - (Nscale * h)
#initial integral values
intVals = get_intVals(h, Nscale)

#set up the object to store the transit parameters
params = batman.TransitParams()
params.t0 = t0                 #time of inferior conjunction
params.per = 1		       #orbital period - set to 1 to keep within phase definitions
params.rp = Rp * Rx / Rs   #planet radius (in units of stellar radii)
params.a = a * Rx / Rs         #semi-major axis (in units of stellar radii)
params.inc = tabPlanets.loc[pl_name]['pl_orbincl']         #orbital inclination (in degrees)
params.ecc = tabPlanets.loc[pl_name]['pl_orbeccen']                #eccentricity
params.w = tabPlanets.loc[pl_name]['pl_orblper']                 #longitude of periastron (in degrees)
params.limb_dark = "custom"    #limb darkening model
params.u = [0]*6               #limb darkening coefficients
params.u[0] = intVals
 
#Setting up model, binning to 60 to match noise (scaling down high-res)
phaSt, phaFi = 0.9, 1.1 #phases to model
numBins = 60 #bins
binPhases = np.linspace(phaSt, phaFi, numBins) #bin centers 
sampRate = binPhases[1] - binPhases[0] #sampling rate
mSt, mFi = phaSt - (sampRate/2), phaFi + (sampRate/2) #start of first bin, end of last bin
mFactor = 21 #how much higher resolution the model will be calculated at - MUST BE AN ODD NUMBER! (else high res time array won't work properly!!!)
eitSide = (mFactor - 1) / 2 #num of data points either side of each point in data
offsets = (sampRate/mFactor) * np.arange(-eitSide, eitSide+0.5) #offsets about data times
tHR = np.sort(  np.concatenate( binPhases + offsets[:,None] )  ) #high resolution time array
tBin = np.concatenate( np.ones(mFactor) * binPhases[:,None] ) #bin phase associated with each high res phase
tab = Table( [tHR, tBin], names=('tHR','tBin'), dtype=('f8','f8') ) #table of model bins and hr phases 
####set up the batman model####
mod = batman.TransitModel(params, tab['tHR']) #initialize a batman model at these times
tab['model'] = mod.light_curve(params) #fluxes 

tabGrp = tab.group_by('tBin') #group the table
tabBin = tabGrp.groups.aggregate(np.mean) #bin the table

#Getting noise, 'data', other info
obsp = 500. #observation period (sec)
counts = tabPlanets.loc[pl_name]['Count Rate'] * obsp #counts per bin = count rate * bin observation period
print('Counts:', counts)

#Remove datapoints to account for flares
doFlare = False
if flarerate != 0:
    doFlare=True
    
#Getting noise and data arrays
allnoise, alldata = get_noise_data(numstack, counts, tabBin['model'], numBins, doFlare=doFlare)

nan_count = np.count_nonzero(np.isnan(alldata), axis=0)
noise = np.nanmean(allnoise, axis=0) #Avg all noise arrays for stacked transit

def divide(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0

#Make array of errors, taking NaNs into account
error = divide(np.sqrt(noise), (np.sqrt(numstack - nan_count) * np.sqrt(counts))) #error array
error[error==0] = 1.0 #Set errors at 0 points to 1
    
data = np.nanmean(alldata, axis=0) #Stacking transit data

#Fitting for constant count rate
constfitguess = [1] #Finding best fit horizontal line
constfit = scipy.optimize.minimize(get_chisq_fit_const, constfitguess, args=(data,error,tabBin)) #Best fit horizontal line

chisqdatanotran = get_chisq(data, np.ones(len(data)) * constfit.x[0], error) #Get chi squared from data and no transit (best fit constant count rate)

chisqmodel = get_chisq(tabBin['model'], data, error) #Get chi squared from data and model

#Add data and error to table
tabBin['data1'] = data
tabBin['error1'] = error

#Print model info
print('MODEL INFO - - - - - - - - - - - - - - - - -')
print('Chi Squared (no transit - data): %s' % (chisqdatanotran))
print('Chi Squared (model - data): %s' % (chisqmodel))

plt.rc('xtick', labelsize=12)# Set the font size for y tick labels
plt.rc('ytick', labelsize=12)# Set the legend font size

if not fitbool: #If no fit, just plot the transit data and model
    print('Model Transit Depth:', 100*(1-min(tabBin['model'])))
    
    #Plot coronal intensity profile
    plt.plot(np.linspace(0, 1, len(intVals)), intVals, color='navy') 
    plt.axvline(Rx, color='navy', ls='dashed', label='Rx')
    plt.xlabel('X')
    plt.ylabel('Relative Coronal Intensity')
    plt.title('%s 1D Coronal Intensity Profile' % pl_name)
    plt.legend()
    plt.savefig('%s_intVals.pdf' % pl_name )
    plt.show()

    #Plot transit model and data
    plt.errorbar(tabBin['tBin'], data, yerr=error, ls='', marker='.', label='Simulated Data', color='gray') #Data
    plt.xlabel("Phase")
    plt.ylabel("Normalized Counts")

    plt.plot(tabBin['tBin'], tabBin['model'], label='Original model', color='navy') #Original model

    plt.legend()
    if figsave:
        plt.savefig('%s_%sStack_Transit_Fig_%sRad_%s.pdf' % (pl_name, numstack, radchange, scope)) #Save figure if figsave=Y
    plt.show()

if fitbool: #If fitting:

    #Fit the data with a model 
    guess = [Rp * Rx / Rs] #Fit planet radius, in batman coords
    
    #Fixed model parameters:
    fixedparams = [t0, 1, a * Rx / Rs, tabPlanets.loc[pl_name]['pl_orbincl'], tabPlanets.loc[pl_name]['pl_orbeccen'], tabPlanets.loc[pl_name]['pl_orblper'], 
    tabPlanets.loc[pl_name]['Temp'], tabPlanets.loc[pl_name]['st_mass'], tabPlanets.loc[pl_name]['st_rad']] 

    #use scipy.optimize to minimize chi-squared:
    fit = scipy.optimize.minimize(get_chisq_fit, guess, args=(fixedparams, Nscale, numBins, tabBin, data, error)) 

    paramsfit = batman.TransitParams() #Setting up a new model using the parameters found in the fit
    paramsfit.t0 = fixedparams[0]
    paramsfit.per = fixedparams[1]
    paramsfit.rp = fit.x[0]
    paramsfit.a = fixedparams[2]
    paramsfit.inc = fixedparams[3]
    paramsfit.ecc = fixedparams[4]
    paramsfit.w = fixedparams[5]
    paramsfit.limb_dark = 'custom'
    paramsfit.u = [0]*6
    
    hfitted = get_h(Nscale, fixedparams[6], fixedparams[7], fixedparams[8]) #Setting scale height for fit
    
    paramsfit.u[0] = get_intVals(hfitted, Nscale) #Setting limb darkening coefficients for fit
    
    model2 = mod.light_curve(paramsfit) #fit fluxes 
    
    model2 = model2.reshape(numBins,21) #Rebinning to match original length
    model2 = model2.mean(axis=1)

    tabBin['model2'] = model2 #Adding fit model to table
    
    chisqdatafit = get_chisq(tabBin['model'], data, error) #calculate chi-squared
    
    smax = fixedparams[2] / (Rx / Rs) * u.cm.to(u.au) #Converting semi major axis to au for printing
    rpfit = paramsfit.rp / (Rx/Rs) * u.cm.to(u.earthRad) #Converting fitted planet radius to earth radii for printing
    rpmodel = params.rp / (Rx/Rs) * u.cm.to(u.earthRad) #Converting literature planet radius to earth radii for printing
    kTCorTemp = fixedparams[6] * 8.617e-5 / 1000 #Converting temp to keV for printing
    
    print('FIT INFO - - - - - - - - - - - - - - - - -')
    print('Chi Squared (data - fit): %s' % (chisqdatafit))
    print('Orbital period: %s days \n Best-fit planet radius: %s earth radii, %s \n Semi-major axis: %s au' % (fixedparams[1], rpfit, rpmodel, smax))
    print('Inclination: %s deg \n Eccentricity: %s \n Longitude of periastron: %s deg' % (fixedparams[3], fixedparams[4], fixedparams[5]))
    print('Coronal scale height: %s \n Stellar mass: %s solar masses\n Stellar radius: %s solar radii \n Coronal temp: %s keV' % (hfitted, fixedparams[7], fixedparams[8], kTCorTemp))

    #Plotting best fit
    plt.errorbar(tabBin['tBin'], data, yerr=error, ls='', marker='.', label='Simulated Data', color='gray') #Data
    plt.xlabel("Phase")
    plt.ylabel("Normalized Counts")
    plt.plot(tabBin['tBin'], tabBin['model2'], color='purple', label='Best fit to data', ls='dashed') #Best fit
    plt.plot(tabBin['tBin'], tabBin['model'], label='Original model', color='navy') #Original model
    plt.legend()
    if figsave:
        plt.savefig('%s_%sStack_Transit_Fig_Fit_%sRad_%s.pdf' % (pl_name, numstack, radchange, scope)) #Save figure if figsave=T
    plt.show()

    ######Fit and transit analysis: Calculate Likelihood ratios######
    
    LtL0data = chisqdatafit / chisqdatanotran #Lt/L0 for data = Chisq data-fit / chisq data-notransit
    
    #Getting Lt/L0 distribution for noise
    numdist = 1000000 #Number of Lt/L0-s for distribution
    
    #Making numdist*numstack noise distributions
    allnoisedist, alldatadist = get_noise_data(numdist*numstack, counts, 1, numBins, doFlare=False, doLEO=False) 
    
    #Stacking noise
    allnoisedistr = allnoisedist.reshape(numdist, numstack, numBins)
    allnoisedist = np.average(allnoisedistr, axis=1) #This is an array of numdist numbins-length noise arrays (every numstack was averaged)
 
    #Calculating error for noise
    noiseerror = np.sqrt(allnoisedist) / (np.sqrt(numstack) * np.sqrt(counts))
    noiseerror[noiseerror==0] = 1.0
    
    #Calculating chi-squared for noise-model
    modeldifferenceFit = allnoisedist - tabBin['model']
    chisqnoisefitarray = np.sum(modeldifferenceFit**2 / noiseerror**2, axis=1)
    
    #Calculating chi-squared for noise-nulltransit
    modeldifferenceNoise = allnoisedist - np.ones(len(tabBin['model']))
    chisqnoiseonearray = np.sum(modeldifferenceNoise**2 / noiseerror**2, axis=1)
    
    LtL0noise = chisqnoisefitarray / chisqnoiseonearray #Lt/L0 for noise
    LtL0noise_stdev = np.std(LtL0noise) 
    LtL0noise_median = np.median(LtL0noise)

    #Printing Lt/L0 comparison info
    print('FIT ANALYSIS - - - - - - - - - - - - - - - - -')
    print('Lt/L0 for data:', LtL0data)
    print('Mean Lt/L0 for noise: %s \n Upper/lower 1 stdev Lt/L0: %s, %s' % (np.mean(LtL0noise), np.mean(LtL0noise) + LtL0noise_stdev, np.mean(LtL0noise - LtL0noise_stdev))) 
    
    if ((LtL0noise_stdev != 0) & (min(tabBin['model2']) < 1.0)): #If the fit has a transit
        num_stdev_diff = ((np.mean(LtL0noise) - LtL0data) / LtL0noise_stdev)
        print('# of stdev away:', ((np.mean(LtL0noise) - LtL0data) / LtL0noise_stdev))
        print('% of stdev to 0:', ((((np.mean(LtL0noise) - LtL0data) / LtL0noise_stdev) / (np.mean(LtL0noise)/ LtL0noise_stdev)) * 100) )
        print('Stdev from 0:', (np.mean(LtL0noise) / LtL0noise_stdev))
        
        #Plotting histogram of LtL0noise for comparison to LtL0data
        print('Lt, L0', chisqdatafit, chisqdatanotran, LtL0data)
        plt.hist(LtL0noise, 100, color='navy')
        plt.axvline(x=LtL0data, color='deepskyblue', label='Lt/L0 for data-fit')

        plt.xlabel('Lt/L0')
        plt.ylabel('# Of occurrences')
        plt.legend()
        plt.title('Lt/L0 Noise Distribution')
        if figsave:
            plt.savefig('%s_LtL0_%sStack_%sRad.pdf' % (pl_name, numstack, radchange), bbox_inches='tight')
        plt.show()

        #Calculate p-value:
        try:
            LtL0Percentile, pval = get_percentiles(LtL0data, LtL0noise)
        except ValueError:
            print('Value outside percentile range, set to minimum value')
            LtL0Percentile = 10**-3
            pval = 10**-5
            
        if ((num_stdev_diff < 0) | (num_stdev_diff > 10) | ((LtL0noise_stdev / max(LtL0noise)) < 0.001)) & (pval == 10**-5):
            pval = 1
            LtL0Percentile = 100
            
        print('Percentile: %s \n p-value: %s' % (LtL0Percentile, pval))
        
       
    
    else:
        print('fit is no transit, Lt/L0 data = Lt/L0 noise')
        
