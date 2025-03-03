Code to simulate X-ray transit observations in AXIS and NewAthena for a given planet is provided in file 'AXISTransitGenerator-InputPlanet.py'. The script can be used to simulate transits for any of the planets in the included data tables.

The included options in the script are:
1. name of planet to simulate (required)
2. Number of observations (required)
3. Whether or not to fit the data with a transit model (default False)
4. Radius multiplier (default 1)
5. Whether or not to save figures (default False)
6. Which telescope to simulate data for (default AXIS)
7. Whether or not to simulate AXIS LEO (default False)
8. Flare rate per 500 seconds (default 0)
9. Average flare duration in ks (default 0)

This script uses data from the two csv files, one for AXIS (tabPlanetsAXIS.csv) and one for NewAthena (tabPlanetsNewAthena.csv). The data files include system information:
1. planet name
2. star name
3. semi-major axis
4. planet radius
5. eccentricity
6. inclination
7. longitude of periastron
8. stellar count rate in AXIS or NewAthena
9. coronal scale height
10. stellar temperature
11. stellar mass

These parameters were gathered from the literature or calculated. Count rates were calulated using response matrices and XSPEC. More information can be found in the associated papers: Cilley et al. (2024) and King et al. (2024).



