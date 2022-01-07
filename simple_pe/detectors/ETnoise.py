from numpy import arange

def getstrain(tech, df=2/4096.):

    f = arange(0. + df/2., 4096.+df/2., df)

    if tech == 'ETB':
        xt = f/100.;
        fit = 2.39/(10**27*xt**15.64) + 0.349/xt**2.145 + 1.76/xt**0.12 + 0.409*xt**1.1;
        strain = fit**2/10.**50;

    if tech == 'CosmicExplorer':
        strain = 10**(-50)*((f/10.5)**(-50) + (f/25.)**(-10) + 1.26*(f/50.)**(-4) + 2*(f/80.)**(-2) + 5 + 2*(f/100.)**2)

    if tech == 'LIGOBlueBird':
        strain = 8.*10**(-48)*((f/15.)**(-20.) + (f/20.)**(-8) + (f/22.)**(-4.)  + (f/3.)**(-1.) + 0.063 + 0.7*(f/2000.)**2.)

    return strain
