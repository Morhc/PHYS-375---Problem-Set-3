import os
import numpy as np
from matplotlib import pyplot as plt, rc

rc('text', usetex=True)

def compute_theta(x, n):
    """Compute the value of theta and its derivative with basic ODE solving.
    INPUTS:
        x - The values to evalute theta at.
        n - The power of theta.
    OUTPUTS:
        theta - An array of the various thetas at x.
        dtheta - The derivative of theta wrt x.
    """

    theta = np.zeros(x.size, dtype=float)
    dtheta = np.zeros(x.size, dtype=float)

    #boundary conditions: theta = 1 and dtheta = 0 @ x = 0
    theta[0] = 1
    dtheta[0] = 0

    dx = x[1] - x[0] #get the x-spacing
    for k in range(1, theta.size):

        #you get this by taking the derivative of the integral form of Lane-Emden equation (see written solution)
        ddtheta = -(theta[k-1]**n + ((2./x[k])*dtheta[k-1]))
        dtheta[k] = dtheta[k-1] + ddtheta * dx
        theta[k] = theta[k-1] + dtheta[k] * dx

    return theta, dtheta

def part_e(savepath=""):
    """Solves the Lane-Emden equation numerically for various n and plots it.
    INPUTS:
        savepath - Path to save the plot to. If nothing is given, it will just display.
    OUTPUTS:
        Outputs a plot with the various solutions for n = [0, 1, 2, 3, 4, 5].
    """

    #step from 0 to 10 for x
    x = np.linspace(0, 10, 1000)
    for n in range(0, 5+1):
        plt.plot(x, compute_theta(x, n)[0], label = rf'$n$ = {n}', zorder=1)

    plt.title(r'$\theta$($x$) vs $x$ for various $n$')
    plt.xlabel(r'$x$ = $\frac{r}{\alpha}$')
    plt.ylabel(r'$\theta$ = $(\frac{\rho}{\rho_c})^{\gamma-1}$')
    plt.legend(loc='upper right')
    plt.ylim(-1,1.1)
    plt.xlim(0,10)
    plt.axhline(0, color='black', zorder=0)

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath)

    plt.close('all')

def part_g():
    """Assuming n=3, Rstar = 1Rsun, and Mstar = 1Mstar; we find alpha, rho_c, and K
    INPUTS:
        savepath - Path to save the plot to. If nothing is given, it will just display.
    OUTPUTS:
        Returns alpha, rho_c, and K
    """

    n = 3
    #grabbed from p. 576 of Ryden
    R = 6.955e8
    M = 1.989e30
    G = 6.673e-11

    x = np.linspace(0, 10, 1000)
    thetas, dthetas = compute_theta(x, n)

    #get the x from the theta closest to zero
    x_0 = x[np.abs(thetas).argmin()]
    alpha = R/x_0

    #grab the dtheta/dx for theta closest to zero
    dtheta_0 = dthetas[np.abs(thetas).argmin()]
    #rearranged the equation from f (thank goodness there's a negative sign in it)
    rho_c = M/(-4*np.pi*(alpha**3)*(x_0**2)*dtheta_0)

    #we can use the alpha (and convert gamma - 2 = 1/n -1)
    K = (alpha**2)*4*np.pi*G/(np.power(rho_c, 1/n - 1)*(n+1))

    return alpha, rho_c, K, x_0

def part_h(alpha, rho_c, K, x_0, savepath=""):
    """Solves the Lane-Emden equation numerically for various n and plots it.
    INPUTS:
        alpha - scaling ratio between x and r.
        rho_c - the central density of the star.
        K - a constant for the polytropic equation of state.
        x_0 - the radius of the star in x.
        savepath - Path to save the plot to. If nothing is given, it will just display.
    OUTPUTS:
        Outputs a plot of the density, pressure, and temperature as a function of r/Rstar
    """

    X, Y = 0.55, 0.4
    mu = np.power(2*X + 3*Y/4, -1) #from Lecture 9a; assuming Z = 0

    #grabbed from p. 576 of Ryden
    R = 6.955e8
    #grabbed from p. 575 of Ryden
    mp = 1.673e-27
    k = 1.381e-23

    n = 3

    #note: r/Rstar = x/x_0, so we can plot over that scale :)
    x_reduced = np.linspace(0, x_0, 1000)/x_0

    thetas, dthetas = compute_theta(x_reduced, n)

    rho = (np.power(thetas, n)*rho_c)

    #since we assume a polytropic equation, we know P(r) = K*p(r)^(1/n + 1)
    P = K * np.power(rho, 1/n + 1)

    #we get temperature by rearranging Equation 15.79 from Ryden
    T = P*mu*mp/(k*rho)

    plt.plot(x_reduced, T, label=r'$T$($r$)')
    plt.plot(x_reduced, P, label=r'$P$($r$)')
    plt.plot(x_reduced, rho, label=r'$\rho$($r$)')

    plt.legend(loc='best')

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath)

    plt.close('all')

def main():

    here = os.path.dirname(os.path.realpath(__file__))

    e_path = os.path.join(here, 'PS3-Q3e.png')
    part_e(e_path)

    alpha, rho_c, K, x_0 = part_g()
    print(f'The solved variables are:\n\talpha = {alpha}\n\trho_c = {rho_c}\n\tK = {K}\n')

    h_path = os.path.join(here, 'PS3-Q3h.png')
    part_h(alpha, rho_c, K, x_0, h_path)

if __name__ == '__main__':
    main()
