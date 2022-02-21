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

def part_g(R, M):
    """Assuming n=3, Rstar = 1Rsun, and Mstar = 1Mstar; we find alpha, rho_c, and K
    INPUTS:
        R - The radius of the star.
        M - The mass of the star.
        savepath - Path to save the plot to. If nothing is given, it will just display.
    OUTPUTS:
        Returns alpha, rho_c, and K
    """

    n = 3
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
    """Calculates the density, pressure, and temperature as a function of r/Rstar and plots it.
    INPUTS:
        alpha - scaling ratio between x and r.
        rho_c - the central density of the star.
        K - a constant for the polytropic equation of state.
        x_0 - the radius of the star in x.
        savepath - Path to save the plot to. If nothing is given, it will just display.
    OUTPUTS:
        Outputs a plot of the density, pressure, and temperature as a function of r/Rstar.
        rho - the density as a function of r/Rstar.
        T - the temperature as a function of r/Rstar.
    """

    X, Y = 0.55, 0.4
    mu = np.power(2*X + 3*Y/4, -1) #from Lecture 9a; assuming Z = 0


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

    #plot the temperature
    plt.title(r'$T$($r/R_{star}$) vs $r/R_{star}$')
    plt.plot(x_reduced, T)
    plt.ylabel(r'$T$($r/R_{star}$) (K/m)')
    plt.xlabel(r'r/$R_{star}$')

    plt.xlim(0,1)

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath.replace('.png', '_T.png'))

    plt.close('all')

    #plot the pressure
    plt.title(r'$P$($r/R_{star}$) vs $r/R_{star}$')
    plt.plot(x_reduced, P)
    plt.ylabel(r'$P$($r/R_{star}$) (Pa/m)')
    plt.xlabel(r'r/$R_{star}$')

    plt.xlim(0,1)

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath.replace('.png', '_P.png'))

    plt.close('all')

    #plot the density
    plt.title(r'$\rho$($r/R_{star}$) vs $r/R_{star}$')
    plt.plot(x_reduced, T)
    plt.ylabel(r'$\rho$($r/R_{star}$) (kg/m$^3$ / m)')
    plt.xlabel(r'r/$R_{star}$')

    plt.xlim(0,1)

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath.replace('.png', '_rho.png'))

    plt.close('all')

    return rho, T

def part_i(rho, T, alpha, x_0, savepath=""):
    """Calculate and plot the energy generation rates and the change in luminosity over r/Rstar
    INPUTS:
        rho - the density of the star as a function of r/Rstar.
        T - the temperature of the star as a function of r/Rstar.
        alpha - the scaling coefficient between x and r.
        x_0 - the radius of the star in x.
        savepath - The path to save the plot to. If none is supplied then it is displayed.
    OUTPUTS:
        Outputs a plot of the luminosity change and energy generation rates as a function of r/Rstar
    """

    X = 0.55
    Xcno = 0.03*X

    x_reduced = np.linspace(0, x_0, 1000)/x_0

    #from Lecture 12b
    eps_pp = 1.07e-7 * (X**2) * (rho/1e5) * np.power(T/1e6, 4)
    eps_cno = 8.24e-26 * X * Xcno * (rho/1e5) * np.power(T/1e6, 19.9)

    #calculate luminosity from Equation 15.78 | assuming eps = eps_pp + eps_cno
    dL_dr = 4*np.pi*(x_reduced**2)*rho*(eps_pp + eps_cno)

    #plot the proton-proton energy generation
    plt.plot(x_reduced, eps_pp)
    plt.title(r'$\epsilon_{pp}$($r/R_{star}$) vs $r/R_{star}$')
    plt.ylabel(r'$\epsilon_{pp}$($r/R_{star}$) (W/kg / m)')
    plt.xlabel(r'r/$R_{star}$')

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath.replace('.png', '_pp.png'))

    plt.close('all')

    #plot the CNO energy generation
    plt.plot(x_reduced, eps_cno)
    plt.title(r'$\epsilon_{CNO}$($r/R_{star}$) vs $r/R_{star}$')
    plt.ylabel(r'$\epsilon_{CNO}$($r/R_{star}$) (W/kg / m)')
    plt.xlabel(r'r/$R_{star}$')

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath.replace('.png', '_cno.png'))

    plt.close('all')

    #plot dL/dr
    plt.plot(x_reduced, dL_dr)
    plt.title(r'$\frac{dL}{dr}$($r/R_{star}$) vs $r/R_{star}$')
    plt.ylabel(r'$\frac{dL}{dr}$($r/R_{star}$) (W/m / m)')
    plt.xlabel(r'r/$R_{star}$')

    if savepath == "":
        plt.show()
    else:
        plt.savefig(savepath.replace('.png', '_L.png'))

    plt.close('all')

    #Since dL = integral(4pi r^2 rho(r) * eps(r) dr) we can reasonably convert this to a sum
    u = alpha * x_0
    L = np.sum(dL_dr * u) * (x_reduced[1] * u - x_reduced[0] * u) * u
    print(f'The total luminosity for this star is: {L} Watts')



def main():

    here = os.path.dirname(os.path.realpath(__file__))

    e_path = os.path.join(here, 'PS3-Q3e.png')
    part_e(e_path)

    for r_scale, m_scale in zip([10, 0.6, 1], [20, 0.5, 1]):
        print(f'R_scale: {r_scale}; M_scale: {m_scale}')

        #grabbed from p. 576 of Ryden
        R = r_scale * 6.955e8
        M = m_scale * 1.989e30

        alpha, rho_c, K, x_0 = part_g(R, M)
        print(f'The solved variables are:\n\talpha = {alpha}\n\trho_c = {rho_c}\n\tK = {K}\n')

        h_path = os.path.join(here, 'PS3-Q3h.png')
        rho, T = part_h(alpha, rho_c, K, x_0, h_path)

        i_path = os.path.join(here, 'PS3-Q3i.png')
        part_i(rho, T, alpha, x_0, i_path)

if __name__ == '__main__':
    main()
