
import matplotlib.pyplot as plt

import cmath
import numpy as np
import scipy.optimize as optimize

# import scipy.ftpack as ft
from scipy.signal import find_peaks
from time import time
from timeit import Timer

#===========================================================================
### Plot and Axis styling params

plt.rcParams['figure.figsize']  = 15, 15
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['font.family']     = 'serif'
plt.rcParams['font.weight']     = 'bold'
plt.rcParams['font.size']       = 11
plt.rcParams['font.sans-serif'] = 'serif'
plt.rcParams['text.usetex']     = True

plt.rcParams['axes.linewidth']  = 0.7
plt.rcParams['axes.titlesize']  = 11
plt.rcParams['axes.labelsize']  = 11
plt.rcParams['legend.loc']      = 'lower right'

plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.major.pad']  = 3
plt.rcParams['xtick.minor.pad']  = 3
plt.rcParams['xtick.color']      = 'k'
# plt.rcParams['xtick.labelsize']  = 'medium'
# plt.rcParams['xtick.direction']  = 'in'

plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.major.pad']  = 3
plt.rcParams['ytick.minor.pad']  = 3
plt.rcParams['ytick.color']      = 'k'
# plt.rcParams['ytick.labelsize']  = 'medium'
# plt.rcParams['ytick.direction']  = 'in'



#===========================================================================
### Global Parameters


sig = 6.41 ##angstrom
eps = 0.8997 #kj/mol
za = 4.19 * sig ##Distance from molecule to cantilever

m = 1.0
k = np.pi**2
F0 = 1.5 * sig
Q = 20.0

b = np.sqrt(m * k) / Q
w0 = np.sqrt(k / m)
zeta = 1 / (2 * Q)

w0_res = w0 * np.sqrt(1 - 2 * zeta**2)
f0_res = w0_res / (2 * np.pi)
T0_res = 2 * np.pi / w0_res

num_points_t = int(1e3) # number of time points
num_osc = 20
# t_data = np.linspace(0, num_osc * T0_res, num_points_t, endpoint=False)
# t_step = t_data[1] - t_data[0]

delta = 0.5
w_step = 0.005
num_points_w = int(2 * delta / w_step)

print("-----------------------------------------------------------")
print(" Global Parameters: ")
print("")
print("num_points_tum time points: ", num_points_t)
print("num_points_tum freq points: ", num_points_w)
print("Resonance freq without interaction: ", w0_res)
print("")
print("-----------------------------------------------------------")


#============================================================================
### Fig and Axis Objects

fig = plt.figure()

span_r = 6 # grid span of subplots vertically
span_c = 6 # grid span of subplots horizontally

num_r = 2 # number of rows of subplots
num_c = 2 # number of columns of subplots

space = 1 # grid gap between subplots
space_r = (num_r + 1) * space
space_c = (num_c + 1) * space

gridsize = (num_r * span_r + space_r, num_c * span_c + space_c)

v_r = [(i+1)*space + i*span_r for i in range(num_r)] # row vertices of subplots
v_c = [(i+1)*space + i*span_c for i in range(num_c)] # col vertices of subplots

## Column 0
spec_ax     = plt.subplot2grid(
                gridsize, (v_r[0], v_c[0]), rowspan = span_r, colspan = span_c)
osc_ax0     = plt.subplot2grid(
                gridsize, (v_r[1], v_c[0]), rowspan = span_r, colspan = span_c)

## Column 1
frc_ax      = plt.subplot2grid(
                gridsize, (v_r[0], v_c[1]), rowspan = span_r, colspan = span_c)
osc_ax1     = plt.subplot2grid(
                gridsize, (v_r[1], v_c[1]), rowspan = span_r, colspan = span_c)

#============================================================================
### Text Strings

SPEC_TITLE = r'$\mathbf{A \; vs \; \omega \; \; Spectrum}$'
SPEC_X_LABEL = r'$\mathrm{\omega \; (rad/s)}$'
SPEC_Y_LABEL = r'$\mathrm{z/\sigma}$'

SPEC_LABEL_0 = r'$w_{res} \; with \; interaction$'
SPEC_LABEL_1 = r'$w_{res} \; without \; interaction$'


OSC0_TITLE = r'$\mathbf{Frequency \; Effect \; On \; Steady \; State \; Oscillations}$'
OSC0_X_LABEL = r'$\mathrm{t \; (s)}$'
OSC0_Y_LABEL = r'$\mathrm{z/\sigma}$'


FRC_TITLE = r'$\mathbf{Interaction \; Force}$'
FRC_X_LABEL = r'$\mathrm{t \; (s)}$'
FRC_Y_LABEL = r'$\mathrm{F \sigma / \epsilon}$'

FRC_LABEL_0 = r'$\mathrm{analytical}$'
FRC_LABEL_1 = r'$\mathrm{iFFT}$'


OSC1_TITLE = r'$\mathbf{Interaction \; Effect \; On \; Steady \; State \; Oscillations}$'
OSC1_X_LABEL = r'$\mathrm{t \; (s)}$'
OSC1_Y_LABEL = r'$\mathrm{z/\sigma}$'

OSC1_LABEL_0 = r'$\mathrm{without \; interaction}$'
OSC1_LABEL_1 = r'$\mathrm{with \; interaction}$'


#============================================================================
### Axis titles and labels

# Spectrum Axis
spec_ax.set_title(SPEC_TITLE)
spec_ax.set_xlabel(SPEC_X_LABEL)
spec_ax.set_ylabel(SPEC_Y_LABEL)

# Oscillation_0 Axis
osc_ax0.set_title(OSC0_TITLE)
osc_ax0.set_xlabel(OSC0_X_LABEL)
osc_ax0.set_ylabel(OSC0_Y_LABEL)

# Interaction Force Axis
frc_ax.set_title(FRC_TITLE)
frc_ax.set_xlabel(FRC_X_LABEL)
frc_ax.set_ylabel(FRC_Y_LABEL)

# Oscillation_1 Axis
osc_ax1.set_title(OSC1_TITLE)
osc_ax1.set_xlabel(OSC1_X_LABEL)
osc_ax1.set_ylabel(OSC1_Y_LABEL)


#============================================================================
### Module Functions

def amplitude(w, F):
    """
    Arguments:
        F   : arraylike, Driving Force amplitudes
        w   : arraylike, Driving angular frequencies

    Note:
        All input parameters must of same shape
    """
    global m, w0, Q

    Z = np.sqrt((w0 * w / Q)**2 + (w0**2 - w**2)**2)

    A = F / (m * Z)
    return A


def driving_force(F, w, ph, t):
    """
    Arguments:
        F   : arraylike, Driving Force amplitudes
        w   : arraylike, Driving angular frequencies
        ph  : arraylike, Driving force phase
        t   : arraylike, time

    Note:
        All input parameters must of same shape
    """
    force = F * np.cos(w * t + ph)
    return force


def theta(w):
    """
    Arguments:
        w   : arraylike, Driving angular frequencies
    """
    global w0, Q

    th = np.arctan2((w0 * w / Q), (w0**2 - w**2))
    return th


def steady_state_z(F, w, ph, t):
    """
    Arguments:
        F   : arraylike, Driving Force amplitudes
        w   : arraylike, Driving angular frequencies
        ph  : arraylike, Driving force phase
        t   : arraylike, time

    Note:
        All input parameters must of same shape
    """
    A   = amplitude(w, F)
    th  = theta(w) - ph

    z = A * np.cos(w * t - th)
    return z


def interaction_force(z):
    """
    Arguments:
        z   : arraylike, relative position values
    """
    global sig, eps, za

    r_hat   = (z + za) / sig
    F_0     = 24 * eps / sig

    F_repulsive     = 2 * F_0 * (1/r_hat)**13
    F_attractive    = -1 * F_0 * (1/r_hat)**7

    return F_attractive + F_repulsive

def fft_spectrum(z, t_step):

    N = num_points_t
    y = np.fft.fft(z,axis=-1)[:,:,0:N//2] # select only positive freq parts

    ph = np.angle(y) # calculate phase
    y = 2 * np.abs(y) / N # calculate amplitude

    # Remove noise from FFT signal
    threshold = np.max(y) / 1e10
    y = y - threshold
    y = np.where(y < 0, 0, y)

    f = np.fft.fftfreq(N, d=t_step)[:,:,0:N//2]
    w = 2 * np.pi * f

    return w, y, ph


def pert_time_evol(w):
    print(np.shape(w))
    """
    Arguments:
        w          : arraylike, driving frequency[axis=0]
        t          : time array                  [axis=2]
        peak_index : fft peaks of positions      [axis=1]

    We are using
    array vector==[FREQUENCY][PEAKS][TIME]
    """
    global F0, sig, eps
    global num_points_t, num_osc

    T = 2 * np.pi / w   # Time Period

    t_start = np.zeros_like(w)
    t_end = num_osc * T

    # calculate time points for each w
    # (Nw x Nt) 3D array
    t = np.linspace(t_start, t_end , num_points_t, endpoint=False, axis=-1)
    # Reshape w, t to compute operations vectorially
    w = w.reshape(w.shape[0], 1, 1)
    t = t.reshape(t.shape[0], 1, t.shape[1])
    t_step = t[:,:,1]-t[:,:,0]
    t_step = t_step.reshape(t_step.shape[0],t_step.shape[1],1)
    # Find steady state z0.
    # (Nw x 1 x Nt) 3D array
    z0 = steady_state_z(F0, w, 0., t) # initial driving force phase is zero
    # Find first order interaction force.
    # (Nw x 1 x Nt) 3D array
    force = interaction_force(z0)
    # Defining number of peaks to consider for FT calculation
    w_fft, y_fft, ph_fft = fft_spectrum(force, t_step)
    ####################################################################################################33
    """
    Finding peaks:

    calculating w[i] manually
    w[i]=i*2*pi/(n*dt)
    n*dt=numosc*2*pi/w
    w[i]=i*w/numosc
    For multiples of w, 2w , 3w... Mw etc
    Peak indices, I=M*numosc
    constructing three peaks for I, I+1, I-1
    keeping M till 12


    """
    ####################################################################################################33333
    ########################################################################################################33
    tp=12####total peaks
    peak_indices=np.arange(0,3*tp+2)*num_osc
    peak_indices[0:tp+1]=np.arange(0,tp+1)*num_osc
    peak_indices[tp+1:2*tp+2]=np.arange(0,tp+1)*num_osc+1
    peak_indices[2*tp+2:3*tp+2]=np.arange(1,tp+1)*num_osc-1
    peak_index   = peak_indices.reshape(1,peak_indices.shape[0],1)

    ft_amp = y_fft[:,:,peak_index]
    ft_w = w_fft[:,:,peak_index]
    ft_ph = ph_fft[:,:,peak_index]
    ft_amp = ft_amp.reshape(ft_amp.shape[0],ft_amp.shape[3],1)
    ft_w = ft_w.reshape(ft_w.shape[0],ft_w.shape[3],1)
    ft_ph = ft_ph.reshape(ft_ph.shape[0],ft_ph.shape[3],1)
    ft_amp[:,0,:] = ft_amp[:,0,:]/2

    force_ifft = np.sum(driving_force(ft_amp, ft_w, ft_ph, t),axis=1)

    # Calculate firt order correction to z
    # (Nw x Nt) 2D array
    z_corr = steady_state_z(ft_amp, ft_w, ft_ph, t)
    z_corr = np.sum(z_corr, axis=1) # sum along w' axis


    # Removing extra dimensions
    z0      = z0[:, 0, :]     # (Nw x Nt) 2D array
    t       = t[:, 0, :]      # (Nw x Nt) 2D array
    force   = force[:, 0, :]  # (Nw x Nt) 2D array

    # All return values are (Nw x Nt) 2D array
    return t, z0, z_corr, force, force_ifft


#============================================================================
### Calculating solution by perturbation

print("-------------------------------------------------------")
print("## Calculating solution by perturbation")

t_i = time()


w_sweep = np.linspace(w0_res-delta, w0_res+delta, num_points_w)

# All variables are (Nw x Nt) 2D arrays
t_23= time()
t, z0, z_corr, force, force_ift = pert_time_evol(w_sweep)
t_32=time()
# Calculate steady state solution with interaction
z = z0 + z_corr

# Calculate max amplitude for each frequency w
# It is (Nw,) 1D array
amplitudes = (np.amax(z, axis=-1) - np.amin(z, axis=-1)) / 2


# Calculate resonance frequency and amplitude
res_idx = np.argmax(amplitudes)
w_res   = w_sweep[res_idx]
res_amp = amplitudes[res_idx]


# Calculate shift in resonance frequency fue to interaction
delta_w     = w_res - w0_res
percent_w   = (delta_w / w0_res) * 100

delta_w     = np.round(delta_w, 4)
percent_w   = np.round(percent_w, 4)



t_f = time()
calc_time = t_f - t_i

print("Calculation Time: ", calc_time)
print("MainCalculation",t_32-t_23)
print("-------------------------------------------------------")


#============================================================================
### Plotting A vs w Spectrum

print("-------------------------------------------------------")
print("## Plotting A vs w Spectrum")

t_i = time()


A_vs_w_line, = spec_ax.plot([], [], 'b')
A_vs_w_line.set_data(w_sweep, amplitudes / sig)


x_min, x_max = w0_res - 0.5*delta, w0_res + 0.5*delta
y_min, y_max = 0., 1.2 * res_amp / sig

spec_ax.set_xlim(xmin=x_min, xmax=x_max)
spec_ax.set_ylim(ymin=y_min, ymax=y_max)

spec_ax.axvline(x=w_res, c='b', ls='--', label=SPEC_LABEL_0)
spec_ax.axvline(x=w0_res, c='k', ls='--', label=SPEC_LABEL_1)


text = (r"$\mathrm{{\Delta \omega_{{resonace}}}} \; = \; {0} \\"
       + r"\mathrm{{ \% \Delta \omega}} \; = \; {1} \%$").format(
        delta_w, percent_w)

spec_ax.text(
    0.15, 0.85, text, transform = spec_ax.transAxes,
    bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})

spec_ax.legend(loc='upper right')
spec_ax.grid()



t_f = time()
plot_time = t_f - t_i

print("Plot Time: ", plot_time)
print("-------------------------------------------------------")

#============================================================================
### Plotting Oscillations for different frequencies

print("-------------------------------------------------------")
print("## Plotting Oscillations for different frequencies")

t_i = time()


indices = np.array([0, res_idx, -1])

for k, index in enumerate(indices):
    z_k = z[index][:] / sig
    t_k = t[index][:]
    w_k = w_sweep[index]
    osc_ax0.plot(t_k, z_k, label=(r'$\mathrm{{\omega\; : {0}}}$').format(w_k))


x_min, x_max = 0., 5 * T0_res
y_min, y_max = -1.2 * res_amp / sig, 1.2 * res_amp / sig

osc_ax0.set_xlim(xmin=x_min, xmax=x_max)
osc_ax0.set_ylim(ymin=y_min, ymax=y_max)

osc_ax0.legend(loc='upper right')
osc_ax0.grid()



t_f = time()
plot_time = t_f - t_i

print("Plot Time: ", plot_time)
print("-------------------------------------------------------")


#============================================================================
### Plotting Oscillations with and without interaction at resonace

print("-------------------------------------------------------")
print("## Plotting Oscillations with and without interaction at resonace")

t_i = time()


t_res   = t[res_idx][:]
z0_res  = z0[res_idx][:] / sig
z_res   = z[res_idx][:] / sig

osc_ax1.plot(t_res, z0_res, 'b', label=OSC1_LABEL_0)
osc_ax1.plot(t_res, z_res, 'r-.', label=OSC1_LABEL_1)

x_min, x_max = 0., 5 * T0_res
y_min, y_max = -1.2 * res_amp / sig, 1.2 * res_amp / sig

osc_ax1.set_xlim(xmin=x_min, xmax=x_max)
osc_ax1.set_ylim(ymin=y_min, ymax=y_max)

osc_ax1.legend(loc='upper right')
osc_ax1.grid()



text = (r"$\mathrm{{\omega_{{drive}} \; = \;"
        + r"\omega_{{resonace}}}} \; = \; {0} $").format(np.round(w_res, 4))

osc_ax1.text(
    0.65, 0.15, text, transform = osc_ax1.transAxes,
    bbox={'facecolor':'white', 'alpha':1.0, 'pad':10})



t_f = time()
plot_time = t_f - t_i

print("Plot Time: ", plot_time)
print("-------------------------------------------------------")


#============================================================================
### Plotting Interaction Force and its inverse FFT


print("-------------------------------------------------------")
print("## Plotting Interaction Force and its inverse FFT")

t_i = time()


t_res       = t[res_idx][:]
frc_res     = force[res_idx][:] * sig / eps
frc_ift_res = force_ift[res_idx][:] * sig / eps

frc_ax.plot(t_res, frc_res, 'orange', label=FRC_LABEL_0)
frc_ax.plot(t_res, frc_ift_res, 'g-.', label=FRC_LABEL_1)

frc_res_amp  = np.amax(np.abs(frc_res))

x_min, x_max = 0., 5 * T0_res
y_min, y_max = -1.2 * frc_res_amp, 1.2 * frc_res_amp

frc_ax.set_xlim(xmin=x_min, xmax=x_max)
frc_ax.set_ylim(ymin=y_min, ymax=y_max)

frc_ax.legend(loc='upper right')
frc_ax.grid()


text = (r"$\mathrm{{\omega_{{drive}} \; = \;"
        + r"\omega_{{resonace}}}} \; = \; {0} $").format(np.round(w_res, 4))

frc_ax.text(
    0.65, 0.15, text, transform = frc_ax.transAxes,
    bbox={'facecolor':'white', 'alpha':1.0, 'pad':10})



t_f = time()
plot_time = t_f - t_i

print("Plot Time: ", plot_time)
print("-------------------------------------------------------")

#============================================================================
### Show the genrated Plots

plt.tight_layout()
plt.show()
