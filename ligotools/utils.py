
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  


def plot_snr_panels(time, SNR, timemax, det, color, fig_dir, eventname, plottype):
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time - timemax, SNR, color, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.title(f"{det} matched filter SNR around event")
    plt.subplot(2,1,2)
    plt.plot(time - timemax, SNR, color, label=f"{det} SNR(t)")
    plt.grid(True)
    plt.ylabel("SNR")
    plt.xlim([-0.15, 0.05])
    plt.xlabel(f"Time since {timemax:.4f}")
    plt.legend(loc="upper left")
    plt.savefig(str(fig_dir / f"{eventname}_{det}_SNR.{plottype}"))

def plot_whitened_panels(time, tevent, strain_whitenbp, template_match,
                         det, color, fig_dir, eventname, plottype):
    plt.figure(figsize=(10,8))
   
    plt.subplot(2,1,1)
    plt.plot(time - tevent, strain_whitenbp, color, label=f"{det} whitened h(t)")
    plt.plot(time - tevent, template_match, 'k', label="Template(t)")
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f"Time since {tevent:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} whitened data around event")
    # Residual
    plt.subplot(2,1,2)
    plt.plot(time - tevent, strain_whitenbp - template_match, color, label=f"{det} resid")
    plt.ylim([-10, 10])
    plt.xlim([-0.15, 0.05])
    plt.grid(True)
    plt.xlabel(f"Time since {tevent:.4f}")
    plt.ylabel("whitened strain (units of noise stdev)")
    plt.legend(loc="upper left")
    plt.title(f"{det} Residual whitened data after subtracting template around event")
    plt.savefig(str(fig_dir / f"{eventname}_{det}_matchtime.{plottype}"))

def plot_asd_template_overlay(datafreq, template_fft, d_eff,
                              freqs, data_psd, det, color,
                              fs, fig_dir, eventname, plottype):
    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.figure(figsize=(10,6))
    plt.loglog(np.abs(datafreq), template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd), color, label=f"{det} ASD")
    plt.xlim(20, fs/2)
    plt.ylim(1e-24, 1e-20)
    plt.grid(True)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(f"{det} ASD and template around event")
    plt.savefig(str(fig_dir / f"{eventname}_{det}_matchfreq.{plottype}"))

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    hf = np.fft.rfft(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(filename, fs, data):
    """Scale to int16 and write wav."""
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename, int(fs), d)

def reqshift(data, fshift=100, sample_rate=4096):
    """
    Constant frequency shift by fshift (Hz) via FFT bin roll.
    """
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.0
    z = np.fft.irfft(y)
    return z

def plot_asds(strain_H1, strain_L1, fs, f_min=20.0, f_max=2000.0,
              label_H1='H1 strain', label_L1='L1 strain', smooth_model=True,
              title='Advanced LIGO strain data', savepath=None):

    NFFT = 4 * fs
    Pxx_H1, freqs = mlab.psd(strain_H1, Fs=fs, NFFT=NFFT)
    Pxx_L1, freqs = mlab.psd(strain_L1, Fs=fs, NFFT=NFFT)

    psd_H1 = interp1d(freqs, Pxx_H1)
    psd_L1 = interp1d(freqs, Pxx_L1)

    psd_smooth_interp = None
    if smooth_model:
        Pxx = (1.0e-22 * (18.0 / (0.1 + freqs)) ** 2) ** 2 + 0.7e-23 ** 2 + ((freqs / 2000.0) * 4.0e-23) ** 2
        psd_smooth_interp = interp1d(freqs, Pxx)

    plt.figure(figsize=(10, 8))
    plt.loglog(freqs, np.sqrt(Pxx_L1), 'g', label=label_L1)
    plt.loglog(freqs, np.sqrt(Pxx_H1), 'r', label=label_H1)
    if smooth_model:
        plt.loglog(freqs, np.sqrt(Pxx), 'k', label='H1 smooth model (O1)')
    plt.axis([f_min, f_max, 1e-24, 1e-19])
    plt.grid(True)
    plt.ylabel('ASD (strain/rtHz)')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc='upper center')
    plt.title(title)

    if savepath is not None:
        plt.savefig(str(savepath), bbox_inches='tight')

    return psd_H1, psd_L1, psd_smooth_interp
