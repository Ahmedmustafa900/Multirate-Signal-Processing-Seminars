############# 
## MRSP 
## Seminar I
#############

#TestTest

#import section
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig 
import math
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import src.sound as sound
import src.MRSP as MRSP 



     



## User options Setting
WaveName           = "Track32.wav"
#filter option
NumFilterBanks     = 8
OverlapFactor      = 1/16
NTaps = 420 ##420 for fir, 9 for iir
# NTaps = 9

## load wav
fs, StereoSound = wav.read(WaveName)
## Select left channel
Sound = StereoSound[:,0]

## Play Original Audio
print("Play original audio")
#sound.sound(Sound , fs)


#Measure Tine and Number of Samples
LengthSample = Sound.shape[0]
TimeSample = LengthSample/fs

h, w, H = MRSP.GenerateFilterBankFIR(NumFilterBanks, OverlapFactor, NTaps)

##Plot Frequency Response
fig0, (ax0,ax1) = plt.subplots(2,1, figsize=(10,18))
fig0.tight_layout(pad=3.0)
    # plt.plot(w1/(2*np.pi),20*np.log10(np.abs(H1)))

# ax.plot(np.linspace(0,fs, LengthSample), np.abs(np.fft.fftshift(fft(Sound))))
# ax.set_xlabel("Frequency")
# ax.set_ylabel("Amplitude")
# ax.set_title("Original Spectrum")


for i in range(8):
  ax0.plot(np.abs(w[i])/(2*math.pi), 20*np.log10(np.abs(H[i])) )

ax0.set_ylim(-80, 10 )
ax0.set_xlabel("Frequency")
ax0.set_ylabel("Magnitude [dB]")
ax0.set_title("Frequency response")

#Plot Frequency of Filtered Sound
FilteredSound = np.zeros((NumFilterBanks, Sound.shape[0]))

#fArray=np.linspace(-fs/2, fs/2, 2**15)

for nFilterBank in range(NumFilterBanks):
    FilteredSound[nFilterBank] = (np.convolve(h[nFilterBank], Sound))[:LengthSample]
    ax1.plot(range(h.shape[1]),np.real(h[nFilterBank]))

ax1.set_xlim(0,h.shape[1])
ax1.set_xlabel("Sample")
ax1.set_ylabel("Amplitude")
ax1.set_title("Filter Impulse Response")


##Test Play Filtered Sounds
print("Play filtered Audio")
#sound.sound(FilteredSound[0]/(FilteredSound[0]).max()* 2**15, fs)
#sound.sound(FilteredSound[3]/(FilteredSound[3]).max()* 2**15, fs)

#Downlampling and Testing Audio
print("Downsample Audio")
DownsampledAudio = FilteredSound[:,::NumFilterBanks]

print("Play downsampled Audio")
#sound.sound(DownsampledAudio[0]/(DownsampledAudio[0]).max()* 2**15, (int(fs/NumFilterBanks)))
#sound.sound(DownsampledAudio[3]/(DownsampledAudio[3]).max()* 2**15, (int(fs/NumFilterBanks)))

print("Upsample Audio")
##upsampling again
UpsampledAudio = np.zeros_like(FilteredSound)
UpsampledAudio[:,::NumFilterBanks] = DownsampledAudio

##Filter Frequencybands
FilteredUpsampled = np.zeros((NumFilterBanks, UpsampledAudio.shape[1]))



for nFilterBank in range(0,NumFilterBanks):
    FilteredUpsampled[nFilterBank] = np.convolve(h[nFilterBank], UpsampledAudio[nFilterBank])[:LengthSample]
#    ax2.plot(fArray,np.abs(np.fft.fftshift(fft(FilteredUpsampled[nFilterBank,0:2**15]))))

# sd.play(FilteredUpsampled[0]/(FilteredUpsampled).max(), fs)
# sd.play(FilteredUpsampled[3]/(FilteredUpsampled).max(), fs)


# ax2.set_title=("Spectrum Upsampled and Filtred Sound")
# ax2.set_xlabel("Frequency")
# ax2.set_ylabel("Amplitude")
ReconstructedSound = np.sum(FilteredUpsampled, 0)

fig0.show()

plt.figure()
plt.plot(Sound/np.max(np.abs(Sound)))
plt.plot(ReconstructedSound/np.max(np.abs(ReconstructedSound)))
plt.title("Original and Reconstructed Audio")
# ax3.set_xlabel("Frequency")
# ax3.set_ylabel("Amplitude")
plt.show()


print("Play Reconstructed Audio")
#sound.sound(ReconstructedSound/ ReconstructedSound.max()* 2**15 , fs) 
