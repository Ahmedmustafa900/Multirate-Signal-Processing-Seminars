#-- ====================================================================
#--
#-- --------------------------------------------------------------------
#-- Title: Process Improved Audio
#-- --------------------------------------------------------------------
#-- Description:   
#-- 
#-- 
#-- ====================================================================
#-- Import necessary packages
try :
    import scipy.signal as sig
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wav
    import src.sound as sound
    import numpy as np
    import math
    import src.MRSP as MRSP
    from  pathlib import Path
except :
    print("\n[INFO] --- Packages not loaded correctly\n")
#-- Set print options
np.set_printoptions(precision=3)
if __name__== "__main__":
#-- -------------------------------------
#-- Main Method
#-- -------------------------------------
    #-- Set Filter Bank Characteristics
    nr_of_bands     = 4
    fs              = 44100         #-- Sample Rate
    filter_length   = 128           #-- Filter Order
    gap             = 0.005
    window          = 1             #-- Default: Rectangle

    #-- Initialise Arrays
    h_bank = np.zeros((nr_of_bands, filter_length))
    H_bank = np.zeros((nr_of_bands, 512))
    w_bank = np.zeros((nr_of_bands, 512))
#-- -------------------------------------
#-- Analysis of Audio Signal
#-- -------------------------------------
    audio_name      = 'Track32.wav'
    #-- Read Mono Signal
    rate_stereo, audio_stereo = wav.read(
      audio_name
    )
    #-- File Characteristics
    MRSP.print_head("Stereo Audio File")
    print("\t - Sampling Rate: %d" %rate_stereo)
    print("\t - Size of stereo audio file: %d KBits" %(np.size(audio_stereo)/1000.0))
    print("\t - Shape of stereo audio file: ", np.shape(audio_stereo))
    print("\t - Dimension: %d\n" %np.ndim(audio_stereo))
    #-- Merge both pans to mono layer
    print("\n[INFO] --- Merge pans to mono layer ---\n")
    audio_mono = audio_stereo[:,0] #+ audio_stereo[:,1]
    rate_mono = rate_stereo
    print("\t - Dimension: %d\n" %np.ndim(audio_mono))
#-- -------------------------------------
#-- Plot Mono 
#-- -------------------------------------
    N = audio_mono.shape[0]
    L = N/rate_mono
    #-- Display
    MRSP.print_head("Mono Audio Signal")
    print("\t - Number of Samples: %d " %N)
    print(f"\t - Audio Length: {L:.2f} [s]")
    fig = plt.figure('Mono Audio Signal')
    ax = fig.add_subplot(111)
    ax.plot(np.arange(N)/rate_mono, audio_mono)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal Amplitude [unknown]')
    ax.set_title=("Audio Signal")
    plt.show()
#-- -------------------------------------
#-- Create Filterbank 
#-- -------------------------------------
    h_bank = MRSP.create_improved_filter_bank (
        nr_of_bands,
        filter_length,
        gap,
        window,
        rate_mono
    )
    print("\n[INFO] --- Plot filterbank impulse/frequency response ---\n")
    MRSP.plot_bank(h_bank, nr_of_bands, audio_mono, 'Filter Bank')
    plt.show()
#-- -------------------------------------
#-- Create Different Windows
#-- -------------------------------------
    #-- Rectangular Window
    window_rect = 1
    #-- Hanning Window: Raised Cosine-Function
    if ((filter_length%2) == 0) :   #-- Even filter length
        window_hann = 0.5-(0.5*np.cos((2*np.pi/filter_length)*(np.arange(filter_length)+0.5)))
    else :  #-- Odd filter length
        window_hann = 0.5-(0.5*np.cos((2*np.pi/filter_length)*(np.arange(filter_length)+1.0)))
    #-- Sine Window
    if ((filter_length%2) == 0) : #-- Even filter length
        window_sine = np.sin((np.pi/filter_length)*(np.arange(filter_length)+0.5))
    else :  #-- Odd filter length
        window_sine = np.sin((np.pi/(filter_length+1))*(np.arange(filter_length)+1.0))
    #-- Kaiser Window
    window_kaiser = np.kaiser(filter_length, 8)
    #-- Vorbis Window
    window_vorbis = np.sin((np.pi/2)*np.sin((np.pi/filter_length)*(np.arange(filter_length)+0.5))**2)
#-- -------------------------------------
#-- Windowed Filter Banks
#-- -------------------------------------
    h_bank_rect = MRSP.create_improved_filter_bank(
        nr_of_bands,
        filter_length,
        gap,
        window_rect,
        rate_mono
    )
    h_bank_hann = MRSP.create_improved_filter_bank(
        nr_of_bands,
        filter_length,
        gap,
        window_hann,
        rate_mono
    )
    h_bank_sine = MRSP.create_improved_filter_bank(
        nr_of_bands,
        filter_length,
        gap,
        window_sine,
        rate_mono
    )
    h_bank_kaiser = MRSP.create_improved_filter_bank(
        nr_of_bands,
        filter_length,
        gap,
        window_kaiser,
        rate_mono
    )
    h_bank_vorbis = MRSP.create_improved_filter_bank(
        nr_of_bands,
        filter_length,
        gap,
        window_vorbis,
        rate_mono
    )
#-- -------------------------------------
#-- Plot Impulse/Frequency Response 
#-- -------------------------------------
    print("\n[INFO] --- Plot filterbank impulse/frequency response ---\n")
    MRSP.plot_bank(h_bank_rect, nr_of_bands, audio_mono, 'Rectangular Bank')
    MRSP.plot_bank(h_bank_hann, nr_of_bands, audio_mono, 'Hanning Bank')
    MRSP.plot_bank(h_bank_sine, nr_of_bands, audio_mono, 'Sine Bank')
    MRSP.plot_bank(h_bank_kaiser, nr_of_bands, audio_mono, 'Kaiser Bank')
    MRSP.plot_bank(h_bank_vorbis, nr_of_bands, audio_mono, 'Vorbis Bank')
    
    plt.show()
#-- -------------------------------------
#-- Downsample
#-- -------------------------------------
    #-- Convolve Audio with Filterbank
    audio_bands_rect = MRSP.downsample_filter(audio_mono, h_bank_rect, nr_of_bands)
    audio_bands_hann = MRSP.downsample_filter(audio_mono, h_bank_hann, nr_of_bands)
    audio_bands_sine = MRSP.downsample_filter(audio_mono, h_bank_sine, nr_of_bands)
    audio_bands_kaiser = MRSP.downsample_filter(audio_mono, h_bank_kaiser, nr_of_bands)
    audio_bands_vorbis = MRSP.downsample_filter(audio_mono, h_bank_vorbis, nr_of_bands)
    #-- Downsample Audio Signal
    audio_downsampled_rect = audio_bands_rect[:,::nr_of_bands]
    audio_downsampled_hann = audio_bands_hann[:,::nr_of_bands]
    audio_downsampled_sine = audio_bands_sine[:,::nr_of_bands]
    audio_downsampled_kaiser = audio_bands_kaiser[:,::nr_of_bands]
    audio_downsampled_vorbis = audio_bands_vorbis[:,::nr_of_bands]
#-- -------------------------------------
#-- Upsample
#-- -------------------------------------
    #-- Upsample Audio Signal
    audio_upsampled_rect = np.zeros((nr_of_bands, np.size(audio_bands_rect,1)))
    audio_upsampled_hann = np.zeros((nr_of_bands, np.size(audio_bands_hann,1)))
    audio_upsampled_sine = np.zeros((nr_of_bands, np.size(audio_bands_sine,1)))
    audio_upsampled_kaiser = np.zeros((nr_of_bands, np.size(audio_bands_kaiser,1)))
    audio_upsampled_vorbis = np.zeros((nr_of_bands, np.size(audio_bands_vorbis,1)))
    audio_upsampled_rect[:,::nr_of_bands] = audio_downsampled_rect
    audio_upsampled_hann[:,::nr_of_bands] = audio_downsampled_hann
    audio_upsampled_sine[:,::nr_of_bands] = audio_downsampled_sine
    audio_upsampled_kaiser[:,::nr_of_bands] = audio_downsampled_kaiser
    audio_upsampled_vorbis[:,::nr_of_bands] = audio_downsampled_vorbis
    #-- Convolve Audio with Filterbank
    audio_bands_rect = MRSP.upsample_filter(audio_upsampled_rect, h_bank_rect, nr_of_bands)
    audio_bands_hann = MRSP.upsample_filter(audio_upsampled_hann, h_bank_hann, nr_of_bands)
    audio_bands_sine = MRSP.upsample_filter(audio_upsampled_sine, h_bank_sine, nr_of_bands)
    audio_bands_kaiser = MRSP.upsample_filter(audio_upsampled_kaiser, h_bank_kaiser, nr_of_bands)
    audio_bands_vorbis = MRSP.upsample_filter(audio_upsampled_vorbis, h_bank_vorbis, nr_of_bands)
    #-- Audio Reconstruction
    audio_reconstructed_rect = MRSP.reconstruct(audio_bands_rect, nr_of_bands)
    audio_reconstructed_hann = MRSP.reconstruct(audio_bands_hann, nr_of_bands)
    audio_reconstructed_sine = MRSP.reconstruct(audio_bands_sine, nr_of_bands)
    audio_reconstructed_kaiser = MRSP.reconstruct(audio_bands_kaiser, nr_of_bands)
    audio_reconstructed_vorbis = MRSP.reconstruct(audio_bands_vorbis, nr_of_bands)
#-- -------------------------------------
#-- Analysis of Reconstructed Audio Signal
#-- -------------------------------------
    MRSP.print_head("Analysis of Reconstructed Audio Signal")
    print("\n[INFO] --- Play Audio after Reconstruction with Rectangular Window ---\n")
    #sound.sound(5*audio_reconstructed_rect,rate_mono)
    print("\n[INFO] --- Play Audio after Reconstruction with Hanning Window ---\n")
    #sound.sound(5*audio_reconstructed_hann,rate_mono)
    print("\n[INFO] --- Play Audio after Reconstruction with Sine Window ---\n")
    #sound.sound(5*audio_reconstructed_sine,rate_mono)
    print("\n[INFO] --- Play Audio after Reconstruction with Kaiser Window ---\n")
    #sound.sound(5*audio_reconstructed_kaiser,rate_mono)
    print("\n[INFO] --- Play Audio after Reconstruction with Vorbis Window ---\n")
    #sound.sound(5*audio_reconstructed_vorbis,rate_mono)
    
    plt.figure()
    plt.plot(audio_mono/np.max(np.abs(audio_mono)))
    plt.plot(audio_reconstructed_rect/np.max(np.abs(audio_reconstructed_rect)))
    plt.title("Original and Reconstructed Audio")
    # ax3.set_xlabel("Frequency")
    # ax3.set_ylabel("Amplitude")
    plt.show()
