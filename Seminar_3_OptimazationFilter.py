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
    import src.MrspPkg as MRSP
except :
    print("\n[INFO] --- Packages not loaded correctly\n")
#-- Set print options
np.set_printoptions(precision=3)

#np.random.seed(6)
if __name__== "__main__":
#-- -------------------------------------
#-- Main Method
#-- -------------------------------------
    #-- Set Filter Bank Characteristics
    
    nr_of_bands     = 8
    fs              = 44100         #-- Sample Rate
    filter_length   = 32           #-- Filter Order
    gap             = 0.03
    window          = 1             #-- Default: Rectangle

    #-- Initialise Arrays
    h_bank = np.zeros((nr_of_bands, filter_length))
    H_bank = np.zeros((nr_of_bands, 512))
    w_bank = np.zeros((nr_of_bands, 512))
#-- -------------------------------------
#-- Analysis of Audio Signal
#-- -------------------------------------
    audio_name      = 'data/Track32.wav'
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
    # h_bank = MRSP.create_improved_filter_bank (
    #     nr_of_bands,
    #     filter_length,
    #     gap,
    #     window,
    #     rate_mono
    # )
    # print("\n[INFO] --- Plot filterbank impulse/frequency response ---\n")
    # MRSP.plot_bank(h_bank, nr_of_bands, audio_mono, 'Filter Bank')
    # plt.show()
#-- -------------------------------------
#-- Create Window with optimasazion
#-- -------------------------------------

    WindowOpt  = MRSP.GenerateWindowByOptimazation(filter_length, 1/nr_of_bands/2,tb_width = .7)
    # WindowOpt  = MRSP.GenerateWindowByOptimazation(filter_length, 0.25)
    WindowRect = 1
#-- -------------------------------------
#-- Windowed Filter Banks
#-- -------------------------------------
    h_bank_rect = MRSP.create_improved_filter_bank(
        nr_of_bands,
        filter_length,
        gap,
        WindowRect,
        rate_mono
    )
    h_bank_opt = MRSP.create_improved_filter_bank(
        nr_of_bands,
        filter_length,
        gap,
        WindowOpt,
        rate_mono
    )
    
#-- -------------------------------------
#-- Plot Impulse/Frequency Response 
#-- -------------------------------------
    print("\n[INFO] --- Plot filterbank impulse/frequency response ---\n")
    MRSP.plot_bank(h_bank_rect, nr_of_bands, audio_mono, 'Rectangular Bank')
    MRSP.plot_bank(h_bank_opt , nr_of_bands, audio_mono, 'Optimazation Bank')
    
    plt.show()
#-- -------------------------------------
#-- Downsample
#-- -------------------------------------
    #-- Convolve Audio with Filterbank
    audio_bands_rect = MRSP.downsample_filter(audio_mono, h_bank_rect, nr_of_bands)
    audio_bands_opt  = MRSP.downsample_filter(audio_mono, h_bank_opt , nr_of_bands)
    
    #-- Downsample Audio Signal
    audio_downsampled_rect = audio_bands_rect[:,::nr_of_bands]
    audio_downsampled_opt  = audio_bands_opt[:,::nr_of_bands]
    
#-- -------------------------------------
#-- Upsample
#-- -------------------------------------
    #-- Upsample Audio Signal
    audio_upsampled_rect = np.zeros((nr_of_bands, np.size(audio_bands_rect,1)))
    audio_upsampled_opt  = np.zeros((nr_of_bands, np.size(audio_bands_opt ,1)))
    
    audio_upsampled_rect[:,::nr_of_bands] = audio_downsampled_rect
    audio_upsampled_opt[ :,::nr_of_bands] = audio_downsampled_opt
    
    #-- Convolve Audio with Filterbank
    audio_bands_rect = MRSP.upsample_filter(audio_upsampled_rect, h_bank_rect, nr_of_bands)
    audio_bands_opt  = MRSP.upsample_filter(audio_upsampled_opt , h_bank_opt , nr_of_bands)
    
    #-- Audio Reconstruction
    audio_reconstructed_rect = MRSP.reconstruct(audio_bands_rect, nr_of_bands)
    audio_reconstructed_opt  = MRSP.reconstruct(audio_bands_opt , nr_of_bands)
#-- -------------------------------------
#-- Analysis of Reconstructed Audio Signal
#-- -------------------------------------
    MRSP.print_head("Analysis of Reconstructed Audio Signal")
    print("\n[INFO] --- Play Audio after Reconstruction with Rectangular Window ---\n")
    #sound.sound(5*audio_reconstructed_rect,rate_mono)
    print("\n[INFO] --- Play Audio after Reconstruction with Optimazation Window ---\n")
    #sound.sound(1024*audio_reconstructed_opt/np.max(np.abs(audio_reconstructed_opt)),rate_mono)
    
    fig = plt.figure('Audio Signal before/after')
    ax = fig.add_subplot(111)
    ax.plot(np.arange(N)/rate_mono, audio_mono/np.max(np.abs(audio_mono)))
    ax.plot(np.arange(N)/rate_mono, audio_reconstructed_opt[0:N] / np.max (np.abs(audio_reconstructed_opt)))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal Amplitude normalized')
    ax.set_title=("Audio Signal")
    ax.legend(('Original','Optimized Window'))
    plt.show()
    
    spec_bef = np.fft.fft(audio_mono)
    spec_bef = np.abs(spec_bef)/np.max(np.abs(spec_bef))
    spec_after = np.fft.fft(audio_reconstructed_opt)
    spec_after = np.abs(spec_after)/np.max(np.abs(spec_after))
    spec_rect = np.fft.fft(audio_reconstructed_rect)
    spec_rect = np.abs(spec_rect)/np.max(np.abs(spec_rect))
    
    fig = plt.figure('Audio Spectrum before/after')
    ax = fig.add_subplot(111)
    ax.plot(np.arange(fs)/fs/2,20*np.log10(spec_bef[0:int(fs)]))
    
    # ax.set_xlabel('freq [Hz]')
    # ax.set_ylabel('Amplitude dBFS')
    # ax.set_title=("Audio Signal")
    
    #ax = fig.add_subplot(211)
    ax.plot(np.arange(fs)/fs/2,20*np.log10(spec_rect[0:int(fs)]))
    # ax.set_xlabel('freq [Hz]')
    # ax.set_ylabel('Amplitude dBFS')
    # ax.set_title=("Audio Signal")
    
    #ax = fig.add_subplot(311)
    ax.plot(np.arange(fs)/fs/2,20*np.log10(spec_after[0:int(fs)]))
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Amplitude dBFS')
    ax.set_title=("Audio Signal")
    ax.legend(('Original','Rect Window','Optimized Window'))
    
    plt.show()
