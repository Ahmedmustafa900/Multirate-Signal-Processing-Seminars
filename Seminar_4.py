#-- ====================================================================
#--
#-- --------------------------------------------------------------------
#-- Title: DFT Filter Bank
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
    import src.sound
    import numpy as np
    import math
    from  pathlib import Path
    import src.MrspPkg as MRSP    
except :
    print("\n[INFO] --- Packages not loaded correctly\n")

#-- Set print options
np.set_printoptions(precision=3)
if __name__== "__main__":
#-- -------------------------------------
#-- Main Method
#-- -------------------------------------
    #-- Set Filter Bank Characteristics
    nr_of_bands     = 8
    filter_length   = 32            #-- Filter Order
    gap             = 0.03
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
    ScriptPath = Path.cwd()
    rate_stereo, audio_stereo = wav.read(
      ScriptPath / 'data' / audio_name
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
    ax.plot(np.arange(N)/rate_mono, MRSP.normalize(audio_mono))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal Amplitude [unknown]')
    ax.set_title=("Audio Signal")
    plt.show()
#-- -------------------------------------
#-- Create FFT Filterbank 
#-- -------------------------------------
    block_size      = nr_of_bands*2
    nr_of_blocks    = math.floor(np.size(audio_mono)/block_size)
    #-- Divide audio signal into blocks
    signal_block        = np.zeros((nr_of_blocks,block_size))
    signal_transform    = np.zeros((nr_of_blocks,block_size))
    signal_block        = np.reshape(audio_mono[0:nr_of_blocks*block_size],(nr_of_blocks,block_size))
    #-- Apply FFT
    reverse         = False
    
    #-- Applied FFT Filter
    T   = np.fft.fft(np.eye(block_size))

    fig     = plt.figure('Filter Bank') 
    ax1     = fig.add_subplot(2,1,1)
    for i in range(nr_of_bands*2) :
        w,H = sig.freqz(np.flipud(T[:,i]),whole=True)    
        ax1.plot(w/2/np.pi, 20*np.log10(np.abs(H)))
    ax1.set_ylim(-40, 35)
    #ax1.set_xlim(0, 0.5*1)
    ax1.grid(True)
    ax1.set_xlabel('Normalized Frequency')
    ax1.set_ylabel('Gain[dB]')
    ax1.set_title('Frequency Response FFT Filter Bank')
#-- -------------------------------------
#-- Create Window with optimisation
#-- -------------------------------------
    WindowOpt  = MRSP.GenerateWindowByOptimazation(filter_length, 1/nr_of_bands/2,tb_width = .7)
#-- -------------------------------------
#-- Create Filterbank with Optimised Window 
#-- -------------------------------------
    
    ax2     = fig.add_subplot(2,1,2)
    h_bank = MRSP.create_improved_filter_bank (
        nr_of_bands,
        filter_length,
        gap,
        WindowOpt,
        rate_mono
    )
    for i in range(nr_of_bands) :
        w,H = sig.freqz(h_bank[i,:],whole=True)    
        ax2.plot(w/2/np.pi, 20*np.log10(np.abs(H)))
    ax2.set_ylim(-55, 20)
    #ax2.set_xlim(0, 0.5*1)
    ax2.grid(True)
    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Gain[dB]')
    ax2.set_title('Frequency Response Optimized Filter Bank')
    plt.tight_layout()
    #print("\n[INFO] --- Plot filterbank impulse/frequency response ---\n")
    #MRSP.plot_bank(h_bank, nr_of_bands, audio_mono, 'Filter Bank with Optimised Window')
    #plt.show()
    
#-- -------------------------------------
#-- Apply DFT Filterbank 
#-- -------------------------------------
    for blocks in range (nr_of_blocks) :
        signal_transform[blocks,:] = MRSP.fft_transform(signal_block[blocks,:],reverse=False)
    
    fig     = plt.figure('FFT Subbands')
    ax      = fig.add_subplot(1,1,1)
    for i in range(nr_of_bands*2) :
        w,H = sig.freqz(np.flipud(signal_transform[:,i]),whole=True) 
        ax.plot(1.0*w/(2*np.pi), 20*np.log10(np.abs(H)))
    ax.grid(True)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Gain[dB]')
    ax.set_title('Subband Signals')

#-- -------------------------------------
#-- Reconstruct Signal
#-- -------------------------------------
    signal_inversed     = np.zeros((nr_of_blocks,block_size))
    reverse             = True
    for blocks in range (nr_of_blocks) :
        signal_inversed[blocks,:] = MRSP.fft_transform(signal_transform[blocks,:],reverse=True)
    #-- Reshape signal
    signal_reconstructed = np.reshape(signal_inversed, (nr_of_blocks*block_size,))

    fig     = plt.figure('Reconstructed Mono Audio Signal')
    ax      = fig.add_subplot(111)
    ax.plot(np.arange(N)/rate_mono, MRSP.normalize(audio_mono))
    ax.plot(np.arange(signal_reconstructed.shape[0])/rate_mono, MRSP.normalize(signal_reconstructed))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal Amplitude [unknown]')
    ax.set_title = ("Audio Signal")
    plt.show()
#-- -------------------------------------
#-- Analysis of Reconstructed Audio Signal
#-- -------------------------------------
    MRSP.print_head("Analysis of Reconstructed Audio Signal")
    print("\n[INFO] --- Play Audio after Reconstruction with Rectangular Window ---\n")
    #src.sound.sound(signal_reconstructed,rate_mono)