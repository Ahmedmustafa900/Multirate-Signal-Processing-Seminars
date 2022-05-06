# -- ====================================================================
# --
# -- --------------------------------------------------------------------
# -- Title: Process Audio with Error-Function
# -- --------------------------------------------------------------------
# -- Description:
# --
# --
# -- ====================================================================
# -- Import necessary packages
try:
    import sys
    sys.path.append("./src")
    import scipy.signal as sig
    import matplotlib.pyplot as plt
    import scipy.io.wavfile as wav
    import scipy.optimize as opt
    from importlib import reload
    import MrspPkg as dsp
    reload(dsp)
    import numpy as np
    import math
    import src.sound
except:
    print("\n[INFO] --- Packages not loaded correctly\n")
# -- Set print options
np.set_printoptions(precision=3)
if __name__ == "__main__":
    # -- -------------------------------------
    # -- Main Method
    # -- -------------------------------------
    # -- Set Filter Bank Characteristics
    nr_of_bands = 8


# -- -------------------------------------
# -- Analysis of Audio Signal
# -- -------------------------------------
    audio_name = 'Track32.wav'
    # -- Read Mono Signal
    rate_stereo, audio_stereo = wav.read(
        'data/' + audio_name
    )
    # -- File Characteristics
    dsp.print_head("Stereo Audio File")
    print("\t - Sampling Rate: %d" % rate_stereo)
    print("\t - Size of stereo audio file: %d KBits" %
          (np.size(audio_stereo)/1000.0))
    print("\t - Shape of stereo audio file: ", np.shape(audio_stereo))
    print("\t - Dimension: %d\n" % np.ndim(audio_stereo))
    # -- Merge both pans to mono layer
    print("\n[INFO] --- Merge pans to mono layer ---\n")
    audio_mono = audio_stereo.mean(1)
    rate_mono = rate_stereo
    print("\t - Dimension: %d\n" % np.ndim(audio_mono))
# -- -------------------------------------
# -- Plot Mono
# -- -------------------------------------
    N = audio_mono.shape[0]
    L = N/rate_mono
    #-- Display
    dsp.print_head("Mono Audio Signal")
    print("\t - Number of Samples: %d " % N)
    print(f"\t - Audio Length: {L:.2f} [s]")
    fig, ax = plt.subplots()
    ax.plot(np.arange(N)/rate_mono, audio_mono)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal Amplitude [unknown]')
    ax.set_title = ("Audio Signal")
    plt.show()


# -- -------------------------------------
# -- Get DCT mtx and freq response
# -- -------------------------------------

    T_DCT4 = dsp.DctType4(nr_of_bands)
    plt_subband = 0
    #analysis -> Time Reversed -> flip
    #synthesis -> notTime Reversed -> no flip
    dsp.plotFrqResp(np.flipud(T_DCT4[:,plt_subband]),"Magnitude Response for DCT4 suband k={}, Analysis Filter".format(plt_subband))
    plt.show()
    dsp.plotFrqResp(         (T_DCT4[:,plt_subband]),"Magnitude Response for DCT4 suband k={}, Synthesis Filter".format(plt_subband))
    plt.show()


    ## Test with Ramp
    L_test = 2**8
    test_ramp = np.arange(L_test)
    test_ramp = np.mod(test_ramp, 10)-4.5
    # Convolve Ramp with Filterbank
    ramp_bands = dsp.downsample_filter(test_ramp, np.flipud(T_DCT4), nr_of_bands)
    #-- Downsample ramp Signal
    ramp_downsampled = ramp_bands[:,::nr_of_bands]

    #resample ramp signal
    ramp_upsampled = np.zeros((nr_of_bands, np.size(ramp_bands,1)))

    ramp_upsampled[:,::nr_of_bands] = ramp_downsampled
    #-- Convolve ramp with Filterbank
    ramp_bands_up = dsp.upsample_filter(ramp_upsampled, T_DCT4, nr_of_bands)
    #-- ramp Reconstruction
    ramp_reconstructed = dsp.reconstruct(ramp_bands_up, nr_of_bands)
    plt.plot(ramp_bands_up[0,:])
    # dsp.PlotFFT(test_ramp)
    plt.figure()
    plt.plot(ramp_reconstructed[:100])
    plt.figure()
    plt.plot(test_ramp[:100])

    ## Test with Audio
    #-- -------------------------------------
    #-- Downsample
    #-- -------------------------------------
    #-- Convolve Audio with Filterbank
    audio_bands = dsp.downsample_filter(audio_mono, np.flipud(T_DCT4), nr_of_bands)
    #-- Downsample Audio Signal
    audio_downsampled = audio_bands[:,::nr_of_bands]
    #-- -------------------------------------
    #-- Upsample
    #-- -------------------------------------
    #-- Upsample Audio Signal
    audio_upsampled = np.zeros((nr_of_bands, np.size(audio_bands,1)))
    audio_upsampled[:,::nr_of_bands] = audio_downsampled
    #-- Convolve Audio with Filterbank
    audio_bands_rec = dsp.upsample_filter(audio_upsampled, T_DCT4, nr_of_bands)
    #-- Audio Reconstruction
    audio_reconstructed = dsp.reconstruct(audio_bands_rec, nr_of_bands)		
    plt.figure()
    plt.plot(audio_bands[2,:])
    plt.figure()
    dsp.PlotFFT(audio_bands_rec[0,:])

    #-- -------------------------------------
    #-- Setting 2: Compressing
    #-- -------------------------------------

    NumberOfTakesSubbands = 3
    audio_downsampled_comporessed = np.zeros_like(audio_downsampled)
    audio_downsampled_comporessed[:NumberOfTakesSubbands,:] = audio_downsampled[:NumberOfTakesSubbands,:]
     
    #-- -------------------------------------
    #-- Upsample
    #-- -------------------------------------
    #-- Upsample Audio Signal
    audio_upsampled_comporessed = np.zeros((nr_of_bands, np.size(audio_bands,1)))
    audio_upsampled_comporessed[:,::nr_of_bands] = audio_downsampled_comporessed
    #-- Convolve Audio with Filterbank
    audio_bands_rec_comporessed = dsp.upsample_filter(audio_upsampled_comporessed, T_DCT4, nr_of_bands)
    #-- Audio Reconstruction
    audio_reconstructed_comporessed = dsp.reconstruct(audio_bands_rec_comporessed, nr_of_bands)		
    plt.figure()
    plt.plot(audio_bands[2,:])
    plt.figure()

    dsp.PlotFFT(audio_bands_rec_comporessed[0,:])

    if 0:
      #original
      sound.sound(audio_mono,rate_mono)
      #compressed



    if 0:
      #-- -------------------------------------
      #-- Task 2: Polyphase
      #-- -------------------------------------
      filterlength = 16
      ##sine window from S2
      if ((filterlength%2) == 0) : #-- Even filter length
          window_sine = np.sin((np.pi/filterlength)*(np.arange(filterlength)+0.5))
      else :  #-- Odd filter length
          window_sine = np.sin((np.pi/(filterlength+1))*(np.arange(filterlength)+1.0))
      n=np.arange(filterlength)-(filterlength-1)/2
      opt_h = dsp.Si(np.pi*n/nr_of_bands)
      plt.plot(window_sine)
      sinfilt = window_sine*opt_h
      # dsp.plotFrqResp(sinfilt,"")


      ramp_poly    = dsp.x2polyphase(test_ramp,nr_of_bands)
      sinfilt_poly = dsp.x2polyphase(sinfilt, nr_of_bands)


      y_poly = dsp.polmatmult(ramp_poly, sinfilt_poly)  


      
      
      
      
      # #-- -------------------------------------
# #-- Analysis of Reconstructed Audio Signal
# #-- -------------------------------------

#     dsp.print_head("Analysis of Reconstructed Audio Signal")
#     print("\n[INFO] --- Play Audio after Reconstruction with Optimised Window ---\n")
#     sound.sound(audio_reconstructed,rate_mono)
