# Multi-Sensor Signal Conditioning and Filter Design Methodology

## 1. Overview and Necessity

This stage bridges raw sensor hardware and interpretable machine-learning features. The preprocessing pipeline addresses three specific challenges dictated by the dataset structure and hardware constraints:

1. **Data Decoding (Hybrid Channel)**  
   The stretch sensor channel packs both physiological signal data and ground-truth labels into a single integer. This requires bitwise separation before analysis.

2. **Bandwidth Limitation and Analysis Range**  
   The hardware sampling rate is \(F_s = 4800\ \text{Hz}\), so the Nyquist limit is \(2400\ \text{Hz}\). We therefore:
   - Restrict subsequent analysis to physically meaningful bands for each sensor, and  
   - Eliminate high-frequency components that are dominated by electronic noise and cannot carry useful physiological information.  

   **Important clarification:** any *analog* anti-alias filtering must already be implemented at the hardware level before sampling. The *digital* filters described here cannot undo aliasing; they simply band-limit the already sampled signals to the ranges that are physiologically and methodologically relevant, and prepare the data for any optional downsampling.

3. **Artifact Removal**  
   - **Audio channels:** removal of low-frequency baseline drift and strong power-line components.  
   - **Motion channels:** smoothing of sensor jitter and very high-frequency noise while preserving slow posture changes and the dynamics of cough-related body movement.

---

## 2. Data Decoding (Stretch Sensor & Labels)

The **3rd column (stretch sensor)** uses a bitwise encoding scheme. To access usable data, the raw integer must be split into the analog sensor value and the digital boolean label.

**Preprocessing logic applied to Column 3** (conceptual description):

1. **Extract label (ground truth)**  
   - Bitwise AND with 1 keeps the least significant bit.  
   - Interpretation:  
     - 0 → background/silence  
     - 1 → cough event  

2. **Extract stretch signal (chest expansion)**  
   - Bitwise right shift by 1 discards the label bit and recovers the encoded stretch value.

After decoding, the stretch signal can be optionally re-centered (mean removal) and scaled if physical calibration data become available (e.g., strain-to-length or pressure units). For the purposes of this filter design, it is sufficient to work in consistent digital units.

---

## 3. Filter Strategy and Methodology

To determine the filter parameters, a **targeted spectral analysis** was performed:

- For each sensor channel, multiple recordings were segmented into:
  - **Cough segments** (label = 1),
  - **Background segments** (label = 0).
- For each segment type and channel, we computed:
  - **Power Spectral Density (PSD)** estimates, and  
  - **Spectrograms** with adequate time–frequency resolution.

This allowed direct visual comparison between cough and background spectra, and between different sensors, to identify which frequency bands carry most of the discriminative energy.

### Zero-phase vs. causal filtering

For **offline analysis and feature extraction**, all digital filters are implemented as **zero-phase filters** using forward–backward application (conceptually equivalent to applying an IIR/FIR filter forward in time and then backward). This:

- Eliminates phase distortion, and  
- Keeps cough peaks aligned across audio and motion channels, which is important for multi-sensor fusion.

However, this approach is **non-causal** and therefore **not suitable for real-time implementation**. For a real-time wearable system, equivalent **causal** filters with known group delay would be required (see Section 3.3 and the real-time discussion in Section 4.D).

---

## 3.1 Choosing Filter Implementation Details

This section explains how the abstract passbands (e.g. 60–2200 Hz) are turned into concrete digital filters.

### 3.1.1 Filter family

Given the offline, zero-phase context and the moderate sampling rate, a reasonable design choice is:

- **IIR Butterworth filters**, because:
  - They have a maximally flat passband (no ripples),
  - A given transition steepness can be achieved with relatively low order (efficient),
  - In combination with forward–backward application, stability and effective high-order attenuation are typically acceptable.

Alternative families (Chebyshev, elliptic, or FIR designs) are possible, but the justification in this document assumes a Butterworth-like monotonic response.

### 3.1.2 Order and attenuation targets

Filter order should be chosen to satisfy *explicit* design targets, for example:

- **Audio channels (pulmonary & ambient microphones):**
  - Sufficient attenuation at 50 Hz (power-line region) when using a 60 Hz high-pass section.
  - Adequate roll-off near 2200 Hz to suppress noise as we approach the 2400 Hz Nyquist limit, while not distorting the spectrum up to ~2000 Hz.
- **Motion channels (stretch sensor & accelerometer):**
  - Strong attenuation above ~20–30 Hz, since PSDs show these regions are noise dominated.
  - Smooth response in the passband [0–10 Hz] to avoid distorting slow physiological trends.

In practice, an IIR order in the range 2–6 for each section (high-pass/low-pass) is typically sufficient. The **effective order doubles** when applying forward–backward filtering.

### 3.1.3 Per-channel implementation choices

Conceptually, for each sensor:

- **Pulmonary mic (bandpass 60–2200 Hz)**  
  - Implement as cascade:
    - High-pass section with cutoff near 60 Hz.
    - Low-pass section with cutoff near 2200 Hz.
- **Ambient mic (bandpass 60–2200 Hz)**  
  - Use *exactly* the same digital filter coefficients as the pulmonary channel (see Section 4.B for rationale).
- **Stretch sensor (low-pass 20 Hz)**  
  - Single low-pass section with cutoff near 20 Hz.
- **Accelerometer (low-pass 20 Hz, no high-pass)**  
  - Single low-pass section with cutoff near 20 Hz, explicitly preserving DC.

The precise numeric filter coefficients are not listed here, but should be derived from the above design choices and stored/version-controlled (e.g. as part of the codebase) to ensure full reproducibility.

---

## 3.2 Quantitative Validation of Filter Choices

The initial passbands were chosen using PSDs and spectrograms. To justify them more rigorously, the filters must be evaluated quantitatively.

### 3.2.1 Cough/background spectral separation

For each channel:

1. Extract a large set of cough and background segments (based on labels).
2. For each segment, compute a PSD estimate (e.g. Welch’s method).
3. Compute:
   - **Band power** in the intended passband, e.g.  
     - Audio: 60–2200 Hz  
     - Motion: 0–20 Hz  
   - **Out-of-band power**, e.g. 0–60 Hz and 2200–2400 Hz for audio before filtering, and >20 Hz for motion.

Then, for each design candidate (e.g. high-pass at 40, 60, 80 Hz):

- Evaluate how much cough band power is retained vs. removed.
- Compare the ratio of cough power to background power in the passband (a simple SNR-like metric).

### 3.2.2 Effect on downstream classification

Ultimately, the filters serve machine-learning tasks (cough detection, activity/posture classification). Therefore, filter parameters should be treated as **hyperparameters**:

- Define a small grid of candidate designs, e.g.:
  - Pulmonary/ambient high-pass: 40 Hz, 60 Hz, 80 Hz  
  - Motion low-pass: 15 Hz, 20 Hz, 25 Hz.
- For each design:
  1. Filter the full dataset.
  2. Extract your planned features (spectral, temporal, etc.).
  3. Train and evaluate the models using a fixed cross-validation scheme.
- Compare metrics (e.g. F1-score, sensitivity, specificity) across filter configurations.

If model performance changes negligibly between, say, 40 Hz and 60 Hz high-pass, then 60 Hz may be chosen for its stronger hum suppression. If performance degrades at 80 Hz, that suggests meaningful cough energy is present in 60–80 Hz that you should not discard.

---

## 3.3 Edge Behaviour and Segment Handling

Forward–backward filtering (zero-phase) introduces **edge effects** because the algorithm internally pads the signal (typically by reflection) before filtering. These effects are most noticeable near the start and end of a filtered segment.

For this project:

- If you **filter entire recordings first** and only then cut them into cough/background segments based on labels:
  - Edge artifacts are confined to the very beginning and end of each recording.
  - These regions can be safely ignored (or removed) as they usually contain no labelled events anyway.
- If you **filter very short segments** (e.g. individual coughs) independently:
  - The relative impact of padding and edge behaviour can be significant.
  - This can slightly distort amplitude and shape near segment boundaries.

**Recommended practice:**

1. Filter each **full channel recording** as a single continuous signal.
2. Discard a small number of samples at the global start and end if necessary (e.g. corresponding to a small fraction of a second).
3. Perform **all event segmentation and feature extraction** on the already filtered continuous signals.

Under this strategy, edge behaviour is a minor issue and does not need special handling for each cough. If, in future, you require online/causal filtering, you must explicitly model group delay and buffer length, which is a separate design problem.

---

## 3.4 Downsampling Strategy: Do We Need It?

Given \(F_s = 4800\ \text{Hz}\), the raw data are oversampled relative to the useful bandwidth of most sensors, especially the motion channels.

### 3.4.1 Audio channels (pulmonary and ambient mics)

- You restrict frequency content to approximately 60–2200 Hz.
- The Nyquist limit at 4800 Hz is 2400 Hz, so the audio channels are **not grossly oversampled**, and:
  - Further downsampling is optional and mainly impacts computational cost.
  - If you **do** downsample, you must:
    - Apply a sufficiently steep low-pass filter below the new Nyquist, and
    - Choose a new sampling rate that still comfortably covers the upper band of interest (e.g. downsample to 4000 Hz or 4410 Hz equivalent).

For simplicity and to avoid additional complexity, keeping the audio channels at 4800 Hz is acceptable in this project.

### 3.4.2 Motion channels (stretch sensor and accelerometer)

For these channels:

- The **useful energy** is confined largely below ~10 Hz, with a noise floor emerging above ~15–20 Hz.
- Keeping them at 4800 Hz means you are storing and processing a large number of **highly redundant, strongly correlated samples**.

Therefore, for motion channels, **downsampling is recommended** after low-pass filtering:

- Example strategy:
  - Apply a 20 Hz low-pass filter (as justified in Sections 4.C and 4.D).
  - Downsample to 100 Hz or 50 Hz.
- Benefits:
  - Reduced storage and computational cost.
  - More numerically stable and robust features when computed over fixed-duration windows.
  - Easier alignment of motion features with audio features at the **window level** (e.g. 100–200 ms feature windows), even if the raw sampling rates differ.

In summary:

- **Audio channels:** downsampling is optional; current design can stay at 4800 Hz.  
- **Motion channels:** downsampling after appropriate low-pass filtering is advisable and justified.

---

## 4. Sensor-Specific Design Analysis

### 4.A Pulmonary Microphone (Channel 1)

**Objective:** Maximize signal-to-noise ratio for cough acoustics.

#### Spectral observations

- **Very low frequencies (0–50 Hz):**  
  PSDs of multiple recordings show a strong spike in the near-DC region. This reflects baseline drift, slow sensor motion, and possibly environmental low-frequency noise, not cough content.

- **Main cough band (approximately 300–1500 Hz):**  
  Cough events appear as broadband “flames” in spectrograms, with energy concentrated roughly between a few hundred Hz and around 1.5 kHz. Background segments are relatively quiet in this band.

#### Filter decision: bandpass [60 Hz – 2200 Hz]

- **High-pass at 60 Hz**

  - **Power-line region:**  
    - In the local environment, power-line interference occurs at 50 Hz and possibly at harmonics (100, 150 Hz, …).  
  - **Literature guidance (respiratory sounds):**  
    - Reichert et al. (“Analysis of Respiratory Sounds: State of the Art”, 2008) describe normal and adventitious respiratory sounds and note that clinically relevant lung sounds span a broad band well above the very-low-frequency drift region.  
    - They do **not** prescribe a single precise cutoff for cough, but their discussion of spectrum-focused analysis supports the idea that physiologically meaningful information is found well above DC and low-frequency artifacts.

  - **Trade-off decision:**  
    - Our PSDs show that, in this dataset, cough energy below ~60 Hz is comparatively small relative to the strong drift and hum components.  
    - Therefore, we **accept a deliberate trade-off**:
      - We remove the entire 0–60 Hz band, losing any residual low-frequency cough content,
      - In exchange for robust suppression of drift and 50 Hz hum and a cleaner mid-band for subsequent feature extraction.

- **Low-pass at 2200 Hz**

  - **Upper limit of interest:**  
    - Shi et al. (“Theory and Application of Audio-Based Assessment of Cough”, Journal of Sensors, 2018) review multiple cough analysis systems, many of which operate with sampling rates and analysis bands extending to a few kilohertz. They treat cough as a broadband transient with substantial energy up to the low kHz range.
  - **Sampling constraint:**  
    - With \(F_s = 4800\ \text{Hz}\), Nyquist is 2400 Hz.  
  - **Design choice:**  
    - Setting the low-pass at **2200 Hz**:
      - Preserves the bulk of the cough-relevant high-frequency content described in the literature and visible in our PSDs,  
      - Creates a guard band between 2200–2400 Hz to attenuate residual high-frequency noise and reduce sensitivity to any imperfect anti-aliasing at the hardware level.

Overall, the **60–2200 Hz** band is chosen as a pragmatic compromise: it retains the dominant cough content while forcefully suppressing the parts of the spectrum that are dominated by drift, hum, or high-frequency noise.

---

### 4.B Ambient Microphone (Channel 2)

**Objective:** Capture environmental sounds and distinguish patient coughs from background noise.

#### Spectral observations

- PSD and spectrogram behaviour are **very similar** to the pulmonary mic:
  - Strong near-DC energy,
  - Cough-related bursts in roughly the same 100–2000 Hz region.

#### Filter decision: same bandpass [60 Hz – 2200 Hz]

- **Design choice:**  
  The ambient mic uses **exactly the same digital filter coefficients** as the pulmonary microphone.

- **Rationale:**
  - Many potential features involve **comparing** the two audio channels:
    - Energy difference \(E_{\text{pulm}} - E_{\text{amb}}\),
    - Ratios, or other contrasts to decide whether a signal is “body-coupled” or purely environmental.
  - Using identical filters ensures that:
    - Both signals are subjected to the **same amplitude and phase shaping**, and
    - Differences between them are due to **true signal differences**, not differences in filtering.

- **Design choice to be validated:**  
  Using identical bandpasses is a **deliberate but empirically testable** choice:
  - Future experiments should compare downstream performance when:
    - Both mics use the same bandpass, versus
    - Slightly different bandpasses (e.g. allowing more low-frequency content on the ambient channel).
  - If performance does not improve with differentiated filters, the symmetric design remains preferable for its simplicity and interpretability.

---

### 4.C Stretch Sensor (Channel 3)

**Objective:** Measure chest wall expansion during breathing and coughing.

#### Spectral observations

- **PSD:**  
  The stretch sensor shows most of its energy in the **0–5 Hz** region. Beyond ~15–20 Hz, the PSD flattens into a noise floor.
- **Spectrogram:**  
  Useful activity (breathing, cough-related expansion) appears only in the very low-frequency bins; higher-frequency bins show no structured patterns.

#### Filter decision: low-pass [20 Hz]

- **Reasoning based on the data:**
  - In your recordings, almost all clear physiological patterns in the stretch signal lie below ~5–10 Hz.
  - The region above ~15–20 Hz is dominated by electronic and quantization noise.

- **Design choice (corrected statement):**
  - It is **not strictly correct** to claim that “no physiological chest movement occurs faster than 15 Hz.”  
  - A more accurate description is:
    - In this dataset, chest expansion dynamics relevant to breathing and coughing produce low-frequency components predominantly below approximately 5–10 Hz,  
    - While the PSD beyond ~15–20 Hz behaves like unstructured noise.

- **Cutoff at 20 Hz:**
  - Chosen to:
    - Fully include the slow respiratory components and cough-related expansion,
    - Provide a margin above the visually observed useful band, and
    - Strongly attenuate higher-frequency noise.

This aggressive low-pass filter produces a stretch signal that primarily reflects the envelope of chest movement, which is appropriate for both respiration and cough-related analyses.

---

### 4.D Accelerometer (Channel 4)

**Objective:** Track patient orientation (posture) and gross activity level, and provide complementary information about cough-related body motion.

#### Spectral observations

- **Gravity (DC):**  
  PSD displays a massive spike at **0 Hz**, corresponding to the gravity vector (\(\approx 1g\)). The DC level differs between postures (lying, sitting, standing).
- **Low-frequency motion:**  
  Walking and other voluntary movements produce energy predominantly in the low-frequency region (a few Hz).
- **High-frequency tail:**  
  Above ~20 Hz, the PSD is comparatively flat and consistent with sensor noise or very small, high-frequency vibrations.

#### Filter decision: low-pass [20 Hz], no high-pass (DC preserved)

- **Motion frequency ranges in the literature:**
  - Merryn J. Mathie’s work (“Monitoring and Interpreting Human Movement Patterns Using a Triaxial Accelerometer”, PhD thesis, University of New South Wales, 2003) and related accelerometer studies consistently report that the **dominant components of human gait** are located below approximately 3–4 Hz.
  - Karantonis et al. (“Implementation of a Real-Time Human Movement Classifier Using a Triaxial Accelerometer for Ambulatory Monitoring”, IEEE Transactions on Information Technology in Biomedicine, 2006) use similar ranges when designing movement classifiers.

  These sources justify treating frequency content above a few tens of Hz as unlikely to carry meaningful gross movement information in this context.

- **Design rationale (corrected and refined):**
  - A 20 Hz low-pass:
    - Preserves:
      - The **gravity-induced DC component** (0 Hz), which is essential for posture estimation (e.g. sitting vs. lying),
      - The low-frequency dynamics of gait and other voluntary movements (up to a few Hz),
      - The main low-frequency components of cough-related body “jerk”.
    - Attenuates:
      - High-frequency electronic noise, and
      - Minor sensor ringing that does not reflect stable or interpretable movement.

  - It would be misleading to claim that this filter “prevents the classifier from confusing a cough with running” solely by cutting high frequencies. Discriminating cough from other activities will still require appropriate **feature design** and **temporal pattern analysis**. The filter’s role is primarily to deliver a clean, low-frequency movement signal with preserved DC.

- **Missing design decision: handling very slow drift vs. posture:**
  - While DC is required for posture estimation, long recordings may show **very slow baseline drift** due to sensor bias changes, belt tension, or temperature.
  - A future refinement should decide whether to:
    - Use **absolute DC levels** as posture features,  
    - Use **long-window averages** (e.g. per minute) of the accelerometer signal to stabilise posture estimates, or  
    - Calibrate posture per recording (e.g. using labelled calibration segments).
  - This is a **separate design choice** that interacts with filter design but is not yet fully specified.

---

## 5. Summary of Preprocessing Parameters

The table below summarises the empirically derived filter specifications. Exact implementation details (family, order, etc.) follow the guidelines in Section 3.1.

| Sensor Channel     | Filter Type | Cutoff Frequency     | Main Rationale                                                                                     |
| :----------------- | :---------- | :------------------- | :-------------------------------------------------------------------------------------------------- |
| **Pulmonary Mic**  | Bandpass    | **60 Hz – 2200 Hz**  | Suppresses 0–60 Hz drift and hum; retains main cough band; respects 4800 Hz sampling constraints.  |
| **Ambient Mic**    | Bandpass    | **60 Hz – 2200 Hz**  | Uses identical filter as pulmonary mic to allow clean energy comparison between channels.          |
| **Stretch Sensor** | Low-pass    | **20 Hz**            | Retains slow chest movement (0–5 Hz), removes noise above ~15–20 Hz.                               |
| **Accelerometer**  | Low-pass    | **20 Hz**            | Preserves DC gravity for posture; retains gait and gross movement; attenuates high-frequency noise.|

All cutoff values and filter shapes should be regarded as **hyperparameters** that can be tuned based on the quantitative validation strategy in Section 3.2 (both spectral metrics and classification performance).

---

## 6. References and Scientific Justification

This section lists the main sources used to support the physiological and signal-processing assumptions underlying the filter design. The bibliographic entries below follow the form used by the original publications.

1. **Shi, Y., Liu, H., Wang, Y., Cai, M., & Xu, W.** (2018). *Theory and Application of Audio-Based Assessment of Cough*. Journal of Sensors, 2018, Article ID 9845321.  
   DOI: 10.1155/2018/9845321  
   Link: https://doi.org/10.1155/2018/9845321  
   - Provides a comprehensive review of cough acoustics and audio-based cough monitoring systems, including typical bandwidths used in practical systems (up to a few kilohertz).

2. **Reichert, S., Gass, R., Brandt, C., & Andrès, E.** (2008). *Analysis of Respiratory Sounds: State of the Art*. Clinical Medicine: Circulatory, Respiratory and Pulmonary Medicine, 2, 45–58.  
   DOI: 10.4137/CCRPM.S530  
   Link: https://journals.sagepub.com/doi/10.4137/CCRPM.S530  
   - Reviews the characteristics of normal and adventitious respiratory sounds and the frequency ranges typically considered in lung sound analysis.

3. **Mathie, M. J.** (2003). *Monitoring and Interpreting Human Movement Patterns Using a Triaxial Accelerometer*. PhD thesis, University of New South Wales, Sydney, Australia.  
   Link: http://handle.unsw.edu.au/1959.4/27386  
   - Describes accelerometer-based characterisation of human movement, showing that dominant gait frequencies are typically below a few Hz.

4. **Karantonis, D. M., Narayanan, M. R., Mathie, M., Lovell, N. H., & Celler, B. G.** (2006). *Implementation of a Real-Time Human Movement Classifier Using a Triaxial Accelerometer for Ambulatory Monitoring*. IEEE Transactions on Information Technology in Biomedicine, 10(1), 156–167.  
   DOI: 10.1109/TITB.2005.856864  
   Link: https://doi.org/10.1109/TITB.2005.856864  
   - Demonstrates practical accelerometer-based movement classification and supports the choice of low-frequency focus for gross human movement.

5. **Otoshi, T., Nagano, T., Izumi, S., Hazama, D., Katsurada, N., Yamamoto, M., Tachihara, M., Kobayashi, K., & Nishimura, Y.** (2021). *A Novel Automatic Cough Frequency Monitoring System Combining a Triaxial Accelerometer and a Stretchable Strain Sensor*. Scientific Reports, 11, 9973.  
   DOI: 10.1038/s41598-021-89457-0  
   Link: https://doi.org/10.1038/s41598-021-89457-0  
   - Provides empirical evidence that a combination of accelerometer and strain sensor is effective for cough monitoring, supporting the choice of these modalities in this project.

These references are used for **qualitative guidance** (typical frequency ranges, sensor choices, and movement characteristics). The **exact numerical cutoffs** (e.g. 60 Hz, 2200 Hz, 20 Hz) are primarily derived from the project’s own PSD and spectrogram analysis and are explicitly treated as **hyperparameters** to be validated empirically as described in Section 3.2.