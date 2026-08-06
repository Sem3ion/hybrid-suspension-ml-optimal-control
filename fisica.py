# -*- coding: utf-8 -*-
"""
fisica.py — Modello dinamico quarter-car a 2 GDL, integratore RK4, ricostruzione del
profilo strada dai dati e stima causale della frequenza di eccitazione.

LEGENDA VARIABILI (valida per tutto il pacchetto):
    z_r = quota della STRADA sotto la ruota        [m]   (ingresso/disturbo)
    z_u = quota verticale della RUOTA              [m]   (massa non sospesa)
    z_s = quota verticale della CASSA              [m]   (massa sospesa, "il cielo")
    apice '  = velocita' [m/s] ; apice '' = accelerazione [m/s^2]
    a_z = z_s'' : accelerazione verticale della cassa (cio' che misura l'IMU)
    vel_rel = z_s' - z_u' : velocita' RELATIVA di sospensione (corsa)
Stato integrato x = [x1, x2, x3, x4]:
    x1 = z_s - z_u  (corsa sospensione)        x2 = z_s'  (velocita' cassa)
    x3 = z_u - z_r  (deflessione pneumatico)   x4 = z_u'  (velocita' ruota)
"""
import numpy as np
import scipy.signal as signal


def derivata_stato(x, c, forza_att, zr_dot, auto):
    """Derivata dello stato dx/dt del quarter-car. UNICA implementazione della dinamica.

    Funziona sia sul caso singolo, x di forma (4,), sia su un BATCH di casi, x di forma
    (..., 4) con c / forza_att / zr_dot di forma (...) — l'indicizzazione x[..., i] e lo
    stack sull'ultimo asse coprono entrambi senza rami separati.

    E' l'UNICO punto in cui vive il modello fisico. Tenerne due copie, una scalare e una
    vettorizzata, significherebbe che una modifica al modello puo' finire in una sola, con i
    due rami che divergono in silenzio e nessun test che lo noti."""
    m_s, m_u = auto.massa_cassa, auto.massa_ruota
    k_s, k_t = auto.rigid_molla, auto.rigid_gomma
    x = np.asarray(x)
    x1, x2, x3, x4 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    vel_rel = x2 - x4
    return np.stack([
        vel_rel,                                                      # z_s' - z_u'
        (-k_s * x1 - c * vel_rel + forza_att) / m_s,                   # z_s'' (accel. cassa)
        x4 - zr_dot,                                                   # z_u' - z_r'
        (k_s * x1 + c * vel_rel - forza_att - k_t * x3) / m_u,          # z_u'' (accel. ruota)
    ], axis=-1)


def passo_rk4(x, c, forza_att, zr0, zr1, auto, h):
    """Un passo di Runge-Kutta 4 (h = passo). zr0, zr1 = velocita' strada a inizio/fine passo.
    Come derivata_stato, vale sia per uno stato singolo sia per un batch di stati."""
    zrm = 0.5 * (zr0 + zr1)
    k1 = derivata_stato(x, c, forza_att, zr0, auto)
    k2 = derivata_stato(x + 0.5 * h * k1, c, forza_att, zrm, auto)
    k3 = derivata_stato(x + 0.5 * h * k2, c, forza_att, zrm, auto)
    k4 = derivata_stato(x + h * k3, c, forza_att, zr1, auto)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rimuovi_offset(az, fs, fc=0.1):
    """Passa-alto zero-fase per togliere offset/deriva dall'accelerazione misurata."""
    b, a = signal.butter(2, fc / (0.5 * fs), btype="highpass")
    return signal.filtfilt(b, a, az)


def despica_hampel(x, finestra=7, n_sigma=3.0):
    """Filtro di Hampel: sostituisce con la mediana locale i campioni che distano piu' di
    n_sigma*MAD dalla mediana. Toglie i PICCHI anomali del sensore ("valori pericolosi")
    senza appiattire il segnale utile. MAD = deviazione mediana assoluta."""
    from scipy.ndimage import median_filter
    x = np.asarray(x, dtype=float)
    med = median_filter(x, size=finestra, mode="nearest")
    mad = median_filter(np.abs(x - med), size=finestra, mode="nearest")
    soglia = n_sigma * 1.4826 * mad + 1e-9        # 1.4826: converte MAD in sigma (rumore gaussiano)
    fuori = np.abs(x - med) > soglia
    y = x.copy(); y[fuori] = med[fuori]
    return y


def ricostruisci_strada(az, cfg):
    """Doppia integrazione (con passa-alto anti-deriva) di a_z -> profilo strada z_r e velocità z_r'.
    L'AMPIEZZA qui e' solo nominale: verra' CALIBRATA sui dati in controllo.py."""
    fs = cfg.freq_campion
    b, a = signal.butter(2, 0.3 / (0.5 * fs), btype="highpass")
    vel = signal.filtfilt(b, a, np.cumsum(az) / fs)      # 1a integrazione -> velocita'
    zr  = signal.filtfilt(b, a, np.cumsum(vel) / fs)     # 2a integrazione -> spostamento
    rms = np.sqrt(np.mean(zr ** 2)) + 1e-9
    zr = zr * (cfg.strada_rms_nom / rms)                 # scala nominale (poi ricalibrata)
    return zr.astype(float), np.gradient(zr, cfg.passo_t).astype(float)


def _kalman_pos_vel(az, cfg):
    """Filtro di Kalman CINEMATICO a 2 stati x = [z_s, z_s'] (posizione & velocita' cassa).
    L'accelerazione misurata a_z e' l'INGRESSO (rumoroso) che integra il modello:
        z_s(k+1) = z_s + dt*z_s' + 0.5*dt^2*a_z ,   z_s'(k+1) = z_s' + dt*a_z
    Per impedire la DERIVA dell'integrazione si usa una pseudo-misura "posizione ~ 0"
    (lo spostamento di comfort e' a media nulla): agisce come passa-alto OTTIMO.
    sigma_a (rumore accel.) e' stimato DAI DATI (MAD robusto sulle differenze).
    Restituisce la velocita' cassa denoised, causale (usa solo il passato)."""
    dt = cfg.passo_t
    F = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    diff = np.diff(np.asarray(az, dtype=float))
    sigma_a = 1.4826 * np.median(np.abs(diff - np.median(diff))) + 1e-6   # rumore accel dai dati
    Q = (B @ B.T) * (sigma_a ** 2)                       # rumore di processo dall'ingresso rumoroso
    H = np.array([[1.0, 0.0]])                            # osserviamo (fittiziamente) la posizione
    R = np.array([[cfg.kalman_scala_pos ** 2]])           # varianza attesa dello spostamento
    x = np.zeros((2, 1)); P = np.eye(2) * 1e-4; I = np.eye(2)
    N = len(az); vel = np.zeros(N)
    for k in range(N):
        x = F @ x + B * az[k]                            # predizione con a_z come ingresso
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R                              # correzione: pseudo-misura posizione = 0
        K = P @ H.T / S
        x = x + K * (0.0 - (H @ x))
        P = (I - K @ H) @ P
        vel[k] = x[1, 0]
    return vel


def velocita_cassa_misurata(az, cfg):
    """Velocita' verticale della cassa dall'a_z del sensore (grandezza su cui agisce lo
    skyhook). Con cfg.usa_kalman -> filtro di Kalman (denoising ottimo, anti-deriva);
    altrimenti -> integrale singolo passa-alto (piu' semplice, piu' rumoroso)."""
    if cfg.usa_kalman:
        return _kalman_pos_vel(az, cfg)
    b, a = signal.butter(2, 0.3 / (0.5 * cfg.freq_campion), btype="highpass")
    return signal.lfilter(b, a, np.cumsum(az) / cfg.freq_campion)


def _freq_velocita(v, cfg, auto):
    """r = omega/omega_n, con omega = 2π·v/lambda_c (frequenza temporale dell'eccitazione a
    velocita' v). Il CROSSOVER della trasmissibilita' e' SEMPRE a r=√2 (fisica): NON e' una
    velocita' da scegliere. Qui si calcola SOLO r; sono xi_da_r/kp_da_r a pivotare su √2
    (r<√2 amplificazione -> xi alto; r>√2 isolamento -> xi basso).
      - omega_n = sqrt(k_s/m_s) dal VEICOLO;
      - lambda_c (cfg.lambda_c_design) e' l'UNICA assunzione: la scala di lunghezza d'onda del
        legame v->frequenza. La velocita' a cui r=√2 NON e' fissa: cambia con lambda reale.
    Robusto: la legge in v non e' falsata dalle risonanze del veicolo (come la FFT dell'a_z)."""
    omega = 2.0 * np.pi * np.asarray(v, dtype=float) / cfg.lambda_c_design
    return np.clip(omega / auto.puls_nat_cassa, 0.15, cfg.r_max).astype(np.float32)


def stima_frequenza(az, v, cfg, auto):
    """Rapporto r = omega/omega_n. Metodi (cfg.metodo_freq):
       'velocita' -> dalla VELOCITA' (eccitazione stradale ~ v): ADATTIVO e robusto (default);
       'fft'      -> centroide spettrale dell'a_z (falsato dalle risonanze del veicolo);
       'rms'      -> rapporto RMS dell'a_z."""
    if cfg.metodo_freq == "velocita":
        return _freq_velocita(v, cfg, auto)
    if az is None:
        # xi_ottimo chiama con az=None quando gli serve solo la parte in velocita'.
        # Meglio dirlo che schiantarsi con un TypeError oscuro dentro la FFT.
        raise ValueError("stima_frequenza: metodo_freq=%r richiede a_z, ma e' stato passato "
                         "None. Usa metodo_freq='velocita' oppure fornisci a_z."
                         % cfg.metodo_freq)
    if cfg.metodo_freq == "fft":
        return _freq_fft(az, cfg, auto)
    return _freq_rms(az, cfg, auto)


def _freq_rms(az, cfg, auto):
    fs = cfg.freq_campion
    b, a = signal.butter(2, 0.3 / (0.5 * fs), btype="highpass")
    vz = signal.lfilter(b, a, np.cumsum(az) / fs)        # integrale causale di a_z
    w = cfg.finestra_freq

    def rms_trailing(s):
        somma = np.convolve(s * s, np.ones(w), mode="full")[:len(s)] / w
        return np.sqrt(np.maximum(somma, 0.0) + 1e-12)

    omega = rms_trailing(az) / (rms_trailing(vz) + 1e-9)
    return np.clip(omega / auto.puls_nat_cassa, 0.15, cfg.r_max)


def _freq_fft(az, cfg, auto):
    """Centroide spettrale su finestra TRAILING via FFT (batch = veloce, e causale):
    r[i] usa gli ultimi finestra_freq campioni fino a i. La finestra di Hann riduce il
    leakage spettrale (condizionamento in frequenza)."""
    from numpy.lib.stride_tricks import sliding_window_view
    N = len(az); w = cfg.finestra_freq
    if N <= w:
        return np.full(N, 1.0, np.float32)
    W = sliding_window_view(az, w)                       # riga j = finestra che finisce al campione j+w-1
    hann = np.hanning(w)
    Wf = (W - W.mean(axis=1, keepdims=True)) * hann      # togli media + finestratura
    potenza = np.abs(np.fft.rfft(Wf, axis=1)) ** 2
    freq_hz = np.fft.rfftfreq(w, d=cfg.passo_t)
    # considera solo la BANDA dinamica della sospensione (toglie DC/quasi-statico e rumore alto)
    lo, hi = cfg.freq_banda_hz
    banda = (freq_hz >= lo) & (freq_hz <= hi)
    potenza = potenza[:, banda]; freq_hz = freq_hz[banda]
    centro_hz = (potenza * freq_hz).sum(1) / (potenza.sum(1) + 1e-12)   # frequenza media pesata [Hz]
    r_win = np.clip(2 * np.pi * centro_hz / auto.puls_nat_cassa, 0.15, cfg.r_max).astype(np.float32)
    r = np.empty(N, np.float32)
    r[w - 1:] = r_win                                    # allineamento causale (fine finestra)
    r[:w - 1] = r_win[0]
    return r


def rms_accel_passiva(zr_dot, auto, cfg):
    """RMS dell'accelerazione cassa del modello PASSIVO (c_nom, senza attuatore).
    Serve per CALIBRARE l'ampiezza strada sui dati (vedi controllo.py).

    E' l'RMS di accel_passiva_serie, e si limita a calcolarlo: scrivere qui un secondo ciclo
    RK4 che accumula il quadrato duplicherebbe il modello e, dove entrambe vengono chiamate,
    farebbe girare la simulazione due volte per gli stessi numeri."""
    return float(np.sqrt(np.mean(accel_passiva_serie(zr_dot, auto, cfg) ** 2)))


def accel_passiva_serie(zr_dot, auto, cfg):
    """Accelerazione cassa (serie temporale) del modello PASSIVO su un profilo strada.
    Serve a generare l'a_z 'misurata' delle strade SINTETICHE (data augmentation)."""
    N = len(zr_dot); xp = np.zeros(4); az = np.zeros(N); h = cfg.passo_t / cfg.n_sottopassi
    for k in range(N):
        az[k] = (-auto.rigid_molla * xp[0] - auto.smorz_nom * (xp[1] - xp[3])) / auto.massa_cassa
        zr0 = zr_dot[k]; zr1 = zr_dot[k + 1] if k + 1 < N else zr_dot[k]
        for _ in range(cfg.n_sottopassi):
            xp = passo_rk4(xp, auto.smorz_nom, 0.0, zr0, zr1, auto, h)
    return az


def strada_demo(auto, cfg, secondi=None):
    """Profilo strada dimostrativo: fondo liscio + dossi periodici attorno alla risonanza
    (dove l'effetto e' massimo). Restituisce (strada z_r, z_r', velocita' v).

    'secondi' permette di generarne un tratto piu' lungo di cfg.anim_secondi: serve per
    dare all'animazione un warm-up da scartare (la finestra della rete parte piena di zeri).
    E' un banco di prova severo ma AMMISSIBILE: xi* spazia su quasi tutto l'intervallo utile
    (~0.08-0.40) e l'ottimo resta dentro il vincolo di distacco ruota. La severita' e' tarata
    su quel vincolo tramite cfg.anim_dosso, non scelta a occhio: una strada che fa violare il
    vincolo anche all'OTTIMO rende il confronto fra controllori privo di significato, perche'
    a quel punto nessuno dei quattro e' realizzabile su quel fondo."""
    N = int((cfg.anim_secondi if secondi is None else secondi) * cfg.freq_campion)
    rng = np.random.default_rng(0)
    # VELOCITA' che cresce nel tempo: cosi' xi spazia da alto (comfort, bassa v) a basso
    # (isolamento, alta v) -> si VEDE lo smorzamento variabile (non e' un due-stati binario).
    vel = np.linspace(5.0, 25.0, N)                    # ~18 -> ~90 km/h
    x_pos = np.cumsum(vel * cfg.passo_t)
    zr = np.zeros(N)
    for xb in np.arange(8.0, x_pos[-1], 7.0):          # DOSSI periodici
        zr += cfg.anim_dosso * np.exp(-((x_pos - xb) ** 2) / (2 * 0.45 ** 2))
    for xb in np.arange(22.0, x_pos[-1], 24.0):        # BUCHE (avvallamenti) sparse
        xc = xb + rng.uniform(-3, 3)
        zr -= 1.4 * cfg.anim_dosso * np.exp(-((x_pos - xc) ** 2) / (2 * 0.35 ** 2))
    zr += cfg.anim_texture * rng.standard_normal(N)    # texture fine
    return zr, np.gradient(zr, cfg.passo_t), vel
