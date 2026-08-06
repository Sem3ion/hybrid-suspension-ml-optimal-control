# -*- coding: utf-8 -*-
"""
strade_sintetiche.py — GENERAZIONE DI STRADE SINTETICHE come disegno sperimentale.

A COSA SERVE
------------
Le tracce registrate disponibili sono tre, tutte di guida urbana a velocita' simile su
fondo relativamente buono. Con soli quei dati il legame fra rugosita' della strada e
controllo ottimo non e' osservabile: mancano interamente i casi che decidono, cioe' fondo
sconnesso e alta velocita'. Queste strade sintetiche riempiono quel vuoto secondo ISO 8608
e diventano il grosso del dataset di addestramento; le tracce reali restano il metro finale.

IL PUNTO CENTRALE: E' UN DISEGNO SPERIMENTALE, NON UN GENERATORE DI DATI
-----------------------------------------------------------------------
La rete deve imparare a distinguere la RUGOSITA' della strada dalla VELOCITA', perche' il
controllo ottimo dipende molto dalla prima e poco dalla seconda, mentre l'accelerazione che
il sensore misura dipende da entrambe (in prima approssimazione RMS a_z ~ sqrt(rugosita' x v)).
Se nel dataset le due grandezze variano insieme, sono statisticamente indistinguibili e
nessuna rete puo' separarle: impara la combinazione che il dataset le presenta, e sbaglia
appena la incontra scomposta diversamente.

Da qui tre requisiti, che sono il motivo per cui questo file esiste in questa forma:

  1. GRIGLIA FATTORIALE classe x velocita' (5 x 5 = 25 celle) invece di coppie fissate.
     Ogni classe ISO va vista a piu' velocita' e ogni velocita' su piu' classi, corner
     compresi: classe E a 100 km/h e classe A a 20 km/h devono esistere entrambe.
     Le celle riservate alla VALIDAZIONE sono DISGIUNTE da quelle di training, cosi' la
     validazione misura generalizzazione a combinazioni nuove e non memoria.

  2. VELOCITA' VARIABILE DENTRO LA TRACCIA. Meta' delle tracce sono rampe, sweep o profili
     stop-and-go che percorrono la STESSA strada a velocita' molto diverse. E' il dato piu'
     informativo del dataset: rugosita' costante, firma spettrale costante, controllo ottimo
     che cambia — l'unico modo di mostrare alla rete che le due cause sono separabili.

  3. JITTER DI RUGOSITA' indipendente dalla classe (Gd0 moltiplicato per U[0.5, 2]).
     Senza, l'RMS di a_z sarebbe una funzione deterministica della coppia (classe, v) e
     quindi un identificatore della traccia: la rete potrebbe riconoscere quale traccia sta
     guardando invece di stimare la strada.

COSA CONTIENE UNA STRADA
------------------------
  * fondo ISO 8608: Gd(n) = Gd(n0)*(n/n0)^(-2), con n = frequenza spaziale [cicli/m] e
    n0 = 0.1. Sintesi a somma di sinusoidi con fasi casuali e frequenze spaziate in modo
    LOGARITMICO: con una PSD che va come n^-2 la spaziatura lineare metterebbe quasi tutti
    i toni dove l'energia e' trascurabile;
  * ostacoli isolati (buche e dossi), serie ravvicinate tipo washboard, gradini e giunti di
    dilatazione a fronte ripido;
  * orografia: pendenza costante piu' ondulazioni lente.

Ogni traccia restituisce anche il profilo z_r' ESATTO, non solo a_z: le etichette si
calcolano su quello. Ricostruire la strada dall'accelerazione (doppia integrazione con
passa-alto) ne distorce la forma spettrale proprio nella banda bassa da cui dipendono i
vincoli di tenuta e corsa, e sposta il controllo ottimo di quanto vale l'intero errore
della rete.

VINCOLI DI CAMPIONAMENTO
------------------------
Tutto il resto della pipeline lavora a 100 Hz, quindi il contenuto generato deve essere
rappresentabile a quella frequenza:
  * la frequenza spaziale massima e' limitata a margine * fs / v_max, altrimenti a velocita'
    alta il profilo supera la Nyquist e l'aliasing riporta energia spuria in banda bassa —
    proprio dove vive il modo della cassa;
  * la larghezza minima di un ostacolo e' ~2.5 passi spaziali: sotto quella soglia non
    produce un urto, produce solo aliasing;
  * la pendenza dei fianchi e l'ampiezza dell'orografia sono limitate da vincoli fisici
    (una buca profonda e' anche larga; una collina ha un raccordo verticale), altrimenti si
    generano eccitazioni che nessuna strada reale produce.
"""
import numpy as np

from fisica import accel_passiva_serie, despica_hampel

# Gd(n0) [m^3] per classe ISO 8608 (A liscia -> E molto sconnessa), n0 = 0.1 cicli/m
_GD0 = {"A": 16e-6, "B": 64e-6, "C": 256e-6, "D": 1024e-6, "E": 4096e-6}

# tipi di profilo di velocita' disponibili (vedi _profilo_velocita)
_TIPI_V = ("costante", "rampa", "sweep", "urbano")


# --- profilo strada ---
def _profilo_iso8608(gd0, x, rng, n_min=0.011, n_max=2.83, n_freq=400):
    """Profilo strada z_r(x) [m] con PSD ISO 8608 di ampiezza gd0 (Gd al riferimento n0=0.1).

    gd0 e' passato ESPLICITO (non derivato dalla sola classe) cosi' si puo' applicare
    il jitter di rugosita' che rompe il legame deterministico classe -> RMS(a_z).
    Frequenze spaziali LOG-spaziate: con PSD ~ n^-2 la spaziatura lineare mette
    quasi tutti i toni dove l'energia e' trascurabile.
    """
    n = np.logspace(np.log10(n_min), np.log10(n_max), n_freq)   # [cicli/m]
    dn = np.gradient(n)                                         # larghezza di banda per tono
    Gd = gd0 * (n / 0.1) ** (-2)                                # densita' spettrale [m^3]
    amp = np.sqrt(2.0 * Gd * dn)                                # ampiezza di ogni sinusoide
    phi = rng.uniform(0, 2 * np.pi, n_freq)
    return (amp[None, :] * np.cos(2 * np.pi * np.outer(x, n) + phi[None, :])).sum(axis=1)


def _aggiungi_ostacoli(zr, x, rng, cfg, severita=1.0, densita=1.0):
    """Buche, dossi, serie ravvicinate e gradini/giunti. Restituisce (zr, n_eventi).

    Tre vincoli fisici, tutti necessari perche' gli eventi siano DURI ma REALI:
      * DENSITA' PER CHILOMETRO, non per traccia. Un numero fisso di buche su una
        traccia corta (o percorsa piano) darebbe una strada assurdamente martellata.
      * LARGHEZZA >= ~2.5 passi spaziali. Sotto quella soglia l'ostacolo non e'
        campionabile a fs e non produce un urto: produce solo ALIASING.
      * PENDENZA DEL FIANCO limitata (prof/largh <= 0.30). Una buca profonda 15 cm
        e larga 15 cm sarebbe un gradino verticale: le buche profonde sono anche larghe.
    'severita' (0.5..1.7) scala la profondita' con la classe ISO.
    """
    passo_x = float(np.max(np.diff(x)))                 # passo spaziale peggiore [m]
    largh_min = 2.5 * passo_x                           # ostacolo rappresentabile a fs
    lung = max(float(x[-1] - x[0]) - 10.0, 1.0)         # tratto utile [m]
    per_km = lung / 1000.0
    n_eventi = 0

    def _posa(xb, prof, largh):
        largh = max(largh, largh_min)
        # fianco non verticale E profondita' fisicamente possibile (max 12 cm)
        prof = float(np.sign(prof)) * min(abs(prof), 0.30 * largh, 0.12)
        return prof * np.exp(-((x - xb) ** 2) / (2 * largh ** 2))

    # --- buche e dossi isolati: ~5 (strada buona) .. ~16 (strada rovinata) per km ---
    n_iso = int(rng.poisson(max(0.5, (1 + 8 * severita) * per_km * densita)))
    for _ in range(n_iso):
        xb = rng.uniform(x[0] + 5, x[-1] - 5)
        prof = rng.uniform(0.03, 0.12) * severita * rng.choice([-1.0, 1.0])
        zr = zr + _posa(xb, prof, rng.uniform(0.20, 0.80))
        n_eventi += 1

    # --- serie ravvicinate (tratto sconnesso / washboard): ~1 ogni km ---
    for _ in range(int(rng.poisson(max(0.15, 1.0 * per_km * densita)))):
        x0 = rng.uniform(x[0] + 5, max(x[0] + 6, x[-1] - 20))
        passo = rng.uniform(1.0, 2.5)                   # distanza fra le buche [m]
        for j in range(int(rng.integers(3, 7))):
            prof = -rng.uniform(0.02, 0.09) * severita
            zr = zr + _posa(x0 + j * passo, prof, rng.uniform(0.20, 0.55))
            n_eventi += 1

    # --- gradini / giunti di dilatazione: fronte ripido (transitorio ad alta banda) ---
    for _ in range(int(rng.poisson(max(0.15, 2.0 * per_km * densita)))):
        xb = rng.uniform(x[0] + 5, x[-1] - 5)
        alt = rng.uniform(0.008, 0.035) * severita * rng.choice([-1.0, 1.0])
        # coppia di fronti (sale e ridiscende): giunto/tombino di lunghezza finita
        larg_f = max(0.12, largh_min * 0.6)             # ripidita' del fronte [m]
        lun_g = rng.uniform(0.3, 1.2)
        zr = zr + alt * 0.5 * (np.tanh((x - xb) / larg_f) - np.tanh((x - xb - lun_g) / larg_f))
        n_eventi += 1

    return zr, n_eventi


def _salite_discese(x, rng):
    """Orografia: pendenza costante + 2-3 ondulazioni lente (lambda 40..300 m).

    La componente quasi-statica viene tolta dal passa-alto a valle: quello che conta
    per a_z e' la VARIAZIONE di pendenza (i raccordi in cima/fondo a salite e discese),
    che con piu' armoniche diventa molto piu' ricca di una sola sinusoide.

    Due vincoli fisici:
      * lambda >= 40 m. Un'ondulazione da 20 m percorsa a 27 m/s vale 1.35 Hz, cioe'
        proprio la frequenza propria della cassa (1.5 Hz): non e' una collina, e'
        un'eccitazione in risonanza, e da sola generava piu' accelerazione di tutta
        la rugosita' ISO. L'orografia deve stare SOTTO la banda della sospensione.
      * ampiezza limitata dal RACCORDO VERTICALE. Su un profilo sinusoidale (A, lambda)
        percorso a v l'accelerazione verticale imposta e' A*(2*pi*v/lambda)^2. Imponendo
        che a v_rif = 30 m/s resti sotto ~0.5 m/s^2 (quello che si sente su un dosso
        stradale raccordato a norma) si ottiene A <= 1.4e-5 * lambda^2. Senza questo
        vincolo una "collina" da 40 m e 30 cm sarebbe un salto da rally.
    """
    z = rng.uniform(-0.06, 0.06) * (x - x[0])           # pendenza costante fino al 6%
    for _ in range(int(rng.integers(2, 4))):
        lam = rng.uniform(40.0, 300.0)                  # lunghezza d'onda collina [m]
        amp = rng.uniform(0.3, 1.0) * 1.4e-5 * lam ** 2  # dislivello [m] (raccordo verticale)
        z = z + amp * np.sin(2 * np.pi * x / lam + rng.uniform(0, 2 * np.pi))
    return z


# --- profilo velocita' ---
def _v_crossover(auto, cfg):
    """Velocita' [m/s] a cui r = √2 (crossover della trasmissibilita'), dai parametri
    del veicolo e da lambda_c. Serve per costruire rampe che lo ATTRAVERSANO."""
    return np.sqrt(2.0) * auto.puls_nat_cassa * cfg.lambda_c_design / (2.0 * np.pi)


def _rumore_lento(N, rng, tau_campioni=200):
    """Rumore correlato a media nulla e ampiezza ~1 (variazioni realistiche di guida)."""
    b = rng.normal(0, 1, N)
    k = np.ones(int(tau_campioni)) / float(tau_campioni)
    s = np.convolve(b, k, mode="same")
    return s / (np.std(s) + 1e-9)


def _profilo_velocita(tipo, v0, N, cfg, auto, rng):
    """Profilo v(t) [m/s]. E' QUI che si disaccoppiano velocita' e rugosita':
    con rampe/sweep la STESSA strada (stessa firma spettrale di a_z) viene percorsa
    a velocita' molto diverse, quindi la stessa a_z corrisponde a xi diversi e la
    scorciatoia 'a_z -> xi' diventa non identificabile."""
    v_cross = _v_crossover(auto, cfg)                   # ~14.3 m/s ≈ 51 km/h

    if tipo == "costante":
        v = np.full(N, v0) + 0.3 * _rumore_lento(N, rng)

    elif tipo == "rampa":
        # arrivo scelto DALL'ALTRA PARTE del crossover -> la traccia lo attraversa
        if v0 < v_cross:
            v1 = rng.uniform(v_cross + 3.0, 31.0)
        else:
            v1 = rng.uniform(3.0, max(3.5, v_cross - 3.0))
        if rng.random() < 0.5:
            v0, v1 = v1, v0                             # accelerazione o decelerazione
        v = np.linspace(v0, v1, N) + 0.4 * _rumore_lento(N, rng)

    elif tipo == "sweep":
        # accelera/decelera piu' volte attorno al crossover: copre tutto il range di r
        amp = rng.uniform(6.0, 12.0)
        periodo = rng.uniform(12.0, 30.0)               # [s]
        t = np.arange(N) * cfg.passo_t
        centro = 0.5 * (v0 + v_cross)
        v = centro + amp * np.sin(2 * np.pi * t / periodo + rng.uniform(0, 2 * np.pi))
        v = v + 0.4 * _rumore_lento(N, rng)

    else:  # "urbano": stop-and-go, forti accelerazioni longitudinali
        t = np.arange(N) * cfg.passo_t
        v = np.zeros(N)
        for _ in range(int(rng.integers(2, 5))):
            per = rng.uniform(8.0, 20.0)
            v = v + rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * t / per + rng.uniform(0, 2 * np.pi))
        v = v0 * (0.55 + 0.75 * (v - v.min()) / (np.ptp(v) + 1e-9))
        v = v + 0.3 * _rumore_lento(N, rng)

    return np.clip(v, 3.0, 33.0)


# --- griglia fattoriale ---
def _griglia(cfg):
    """Tutte le celle (classe, velocita') del disegno fattoriale."""
    classi = list(cfg.aug_classi)
    velocita = list(cfg.aug_velocita)
    return [(c, v) for c in classi for v in velocita]


def _celle(cfg, n, per_validazione=False):
    """Estrae n celle dalla griglia. Il RESHUFFLE e' deterministico (aug_seed) e la
    coda della permutazione e' riservata alla validazione: training e validazione
    usano celle DISGIUNTE, quindi la validazione misura generalizzazione vera
    (combinazioni classe x velocita' mai viste), non memorizzazione."""
    g = _griglia(cfg)
    rng = np.random.default_rng(cfg.aug_seed)
    perm = list(rng.permutation(len(g)))
    n_val = int(cfg.n_tracce_sint_val)
    idx_val = perm[-n_val:] if n_val else []
    idx_tr = [i for i in perm if i not in idx_val]
    scelti = idx_val if per_validazione else idx_tr
    if not per_validazione and n > len(scelti):         # se servono piu' tracce, si cicla
        scelti = (scelti * (n // len(scelti) + 1))
    return [g[i] for i in scelti[:n]]


# --- generazione ---
def profilo_come_training(classe, v, cfg, rng, jitter=None):
    """Profilo z_r con gli STESSI ingredienti delle strade di addestramento: spettro ISO 8608
    della classe richiesta, piu' jitter di rugosita', salite/discese e ostacoli isolati.

    PERCHE' SERVE UNA FUNZIONE CONDIVISA. La diagnostica generava le sue strade con
    _profilo_iso8608 puro, cioe' senza ostacoli e senza jitter, mentre le strade di training
    li hanno sempre. A parita' di classe e velocita' le due popolazioni non coincidono: a
    97 km/h la classe A vale RMS a_z 0.46 m/s^2 pulita e 1.05 con ostacoli, la classe B 0.91
    contro 1.89. Circa un fattore due, cioe' uno scalino di classe intero.
    Conseguenza: la mappa closed-loop misurava la rete su una popolazione di strade che la
    rete non ha mai visto, e l'errore |xi - xi*| che ne usciva non era interpretabile — una
    parte era distanza fra i due GENERATORI, non fra rete ed esperto.
    Passando da qui, diagnostica e addestramento campionano la stessa popolazione."""
    x = np.cumsum(np.asarray(v, float) * cfg.passo_t)
    n_max = min(cfg.aug_freq_spaziale_max,
                cfg.aug_margine_nyquist * cfg.freq_campion / max(float(np.max(v)), 1e-6))
    lo, hi = cfg.aug_jitter_rugosita
    fattore = float(rng.uniform(lo, hi)) if jitter is None else float(jitter)
    gd0 = _GD0[classe] * fattore
    zr = _profilo_iso8608(gd0, x, rng, n_max=n_max)
    zr = zr + _salite_discese(x, rng)
    severita = 0.5 + 1.2 * (np.log10(gd0 / _GD0["A"]) / np.log10(_GD0["E"] / _GD0["A"]))
    zr, n_ev = _aggiungi_ostacoli(zr, x, rng, cfg, severita=severita)
    return zr, dict(jitter=fattore, n_ostacoli=n_ev, n_max=n_max)


def _una_traccia(classe, v0, tipo, auto, cfg, rng, verbose=True, etichetta=""):
    """Genera una singola traccia (a_z, v) + metadati."""
    N = int(cfg.aug_secondi * cfg.freq_campion)

    v = _profilo_velocita(tipo, v0, N, cfg, auto, rng)
    x = np.cumsum(v * cfg.passo_t)                      # avanzamento longitudinale [m]

    # ANTI-ALIASING: n * v_max <= margine * fs  (fs = 100 Hz -> Nyquist 50 Hz)
    n_max = min(cfg.aug_freq_spaziale_max,
                cfg.aug_margine_nyquist * cfg.freq_campion / max(float(np.max(v)), 1e-6))

    # JITTER di rugosita': rompe il legame deterministico classe -> RMS(a_z)
    lo, hi = cfg.aug_jitter_rugosita
    fattore = float(rng.uniform(lo, hi))
    gd0 = _GD0[classe] * fattore

    zr = _profilo_iso8608(gd0, x, rng, n_max=n_max)
    zr = zr + _salite_discese(x, rng)
    severita = 0.5 + 1.2 * (np.log10(gd0 / _GD0["A"]) / np.log10(_GD0["E"] / _GD0["A"]))
    zr, n_ev = _aggiungi_ostacoli(zr, x, rng, cfg, severita=severita)

    zr_dot = np.gradient(zr, cfg.passo_t)
    az_pulita = accel_passiva_serie(zr_dot, auto, cfg)  # a_z del modello PASSIVO

    # rumore del sensore realistico: gaussiano + qualche picco anomalo
    az = az_pulita + rng.normal(0, 0.05, N)
    for _ in range(int(rng.integers(2, 6))):
        az[rng.integers(N)] += rng.uniform(2, 6) * rng.choice([-1.0, 1.0])
    # DESPIKING anche qui. Prima i picchi venivano iniettati "tanto li togliera' il
    # despiking", ma despica_hampel era applicato SOLO alle tracce reali (in dati.py):
    # le sintetiche arrivavano alla rete con outlier che le reali non hanno, cioe' due
    # distribuzioni di ingresso diverse dentro lo stesso dataset.
    if cfg.despica:
        az = despica_hampel(az, cfg.hampel_finestra, cfg.hampel_sigma)

    # la traccia porta con se' z_r' ESATTO: le etichette non devono ricostruirlo da a_z
    meta = dict(classe=classe, tipo=tipo, fattore_rugosita=fattore,
                v_min=float(np.min(v)), v_max=float(np.max(v)), v_media=float(np.mean(v)),
                rms_az=float(np.sqrt(np.mean(az_pulita ** 2))), n_ostacoli=n_ev,
                n_max_spaziale=float(n_max))
    if verbose:
        print(f"      {etichetta} classe {classe} (x{fattore:.2f})  v {meta['v_min']*3.6:4.0f}"
              f"-{meta['v_max']*3.6:4.0f} km/h [{tipo:8s}]  RMS a_z={meta['rms_az']:5.2f} m/s^2"
              f"  ostacoli={n_ev:2d}  n_max={n_max:.2f} c/m")
    nome = (f"{etichetta.strip().rstrip(':')} ISO {classe}(x{fattore:.2f}) "
            f"{meta['v_min']*3.6:.0f}-{meta['v_max']*3.6:.0f} km/h [{tipo}]")
    return (az.astype(float), v.astype(float), zr_dot.astype(float), nome), meta


def genera_tracce_sintetiche(auto, cfg, ritorna_meta=False):
    """Tracce sintetiche per il TRAINING: disegno fattoriale classe x velocita',
    con meta' delle tracce a velocita' variabile che attraversa il crossover r=√2."""
    rng = np.random.default_rng(cfg.aug_seed)
    celle = _celle(cfg, int(cfg.n_tracce_sintetiche), per_validazione=False)

    # meta' (o cfg.aug_frazione_variabile) delle tracce ha velocita' VARIABILE
    n = len(celle)
    n_var = int(round(cfg.aug_frazione_variabile * n))
    tipi = ["costante"] * (n - n_var)
    for i in range(n_var):
        tipi.append(("rampa", "sweep", "urbano")[i % 3])
    tipi = list(rng.permutation(tipi))

    tracce, metas = [], []
    for i, ((classe, v0), tipo) in enumerate(zip(celle, tipi)):
        tr, meta = _una_traccia(classe, v0, tipo, auto, cfg, rng,
                                etichetta=f"train {i+1:2d}:")
        tracce.append(tr); metas.append(meta)
    return (tracce, metas) if ritorna_meta else tracce


def genera_tracce_aggregazione(auto, cfg, ritorna_meta=False):
    """Strade NUOVE per i rollout di DAgger: stesse celle (classe x velocita') del training,
    ma SEME diverso, quindi realizzazioni diverse dello stesso spettro ISO 8608.

    PERCHE' NON RIUSARE LE STRADE DI TRAINING. Facendo i rollout sulle strade su cui la rete
    si e' gia' addestrata, l'aggregazione aggiunge la distribuzione closed-loop ma nessuna
    varieta' di strada: la rete puo' riconoscere la firma spettrale di QUELLA realizzazione e
    ricordarsi l'etichetta associata, invece di imparare la relazione fra rugosita' e xi*. E'
    la stessa scorciatoia che il disegno fattoriale serviva a chiudere, riaperta dal lato dei
    dati aggregati. Con un seme diverso le celle restano quelle (si vuole coprire il dominio
    di addestramento, non estrapolare) ma i profili sono realizzazioni mai viste."""
    rng = np.random.default_rng(cfg.aug_seed + int(cfg.dagger_seme_strade))
    celle = _celle(cfg, int(cfg.n_tracce_sintetiche), per_validazione=False)
    n = len(celle)
    n_var = int(round(cfg.aug_frazione_variabile * n))
    tipi = (["rampa", "sweep", "urbano"] * (n_var // 3 + 1))[:n_var] + ["costante"] * (n - n_var)
    rng.shuffle(tipi)
    tracce, metas = [], []
    for k, ((classe, v0), tipo) in enumerate(zip(celle, tipi), 1):
        t, m = _una_traccia(classe, v0, tipo, auto, cfg, rng, verbose=False,
                            etichetta=f"aggreg{k:2d}")
        tracce.append(t); metas.append(m)
    return (tracce, metas) if ritorna_meta else tracce


def genera_tracce_validazione(auto, cfg, ritorna_meta=False):
    """Tracce sintetiche per la VALIDAZIONE: celle (classe, velocita') DISGIUNTE da
    quelle di training e seed diverso. Serve a misurare se la rete generalizza a
    combinazioni rugosita' x velocita' mai viste — esattamente il caso in cui la
    scorciatoia 'a_z -> xi' si rompe."""
    rng = np.random.default_rng(cfg.aug_seed + 9973)
    celle = _celle(cfg, int(cfg.n_tracce_sint_val), per_validazione=True)
    tipi = ["rampa", "costante", "sweep", "urbano"]
    tracce, metas = [], []
    for i, (classe, v0) in enumerate(celle):
        tr, meta = _una_traccia(classe, v0, tipi[i % len(tipi)], auto, cfg, rng,
                                etichetta=f"valid {i+1:2d}:")
        tracce.append(tr); metas.append(meta)
    return (tracce, metas) if ritorna_meta else tracce


def riepilogo_copertura(auto, cfg):
    """Stampa la copertura del disegno fattoriale (quante celle, quali riservate)."""
    g = _griglia(cfg)
    tr = _celle(cfg, int(cfg.n_tracce_sintetiche), False)
    va = _celle(cfg, int(cfg.n_tracce_sint_val), True)
    print(f"    Griglia fattoriale: {len(cfg.aug_classi)} classi x {len(cfg.aug_velocita)} velocita'"
          f" = {len(g)} celle | training {len(set(tr))} celle | validazione {len(set(va))} celle"
          f" (DISGIUNTE)")
    print(f"    Crossover r=√2 a v ≈ {_v_crossover(auto, cfg)*3.6:.0f} km/h:"
          f" le tracce a velocita' variabile lo attraversano a rugosita' costante")
