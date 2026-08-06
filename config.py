# -*- coding: utf-8 -*-
"""
config.py — Parametri FISICI dell'auto e scelte di progetto del controllore.

Qui dentro si trovano i  parametri di MODELLO (specifiche del veicolo) e di PROGETTO del controllo
closed-loop per comandare il sistema ibrido di sospensione skyhook + smorzatore variabile.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Auto:
    """Peugeot 207 come quarter-car a 2 gradi di liberta' (cassa + ruota)."""
    massa_cassa: float = 260.0        # m_s  massa sospesa (carrozzeria)          [kg]
    massa_ruota: float = 38.0         # m_u  massa non sospesa (ruota + mozzo)    [kg]
    rigid_molla: float = 23000.0      # k_s  rigidezza molla sospensione          [N/m]
    rigid_gomma: float = 190000.0     # k_t  rigidezza verticale pneumatico       [N/m]
    smorz_min:   float = 400.0        # c_min ammortizzatore: minimo realizzabile [Ns/m]
    smorz_max:   float = 3500.0       # c_max ammortizzatore: massimo realizzabile[Ns/m]
    smorz_nom:   float = 1500.0       # c_nom valore passivo di riferimento       [Ns/m]

    @property
    def puls_nat_cassa(self):
        """omega_n = sqrt(k_s/m_s): pulsazione naturale della cassa [rad/s] (~9.4 = 1.5 Hz)."""
        return np.sqrt(self.rigid_molla / self.massa_cassa)

    @property
    def smorz_critico(self):
        """c_crit = 2*sqrt(k_s*m_s): smorzamento critico [Ns/m]. Serve per xi = c/c_crit."""
        return 2.0 * np.sqrt(self.rigid_molla * self.massa_cassa)

    @property
    def xi_min(self):
        """xi MINIMO fisicamente realizzabile = c_min/c_crit. NON e' una scelta: e' il
        limite del damper reale (smorz_min). Usato sopra r=√2 (isolamento)."""
        return self.smorz_min / self.smorz_critico

    @property
    def xi_max(self):
        """xi MASSIMO fisicamente realizzabile = c_max/c_crit. NON e' una scelta: e' il
        limite del damper reale (smorz_max). Usato sotto r=√2 (comfort/controllo)."""
        return self.smorz_max / self.smorz_critico

    def c_da_xi(self, xi):
        """Da fattore di smorzamento adimensionale xi al coefficiente c [Ns/m]."""
        return xi * self.smorz_critico

    def xi_da_c(self, c):
        """Da coefficiente c [Ns/m] al fattore di smorzamento adimensionale xi."""
        return c / self.smorz_critico


@dataclass
class Config:
    """Scelte di progetto del controllore e dell'addestramento.

    PROVENIENZA di ogni valore (per capire cosa e' un magic number e cosa no):
      [FISICO]   spec hardware/veicolo misurabile          (non si inventa)
      [DERIVATO] calcolato da altri parametri via formula   (non si tocca a mano)
      [DATI]     stimato a runtime dalle registrazioni       (nessun numero fisso)
      [STANDARD] convenzione/standard (ISO, Nyquist, 3σ...)  (giustificato dalla teoria)
      [MANOPOLA] vera scelta di taratura libera             (l'unico "magic number" onesto)
      [NUMERICO] dettaglio del solver/implementazione
      [GRAFICA]  solo resa della demo, non tocca il controllo
    """
    # --- campionamento e integrazione ---
    freq_campion: float = 100.0     # [FISICO] frequenza di campionamento del sensore/controllo [Hz]
    passo_t: float = 0.01           # [DERIVATO] dt = 1/freq_campion (tenere coerente)          [s]
    n_sottopassi: int = 2           # [NUMERICO] sotto-passi RK4 per stabilita' della ODE
    seq_len: int = 200              # [DERIVATO] finestra a_z alle reti = 2.0 s, tenuta UGUALE a
                                    # xi_ott_finestra_s: l'etichetta xi* e' un RMS mobile su
                                    # quella stessa finestra, quindi la rete riceve esattamente
                                    # l'informazione che serve a calcolarla. Se le due
                                    # divergono, una quota dell'errore diventa irrecuperabile
                                    # per costruzione e indistinguibile dagli errori veri.
    seq_len_forza: int = 100        # [DERIVATO] la rete FORZA usa solo gli ultimi 100 campioni
                                    # (1.0 s) della finestra. L'etichetta e' F = -kp*(t)*z_s'(t)
                                    # e la velocita' di cassa si ricostruisce in ~1 s: i 2 s
                                    # servono a xi (per allinearsi alla finestra di costo), alla
                                    # forza darebbero solo modo di memorizzare, e si paga in
                                    # sovradattamento. None = usa tutta la finestra.
    finestra_freq: int = 100        # [DERIVATO] finestra per la stima causale della frequenza (1.0 s)
    passo_finestre: int = 1         # [NUMERICO] si tiene 1 finestra ogni N. Finestre consecutive
                                    # condividono seq_len-1 campioni su seq_len (199 su 200):
                                    # sono quasi duplicati, quindi sottocampionarle non toglie
                                    # informazione ma dimezza memoria e tempo per epoca.
                                    # Con seq_len=200 il dataset pesa ~1.2 GB fra numpy e copia
                                    # sul device, e passo 2 lo dimezza. Tenuto a 1 perche' la
                                    # rete forza e' sensibile al rapporto dati/parametri: se la
                                    # memoria stringe e' la prima manopola da girare, ma va
                                    # guardato il divario Train/Valid della forza.

    # --- limiti FISICI dell'attuatore/strada (non normalizzazione) ---
    forza_max: float = 1500.0       # [FISICO] saturazione forza attuatore [N] (spec hardware)
    strada_rms_nom: float = 0.010   # [IRRILEVANTE se calibra_strada=True] ampiezza solo di
                                    # partenza: viene SOVRASCRITTA dalla calibrazione sui dati
                                    # (fisica.ricostruisci_strada -> controllo.genera). Conta
                                    # solo quando calibra_strada=False.
    calibra_strada: bool = True     # calibra l'ampiezza strada sui DATI (RMS a_z misurato)
    r_max: float = 5.0              # [MANOPOLA] clip di r = omega/omega_n per robustezza

    # --- condizionamento del segnale (pulizia dati) ---
    despica: bool = True            # filtro di Hampel: rimuove i picchi anomali del sensore
    hampel_finestra: int = 7        # [MANOPOLA] ampiezza (dispari) finestra mediana per il despiking
    hampel_sigma: float = 3.0       # [STANDARD] regola 3σ: outlier se |x-mediana| > 3·MAD
    metodo_freq: str = "velocita"   # "velocita" (r da v: adattivo, non falsato dal veicolo) | "fft" | "rms"
    # La scheda usa r = omega/omega_n; il crossover e' SEMPRE a r=√2 (fisica, in xi_da_r/kp_da_r):
    # NON e' una velocita' da scegliere. Nel metodo "velocita" omega = 2π·v/lambda_c, con
    # omega_n dal veicolo e lambda_c come UNICA assunzione (scala di lunghezza d'onda del legame
    # v->frequenza). La velocita' a cui r=√2 NON e' fissa: cambia con la lambda reale della strada.
    #
    # PERCHE' lambda_c e' di PROGETTO e non stimata dai dati (verifica sperimentale sul dataset):
    #   - lo spettro di a_z ha una parte fissa (risonanze veicolo) + una debole parte che scala
    #     con v; togliendo le risonanze note per DECONVOLUZIONE (1/|H(w)|^2 del quarter-car) la
    #     dipendenza dalla velocita' SPARISCE (pendenza ~0). Fisicamente: una strada ISO e' a
    #     banda larga con FORMA spettrale fissa (PSD spostamento ∝ f^-2), quindi accelerando si
    #     scala l'AMPIEZZA dell'eccitazione, non il suo baricentro in frequenza -> nessuna
    #     "lunghezza d'onda dominante" osservabile. La debole pendenza residua nel grezzo da'
    #     comunque lambda_c ≈ 7 m, coerente col valore di progetto qui sotto.
    lambda_c_design: float = 7.0    # [DESIGN] scala di lunghezza d'onda [m] del legame v->frequenza
                                    # (~7 m dalla pendenza grezza a_z vs v; non piu' estraibile di cosi')
    freq_banda_hz: tuple = (0.3, 6.0)   # (solo per "fft") banda del modo cassa
    freq_banda_hz: tuple = (0.3, 6.0)   # (solo per "fft") banda del modo cassa

    # --- struttura della rete FORZA ---
    forza_due_uscite: bool = True   # la rete forza predice i DUE FATTORI dell'etichetta
                                    # (z_s' e kp) e li ricompone: F = clip(-kp*z_s').
                                    # Una singola regressione su F deve inseguire il
                                    # prodotto, che ha curtosi 26 e saturazione: mediare fra
                                    # le due modalita' la fermava a R^2 0.62.
                                    # False = uscita singola su F (utile come ablation).
    zs_max: float = 2.0             # [DATI] limite della velocita' di cassa stimata [m/s].
                                    # Sulle tracce sintetiche piu' severe l'RMS e' ~0.37 e il
                                    # massimo ~1.4: 2.0 lascia margine senza rendere la Tanh
                                    # inutilmente piatta.
    peso_aux_forza: float = 0.3     # [MANOPOLA] peso delle loss AUSILIARIE sui due fattori.
                                    # La loss principale resta su F (e' quella che conta),
                                    # ma supervisionare anche z_s' e kp separatamente evita
                                    # che la rete trovi combinazioni sbagliate il cui
                                    # prodotto e' per caso giusto.

    # --- attivazioni delle reti (relu/gelu/silu/mish/elu/leaky) ---
    att_xi: str = "silu"            # attivazione rete xi (SiLU liscia e veloce; ReLU va bene uguale)
    att_forza: str = "mish"         # forza: TCN + silu (regressione liscia)

    # --- ORIGINE DELL'ETICHETTA xi ---------------------------------------------------
    # "sqrt2"  : regola della trasmissibilita' √2. ATTENZIONE: con metodo_freq="velocita"
    #            quell'etichetta e' xi = f(v) e basta — verificato, sigmoid(L(v)) la
    #            riproduce con errore 4e-8. E' un ORACOLO: la finestra di a_z non porta
    #            nessuna informazione sul target, quindi non c'e' niente da imparare e
    #            tutto cio' che la rete estrae da a_z e' scorciatoia.
    # "ottimo" : xi* e' il risultato di un'OTTIMIZZAZIONE (vedi xi_ottimo.py): per ogni
    #            tratto, lo smorzamento che minimizza comfort + tenuta + corsa. Dipende
    #            dalla strada e non solo dalla velocita' -> la rete ha un compito vero.
    # NB: con "ottimo" si ottimizzano INSIEME xi (damper) e kp (guadagno skyhook). Erano
    # due canali che agiscono sulla stessa massa: tenerne uno sulla regola √2 e l'altro
    # sull'ottimo produce un sistema incoerente (damper morbido dove l'attuatore e' debole
    # e viceversa, perche' la √2 metteva kp basso proprio dove dava per scontato un damper
    # duro). Ottimizzarli separatamente e' anche circolare: xi* dipende dal kp usato.
    metodo_xi: str = "ottimo"       # "ottimo" | "sqrt2"
    xi_ott_n_candidati: int = 19    # [NUMERICO] valori di xi provati fra xi_min e xi_max - INIZIALE ERA 11
    kp_ott_n_candidati: int = 13    # [NUMERICO] valori di kp provati (griglia 11x7 = 77) - INIZIALE ERA 7
    kp_min_ott: float = 300.0       # [FISICO] guadagno skyhook MINIMO [Ns/m]. NON zero:
                                    # con kp_min = 0 l'ottimo spegne del tutto l'attuatore
                                    # su fondo sconnesso (misurato kp*=0 nel 35-60% dei
                                    # campioni su classe D/E), e l'etichetta della forza
                                    # diventa bimodale — una massa esattamente in zero piu'
                                    # una coda fino a saturazione, curtosi in eccesso 26.
                                    # Un guadagno minimo non nullo la rende continua senza
                                    # togliere fisica: un attuatore reale ha comunque una
                                    # banda morta e un guadagno minimo utile.
    kp_max_ott: float = 6000.0      # [FISICO] guadagno skyhook massimo ammesso [Ns/m]
                                    # (il limite vero che morde e' la saturazione di forza)
    xi_ott_finestra_s: float = 2.0  # [DERIVATO] finestra su cui si valuta il costo [s].
                                    # Tenuta UGUALE a seq_len/freq_campion: cosi' xi* e'
                                    # una funzione di cio' che la rete vede davvero.
    xi_ott_causale: bool = True     # finestra TRAILING (solo passato). Con finestra
                                    # centrata l'etichetta userebbe meta' finestra di
                                    # FUTURO, che la rete non puo' conoscere: una quota
                                    # dell'errore diventerebbe irrecuperabile per
                                    # costruzione, indistinguibile dagli errori veri.
    fattore_picco_default: float = 3.3  # [DATI] RIPIEGO per max|x|/RMS, usato solo quando
                                    # non si puo' misurare sui segnali. Il valore vero viene
                                    # MISURATO in xi_ottimo.ottimo (mediana fra i candidati):
                                    # 3.3-3.7 per la deflessione gomma, 3.0-3.4 per la corsa.
                                    # NON e' sqrt(2)=1.41 (quello vale per una sinusoide, e la
                                    # risposta a una strada ISO non lo e') ne' esattamente 3
                                    # (la convenzione dei 3 sigma per un gaussiano): dipende
                                    # dallo spettro, quindi si misura.
    xi_ott_comfort_rif: float = 0.315   # [STANDARD] soglia ISO 2631-1 della scala di
                                    # comfort ("a little uncomfortable") [m/s^2]: e' solo
                                    # la SCALA del termine di comfort, non un limite.
    corsa_disponibile: float = 0.08 # [FISICO] corsa utile della sospensione (rattle space)
                                    # prima del fine corsa [m]. Il limite di distacco
                                    # ruota NON e' qui: e' DERIVATO da (m_s+m_u)g/k_t.
    xi_ott_pesi: tuple = (1.0, 1.0, 1.0)   # (comfort, tenuta, corsa). Comfort e' l'obiettivo,
                                    # gli altri due sono VINCOLI: il loro peso decide solo
                                    # quanto duro e' il vincolo, non un compromesso.
    xi_ott_penalita: float = 1000.0  # [MANOPOLA] durezza dei vincoli. E' l'UNICA vera
                                    # taratura del costo, e va dichiarata: sotto ~300 il
                                    # comfort vince sempre e i vincoli sono un suggerimento;
                                    # sopra ~1000 il risultato satura (verificato su griglia
                                    # classe x velocita': l'escursione di xi* passa da 0.16
                                    # a 0.27 fra P=100 e P=1000, poi si ferma a 0.29).
                                    # Grande = vincolo quasi
                                    # rigido. NB: una semplice somma pesata NON funziona —
                                    # il quarter-car e' lineare e un costo quadratico e'
                                    # omogeneo, quindi l'ottimo non dipenderebbe
                                    # dall'ampiezza della strada (vedi xi_ottimo.py).

    # --- VINCOLO FISICO STRUTTURALE sulla rete xi ---
    # Problema: la finestra a_z entra nella testa con ~6000 feature, la velocita' con UNA
    # sola. A parita' di loss la discesa del gradiente passa per a_z, e in closed-loop
    # l'accelerazione "sovrascrive" la schedulazione in velocita'. Qui la monotonia in v
    # smette di essere una speranza statistica e diventa una GARANZIA algebrica:
    #     xi = xi_min + (xi_max-xi_min) * sigmoid( logit_sched(v) + beta*tanh(f(a_z,v)) )
    # logit_sched(v) e' esattamente il logit del target xi_da_r(r(v)): la rete parte gia'
    # sulla curva fisica e puo' solo MODULARLA di +-beta in spazio logit.
    # Struttura: BASELINE FISICO + CORREZIONE APPRESA, con il baseline che NON domina.
    #     xi = sigmoid( L(v) + alpha * Delta_theta(a_z, v) )
    # L(v) e' il logit della regola √2 (la tendenza in velocita', gratis e sempre giusta
    # come andamento); Delta_theta e' la correzione che la rete impara — ed e' dove ora
    # vive il contenuto informativo, perche' xi* ottimo si discosta da L(v) in funzione
    # della STRADA. alpha e' l'ampiezza concessa alla correzione, in scala logit.
    #
    # CURRICULUM: alpha parte piccolo (la rete resta vicina al baseline e non insegue
    # subito il rumore) e cresce durante l'addestramento fino ad alpha_finale. Il valore
    # finale non e' scelto a occhio: main.py stampa l'alpha SUGGERITO dai dati, cioe' il
    # 95esimo percentile dello scarto |logit(xi*) - L(v)| misurato sulle etichette.
    xi_vincolo_fisico: bool = True  # False = rete libera (per l'ablation nel report)
    xi_baseline: str = "comfort"    # "comfort" | "sqrt2" — DEVE essere coerente con
                                    # metodo_xi: "sqrt2" col target √2, "comfort" col
                                    # target ottimo. Incrociarli fa combattere baseline e
                                    # etichetta (servirebbe alpha ~ 10 per annullare il
                                    # baseline invece che per imparare qualcosa).
    xi_margine_logit: float = 0.03  # [NUMERICO] il baseline "comfort" sta a xi_min + 3%
                                    # del range: il logit diverge esattamente sul bordo.
    autorita_az: float = 7.0        # alpha FINALE = TETTO della correzione, non una
                                    # statistica da centrare. Va tenuto >= alpha_suggerito
                                    # (main.py lo alza da solo): con un tetto troppo basso la
                                    # rete SATURA e resta piatta sulle strade peggiori, cioe'
                                    # quelle che decidono la tenuta. Il sintomo e'
                                    # riconoscibile: xi identico su ogni riga e ogni colonna
                                    # della mappa diagnostica, pari a sigmoid(L0 + alpha).
                                    # NOTA ONESTA: con baseline "comfort" (costante) e alpha
                                    # cosi' ampio il vincolo strutturale non e' piu' un prior
                                    # fisico, e' una riparametrizzazione che schiaccia
                                    # l'uscita nel range del damper. Ed e' giusto cosi': il
                                    # prior serviva a impedire la scorciatoia quando a_z non
                                    # portava informazione. Ora a_z E' il segnale, quindi
                                    # limitarne l'ampiezza fa solo danno.
    autorita_az_iniziale: float = 0.3   # alpha alla prima epoca (curriculum)
    autorita_az_rampa: float = 0.6  # [MANOPOLA] frazione delle epoche su cui alpha sale
                                    # da iniziale a finale (poi resta costante)

    # --- filtro di Kalman per la velocita' cassa (denoising ottimo) ---
    usa_kalman: bool = True         # stima z_s' con Kalman invece dell'integrale semplice
    kalman_scala_pos: float = 0.02  # [MANOPOLA] scala tipica spostamento comfort ~2 cm [m] (anti-deriva)

    # bilancio energetico rigenerativo (solo per la stampa in main.py, non tocca il controllo)
    eta_generatore: float = 0.60    # [MANOPOLA] rendimento in RECUPERO (generatore): quota
                                    # dell'energia meccanica assorbita che arriva in batteria [-]
    eta_attuatore: float = 1.0      # [MANOPOLA] rendimento in TRAZIONE (motore): per consegnare
                                    # E_inj joule meccanici la batteria ne fornisce
                                    # E_inj/eta_attuatore, cioe' DI PIU'.
                                    # Tenuto a 1.0 perche' non abbiamo la specifica
                                    # dell'attuatore e non si inventa un numero: significa che
                                    # il costo stampato e' un LIMITE INFERIORE, non una stima.
                                    # Il bilancio riporta accanto anche il caso con lo stesso
                                    # rendimento del generatore applicato in trazione, e su
                                    # questi numeri il SEGNO del netto cambia fra le due
                                    # ipotesi: non e' un dettaglio, e' la conclusione.
                                    # Applicare eta al solo recupero, come si faceva prima,
                                    # assume un attuatore perfetto quando spinge e imperfetto
                                    # quando frena: non e' coerente.
    damper_rigenerativo: bool = False
                                    # [SCELTA HARDWARE] L'ammortizzatore di questo progetto e'
                                    # IDROPNEUMATICO a smorzamento variabile: il lavoro che fa
                                    # diventa calore nell'olio e si perde. NON e' recuperabile,
                                    # per quanto grande sia.
                                    # Contarlo fra i recuperi e' l'errore che fa apparire il
                                    # sistema auto-alimentato: nella run di riferimento
                                    # includerlo dava +1.6 J (ATTIVO), escluderlo -0.8 J (COSTO),
                                    # e la seconda e' la risposta giusta per questo hardware.
                                    # Va messo True SOLO se si dichiara di sostituire il damper
                                    # con uno ELETTROMAGNETICO, che e' un progetto diverso: e in
                                    # quel caso eta_generatore va rimisurata su quel dispositivo,
                                    # non riusata.

    # --- data augmentation: strade sintetiche ISO 8608 (+ salite/discese + buche) ---
    # DISEGNO FATTORIALE: la vecchia versione accoppiava 1 classe <-> 1 velocita' (6 celle
    # su una diagonale), creando la correlazione spuria "a_z grande -> xi alto" che la rete
    # ha imparato come scorciatoia. Ora classe e velocita' sono INCROCIATE.
    usa_augmentation: bool = True
    n_tracce_sintetiche: int = 48   # [MANOPOLA] strade sintetiche di TRAINING (griglia 5x5).
                                    # ALZATO DA 24 per il divario train/valid della rete xi:
                                    # 0.00059 contro 0.00759, cioe' 13 volte. Un divario cosi'
                                    # non e' capacita' che manca (la rete il target lo
                                    # rappresenta benissimo), e' la rete che impara le singole
                                    # REALIZZAZIONI invece della relazione fra rugosita' e xi*.
                                    # Con 25 celle e 48 strade ogni cella ha in media due
                                    # realizzazioni diverse, e memorizzarle costa piu' che
                                    # generalizzare. Costo: generazione etichette e tempo per
                                    # epoca crescono in proporzione (la run da ~32 min va verso
                                    # l'ora). E' la prima manopola da riabbassare se stringe.
    n_tracce_sint_val: int = 4      # [MANOPOLA] celle riservate alla VALIDAZIONE (disgiunte)
    aug_secondi: float = 45.0       # [MANOPOLA] durata di ogni traccia sintetica [s]
    aug_seed: int = 1234            # [NUMERICO] seed del disegno sperimentale (riproducibile)
    aug_classi: tuple = ("A", "B", "C", "D", "E")        # classi ISO 8608 (liscia -> sconnessa)
    aug_velocita: tuple = (6.0, 11.0, 16.0, 21.0, 27.0)  # velocita' nominali [m/s] (22..97 km/h)
    aug_frazione_variabile: float = 0.5   # [MANOPOLA] quota di tracce a velocita' VARIABILE
                                    # (rampe/sweep che ATTRAVERSANO r=√2 a rugosita' costante:
                                    #  e' il dato che rende non identificabile la scorciatoia)
    aug_jitter_rugosita: tuple = (0.5, 2.0)   # Gd0 moltiplicato per U[lo,hi]: cosi' l'RMS di a_z
                                    # NON e' piu' una funzione deterministica di (classe, v)
    aug_freq_spaziale_max: float = 2.83   # [STANDARD] n_max ISO 8608 [cicli/m]
    aug_margine_nyquist: float = 0.4      # [STANDARD] n*v_max <= 0.4*fs -> niente ALIASING
                                    # (prima: n_max*v = 71 Hz a 25 m/s contro Nyquist 50 Hz,
                                    #  le due tracce piu' veloci erano le uniche corrotte)

    # --- regola della trasmissibilita' √2 (scheduling di xi e kp) ---
    # NB: gli estremi di xi NON stanno piu' qui: sono DERIVATI dal damper reale
    # (Auto.xi_min = c_min/c_crit ≈ 0.164, Auto.xi_max = c_max/c_crit ≈ 0.654).
    # xi(r) interpola fra questi due limiti fisici (vedi controllo.xi_da_r), quindi per
    # cambiarli si cambia il damper (smorz_min/smorz_max), non un numero scelto a mano.
    pendenza_sched: float = 4.0     # [MANOPOLA] ripidita' della transizione attorno a r=√2 (solo forma)
    kp_basso: float = 1500.0        # [MANOPOLA] guadagno skyhook attuatore per r < √2 [Ns/m]
    kp_alto:  float = 5000.0        # [MANOPOLA] guadagno skyhook attuatore per r > √2 [Ns/m]

    # --- DAgger: allineamento fra dati di addestramento e stati davvero visitati ---
    # Le etichette e gli ingressi vengono dal modello PASSIVO, ma in closed-loop la rete
    # legge l'accelerazione GIA' CONTROLLATA, che e' circa il 47% di quella (misurato).
    # Con le etichette √2 non contava (xi non dipendeva da a_z); con quelle ottime e' il
    # problema dominante. Vedi aggregazione.py.
    dagger_seme_strade: int = 4441  # [NUMERICO] scostamento del seme per le strade usate nei
                                    # rollout: stesse celle classe x velocita' del training,
                                    # realizzazioni DIVERSE. Rifare i rollout sulle strade di
                                    # addestramento aggiunge distribuzione closed-loop ma zero
                                    # varieta', e lascia aperta la memorizzazione della firma
                                    # spettrale della singola realizzazione.
    usa_dagger: bool = True
    dagger_iterazioni: int = 5      # [MANOPOLA] giri di raccolta + riaddestramento
    dagger_secondi: float = 20.0    # [MANOPOLA] durata del rollout per strada [s]
    dagger_epoche: int = 2          # [MANOPOLA] epoche di riaddestramento per giro.
                                    # TARATO SULLE CURVE, non a occhio: con 4 epoche per giro
                                    # la validazione della rete xi toccava il minimo alla
                                    # SECONDA e poi risaliva (0.0056 -> 0.0073 -> 0.0091),
                                    # quindi meta' delle epoche di ogni giro peggiorava il
                                    # modello e veniva scartata dall'early stopping. Meglio
                                    # piu' giri corti: si raccolgono stati nuovi piu' spesso e
                                    # ogni riaddestramento si ferma prima di sovradattare.
    dagger_beta: float = 0.4        # [MANOPOLA] decadimento della guida dell'ESPERTO:
                                    # giro 1 sempre beta=1 (guida l'esperto, distribuzione
                                    # bersaglio), poi beta = dagger_beta^(giro-1) e il
                                    # controllo passa alla rete, cosi' il dataset copre
                                    # anche gli stati raggiunti per i suoi stessi errori.

    # --- addestramento => Metodo Monte Carlo ---
    diag_ripetizioni: int = 5       # [NUMERICO] realizzazioni di strada per ogni cella della
                                    # mappa closed-loop della diagnostica. Con 1 sola la
                                    # tabella e' rumorosa e il rumore sta nel TARGET: ogni
                                    # cella dura pochi secondi con ostacoli piazzati a caso, e
                                    # se un dosso cade dentro la finestra di costo il vincolo
                                    # si attiva e xi* salta. Misurato: xi* variava di 0.240
                                    # dentro la stessa classe, sette volte l'errore della rete.
                                    # Mediando su 4 il rumore scende di 2x (va come 1/sqrt(n))
                                    # e viene stampata la barra d'errore, cosi' si vede quali
                                    # differenze sono vere. Costa: il batch della diagnostica
                                    # passa da 25 a 100 strade, ma e' UNA forward per passo
                                    # temporale comunque, quindi pesa poco.

    # --- regolarizzazione (contro il sovradattamento, non contro l'errore di training) ---
    wd_xi: float = 3e-4             # [MANOPOLA] weight decay della rete xi. Compromesso
                                    # bias/varianza TARATO SU DUE RUN, non scelto:
                                    #   1e-4 -> divario train/valid 13x (memorizza le strade)
                                    #   1e-3 -> divario 6.7x e R^2 su celle mai viste da 0.80
                                    #           a 0.91, ma la funzione si liscia troppo. Il
                                    #           target e' quasi un gradino (vincolo attivo o
                                    #           no) e un'approssimazione liscia sta SOPRA il
                                    #           gradino sul lato basso: bias +0.045 nella zona
                                    #           di transizione (xi* fra 0.11 e 0.28), la rete
                                    #           non scendeva piu' sotto 0.114 sulla demo
                                    #           (prima 0.082) e |xi_ML - xi*| li' passava da
                                    #           0.032 a 0.062.
                                    # 3e-4 sta in mezzo. NB: il bias NON e' saturazione della
                                    # sigmoide in uscita — misurato, il fondo lo tocca a
                                    # +0.0002 da xi_min, e nelle celle col target al minimo il
                                    # bias vale +0.004 contro +0.045 in transizione. Allargare
                                    # il range con un margine e un clamp finale mirerebbe al
                                    # posto sbagliato, e il clamp ha gradiente nullo fuori
                                    # dall'intervallo (zona morta in addestramento).
    wd_forza: float = 2e-5          # [MANOPOLA] weight decay della rete forza. TENUTO BASSO:
                                    # il suo divario e' 3.7x (0.00075 contro 0.00276), molto
                                    # meno grave, e la forza deve inseguire una grandezza
                                    # istantanea — irrigidirla troppo le toglie prontezza.
    lr_xi: float = 2e-5            # [MANOPOLA] passo di apprendimento della rete xi
    lr_forza: float = 4e-5          # [MANOPOLA] passo della rete forza: piu' piccolo perche'
                                    # il suo target ha code (curtosi in eccesso 6.5)

    n_epoche: int = 12              # [MANOPOLA] numero di epoche per l'addestramento
    batch: int = 256                # [MANOPOLA] batch grande = meno overhead su GPU/MPS
    usa_gpu: bool = True            # usa MPS (Apple Silicon) / CUDA se disponibili
    closed_loop_su_cpu: bool = True  # le simulazioni closed-loop a STRADA SINGOLA girano su
                                    # CPU. Sono sequenziali con batch 1: su GPU ogni passo
                                    # paga lancio del kernel + sincronizzazione per leggere
                                    # il risultato, e con reti piccole quell'overhead supera
                                    # il calcolo. La mappa closed-loop della diagnostica usa
                                    # invece simula_closed_loop_batch (25 strade in un batch),
                                    # che sul device ha senso.
    n_core_forza: int = -1          # -1 = auto (solo performance core); >0 = forza N core
    comfort_secondi: float = 60.0   # durata segmento su cui misurare il comfort ML (closed-loop)

    # --- output ---
    salva_figura: bool = True       # salva la figura si = True | no = False
    mostra_anim: bool  = True       # apre la finestra interattiva dell'animazione a fine simulazione
    salva_video: bool  = True       # salva anche l'MP4 (richiede ffmpeg)
    video_file: str    = "confronto_sospensione.mp4"
    figura_file: str   = "risultati_sospensione.png"

    # --- animazione (parametri di sola resa grafica) ---
    anim_confronto: bool = True     # simula TUTTI i controllori (ML, ottimo, regola √2)
                                    # sulla stessa strada e mette i pulsanti per commutarli
                                    # dal vivo nell'animazione. Le traiettorie sono
                                    # precalcolate: il pulsante scambia solo cosa si
                                    # disegna, perche' un passo RK4 con forward della rete
                                    # non gira in tempo reale.
    anim_usa_ml: bool = True        # (solo se anim_confronto=False) True = anima il ML,
                                    # False = anima l'ottimo (xi*, kp*)
    anim_secondi: float = 8.0       # durata simulata da animare [s]
    anim_dosso: float = 0.021       # [GRAFICA] altezza dossi della strada demo [m].
                                    # TARATO SUL VINCOLO, non a occhio. Con 0.060 la strada
                                    # equivaleva a una classe E e il vincolo di distacco ruota
                                    # era violato da TUTTI i controllori, ottimo compreso
                                    # (margine 2.45 > 1): il confronto nella tabella
                                    # dell'animazione non dimostrava piu' niente, perche' la
                                    # strada e' oltre cio' che l'hardware puo' gestire.
                                    # A 0.021 l'ottimo resta dentro (margine 0.88) e xi* spazia
                                    # ancora 0.082-0.399, cioe' quasi tutto l'intervallo utile:
                                    # severa quanto serve a far muovere xi, non di piu'.
    anim_texture: float = 0.0021    # [GRAFICA] rugosita' fine sovrapposta ai dossi [m],
                                    # scalata con anim_dosso per tenere lo stesso rapporto
    anim_gain: float = 6.0          # ingrandimento verticale del movimento (resa grafica)
    anim_rallenta: float = 1.5      # [GRAFICA] RALLENTATORE: quante volte l'animazione va piu'
                                    # lenta del tempo reale. A 80 km/h il veicolo attraversa la
                                    # finestra da 6 m in 0.27 s: in tempo reale non si vede
                                    # niente e i valori nel riquadro non si riescono a leggere.
                                    # Con 3.0 la stessa scena dura 0.81 s.
                                    # NB: non e' un sotto-campionamento. Rallentare mostrando
                                    # meno campioni farebbe scattare l'immagine; qui i
                                    # fotogrammi sono PIU' dei campioni e le grandezze vengono
                                    # interpolate sulla griglia dei fotogrammi.
    anim_fps: int = 60              # [GRAFICA] fotogrammi al secondo. Numero di fotogrammi
                                    # totali = durata * anim_rallenta * anim_fps, quindi alzarlo
                                    # allunga il salvataggio dell'MP4 in proporzione.
