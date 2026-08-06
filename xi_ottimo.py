# -*- coding: utf-8 -*-
"""
xi_ottimo.py — L'ESPERTO: calcola il controllo ottimo che le reti devono imparare.

COS'E' L'ESPERTO E PERCHE' SERVE
--------------------------------
Le reti non inventano una legge di controllo: imparano a riprodurre un riferimento calcolato
qui. Questo modulo e' quel riferimento. Conosce cose che le reti non hanno — il profilo
stradale esatto e il futuro — e con quelle risolve, per ogni istante, il problema di
progetto vero:

    (xi*, kp*) = argmin  RMS_Wk(a_z)                        <- COMFORT, ponderato ISO 2631-1
                 soggetto a   RMS(z_u - z_r) <= limite      <- la ruota resta a terra
                              RMS(z_s - z_u) <= limite      <- non si tocca il fine corsa
                              RMS(F)         <= limite      <- l'attuatore non satura

La ricerca e' per enumerazione su una griglia di coppie (xi, kp) simulate tutte in
parallelo, con una finestra mobile che risponde alla domanda "quale coppia costante sarebbe
stata la migliore su questo tratto". E' la formulazione classica del punto di progetto di
uno scheduling.

xi e kp SI OTTIMIZZANO INSIEME
------------------------------
Sono due comandi che agiscono sulla stessa massa. Schedularne uno con un criterio e l'altro
con un altro produce un sistema incoerente: si finisce con il damper morbido dove
l'attuatore e' debole e viceversa, perche' ogni regola presuppone un comportamento
dell'altro canale che non e' quello effettivo. Ottimizzarli separatamente e' anche
circolare, dato che l'ottimo di ciascuno dipende dal valore usato per l'altro.

PERCHE' IL COSTO E' VINCOLATO E NON UNA SOMMA PESATA
----------------------------------------------------
La formulazione naturale — sommare comfort, tenuta e corsa con dei pesi, ciascuno diviso per
il proprio limite — NON funziona, e il motivo e' strutturale, non di taratura.

Il quarter-car e' LINEARE: raddoppiando l'ampiezza della strada raddoppiano tutte le
risposte. Un costo quadratico e' quindi omogeneo di grado 2, e moltiplicarlo per una
costante non sposta l'argmin. Verificato direttamente: la stessa strada ad ampiezza x1 e x4
da' identico xi*. Un costo cosi' puo' dipendere solo dalla FORMA spettrale
dell'eccitazione, cioe' in pratica solo dalla velocita' — e un'etichetta che dipende solo
dalla velocita' non contiene nulla che la rete possa imparare dall'accelerometro.

Cio' che rompe l'omogeneita' sono i LIMITI ASSOLUTI, che non scalano con la strada:
    la ruota STACCA da terra quando |z_u - z_r| supera la deflessione statica del
        pneumatico, d_max = (m_s+m_u)*g/k_t, cioe' il punto in cui la forza di contatto si
        annulla;
    la sospensione arriva a FINE CORSA quando |z_s - z_u| supera il rattle space;
    l'attuatore SATURA a F_max.
Sono soglie in metri e newton, non rapporti: superarle su fondo sconnesso e' un evento
fisico, su fondo liscio e' impossibile. E' li' che entra l'ampiezza della strada, ed e' per
questo che xi* dipende dalla rugosita' e non solo dalla velocita'.

Il limite sulla FORZA non e' un dettaglio: senza di esso il guadagno skyhook migliorerebbe
il comfort in modo monotono — lo skyhook ideale e' smorzamento infinito verso un
riferimento fermo — e kp* finirebbe sempre sul bordo superiore della griglia.

DAI LIMITI DI PICCO AI LIMITI DI RMS: I FATTORI DI PICCO
--------------------------------------------------------
I limiti sono di picco, il costo penalizza l'RMS. La conversione richiede il fattore di
picco p = max|x|/RMS, che NON si assume: si misura sui segnali di ogni traccia.
    sqrt(2) = 1.41   e' il fattore di una SINUSOIDE. La risposta a una strada ISO non lo e',
                     e usarlo sottostimerebbe di oltre il doppio.
    3.00             e' la convenzione dei 3 sigma per un gaussiano: vicino, ma non esatto.
    3.3 - 3.7        e' quanto si MISURA sulla deflessione del pneumatico; 3.0 - 3.4 sulla
                     corsa. Dipende dallo spettro, quindi si misura.
Si prende la mediana fra i candidati: il fattore dipende debolmente da (xi, kp), e usare
quello del candidato in esame renderebbe il criterio circolare — il costo definirebbe il
proprio metro.

I vincoli entrano come penalita' a cerniera: valgono ZERO finche' si e' dentro e crescono in
fretta appena si esce. Si comportano come vincoli rigidi ma lasciano il costo continuo,
quindi xi* varia con continuita' invece che a scatti.

IL COMPORTAMENTO CHE NE ESCE
----------------------------
  * su fondo liscio i vincoli sono inattivi, conta solo il comfort, e xi* va al MINIMO del
    damper. Con eccitazione a banda larga il comfort preferisce sempre poco smorzamento:
    l'ottimo di comfort puro cade sotto il minimo realizzabile;
  * man mano che il fondo peggiora la deflessione del pneumatico si avvicina al distacco, il
    vincolo si attiva e xi* RISALE verso l'ottimo di tenuta;
  * la soglia a cui questo avviene dipende dall'AMPIEZZA della strada, che sta nella storia
    di a_z. E' esattamente l'informazione che la rete deve estrarre.

NATURA DELL'ETICHETTA
---------------------
xi* e' un ORACOLO: usa il profilo stradale esatto e una finestra di storia. La finestra e'
CAUSALE (solo passato) e lunga quanto quella che la rete riceve, cosi' il target e' una
funzione di cio' che la rete vede davvero. Con una finestra centrata l'etichetta userebbe
anche il futuro, e una quota dell'errore diventerebbe irrecuperabile per costruzione —
indistinguibile dagli errori veri di stima.

FILTRO ISO 2631-1 Wk
--------------------
Il comfort non e' l'RMS grezzo dell'accelerazione: il corpo umano e' molto piu' sensibile
fra 4 e 8 Hz che a 0.5 Hz, e ottimizzare l'RMS non ponderato significa ottimizzare la cosa
sbagliata. Il filtro e' la cascata della norma,
    H(s) = Hh(s) * Hl(s) * Ht(s) * Hs(s)
con f1 = 0.4 Hz e f2 = 100 Hz per la limitazione di banda (Q = 1/sqrt(2)), f3 = f4 = 12.5 Hz
e Q4 = 0.63 per la transizione accelerazione-velocita', f5 = 2.37 Hz e f6 = 3.35 Hz con
Q = 0.91 per il gradino che alza la sensibilita' fra 4 e 8 Hz. A 100 Hz di campionamento il
passa-basso a 100 Hz cade sopra Nyquist ed e' di fatto inattivo: resta nella cascata per
fedelta' alla norma.
verifica_wk() confronta il guadagno con i fattori tabellati (Tabella 3) e viene chiamata
all'avvio: un filtro montato male non produce errori, produce solo l'ottimo di un altro
problema.
"""
import numpy as np
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d

from controllo import kp_da_r, xi_da_r
from fisica import stima_frequenza, derivata_stato, passo_rk4

G = 9.80665


## filtro di ponderazione ISO 2631-1 Wk
def filtro_wk(fs):
    """SOS del filtro Wk (ponderazione in frequenza per vibrazione verticale, ISO 2631-1).

    Restituisce i coefficienti second-order-sections per filtrare a_z prima di calcolarne
    l'RMS: senza ponderazione si darebbe lo stesso peso a 0.5 Hz e a 8 Hz, mentre il corpo
    umano e' molto piu' sensibile fra 4 e 8 Hz."""
    w1, w2 = 2 * np.pi * 0.4, 2 * np.pi * 100.0
    w3, w4 = 2 * np.pi * 12.5, 2 * np.pi * 12.5
    w5, w6 = 2 * np.pi * 2.37, 2 * np.pi * 3.35
    q1 = q2 = 1.0 / np.sqrt(2.0)
    q4, q5, q6 = 0.63, 0.91, 0.91

    # Hs NON porta il fattore (w6/w5)^2: la norma normalizza a 1 il PIANEROTTOLO ALTO,
    # quindi il gradino vale w5^2/w6^2 = 0.5 in continua e 1 sopra i ~4 Hz. Mettendo quel
    # fattore la ponderazione risulterebbe esattamente doppia ovunque.
    sezioni = [
        ([1.0, 0.0, 0.0], [1.0, w1 / q1, w1 ** 2]),                    # Hh passa-alto
        ([0.0, 0.0, w2 ** 2], [1.0, w2 / q2, w2 ** 2]),                # Hl passa-basso
        ([0.0, w4 ** 2 / w3, w4 ** 2], [1.0, w4 / q4, w4 ** 2]),       # Ht transizione a-v
        ([1.0, w5 / q5, w5 ** 2], [1.0, w6 / q6, w6 ** 2]),            # Hs gradino in salita
    ]
    sos = []
    for num, den in sezioni:
        b, a = signal.bilinear(num, den, fs)
        sos.append(np.concatenate([b, a]))
    return np.array(sos, dtype=float)


def pondera_wk(az, fs):
    """Applica Wk a media nulla e fase nulla (filtfilt: l'etichetta e' offline, non causale,
    quindi non vogliamo introdurre ritardo nel confronto fra candidati)."""
    return signal.sosfiltfilt(filtro_wk(fs), np.asarray(az, dtype=float))


#: fattori di ponderazione Wk tabellati in ISO 2631-1:1997, Tabella 3 (asse z, seduto)
_WK_TABELLA = {0.5: 0.418, 0.63: 0.459, 0.8: 0.477, 1.0: 0.482, 1.25: 0.484,
               1.6: 0.494, 2.0: 0.531, 2.5: 0.631, 3.15: 0.804, 4.0: 0.967,
               5.0: 1.039, 6.3: 1.054, 8.0: 1.036, 10.0: 0.988, 12.5: 0.902,
               16.0: 0.768, 20.0: 0.636, 25.0: 0.513, 31.5: 0.405, 40.0: 0.314}


def verifica_wk(fs=1000.0):
    """Confronta il guadagno del filtro con i fattori tabellati nella norma.
    E' il test che dice se la cascata e' montata bene: se lo fosse solo a meta', il termine
    di comfort peserebbe le frequenze sbagliate e xi* sarebbe ottimo per il problema errato.
    Si verifica a fs alta (1 kHz) perche' a 100 Hz la banda utile arriva solo a 50 Hz."""
    f = np.array(sorted(_WK_TABELLA))
    _, h = signal.sosfreqz(filtro_wk(fs), worN=2 * np.pi * f / fs)
    return f, np.abs(h), np.array([_WK_TABELLA[x] for x in f])


## dinamica vettorizzata (tutti i candidati xi in parallelo)
# la dinamica vive in UN SOLO posto: fisica.derivata_stato / fisica.passo_rk4, che
# funzionano sia su uno stato singolo sia su un batch di stati (..., 4)
_derivata = derivata_stato
_rk4 = passo_rk4


def griglia_candidati(auto, cfg):
    """Tutte le coppie (xi, kp) provate. xi = smorzamento del damper passivo variabile,
    kp = guadagno skyhook dell'attuatore. Si ottimizzano INSIEME perche' agiscono sulla
    stessa massa: ottimizzare uno tenendo l'altro su una regola diversa produce un
    sistema incoerente (damper morbido dove l'attuatore e' debole, e viceversa)."""
    xi = np.linspace(auto.xi_min, auto.xi_max, int(cfg.xi_ott_n_candidati))
    kp = np.linspace(float(cfg.kp_min_ott), float(cfg.kp_max_ott), int(cfg.kp_ott_n_candidati))
    XI, KP = np.meshgrid(xi, kp, indexing="ij")
    return xi, kp, XI.ravel(), KP.ravel()


def _simula_griglia(zr_dot, xi_cand, kp_cand, auto, cfg):
    """Simula il quarter-car per TUTTE le coppie (xi, kp) contemporaneamente.

    Restituisce (a_z, defl_gomma, corsa, forza) di forma (n_cand, N). Ogni candidato tiene
    la propria coppia COSTANTE: l'etichetta risponde alla domanda "quale coppia costante
    sarebbe stata la migliore su questo tratto", che e' la formulazione classica del punto
    di progetto di uno scheduling."""
    n = len(xi_cand)
    N = len(zr_dot)
    c = auto.c_da_xi(np.asarray(xi_cand, dtype=float))        # (n_cand,)
    kp = np.asarray(kp_cand, dtype=float)
    h = cfg.passo_t / cfg.n_sottopassi

    x = np.zeros((n, 4))
    az = np.zeros((n, N)); gom = np.zeros((n, N)); cor = np.zeros((n, N)); frz = np.zeros((n, N))
    for k in range(N):
        vrel = x[:, 1] - x[:, 3]
        forza = np.clip(-kp * x[:, 1], -cfg.forza_max, cfg.forza_max)
        az[:, k] = (-auto.rigid_molla * x[:, 0] - c * vrel + forza) / auto.massa_cassa
        gom[:, k] = x[:, 2]                                   # z_u - z_r (deflessione gomma)
        cor[:, k] = x[:, 0]                                   # z_s - z_u (corsa sospensione)
        frz[:, k] = forza
        zr0 = zr_dot[k]; zr1 = zr_dot[k + 1] if k + 1 < N else zr_dot[k]
        for _ in range(cfg.n_sottopassi):
            x = _rk4(x, c, forza, zr0, zr1, auto, h)
    return az, gom, cor, frz


## costo e ricerca del minimo
def _media_mobile(u, w, cfg):
    """Media mobile su finestra w, CAUSALE (trailing) se cfg.xi_ott_causale.

    Perche' importa. Con finestra CENTRATA l'etichetta a t usa w/2 secondi di FUTURO,
    mentre la rete vede solo il passato: una parte dell'errore diventa strutturalmente
    non recuperabile, e non c'e' modo di distinguerla dagli errori veri.
    Con finestra TRAILING l'etichetta e' una funzione del solo passato, e se la si tiene
    lunga quanto la finestra della rete (seq_len) il compito diventa BEN POSTO: tutto
    cio' che serve per calcolare xi* e' dentro cio' che la rete riceve."""
    if not getattr(cfg, "xi_ott_causale", True):
        return uniform_filter1d(u, size=w, axis=-1, mode="nearest")
    # origin sposta la finestra: con w dispari e origin=(w-1)//2 copre [i-w+1, i]
    return uniform_filter1d(u, size=w, axis=-1, mode="nearest", origin=(w - 1) // 2)


def fattore_picco(x):
    """max|x| / RMS(x): quante deviazioni standard vale il picco osservato.

    Serve a convertire un limite di PICCO (la ruota stacca, si tocca il fine corsa, la forza
    satura) in un limite sull'RMS, che e' cio' che il costo puo' penalizzare.

    Quanto vale, a seconda del segnale:
        sinusoide pura                        sqrt(2) = 1.41
        gaussiano, convenzione dei 3 sigma            3.00
        MISURATO su questo quarter-car          3.3 - 3.7 (deflessione gomma)
                                                3.0 - 3.4 (corsa sospensione)
    Non e' 1.41 perche' la risposta a una strada ISO non e' una sinusoide, ed e' vicino a 3
    ma non uguale, perche' dipende dallo spettro e dalla durata dell'osservazione. Quindi si
    misura invece di assumerlo: era l'ultimo numero fissato a mano nei vincoli."""
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x ** 2)))
    return float(np.max(np.abs(x)) / rms) if rms > 1e-12 else float("nan")


def riferimenti(auto, cfg, fattori=None):
    """Scala del comfort e i TRE limiti RMS ammissibili = limite di picco / fattore di picco.

    'fattori' = dict con i fattori di picco MISURATI sui segnali di questa traccia, chiavi
    "gomma", "corsa", "forza". Se assente si usa cfg.fattore_picco_default, che va inteso
    come ripiego dichiarato e non come verita': con il ripiego i vincoli possono risultare
    piu' larghi o piu' stretti del giusto di un 15-20%.

    d_max = (m_s+m_u)*g/k_t e' la deflessione statica del pneumatico: oltre quella la forza
    di contatto si annulla e la ruota stacca. s_max e' il rattle space. F_max e' la
    saturazione dell'attuatore, gia' una spec hardware in config. Sono tutte grandezze
    ASSOLUTE: e' la loro non-scalabilita' che rende l'ottimo dipendente dall'ampiezza della
    strada e non solo dalla velocita'.

    Il limite sulla FORZA non e' un dettaglio: senza di esso il guadagno skyhook kp
    migliorerebbe il comfort in modo monotono (lo skyhook ideale e' smorzamento infinito
    verso il cielo) e kp* finirebbe sempre sul bordo superiore della griglia — l'ottimo
    sarebbe degenere esattamente come lo era il costo omogeneo."""
    fp = dict(gomma=None, corsa=None, forza=None)
    fp.update(fattori or {})
    dflt = float(cfg.fattore_picco_default)
    for k in fp:
        v = fp[k]
        fp[k] = dflt if (v is None or not np.isfinite(v) or v <= 0) else float(v)

    a_rif = float(cfg.xi_ott_comfort_rif)                              # ISO 2631-1
    # deflessione statica del pneumatico: oltre quella la forza di contatto si annulla
    d_max = (auto.massa_cassa + auto.massa_ruota) * G / auto.rigid_gomma
    s_max = float(cfg.corsa_disponibile)                               # rattle space
    f_max = float(cfg.forza_max)                                       # saturazione attuatore
    return a_rif, d_max / fp["gomma"], s_max / fp["corsa"], f_max / fp["forza"]


def descrivi_riferimenti(auto, cfg, fattori=None):
    """Le righe di log che spiegano DA DOVE viene ogni limite, formula compresa."""
    fp = dict(gomma=None, corsa=None, forza=None); fp.update(fattori or {})
    dflt = float(cfg.fattore_picco_default)
    val = {k: (dflt if (v is None or not np.isfinite(v) or v <= 0) else float(v))
           for k, v in fp.items()}
    src = {k: ("misurato" if (fp[k] is not None and np.isfinite(fp[k] or np.nan)
                              and (fp[k] or 0) > 0) else "ripiego cfg") for k in fp}
    d_max = (auto.massa_cassa + auto.massa_ruota) * G / auto.rigid_gomma
    a_rif, d_rif, s_rif, f_rif = riferimenti(auto, cfg, fattori)
    return [
        "    Limiti dei vincoli — ognuno = limite di PICCO / fattore di picco:",
        f"      GOMMA  picco {d_max*1000:6.2f} mm = (m_cassa+m_ruota)*g/k_gomma"
        f" = ({auto.massa_cassa:.0f}+{auto.massa_ruota:.0f})*9.807/{auto.rigid_gomma:.0f}"
        f"   [deflessione statica: oltre, la ruota stacca]",
        f"             fattore {val['gomma']:.2f} ({src['gomma']})"
        f"  ->  RMS ammesso {d_rif*1000:.2f} mm",
        f"      CORSA  picco {s_max_str(cfg):>6} = cfg.corsa_disponibile"
        f"   [rattle space — VALORE NON VERIFICATO sulla 207]",
        f"             fattore {val['corsa']:.2f} ({src['corsa']})"
        f"  ->  RMS ammesso {s_rif*1000:.2f} mm",
        f"      FORZA  picco {cfg.forza_max:6.0f} N  = cfg.forza_max"
        f"   [saturazione attuatore, spec hardware]",
        f"             fattore {val['forza']:.2f} ({src['forza']})"
        f"  ->  RMS ammesso {f_rif:.0f} N",
        f"      COMFORT scala {a_rif:.3f} m/s^2 = cfg.xi_ott_comfort_rif"
        f"   [soglia ISO 2631-1, non un limite: e' la scala dell'obiettivo]",
    ]


def s_max_str(cfg):
    return f"{cfg.corsa_disponibile*1000:.0f} mm"


def _costo(az_w, gom, cor, frz, auto, cfg, fattori=None):
    """J(xi, kp, t) = comfort + penalita' a cerniera sui TRE vincoli hardware.

    hinge(u) = max(0, u - 1)^2 vale ZERO finche' il vincolo e' rispettato: su fondo liscio
    il costo e' comfort puro e xi* va al minimo del damper. Appena la deflessione del
    pneumatico, la corsa o la forza si avvicinano al limite, il termine esplode.
    La penalita' e' grande di proposito: emula un vincolo rigido restando continua."""
    w = max(3, int(cfg.xi_ott_finestra_s * cfg.freq_campion))
    a_rif, d_rif, s_rif, f_rif = riferimenti(auto, cfg, fattori)
    w_c, w_h, w_s = cfg.xi_ott_pesi
    P = float(cfg.xi_ott_penalita)

    def rms(u):                                  # RMS mobile, vedi finestra_causale()
        return np.sqrt(_media_mobile(u ** 2, w, cfg))

    def hinge(u):                                # 0 dentro il vincolo, quadratico fuori
        return np.maximum(0.0, u - 1.0) ** 2

    return (w_c * (rms(az_w) / a_rif) ** 2
            + P * w_h * hinge(rms(gom) / d_rif)
            + P * w_s * hinge(rms(cor) / s_rif)
            + P * hinge(rms(frz) / f_rif))


def _lisciatura(u, cfg, secondi=0.3):
    """Media mobile breve: l'argmin puo' oscillare fra candidati quasi equivalenti e
    un'etichetta a scatti non e' riproducibile da nessuna rete."""
    return uniform_filter1d(np.asarray(u, dtype=float),
                            size=max(1, int(secondi * cfg.freq_campion)), mode="nearest")


def ottimo(az_misurata, v, zr_dot, auto, cfg, ritorna_dettagli=False):
    """Etichette OTTIME e COERENTI (xi*, kp*): la coppia damper + skyhook che minimizza il
    comfort ISO 2631 restando dentro i limiti di distacco ruota, fine corsa e forza.

    Ingressi: a_z misurata (non usata dal criterio, tenuta per simmetria di firma),
    velocita' v e velocita' del profilo strada z_r' gia' ricostruita e calibrata dai dati.
    Uscite: xi*(t) adimensionale e kp*(t) in Ns/m, entrambi lisciati nel tempo.
    """
    N = len(zr_dot)
    xi_g, kp_g, xi_cand, kp_cand = griglia_candidati(auto, cfg)

    az, gom, cor, frz = _simula_griglia(zr_dot, xi_cand, kp_cand, auto, cfg)
    az_w = np.stack([pondera_wk(a, cfg.freq_campion) for a in az])     # ponderazione ISO

    # FATTORI DI PICCO misurati su QUESTI segnali, non assunti. Si prende la MEDIANA fra i
    # candidati: il fattore dipende debolmente da (xi, kp), e la mediana e' la scelta stabile
    # che evita di far dipendere il vincolo dal candidato che si sta valutando (sarebbe
    # circolare: il costo definirebbe il proprio metro).
    fattori = dict(gomma=float(np.median([fattore_picco(g) for g in gom])),
                   corsa=float(np.median([fattore_picco(c_) for c_ in cor])),
                   forza=float(np.median([fattore_picco(f_) for f_ in frz
                                          if np.any(np.abs(f_) > 1e-9)] or [np.nan])))
    J = _costo(az_w, gom, cor, frz, auto, cfg, fattori)                # (n_cand, N)

    i = np.argmin(J, axis=0)
    xi_str = np.clip(_lisciatura(xi_cand[i], cfg), auto.xi_min, auto.xi_max).astype(np.float32)
    kp_str = np.clip(_lisciatura(kp_cand[i], cfg), kp_g[0], kp_g[-1]).astype(np.float32)

    if not ritorna_dettagli:
        return xi_str, kp_str

    a_rif, d_rif, s_rif, f_rif = riferimenti(auto, cfg, fattori)
    w = max(3, int(cfg.xi_ott_finestra_s * cfg.freq_campion))
    n = np.arange(N)

    def rms(u):
        return np.sqrt(_media_mobile(u ** 2, w, cfg))

    return xi_str, kp_str, dict(
        xi_griglia=xi_g, kp_griglia=kp_g, J=J, indice=i,
        fattori_picco=fattori,          # MISURATI su questi segnali, non assunti
        comfort=rms(az_w)[i, n] / a_rif,
        margine_gomma=rms(gom)[i, n] / d_rif,       # >1 = ruota che stacca
        margine_corsa=rms(cor)[i, n] / s_rif,       # >1 = fine corsa
        margine_forza=rms(frz)[i, n] / f_rif,       # >1 = attuatore saturo
        quota_vincolata=float(np.mean((rms(gom)[i, n] / d_rif > 1.0)
                                      | (rms(cor)[i, n] / s_rif > 1.0)
                                      | (rms(frz)[i, n] / f_rif > 1.0))),
        xi_sul_bordo=float(np.mean((xi_str <= xi_g[0] + 1e-6) | (xi_str >= xi_g[-1] - 1e-6))),
        kp_sul_bordo=float(np.mean((kp_str <= kp_g[0] + 1e-6) | (kp_str >= kp_g[-1] - 1e-6))))


def xi_ottimo(az_misurata, v, zr_dot, auto, cfg):
    """Solo xi* (comodita': quando serve la sola schedulazione del damper)."""
    return ottimo(az_misurata, v, zr_dot, auto, cfg)[0]


# --- quanto xi* si discosta dalla regola sqrt(2) -> alpha suggerito ---
def _logit(u):
    # il clip non e' cosmetico: molte etichette cadono ESATTAMENTE su xi_min (fondo liscio,
    # vincoli inattivi) e li' il logit diverge. 0.005 corrisponde a +-5.3 in logit, che una
    # sigmoide raggiunge senza problemi: e' il limite pratico di risoluzione, non una fudge.
    u = np.clip(u, 0.005, 1 - 0.005)
    return np.log(u / (1.0 - u))


def scarto_dal_baseline(xi_str, v, auto, cfg):
    """Distanza fra le etichette e il BASELINE della rete, misurata in logit — cioe' nella
    stessa scala in cui agisce la correzione alpha*Delta. Serve a scegliere alpha dai dati.

    Il baseline confrontato e' quello configurato (cfg.xi_baseline), non sempre la √2:
    con etichette xi* ottime il baseline giusto e' "comfort" (damper morbido)."""
    lo, hi = auto.xi_min, auto.xi_max
    u = (np.asarray(xi_str, dtype=float) - lo) / (hi - lo)
    if getattr(cfg, "xi_baseline", "sqrt2") == "comfort":
        base = np.full_like(u, _logit(np.array([float(cfg.xi_margine_logit)]))[0])
    else:
        xi_fis = np.asarray(xi_da_r(stima_frequenza(None, v, cfg, auto), auto, cfg), dtype=float)
        base = _logit((xi_fis - lo) / (hi - lo))
    d = _logit(u) - base
    # alpha SUGGERITO = il MASSIMO scarto piu' un margine del 10%.
    # alpha non e' una statistica da centrare, e' un TETTO: tararlo su un percentile lascia
    # per costruzione una frazione delle etichette fuori portata, e quella frazione sono le
    # strade piu' sconnesse, cioe' i casi che decidono la tenuta. Il sintomo e' riconoscibile
    # — la rete si ferma sullo stesso valore su OGNI riga e OGNI colonna della mappa, che e'
    # esattamente sigmoid(L0 + alpha): saturazione, non errore di stima.
    # Il margine del 10% serve perche' la sigmoide raggiunge il bordo solo asintoticamente.
    return dict(mediana=float(np.median(np.abs(d))),
                p90=float(np.percentile(np.abs(d), 90)),
                p99=float(np.percentile(np.abs(d), 99)),
                massimo=float(np.max(np.abs(d))),
                medio_con_segno=float(np.mean(d)),
                alpha_suggerito=float(np.max(np.abs(d))) * 1.10,
                baseline=getattr(cfg, "xi_baseline", "sqrt2"))


# retrocompatibilita' con il nome precedente
scarto_dalla_regola = scarto_dal_baseline


# --- confronto fra controllori: regola sqrt(2), xi* ottimo, passivo ---
def simula_con_xi(zr_dot, vel, xi_serie, auto, cfg, kp_serie=None, con_attuatore=True):
    """Quarter-car con xi(t) E kp(t) IMPOSTI dall'esterno (non da una rete). Serve per
    valutare i controllori analitici — regola √2 e ottimo vincolato — sulle stesse strade.

    kp_serie va passato ESPLICITAMENTE: prima veniva ricavato sempre da kp_da_r, cioe' dalla
    regola √2, e quindi il confronto valutava l'ottimo con il guadagno skyhook dell'euristica
    — non l'ottimo. Se None si usa la regola √2 (modalita' di confronto storica)."""
    N = len(zr_dot)
    xi_serie = np.broadcast_to(np.asarray(xi_serie, dtype=float), (N,))
    c = auto.c_da_xi(xi_serie)
    kp = (kp_da_r(stima_frequenza(None, vel, cfg, auto), cfg) if kp_serie is None
          else np.broadcast_to(np.asarray(kp_serie, dtype=float), (N,)))
    h = cfg.passo_t / cfg.n_sottopassi
    x = np.zeros(4)
    az = np.zeros(N); gom = np.zeros(N); cor = np.zeros(N)
    for k in range(N):
        f = float(np.clip(-kp[k] * x[1], -cfg.forza_max, cfg.forza_max)) if con_attuatore else 0.0
        az[k] = (-auto.rigid_molla * x[0] - c[k] * (x[1] - x[3]) + f) / auto.massa_cassa
        gom[k] = x[2]; cor[k] = x[0]
        zr0 = zr_dot[k]; zr1 = zr_dot[k + 1] if k + 1 < N else zr_dot[k]
        for _ in range(cfg.n_sottopassi):
            x = _rk4(x[None, :], np.array([c[k]]), np.array([f]), zr0, zr1, auto, h)[0]
    return az, gom, cor


def metriche(az, gom, cor, auto, cfg):
    """Le tre grandezze che contano, piu' i margini rispetto ai limiti fisici."""
    a_rif, d_rif, s_rif, f_rif = riferimenti(auto, cfg)   # 4: comfort + i tre vincoli
    azw = pondera_wk(az, cfg.freq_campion)
    d_max = (auto.massa_cassa + auto.massa_ruota) * G / auto.rigid_gomma
    return dict(comfort_wk=float(np.sqrt(np.mean(azw ** 2))),
                comfort_grezzo=float(np.sqrt(np.mean(az ** 2))),
                gomma_rms=float(np.sqrt(np.mean(gom ** 2))),
                corsa_rms=float(np.sqrt(np.mean(cor ** 2))),
                margine_gomma=float(np.sqrt(np.mean(gom ** 2)) / d_rif),
                margine_corsa=float(np.sqrt(np.mean(cor ** 2)) / s_rif),
                distacco=float(np.mean(np.abs(gom) > d_max)),
                fine_corsa=float(np.mean(np.abs(cor) > cfg.corsa_disponibile)))
