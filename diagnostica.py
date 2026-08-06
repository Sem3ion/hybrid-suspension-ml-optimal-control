# -*- coding: utf-8 -*-
"""
diagnostica.py — VERIFICA FISICA del controllore appreso.

PERCHE' NON BASTA L'R^2
-----------------------
Un R^2 alto sul dataset dice che la rete riproduce le etichette sui campioni che le sono
stati mostrati. Non dice se il controllore, messo nell'anello, si comporta in modo
fisicamente sensato. Le due cose possono divergere completamente: una rete che riconosce
quale traccia sta guardando dalla firma spettrale dell'accelerazione ottiene un R^2
eccellente e in closed-loop fa scelte assurde, perche' quella firma non e' piu' la stessa.

Questi strumenti guardano il COMPORTAMENTO: cosa comanda la rete al variare della strada e
della velocita', quanto si discosta dal riferimento, e se l'anello chiuso e' stabile.

I CRITERI SI RIBALTANO CON L'ETICHETTA — attenzione
---------------------------------------------------
  cfg.metodo_xi = "sqrt2"    il target e' xi(v), funzione della sola velocita'. L'accelerazione
      non deve contare NIENTE: se conta e' una scorciatoia. La monotonia rispetto a v e' un
      requisito, e l'indice di scorciatoia deve essere ~0.
  cfg.metodo_xi = "ottimo"   il target dipende soprattutto dalla RUGOSITA' della strada.
      L'accelerazione DEVE contare: se non conta, la rete sta ignorando l'informazione utile.
      La monotonia in v non e' piu' un requisito (la tenuta di strada puo' richiedere di
      irrigidire anche ad alta velocita'), e l'indicatore che conta e' |xi_rete - xi*| in
      closed-loop.
Le stampe si adattano da sole; i criteri marcati [OK]/[NO] valgono solo nel modo
corrispondente. Leggere un indicatore col criterio dell'altro modo porta a conclusioni
esattamente rovesciate.

CONTENUTO
  1. mappa_xi_open_loop     xi su una griglia (velocita' x livello di a_z), con finestre di
                            forma spettrale realistica riscalate al livello indicato: un
                            esperimento controllato, una variabile per volta.
                            RIGHE = livelli di a_z, COLONNE = velocita'.
  2. mappa_xi_closed_loop   xi medio con l'anello CHIUSO su strade ISO A..E a varie
                            velocita', confrontato con xi* calcolato su quella stessa strada.
                            E' la misura che conta: include la retroazione.
  3. distribuzioni_dataset  dove vivono i campioni nel piano (v, RMS a_z), come si distribuisce
                            l'etichetta, quanto rugosita' e velocita' sono confuse fra loro, e
                            quanto della griglia e' effettivamente coperto.
  4. statistiche_xi         indici di sanita', interpretati secondo l'etichetta in uso.
  5. sensibilita_feedback   guadagno d'anello rete -> a_z -> rete. Sopra la risonanza piu'
                            smorzamento significa PIU' accelerazione trasmessa, quindi una
                            sensibilita' positiva chiude un anello a retroazione positiva.
                            Con etichette ottime puo' essere legittimo — se xi* stesso sale
                            con la rugosita' — e va giudicato contro xi*, non da solo.

Ogni tabella dichiara SU QUALI DATI e' calcolata e se i numeri sono appresi o calcolati.

USO
    python diagnostica.py                 carica rete_xi.pt / rete_forza.pt + normalizzatore.json
    python diagnostica.py --dataset       ricostruisce anche il dataset (piu' lento)
    python diagnostica.py --no-figure     solo tabelle a schermo
oppure da main.py:  esegui_diagnostica(rete_xi, rete_forza, norm, auto, cfg, device, dati=...)
"""
import argparse
import os
import sys

import numpy as np
import torch

from config import Auto, Config
from controllo import xi_da_r
from fisica import _freq_velocita, accel_passiva_serie
from portabilita import apri_testo, forza_utf8
from reti import crea_reti, norm_a_xi
from simulazione import _device_di, simula_closed_loop
from strade_sintetiche import _GD0, _profilo_iso8608, _v_crossover, profilo_come_training

BASE = os.path.dirname(os.path.abspath(__file__))


# --- utilita' ---
def _predici_xi(rete_xi, finestre, velocita, norm, auto, cfg, device, batch=512):
    """Valuta la rete xi su un blocco di finestre (M, seq) e velocita' (M,) -> xi (M,).

    Il device si legge DAI PESI: l'argomento 'device' e' quello scelto all'avvio, ma le reti
    possono essere state spostate nel frattempo (il closed-loop a strada singola gira su CPU)
    e un tensore su un device diverso dai pesi fa fallire la conv1d."""
    device = _device_di(rete_xi) or device
    rete_xi.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(finestre), batch):
            az = torch.as_tensor(norm.na(finestre[i:i + batch]), dtype=torch.float32,
                                 device=device).unsqueeze(1)
            v = torch.as_tensor(norm.nv(velocita[i:i + batch]), dtype=torch.float32,
                                device=device).view(-1, 1)
            out.append(rete_xi(az, v).cpu().numpy().ravel())
    return norm_a_xi(np.concatenate(out), auto)


def _banca_finestre(auto, cfg, n_finestre=48, seed=7):
    """Banca di finestre di a_z FISICHE (non rumore bianco): simula il quarter-car passivo
    su strade generate ESATTAMENTE come quelle di addestramento (ISO 8608 + jitter di
    rugosita' + salite/discese + ostacoli) e ritaglia finestre di seq_len campioni.
    Serve per la mappa xi(v, a_z): le finestre vengono poi RISCALATE a un RMS bersaglio,
    cosi' si varia il LIVELLO di a_z tenendo una forma spettrale realistica."""
    rng = np.random.default_rng(seed)
    N = int(20.0 * cfg.freq_campion)
    per_combo = max(1, int(np.ceil(n_finestre / 12)))
    finestre = []
    for classe in ("B", "C", "D", "E"):
        for v0 in (8.0, 16.0, 24.0):
            vel = np.full(N, v0)
            zr, _m = profilo_come_training(classe, vel, cfg, rng)   # ostacoli+jitter come nel training
            az = accel_passiva_serie(np.gradient(zr, cfg.passo_t), auto, cfg)
            for _ in range(per_combo):
                i0 = int(rng.integers(cfg.seq_len, N - cfg.seq_len))
                w = az[i0 - cfg.seq_len:i0]
                if np.std(w) > 1e-6:
                    finestre.append(w)
    return np.array(finestre[:n_finestre], dtype=np.float32)


def _riscala(finestre, rms_target):
    """Riscala ogni finestra al RMS bersaglio (a media nulla), preservandone la FORMA."""
    f = finestre - finestre.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(f ** 2, axis=1, keepdims=True)) + 1e-9
    return (f * (rms_target / rms)).astype(np.float32)


def _xi_baseline(v, auto, cfg):
    """Regola √2: funzione della SOLA velocita'.

    Con metodo_xi="ottimo" NON e' piu' il target, e' solo il BASELINE attorno a cui la rete
    impara la correzione. Va letta come tendenza di riferimento, non come risposta giusta."""
    return np.asarray(xi_da_r(_freq_velocita(np.atleast_1d(v), cfg, auto), auto, cfg),
                      dtype=float)


def _etichetta_xi(zr_dot, v, auto, cfg):
    """Il TARGET vero su una strada data: xi* ottimo se metodo_xi='ottimo', altrimenti √2."""
    if getattr(cfg, "metodo_xi", "sqrt2") != "ottimo":
        return _xi_baseline(v, auto, cfg)
    from xi_ottimo import xi_ottimo
    az_pas = accel_passiva_serie(zr_dot, auto, cfg)   # serve solo alla schedulazione di kp
    return np.asarray(xi_ottimo(az_pas, v, zr_dot, auto, cfg), dtype=float)


# --- 1. mappa xi(v, RMS a_z) — open loop ---
def mappa_xi_open_loop(rete_xi, norm, auto, cfg, device,
                       velocita=None, rms_griglia=None, n_finestre=48):
    """xi predetto su una griglia (velocita' x livello di a_z), a forma d'onda fissata.

    E' un esperimento CONTROLLATO: cambia una cosa per volta. La matrice restituita ha
    RIGHE = livelli di RMS a_z e COLONNE = velocita'. Se la rete e' sana varia lungo le
    righe (cioe' con v) e non lungo le colonne (cioe' con a_z): strisce verticali."""
    velocita = np.asarray(velocita if velocita is not None
                          else np.linspace(4.0, 30.0, 14), dtype=np.float32)
    rms_griglia = np.asarray(rms_griglia if rms_griglia is not None
                             else np.geomspace(0.05, 6.0, 12), dtype=np.float32)

    banca = _banca_finestre(auto, cfg, n_finestre)
    M = np.zeros((len(rms_griglia), len(velocita)), dtype=float)
    S = np.zeros_like(M)
    for i, rms in enumerate(rms_griglia):
        fin = _riscala(banca, float(rms))
        for j, v in enumerate(velocita):
            xi = _predici_xi(rete_xi, fin, np.full(len(fin), v, np.float32),
                             norm, auto, cfg, device)
            M[i, j] = float(np.mean(xi)); S[i, j] = float(np.std(xi))
    return dict(xi=M, dev=S, velocita=velocita, rms=rms_griglia,
                xi_fisico=_xi_baseline(velocita, auto, cfg))


# --- 2. mappa xi — closed loop (con la retroazione) ---
def mappa_xi_closed_loop(rete_xi, rete_forza, norm, auto, cfg, device,
                         classi=("A", "B", "C", "D", "E"),
                         velocita=(6.0, 11.0, 16.0, 21.0, 27.0), secondi=12.0, seed=3,
                         ripetizioni=None):
    """Simula il closed-loop su strade ISO di ogni classe a ogni velocita' e riporta xi medio.

    Differenza rispetto alla mappa open-loop: qui la finestra di a_z che la rete legge e'
    l'accelerazione della cassa GIA' CONTROLLATA, quindi l'anello
        rete -> xi/F -> dinamica -> a_z -> rete
    e' chiuso. Se xi cresce con a_z e siamo sopra r=√2 (dove piu' smorzamento significa
    PIU' accelerazione trasmessa) l'anello e' a retroazione positiva e si "aggrappa"
    all'estremo alto: e' il meccanismo del xi ~ 0.55 a 90 km/h su strada sconnessa.

    PERCHE' SI RIPETE OGNI CELLA (cfg.diag_ripetizioni)
    ---------------------------------------------------
    Con una sola realizzazione per cella la tabella e' RUMOROSA, e il rumore e' nel TARGET,
    non nella rete. Ogni strada dura pochi secondi e ha ostacoli piazzati a caso: se un dosso
    cade dentro la finestra di costo il vincolo di distacco ruota si attiva e xi* salta in
    alto, se cade fuori resta al minimo. Misurato su una riga di classe B, xi* faceva
    0.082 -> 0.229 -> 0.180 -> 0.085 -> 0.322 al crescere della velocita': un'escursione di
    0.240 dentro la stessa classe, cioe' quasi sette volte l'errore medio della rete (0.035).
    Confrontare cella per cella in quelle condizioni vuol dire leggere il sorteggio degli
    ostacoli, non il controllore.
    Ripetendo ogni cella con semi diversi e mediando, il rumore scende come 1/sqrt(n), e lo
    scarto fra ripetizioni viene riportato: e' la barra d'errore sotto la quale nessuna
    differenza fra rete e target e' interpretabile."""
    from simulazione import simula_closed_loop_batch
    n_rip = int(ripetizioni if ripetizioni is not None else getattr(cfg, "diag_ripetizioni", 1))
    n_rip = max(1, n_rip)
    N = int(secondi * cfg.freq_campion)
    s = int(0.2 * N)                                  # scarta il transitorio di warm-up

    # TUTTE le combinazioni classe x velocita' x ripetizione in UN SOLO batch. Sono
    # simulazioni INDIPENDENTI: farle in sequenza vorrebbe dire tanti piu' lanci di kernel,
    # ognuno con la propria sincronizzazione, per lo stesso risultato.
    strade, chiavi = [], []
    for rip in range(n_rip):
        rng = np.random.default_rng(seed + 1000 * rip)   # una realizzazione diversa per giro
        for i, classe in enumerate(classi):
            for j, v0 in enumerate(velocita):
                vel = np.full(N, float(v0))
                # jitter=1.0: rugosita' NOMINALE della classe, cosi' la riga della tabella
                # resta interpretabile come "classe C" invece di essere una classe casuale
                # fra due. Ostacoli e pendenze restano: le strade di training li hanno sempre.
                zr, _m = profilo_come_training(classe, vel, cfg, rng, jitter=1.0)
                strade.append(dict(zr_dot=np.gradient(zr, cfg.passo_t), vel=vel))
                chiavi.append((rip, i, j))

    out = simula_closed_loop_batch(strade, rete_xi, rete_forza, norm, auto, cfg, device)

    forma = (n_rip, len(classi), len(velocita))
    XI_R = np.zeros(forma); RMS_R = np.zeros(forma); ATT_R = np.zeros(forma)
    for n, (rip, i, j) in enumerate(chiavi):
        XI_R[rip, i, j] = float(np.mean(out["xi"][n, s:]))
        RMS_R[rip, i, j] = float(np.sqrt(np.mean(out["acc_cassa_ml"][n, s:] ** 2)))
        # TARGET vero su QUESTA strada (non la regola √2): dipende dalla classe
        ATT_R[rip, i, j] = float(np.mean(_etichetta_xi(strade[n]["zr_dot"],
                                                       strade[n]["vel"], auto, cfg)[s:]))
    # incertezza della MEDIA su n_rip realizzazioni = scarto fra realizzazioni / sqrt(n)
    inc = ATT_R.std(axis=0, ddof=1) / np.sqrt(n_rip) if n_rip > 1 else np.zeros(forma[1:])
    return dict(xi=XI_R.mean(axis=0), rms_az=RMS_R.mean(axis=0),
                xi_atteso=ATT_R.mean(axis=0), incertezza_target=inc,
                dispersione_target=(ATT_R.std(axis=0, ddof=1) if n_rip > 1
                                    else np.zeros(forma[1:])),
                ripetizioni=n_rip, secondi=secondi,
                classi=list(classi), velocita=np.asarray(velocita, float),
                baseline=np.asarray([_xi_baseline(x, auto, cfg)[0] for x in velocita]))


# --- 3. distribuzioni del dataset ---
def distribuzioni_dataset(finestre_az, vel, xi_target, nome="TRAINING", max_campioni=150000,
                          ottimo=False, provenienza=""):
    """Statistiche del dataset nel piano (v, RMS a_z). Quantifica il CONFOUNDING:
    se RMS(a_z) e xi_target sono correlati, la rete puo' predire xi senza guardare v."""
    finestre_az = np.asarray(finestre_az)
    n = len(finestre_az)
    if n > max_campioni:                 # sottocampiona: le correlazioni non ne risentono
        sel = np.linspace(0, n - 1, max_campioni).astype(int)
        finestre_az = finestre_az[sel]; vel = np.asarray(vel)[sel]
        xi_target = np.asarray(xi_target)[sel]
    rms = np.sqrt(np.mean(finestre_az.astype(np.float32) ** 2, axis=1)).astype(np.float64)
    v = np.asarray(vel, dtype=np.float64)
    xi = np.asarray(xi_target, dtype=np.float64)

    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return _corr(ra, rb)

    # correlazione PARZIALE fra RMS(a_z) e xi al netto della velocita': quanto resta della
    # correlazione una volta rimosso l'effetto di v (regressione lineare su v)
    A = np.column_stack([np.ones_like(v), v])
    res_rms = rms - A @ np.linalg.lstsq(A, rms, rcond=None)[0]
    res_xi = xi - A @ np.linalg.lstsq(A, xi, rcond=None)[0]

    return dict(rms=rms, v=v, xi=xi, n_totale=n,
                provenienza=provenienza,
                corr_rms_xi=_corr(rms, xi), spearman_rms_xi=_spearman(rms, xi),
                corr_v_xi=_corr(v, xi), corr_rms_v=_corr(rms, v),
                corr_parziale=_corr(res_rms, res_xi), nome=nome, ottimo=ottimo,
                copertura=_copertura(v, rms))


def _copertura(v, rms, n_v=6, n_r=6):
    """Frazione di celle occupate nel piano (v, log RMS): misura quanto il disegno
    sperimentale e' FATTORIALE (celle piene sparse su tutta la griglia) invece che
    DIAGONALE (poche celle allineate, dove v e a_z non sono separabili)."""
    if len(v) == 0:
        return 0.0
    bv = np.linspace(v.min(), v.max() + 1e-9, n_v + 1)
    lr = np.log10(rms + 1e-9)
    br = np.linspace(lr.min(), lr.max() + 1e-9, n_r + 1)
    H, _, _ = np.histogram2d(v, lr, bins=[bv, br])
    return float((H > 0).sum() / H.size)


# --- 4. statistiche e indici ---
def statistiche_xi(mappa):
    """Dalla mappa open-loop ricava gli indici di sanita' fisica.

    indice_scorciatoia in [0,1]
        = variazione di xi lungo a_z / (variazione lungo a_z + variazione lungo v).
        0.0 -> xi dipende SOLO dalla velocita' (comportamento voluto)
        0.5 -> a_z pesa quanto la velocita'
        1.0 -> la velocita' e' ignorata (scorciatoia pura)

    margine_monotonia
        = min(xi alle basse v, su tutte le a_z) - max(xi alle alte v, su tutte le a_z).
        POSITIVO -> nessuna strada sconnessa ad alta velocita' puo' produrre uno xi
        maggiore di una strada liscia a bassa velocita'. E' la proprieta' che serve.

    frazione_monotona
        = quota di coppie di velocita' adiacenti in cui xi scende, mediata sulle a_z.
    """
    M, v, rms = mappa["xi"], mappa["velocita"], mappa["rms"]

    var_lungo_az = float(np.mean(np.std(M, axis=0)))       # a v fissa, variando a_z
    var_lungo_v = float(np.mean(np.std(M, axis=1)))        # a a_z fissa, variando v
    indice = var_lungo_az / (var_lungo_az + var_lungo_v + 1e-12)

    i_lo = v <= np.quantile(v, 0.25)
    i_hi = v >= np.quantile(v, 0.75)
    margine = float(M[:, i_lo].min() - M[:, i_hi].max())

    diffs = np.diff(M, axis=1)
    frazione = float(np.mean(diffs < 0))

    # sensibilita' locale: d(xi)/d(log10 RMS) a velocita' fissa, e d(xi)/dv a a_z fissa
    dlog = np.gradient(np.log10(rms))
    dxi_dlogrms = float(np.mean(np.gradient(M, axis=0) / dlog[:, None]))
    dxi_dv = float(np.mean(np.gradient(M, axis=1) / np.gradient(v)[None, :]))

    # errore rispetto alla regola fisica, mediato sulle a_z
    err = float(np.mean(np.abs(M.mean(axis=0) - mappa["xi_fisico"])))

    return dict(indice_scorciatoia=indice, margine_monotonia=margine,
                frazione_monotona=frazione, var_lungo_az=var_lungo_az,
                var_lungo_v=var_lungo_v, dxi_dlogrms=dxi_dlogrms, dxi_dv=dxi_dv,
                errore_vs_fisica=err)


def sensibilita_feedback(mappa, auto, cfg):
    """Guadagno d(xi)/d(log RMS a_z) separato SOTTO e SOPRA il crossover r=√2.

    Sopra r=√2 la trasmissibilita' PEGGIORA all'aumentare dello smorzamento: se in quella
    zona d(xi)/d(RMS) > 0, l'anello chiuso e' a retroazione POSITIVA (piu' a_z -> piu' xi
    -> piu' a_z) e il controllore si aggrappa all'estremo alto. E' il segno diagnostico
    del comportamento riportato in closed-loop su strade molto sconnesse."""
    M, v, rms = mappa["xi"], mappa["velocita"], mappa["rms"]
    v_cross = _v_crossover(auto, cfg)
    dlog = np.gradient(np.log10(rms))[:, None]
    g = np.gradient(M, axis=0) / dlog
    sotto = v < v_cross
    sopra = v >= v_cross
    return dict(v_crossover=float(v_cross),
                guadagno_sotto=float(np.mean(g[:, sotto])) if sotto.any() else float("nan"),
                guadagno_sopra=float(np.mean(g[:, sopra])) if sopra.any() else float("nan"),
                anello_positivo=bool(sopra.any() and np.mean(g[:, sopra]) > 0.01))


# --- stampa ---
def stampa_mappa(mappa, auto, cfg):
    M, v, rms = mappa["xi"], mappa["velocita"], mappa["rms"]
    print("\n" + "=" * 96)
    print(" 1) MAPPA xi(v, RMS a_z) — OPEN LOOP")
    print("=" * 96)
    print(" DATI      : banca di finestre di a_z generate col modello PASSIVO su profili ISO")
    print("             8608 classi B,C,D,E a 29/58/86 km/h, poi RISCALATE al livello di RMS")
    print("             indicato in riga. La FORMA spettrale resta realistica, cambia solo")
    print("             l'ampiezza: e' un esperimento controllato, una variabile per volta.")
    print(" xi        : APPRESO dalla rete (uscita, non etichetta). Media su tutte le finestre.")
    print(" NB        : e' OPEN LOOP — a_z e' imposta dall'esterno, non prodotta dal controllo.")
    print("             Per il comportamento reale vedi la mappa closed-loop al punto 2.")
    print("    Righe = livello di a_z, colonne = velocita'. Quindi:")
    print("      lungo una RIGA (a_z fissa, v che cresce) xi deve SCENDERE da xi_max a xi_min;")
    print("      lungo una COLONNA (v fissa, a_z che cresce) xi deve restare QUASI COSTANTE.")
    print()
    print("  RMS a_z \\ v [km/h] " + "".join(f"{x*3.6:7.0f}" for x in v))
    print("  " + "-" * (19 + 7 * len(v)))
    for i in range(len(rms) - 1, -1, -1):
        print(f"  {rms[i]:7.2f} m/s^2      " + "".join(f"{M[i, j]:7.3f}" for j in range(len(v))))
    print("  " + "-" * (19 + 7 * len(v)))
    print("  BASELINE √2 (solo v) " + "".join(f"{x:7.3f}" for x in mappa["xi_fisico"]))
    print(f"  [range damper: xi_min={auto.xi_min:.3f}  xi_max={auto.xi_max:.3f}]")


def stampa_closed_loop(cl, auto):
    print("\n" + "=" * 96)
    print(" 2) xi MEDIO IN CLOSED-LOOP su strade ISO 8608 (anello di retroazione CHIUSO)")
    print("=" * 96)
    v = cl["velocita"]
    n_rip = int(cl.get("ripetizioni", 1))
    print(f" DATI      : {len(cl['classi'])} profili ISO 8608 (classi {', '.join(cl['classi'])})"
          f" x {len(v)} velocita' costanti, {cl.get('secondi', 12):.0f} s ciascuno,")
    print(f"             generati come le strade di TRAINING (ostacoli, pendenze, spettro ISO).")
    if n_rip > 1:
        print(f"             Ogni cella e' la MEDIA di {n_rip} realizzazioni con semi diversi:")
        print("             con una sola strada per cella si legge il sorteggio degli ostacoli.")
    else:
        print("             UNA sola realizzazione per cella: i valori sono rumorosi"
              " (vedi cfg.diag_ripetizioni).")
    print("             Scartato il 20% iniziale (warm-up: la finestra parte piena di zeri).")
    print(" xi rete   : APPRESO — la rete legge la propria a_z controllata e comanda.")
    print(" xi* target: CALCOLATO su QUELLA strada per ottimizzazione vincolata (conosce z_r).")
    print("  classe \\ v [km/h]  " + "".join(f"{x*3.6:9.0f}" for x in v))
    print("  " + "-" * (19 + 9 * len(v)))
    for i, c in enumerate(cl["classi"]):
        etich = f"{c} (sconnessa)" if c == "E" else (f"{c} (liscia)" if c == "A" else c)
        riga = "".join(f"{cl['xi'][i, j]:9.3f}" for j in range(len(v)))
        print(f"  {etich:<17}" + riga)
    print("  " + "-" * (19 + 9 * len(v)))
    print("\n  TARGET xi* per classe (l'ottimo su QUELLA strada: cambia riga per riga):")
    for i, c in enumerate(cl["classi"]):
        print(f"    {c}: " + "".join(f"{cl['xi_atteso'][i, j]:8.3f}" for j in range(len(v))))
    print("  baseline √2 (solo v): " + "".join(f"{x:8.3f}" for x in cl["baseline"]))
    print("\n  RMS a_z closed-loop [m/s^2] (per capire quanto e' fuori distribuzione):")
    for i, c in enumerate(cl["classi"]):
        print(f"    {c}: " + "".join(f"{cl['rms_az'][i, j]:8.2f}" for j in range(len(v))))

    err = np.abs(cl["xi"] - cl["xi_atteso"])
    i, j = np.unravel_index(np.argmax(err), err.shape)
    print(f"\n  Errore medio  |xi_rete - xi*| = {float(np.mean(err)):.3f}")
    print(f"  Errore MASSIMO                = {err[i, j]:.3f} "
          f"(classe {cl['classi'][i]}, {v[j]*3.6:.0f} km/h: "
          f"xi={cl['xi'][i, j]:.3f} contro xi*={cl['xi_atteso'][i, j]:.3f})")
    # BARRA D'ERRORE: sotto questa soglia una differenza fra rete e target non e'
    # interpretabile, e' il sorteggio degli ostacoli fra una realizzazione e l'altra
    inc = cl.get("incertezza_target")
    if inc is not None and n_rip > 1:
        disp_t = cl.get("dispersione_target")
        print(f"\n  RUMORE DEL TARGET (da {n_rip} realizzazioni della stessa cella):")
        print(f"    scarto tipo fra realizzazioni     : {float(np.mean(disp_t)):.3f}"
              f"   (max {float(np.max(disp_t)):.3f})")
        print(f"    incertezza sulla MEDIA riportata  : {float(np.mean(inc)):.3f}"
              f"   <- barra d'errore delle celle qui sopra")
        n_sign = int(np.sum(err > 2.0 * np.maximum(inc, 1e-9)))
        print(f"    celle con errore oltre 2 barre    : {n_sign} su {err.size}"
              f"   (le altre sono compatibili col rumore)")
        bias = float(np.mean(cl["xi"] - cl["xi_atteso"]))
        print(f"    bias medio con segno              : {bias:+.3f}"
              f"   ({'rete piu RIGIDA del target' if bias > 0 else 'rete piu MORBIDA del target'})")
    # quanto il target dipende DAVVERO dalla strada: se questa riga e' ~0, xi* e' ancora
    # una funzione della sola velocita' e siamo tornati al problema di partenza
    disp = float(np.mean(np.ptp(cl["xi_atteso"], axis=0)))
    disp_rete = float(np.mean(np.ptp(cl["xi"], axis=0)))
    print(f"\n  Escursione di xi* fra classe A ed E, a parita' di velocita' : {disp:.3f}")
    print(f"  Escursione prodotta dalla RETE                              : {disp_rete:.3f}")
    print("  (se la prima e' grande e la seconda ~0, la rete sta ignorando la strada;")
    print("   se la prima e' ~0, e' il TARGET a non dipendere dalla strada)")


def stampa_dataset(d):
    print("\n" + "=" * 96)
    print(f" 3) DISTRIBUZIONE DATASET ({d['nome']}) — dove vivono i campioni nel piano (v, a_z)")
    print("=" * 96)
    v, rms, xi = d["v"], d["rms"], d["xi"]
    print(f" DATI      : {d.get('provenienza', 'finestre di addestramento')}")
    print(" xi        : ETICHETTA calcolata (target), non uscita della rete")
    print(f"  campioni analizzati: {len(v)}" + (f" (sottocampionati da {d['n_totale']})"
                                               if d.get("n_totale", 0) > len(v) else ""))
    print(f"  v     [km/h] : min {v.min()*3.6:6.1f}  mediana {np.median(v)*3.6:6.1f}  "
          f"max {v.max()*3.6:6.1f}")
    print(f"  RMS a_z      : min {rms.min():6.3f}  mediana {np.median(rms):6.3f}  "
          f"max {rms.max():6.3f}")
    print(f"  xi target    : min {xi.min():6.3f}  mediana {np.median(xi):6.3f}  "
          f"max {xi.max():6.3f}")
    print()
    seg = ("il SEGNALE VERO (xi* dipende dalla rugosita')" if d.get("ottimo")
           else "la SCORCIATOIA disponibile: |.|>0.3 e' gia' sfruttabile")
    print(f"  corr(RMS a_z, xi)          = {d['corr_rms_xi']:+.3f}   <- {seg}")
    print(f"  Spearman(RMS a_z, xi)      = {d['spearman_rms_xi']:+.3f}")
    vv = ("informazione COMPLEMENTARE: serve a convertire a_z in rugosita'" if d.get("ottimo")
          else "il segnale VERO (forte e negativo)")
    print(f"  corr(v, xi)                = {d['corr_v_xi']:+.3f}   <- {vv}")
    print(f"  corr(RMS a_z, v)           = {d['corr_rms_v']:+.3f}   <- quanto a_z e v sono "
          f"confuse fra loro")
    pp = ("quanto conta a_z A PARITA' di velocita' (deve essere grande)" if d.get("ottimo")
          else "residuo dopo aver rimosso l'effetto di v")
    print(f"  corr parziale (a v fissa)  = {d['corr_parziale']:+.3f}   <- {pp}")
    print(f"  copertura griglia (v x a_z)= {d['copertura']:.0%}      <- alta = disegno "
          f"fattoriale, bassa = diagonale")
    print()
    print("  Tabella incrociata: xi target medio per bin di velocita' e di RMS a_z")
    _tabella_incrociata(v, rms, xi)


def _tabella_incrociata(v, rms, xi, n_v=5, n_r=5):
    bv = np.quantile(v, np.linspace(0, 1, n_v + 1))
    br = np.quantile(rms, np.linspace(0, 1, n_r + 1))
    intest = "  RMS \\ v [km/h]  " + "".join(
        f"{0.5*(bv[j]+bv[j+1])*3.6:9.0f}" for j in range(n_v))
    print(intest)
    print("  " + "-" * (len(intest) - 2))
    for i in range(n_r - 1, -1, -1):
        m_r = (rms >= br[i]) & (rms <= br[i + 1])
        riga = f"  {0.5*(br[i]+br[i+1]):14.2f}  "
        for j in range(n_v):
            m = m_r & (v >= bv[j]) & (v <= bv[j + 1])
            riga += f"{np.mean(xi[m]):9.3f}" if m.sum() > 20 else f"{'--':>9}"
        print(riga)
    print("  ('--' = cella VUOTA: combinazione mai vista in addestramento; se le celle piene")
    print("   stanno su una diagonale, velocita' e accelerazione NON sono separabili)")


def stampa_indici(st, fb, auto, cl=None, cfg=None):
    """I criteri CAMBIANO col tipo di etichetta, e vanno letti di conseguenza.

    Con metodo_xi="sqrt2" il target e' xi(v): a_z non deve contare NIENTE, quindi
    l'indice di scorciatoia deve essere ~0 e la monotonia in v e' un requisito.
    Con metodo_xi="ottimo" il target dipende anche dalla strada: a_z DEVE contare, e
    la monotonia stretta in v non e' piu' obbligatoria (su fondo sconnesso la tenuta
    di strada puo' richiedere di irrigidire anche ad alta velocita'). L'indicatore che
    conta in quel caso e' l'errore rispetto a xi*, non l'aderenza alla regola √2."""
    ottimo = cfg is not None and getattr(cfg, "metodo_xi", "sqrt2") == "ottimo"
    print("\n" + "=" * 96)
    print(" 4) INDICI DI SANITA' FISICA"
          + ("   [target = xi* ottimo]" if ottimo else "   [target = regola √2]"))
    print("=" * 96)

    def ok(b):
        return "OK  " if b else "NO  "

    i = st["indice_scorciatoia"]; m = st["margine_monotonia"]; f = st["frazione_monotona"]

    if ottimo:
        print(f"  [    ] quota di xi spiegata da a_z  = {i:.3f}   "
              f"(ora NON deve essere 0: se lo e', la rete ignora la strada)")
        print(f"          variazione di xi lungo a_z  = {st['var_lungo_az']:.4f}")
        print(f"          variazione di xi lungo v    = {st['var_lungo_v']:.4f}")
        if cl is not None:
            err = float(np.mean(np.abs(cl["xi"] - cl["xi_atteso"])))
            disp_t = float(np.mean(np.ptp(cl["xi_atteso"], axis=0)))
            disp_r = float(np.mean(np.ptp(cl["xi"], axis=0)))
            print(f"  [{ok(err < 0.05)}] errore closed-loop |xi - xi*| = {err:.4f}   "
                  f"(e' l'indicatore che conta ora)")
            print(f"  [{ok(disp_t < 1e-6 or disp_r > 0.75 * disp_t)}] "
                  f"escursione con la strada: rete {disp_r:.3f} contro target {disp_t:.3f}"
                  f"  ({100*disp_r/max(disp_t,1e-9):.0f}% del necessario)")
        print(f"  [    ] frazione monotona in v       = {f:.1%}  "
              f"(non piu' un requisito: la tenuta puo' chiedere di irrigidire)")
        print(f"  [    ] distanza dal baseline √2     = {st['errore_vs_fisica']:.4f}   "
              f"(se ~0 la correzione appresa non sta facendo nulla)")
    else:
        print(f"  [{ok(i < 0.25)}] indice di scorciatoia       = {i:.3f}   "
              f"(0 = xi dipende solo da v; >0.5 = a_z domina)")
        print(f"          variazione di xi lungo a_z  = {st['var_lungo_az']:.4f}")
        print(f"          variazione di xi lungo v    = {st['var_lungo_v']:.4f}")
        print(f"  [{ok(m > 0)}] margine di monotonia        = {m:+.3f}   "
              f"(min xi a bassa v  -  max xi ad alta v; deve essere > 0)")
        print(f"  [{ok(f > 0.95)}] frazione monotona in v      = {f:.1%}  "
              f"(quota di passi in v in cui xi scende)")
        print(f"  [{ok(st['errore_vs_fisica'] < 0.05)}] errore medio |xi - regola √2| = "
              f"{st['errore_vs_fisica']:.4f}")

    print()
    print(f"  Sensibilita' d(xi)/d(log10 RMS a_z), crossover r=√2 a "
          f"{fb['v_crossover']*3.6:.0f} km/h:")
    print(f"        sotto il crossover : {fb['guadagno_sotto']:+.4f}")
    print(f"        sopra il crossover : {fb['guadagno_sopra']:+.4f}")
    if ottimo:
        print("        (sopra r=√2 una sensibilita' POSITIVA e' un anello di retroazione")
        print("         positivo, ma ora puo' essere legittimo: se xi* stesso sale con la")
        print("         rugosita', l'anello e' voluto. Va giudicato contro xi*, non da solo.)")
    elif fb["anello_positivo"]:
        print("  [NO  ] ANELLO POSITIVO sopra r=√2: piu' a_z -> piu' xi -> (sopra risonanza)")
        print("         piu' accelerazione trasmessa -> piu' a_z. Il controllore si aggrappa")
        print("         all'estremo alto di xi proprio dove dovrebbe isolare.")
    else:
        print("  [OK  ] nessun anello positivo sopra il crossover")


# --- figura ---
def salva_figura_diagnostica(mappa, cl, dati, percorso):
    """Pannello unico: mappa open-loop, mappa closed-loop, curve xi(v), distribuzioni.

    NB: si usa Figure + FigureCanvasAgg LOCALE, non pyplot. matplotlib.use('Agg') cambia il
    backend GLOBALE del processo: chiamandolo qui si spegneva l'animazione interattiva di
    grafica.py, che gira dopo nella stessa esecuzione ('FigureCanvasAgg is non-interactive').
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from contesto import titola
    n_col = 3 if dati is None else 4
    # layout="constrained": con 8 pannelli, 4 colorbar e 4 legende il posizionamento a mano
    # tagliava i titoli ai bordi. Lo spazio per titolo e note lo riserva contesto.titola.
    fig = Figure(figsize=(5.6 * n_col, 10.2), layout="constrained")
    FigureCanvasAgg(fig)                       # canvas locale: nessun effetto sul backend
    ax = fig.subplots(2, n_col)

    M, v, rms = mappa["xi"], mappa["velocita"], mappa["rms"]

    # (0,0) mappa open loop
    im = ax[0, 0].pcolormesh(v * 3.6, rms, M, shading="auto", cmap="viridis")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("velocita' [km/h]"); ax[0, 0].set_ylabel("RMS a_z [m/s^2]")
    ax[0, 0].set_title("1) xi APPRESO — open-loop\nbande orizzontali = dipende dalla rugosita'",
                       fontsize=10.5)
    fig.colorbar(im, ax=ax[0, 0], label="xi")

    # (1,0) curve xi(v) a vari livelli di a_z + curva fisica
    for i in range(0, len(rms), max(1, len(rms) // 5)):
        ax[1, 0].plot(v * 3.6, M[i], marker="o", ms=3, lw=1, label=f"a_z {rms[i]:.2f}")
    ax[1, 0].plot(v * 3.6, mappa["xi_fisico"], "k--", lw=2.2, label="baseline √2")
    ax[1, 0].set_xlabel("velocita' [km/h]"); ax[1, 0].set_ylabel("xi")
    ax[1, 0].set_title("2) xi(v) a livelli di a_z diversi", fontsize=10.5)
    ax[1, 0].legend(fontsize=7, ncols=2); ax[1, 0].grid(alpha=0.3)

    # (0,1) mappa closed loop
    im = ax[0, 1].imshow(cl["xi"], aspect="auto", origin="lower", cmap="viridis",
                         vmin=M.min(), vmax=M.max())
    ax[0, 1].set_xticks(range(len(cl["velocita"])))
    ax[0, 1].set_xticklabels([f"{x*3.6:.0f}" for x in cl["velocita"]])
    ax[0, 1].set_yticks(range(len(cl["classi"]))); ax[0, 1].set_yticklabels(cl["classi"])
    ax[0, 1].set_xlabel("velocita' [km/h]"); ax[0, 1].set_ylabel("classe ISO 8608")
    ax[0, 1].set_title("3) xi APPRESO — closed-loop\nfeedback chiuso, velocita' costante",
                       fontsize=10.5)
    fig.colorbar(im, ax=ax[0, 1], label="xi")

    # (1,1) xi closed-loop per classe vs atteso
    for i, c in enumerate(cl["classi"]):
        ax[1, 1].plot(cl["velocita"] * 3.6, cl["xi"][i], marker="s", ms=4, label=f"classe {c}")
    for i, c in enumerate(cl["classi"]):
        ax[1, 1].plot(cl["velocita"] * 3.6, cl["xi_atteso"][i], "--", lw=1.2, alpha=0.6)
    ax[1, 1].plot(cl["velocita"] * 3.6, cl["baseline"], "k:", lw=2.0, label="baseline √2")
    ax[1, 1].set_xlabel("velocita' [km/h]"); ax[1, 1].set_ylabel("xi medio")
    ax[1, 1].set_title("4) closed-loop: rete (continuo) vs xi* (tratteggio)", fontsize=10.5)
    ax[1, 1].legend(fontsize=7, ncols=2); ax[1, 1].grid(alpha=0.3)

    # (0,2) scostamento dalla fisica in closed loop
    err = cl["xi"] - cl["xi_atteso"]
    lim = float(np.max(np.abs(err))) + 1e-9
    im = ax[0, 2].imshow(err, aspect="auto", origin="lower", cmap="coolwarm",
                         vmin=-lim, vmax=lim)
    ax[0, 2].set_xticks(range(len(cl["velocita"])))
    ax[0, 2].set_xticklabels([f"{x*3.6:.0f}" for x in cl["velocita"]])
    ax[0, 2].set_yticks(range(len(cl["classi"]))); ax[0, 2].set_yticklabels(cl["classi"])
    ax[0, 2].set_xlabel("velocita' [km/h]"); ax[0, 2].set_ylabel("classe ISO 8608")
    ax[0, 2].set_title("5) errore xi - xi*\nrosso = la rete irrigidisce troppo", fontsize=10.5)
    fig.colorbar(im, ax=ax[0, 2], label="xi - atteso")

    # (1,2) RMS a_z raggiunta in closed loop
    for i, c in enumerate(cl["classi"]):
        ax[1, 2].plot(cl["velocita"] * 3.6, cl["rms_az"][i], marker="o", ms=4, label=f"classe {c}")
    ax[1, 2].set_xlabel("velocita' [km/h]"); ax[1, 2].set_ylabel("RMS a_z [m/s^2]")
    ax[1, 2].set_title("6) RMS a_z raggiunta in closed-loop\ne' l'ingresso che la rete vede davvero",
                       fontsize=10.5)
    ax[1, 2].set_yscale("log"); ax[1, 2].legend(fontsize=7, ncols=2)
    ax[1, 2].grid(alpha=0.3)

    # (0,3)/(1,3) distribuzioni dataset
    if dati is not None:
        sc = ax[0, 3].scatter(dati["v"] * 3.6, dati["rms"], c=dati["xi"], s=2,
                              cmap="viridis", alpha=0.4)
        ax[0, 3].axhline(np.median(dati["rms"]), ls=":", c="crimson", lw=1)
        ax[0, 3].set_yscale("log")
        ax[0, 3].set_xlabel("velocita' [km/h]"); ax[0, 3].set_ylabel("RMS a_z finestra")
        ax[0, 3].set_title(f"7) ETICHETTE dataset {dati['nome']} — copertura (v, a_z)\n"
                           f"corr(a_z, xi*) = {dati['corr_rms_xi']:+.2f}   |   "
                           f"griglia coperta {dati['copertura']:.0%}", fontsize=10.5)
        fig.colorbar(sc, ax=ax[0, 3], label="xi target")

        # ISTOGRAMMA con percentuali: "quante barre" non dice niente, "il 63% dei campioni
        # sta al minimo del damper" dice che il dataset e' sbilanciato verso strade lisce
        cont, bordi, _p = ax[1, 3].hist(dati["xi"], bins=40, color="tab:blue", alpha=0.85)
        tot = max(cont.sum(), 1.0)
        for c, b0, b1 in zip(cont, bordi[:-1], bordi[1:]):
            q = 100.0 * c / tot
            if q >= 4.0:                      # solo le barre significative, o diventa illeggibile
                ax[1, 3].annotate(f"{q:.0f}%", (0.5 * (b0 + b1), c),
                                  textcoords="offset points", xytext=(0, 3),
                                  ha="center", fontsize=7, fontweight="bold")
        q_min = 100.0 * np.mean(dati["xi"] <= dati["xi"].min() + 1e-6)
        ax[1, 3].set_xlabel("xi target (etichetta calcolata)")
        # scala log: con il 55% dei campioni in una sola barra, in scala lineare le altre
        # 39 barre sono invisibili e l'istogramma non dice piu' niente sulla coda
        ax[1, 3].set_yscale("log")
        ax[1, 3].set_ylabel(f"campioni, scala log  (totale {int(tot)})")
        ax[1, 3].set_title(f"8) distribuzione delle ETICHETTE xi*\n"
                           f"{q_min:.0f}% dei campioni al minimo del damper", fontsize=10.5)
        ax[1, 3].grid(alpha=0.3)

    titola(fig, "Diagnostica fisica del controllore APPRESO — xi(v, a_z)",
           sottotitolo=(dati["provenienza"] if dati and dati.get("provenienza") else ""),
           nota=("Pannelli 1-2: finestre di a_z generate col modello PASSIVO su profili ISO 8608"
                 " (classi B-E), riscalate a RMS fissato — esperimento controllato, una variabile"
                 " per volta.   Pannelli 3-6: closed-loop su ISO A-E x 5 velocita', feedback"
                 " chiuso: e' la misura che conta.   Pannelli 7-8: etichette del dataset di"
                 " addestramento (calcolate, non apprese)."),
           dim_titolo=14)
    fig.savefig(percorso, dpi=130)
    print(f"\n  Figura diagnostica salvata: {os.path.basename(percorso)}")


# --- orchestratore ---
class _Doppio:
    """Scrive su schermo E su file (tee): il report resta consultabile dopo la run."""

    def __init__(self, flusso, file_):
        self.flusso = flusso; self.file = file_

    def write(self, s):
        self.flusso.write(s); self.file.write(s)

    def flush(self):
        self.flusso.flush(); self.file.flush()


def esegui_diagnostica(rete_xi, rete_forza, norm, auto, cfg, device,
                       dati=None, figura=True, percorso=None, report=None, provenienza=""):
    """Esegue tutta la batteria, stampa il verdetto e lo salva su file.
    'dati' = (finestre_az, vel, xi_target); 'report' = percorso .txt (None -> default)."""
    import contextlib
    import datetime

    percorso = percorso or os.path.join(BASE, "diagnostica_xi.png")
    report = report if report is not None else os.path.join(BASE, "diagnostica_report.txt")

    # encoding esplicito: il report contiene √ ≈ σ, che cp1252 (Windows) non ha.
    with apri_testo(report, "w") as f, contextlib.redirect_stdout(_Doppio(sys.stdout, f)):
        print("#" * 96)
        _ott = getattr(cfg, "metodo_xi", "sqrt2") == "ottimo"
        print("#  DIAGNOSTICA FISICA — " + ("xi deve seguire xi* ottimo: dipende dalla RUGOSITA'"
                                            " (da a_z) e dalla velocita'" if _ott else
                                            "xi deve seguire la VELOCITA', a_z puo' solo modulare"))
        print(f"#  {datetime.datetime.now():%Y-%m-%d %H:%M}"
              f"   |   vincolo strutturale: "
              f"{'ATTIVO alpha=%.2f' % getattr(rete_xi, 'autorita', float('nan')) if cfg.xi_vincolo_fisico else 'DISATTIVO'}"
              f"   |   metodo_freq: {cfg.metodo_freq}")
        print("#" * 96)

        mappa = mappa_xi_open_loop(rete_xi, norm, auto, cfg, device)
        stampa_mappa(mappa, auto, cfg)

        cl = mappa_xi_closed_loop(rete_xi, rete_forza, norm, auto, cfg, device)
        stampa_closed_loop(cl, auto)

        d = None
        if dati is not None:
            d = distribuzioni_dataset(*dati, ottimo=_ott, provenienza=provenienza)
            stampa_dataset(d)

        st = statistiche_xi(mappa)
        fb = sensibilita_feedback(mappa, auto, cfg)
        stampa_indici(st, fb, auto, cl, cfg)

        if figura:
            salva_figura_diagnostica(mappa, cl, d, percorso)

    print(f"  Report testuale salvato:    {os.path.basename(report)}")
    return dict(mappa=mappa, closed_loop=cl, dataset=d, statistiche=st, feedback=fb)


def _carica_reti(auto, cfg, device):
    """Carica reti + normalizzatore dai file salvati dall'addestramento."""
    def _carica(rete, nome):
        p = os.path.join(BASE, nome)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"\n  Manca {nome}: le reti non sono ancora state addestrate.\n"
                f"  Lancia prima 'python main.py' (la diagnostica parte gia' da sola alla fine).")
        try:
            rete.load_state_dict(torch.load(p, map_location=device))
        except RuntimeError as e:
            raise RuntimeError(
                f"\n  {nome} e' un checkpoint VECCHIO, salvato con l'architettura precedente:\n"
                f"  non e' caricabile e non serve a niente convertirlo.\n"
                f"  Cancellalo e rilancia 'python main.py' per riaddestrare.\n"
                f"  (Sono cambiate sia la testa di fusione a due rami sia, per xi, la\n"
                f"   schedulazione vincolata: i nomi dei pesi non coincidono piu'.)\n\n"
                f"  Dettaglio torch: {str(e)[:300]}") from None
        return rete

    rete_xi, rete_forza = crea_reti(cfg, auto)
    _carica(rete_xi, "rete_xi.pt"); _carica(rete_forza, "rete_forza.pt")
    rete_xi.to(device).eval(); rete_forza.to(device).eval()

    from dati import Normalizzatore
    p = os.path.join(BASE, "normalizzatore.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "\n  Manca normalizzatore.json. Lancia 'python main.py': lo salva insieme alle reti.\n"
            "  Ricostruirlo a mano con valori copiati falsa ogni risultato closed-loop, perche'\n"
            "  sposta la finestra di a_z rispetto a quella su cui la rete e' stata addestrata.")
    return rete_xi, rete_forza, Normalizzatore.carica(p)


def _costruisci_dataset(auto, cfg):
    """Ricostruisce le finestre di training per le distribuzioni (lento: rigenera le etichette)."""
    from dati import carica_tracce, costruisci_finestre
    from strade_sintetiche import genera_tracce_sintetiche
    tr, _ = carica_tracce(cfg)
    if cfg.usa_augmentation:
        tr = tr + genera_tracce_sintetiche(auto, cfg)
    # NB: costruisci_finestre ritorna 8 valori (dopo l'aggiunta dei target ausiliari kp*
    # e z_s'). Spacchettarne un numero fisso e' fragile: qui si prendono per indice i tre
    # che servono, cosi' aggiungere altre uscite non rompe piu' questa chiamata.
    out = costruisci_finestre(tr, auto, cfg)
    return out[0], out[1], out[2]


def main():
    ap = argparse.ArgumentParser(description="Diagnostica fisica del controllore appreso")
    ap.add_argument("--dataset", action="store_true",
                    help="ricostruisce anche il dataset di training (piu' lento)")
    ap.add_argument("--no-figure", action="store_true", help="solo tabelle a schermo")
    args = ap.parse_args()

    auto, cfg = Auto(), Config()
    device = torch.device("cpu")
    rete_xi, rete_forza, norm = _carica_reti(auto, cfg, device)
    print(f"Reti caricate. {norm}")
    print("Vincolo fisico strutturale su xi: "
          + (f"ATTIVO (beta={cfg.autorita_az:.2f})" if cfg.xi_vincolo_fisico else "DISATTIVO"))

    dati = _costruisci_dataset(auto, cfg) if args.dataset else None
    esegui_diagnostica(rete_xi, rete_forza, norm, auto, cfg, device,
                       dati=dati, figura=not args.no_figure)


if __name__ == "__main__":
    forza_utf8()          # vedi portabilita.py
    main()
