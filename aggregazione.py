# -*- coding: utf-8 -*-
"""
aggregazione.py — DAgger (Dataset Aggregation): elimina lo scarto fra i dati di addestramento
e quelli che la rete incontra davvero quando comanda lei.

IL PROBLEMA, MISURATO
---------------------
Le etichette e gli ingressi vengono dal modello PASSIVO. Ma in closed-loop la rete legge
l'accelerazione della cassa GIA' CONTROLLATA, che il controllo stesso ha ridotto:

    RMS a_z      addestramento -> closed-loop
    classe E         3.34      ->    1.69      (22 km/h)
    classe D         1.62      ->    0.76
    mediana su tutta la griglia: closed-loop = 47% dell'addestramento

Conta perche' il segnale utile per xi E' il livello di a_z: la rete impara "RMS 3.3 =
classe E" sui dati passivi, ma in closed-loop quella stessa strada le arriva a 1.69 — che
nella distribuzione di addestramento corrisponde a una classe piu' liscia. Risponde
correttamente a cio' che vede; e' cio' che vede a essere spostato di una classe di strada.

Quanto conta dipende dall'etichetta: se xi non dipendesse da a_z (come con la regola √2) lo
scarto sarebbe irrilevante.

Peggio: l'errore si auto-conferma. Ammorbidisce -> a_z cala -> la strada sembra ancora
piu' liscia -> ammorbidisce ancora.

PERCHE' DAgger E QUANTO COSTA QUI
---------------------------------
DAgger e' la risposta standard: addestra, fai guidare la rete, registra gli stati che
VISITA DAVVERO, rietichettali con l'esperto, aggrega, riaddestra. Qui e' insolitamente
economico per due motivi.

Primo, l'esperto e' automatico: e' l'ottimizzazione vincolata di xi_ottimo.py, non un
essere umano da interpellare.

Secondo, e piu' importante: **xi* dipende solo dalla STRADA, non dal controllore usato**.
E' la soluzione di un problema di ottimizzazione sul profilo stradale, quindi non va
ricalcolata a ogni giro. Cambia solo l'INGRESSO (la finestra di a_z), che e' esattamente
cio' che vogliamo correggere. L'etichetta della FORZA invece va rifatta, perche'
F = -kp*(t) * z_s'(t) dipende dalla velocita' di cassa effettiva, che cambia col
controllore: e' proprio la rietichettatura on-policy che DAgger prescrive.

MISCELAZIONE (beta)
-------------------
Al primo giro i rollout sono guidati dall'ESPERTO (beta=1): la distribuzione che si
raccoglie e' quella che vogliamo che la rete raggiunga, ed e' anche l'unica sensata
finche' la rete e' ancora sbagliata. Poi beta scende e il controllo passa alla rete, cosi'
il dataset copre anche gli stati che la rete raggiunge per i propri errori — che e' il
motivo per cui DAgger esiste. Senza decadimento sarebbe semplice behaviour cloning sulla
traiettoria dell'esperto; senza il primo giro a beta=1 si rischia di aggregare stati
prodotti da una policy ancora incompetente.

COSTO
-----
I rollout girano in BATCH su tutte le strade insieme: a ogni passo temporale si valuta la
rete una volta sola su un batch di finestre invece di una volta per strada. Un giro su 24
strade da 20 s costa quindi ~2000 forward batchati, non 48000 sequenziali.
"""
import numpy as np
import torch

from fisica import ricostruisci_strada, rms_accel_passiva, velocita_cassa_misurata
from reti import norm_a_xi, norm_a_forza
from xi_ottimo import ottimo


# --- preparazione: strada + etichette dell'esperto (calcolate UNA volta) ---
def prepara_strade(tracce, auto, cfg, secondi=None, verbose=True):
    """Da ogni traccia (a_z, v) ricava il profilo strada e le etichette dell'esperto.

    La strada si ricostruisce e si calibra esattamente come in controllo.py (doppia
    integrazione passa-alto + scalatura sull'RMS misurato), cosi' l'esperto qui e' lo
    stesso che ha generato le etichette originali. xi* e kp* si calcolano una volta sola:
    dipendono dalla strada, non dal controllore, quindi valgono per tutti i giri."""
    n = int((secondi or cfg.dagger_secondi) * cfg.freq_campion)
    strade = []
    n_vere = 0
    for i, t in enumerate(tracce):
        # una traccia e' (a_z, v) oppure (a_z, v, z_r') se la strada VERA e' nota
        az, v = t[0], t[1]
        zr_dot_vero = t[2] if len(t) > 2 else None
        nome = t[3] if len(t) > 3 else f"traccia {i+1}"
        m = min(len(az), n)
        az_s, v_s = np.asarray(az[:m], float), np.asarray(v[:m], float)
        if zr_dot_vero is not None:
            # stessa scelta di controllo.py: se la strada e' nota si usa QUELLA, perche'
            # ricostruirla da a_z sposta xi* di 0.064 in media
            zr_dot = np.asarray(zr_dot_vero, float)[:m]
            n_vere += 1
        else:
            zr, zr_dot = ricostruisci_strada(az_s, cfg)
            if cfg.calibra_strada:
                f = np.clip(np.sqrt(np.mean(az_s ** 2))
                            / (rms_accel_passiva(zr_dot, auto, cfg) + 1e-9), 0.2, 20.0)
                zr_dot = zr_dot * f
        xi_e, kp_e = ottimo(az_s, v_s, zr_dot, auto, cfg)
        strade.append(dict(zr_dot=zr_dot, vel=v_s, xi=np.asarray(xi_e, float),
                           kp=np.asarray(kp_e, float), N=m, nome=nome))
    if verbose:
        L = min(s["N"] for s in strade)
        print(f"    Preparate {len(strade)} strade per l'aggregazione "
              f"({L/cfg.freq_campion:.0f} s ciascuna), di cui {n_vere} con strada VERA nota; "
              f"etichette dell'esperto calcolate una volta")
    return strade


# --- rollout in batch (tutte le strade in parallelo) ---
def rollout(strade, rete_xi, rete_forza, norm, auto, cfg, device, beta=0.0):
    """Integra il closed-loop su TUTTE le strade contemporaneamente, con miscelazione
    esperto/rete regolata da beta.

    E' un involucro sottile su simulazione.simula_closed_loop_batch: quel ciclo era scritto
    due volte, qui e la', identico a meno della miscelazione. Ora la dinamica batchata ha un
    solo posto in cui vive, e questa funzione si limita a impacchettare gli ingressi e a
    restituire cio' che serve all'aggregazione."""
    from simulazione import simula_closed_loop_batch
    N = min(s["N"] for s in strade)
    XI_E = np.stack([s["xi"][:N] for s in strade])
    KP_E = np.stack([s["kp"][:N] for s in strade])
    VEL = np.stack([s["vel"][:N] for s in strade])
    out = simula_closed_loop_batch(
        [dict(zr_dot=s["zr_dot"][:N], vel=s["vel"][:N]) for s in strade],
        rete_xi, rete_forza, norm, auto, cfg, device,
        beta=beta, xi_esperto=XI_E, kp_esperto=KP_E)
    return out["acc_cassa_ml"], out["xi"], VEL, XI_E, KP_E


# --- rietichettatura e finestre ---
def finestre_da_rollout(az_cl, VEL, XI_E, KP_E, auto, cfg):
    """Costruisce (finestre, v, xi*, F*, kp*, z_s') dagli stati VISITATI.

    xi_target = xi* dell'esperto, invariato (dipende dalla strada).
    forza_target = RIETICHETTATA: -kp*(t) * z_s'(t) con la velocita' di cassa ricavata
    dall'a_z del rollout. E' il passo che rende DAgger diverso dal semplice aggiungere
    dati: l'azione dell'esperto viene ricalcolata NELLO STATO CHE LA RETE HA VISITATO."""
    from numpy.lib.stride_tricks import sliding_window_view
    passo = max(1, int(getattr(cfg, "passo_finestre", 1)))
    F, V, XI, FZ, KP, ZS = [], [], [], [], [], []
    ns, N = az_cl.shape
    for i in range(ns):
        a = az_cl[i].astype(np.float32)
        vc = velocita_cassa_misurata(a.astype(float), cfg)     # Kalman sull'a_z on-policy
        forza = np.clip(-KP_E[i] * vc, -cfg.forza_max, cfg.forza_max)
        # finestre come VISTA scorrevole: nessuna copia per campione, un solo array finale
        W = sliding_window_view(a, cfg.seq_len)[:-1]            # riga j = finestra che finisce a j+seq-1
        idx = np.arange(cfg.seq_len, N, passo)
        idx = idx[idx - cfg.seq_len < len(W)]
        tieni = idx[VEL[i, idx] >= 2.0]                         # scarta l'auto quasi ferma
        if len(tieni) == 0:
            continue
        F.append(W[tieni - cfg.seq_len]); V.append(VEL[i, tieni])
        XI.append(XI_E[i, tieni]); FZ.append(forza[tieni])
        # anche i due FATTORI, che la rete forza a due uscite usa come target ausiliari
        KP.append(KP_E[i, tieni]); ZS.append(vc[tieni])
    if not F:
        z = np.zeros((0, cfg.seq_len), np.float32); v0 = np.zeros(0, np.float32)
        return z, v0, v0, v0, v0, v0
    return (np.concatenate(F).astype(np.float32), np.concatenate(V).astype(np.float32),
            np.concatenate(XI).astype(np.float32), np.concatenate(FZ).astype(np.float32),
            np.concatenate(KP).astype(np.float32), np.concatenate(ZS).astype(np.float32))



# --- ciclo completo ---
def esegui_dagger(rete_xi, rete_forza, norm, auto, cfg, device, tracce_tr,
                  dati_base, tracce_val, addestra, imposta_validazione=None):
    """Iterazioni di DAgger. 'dati_base' = (az, v, xi, forza) del dataset originale,
    'addestra' = callback addestra(AZ, V, XI, F, epoche) che fa un ciclo di training.

    Il dataset ORIGINALE non si butta: si AGGREGA. Sostituirlo farebbe dimenticare alla
    rete le condizioni gia' imparate (e' il motivo del nome Dataset Aggregation).

    LA VALIDAZIONE VA CAMBIATA, ed e' facile sbagliarlo. Il set di validazione ordinario e'
    costruito con a_z del modello PASSIVO. DAgger aggiunge apposta stati closed-loop, che su
    quel metro sembrano un peggioramento: se il best-checkpoint continua a essere scelto su
    quella validazione, la condizione di salvataggio non si verifica mai e tutte le epoche di
    aggregazione vengono buttate senza che nulla lo segnali. Qui si usa invece un rollout
    dell'esperto sulle strade di VALIDAZIONE (celle classe x velocita' disgiunte dal
    training): la distribuzione bersaglio, mai vista in addestramento."""
    print("\n[2b] DAgger: allineamento fra dati di addestramento e stati visitati")
    strade = prepara_strade(tracce_tr, auto, cfg)

    if tracce_val and imposta_validazione is not None:
        strade_val = prepara_strade(tracce_val, auto, cfg, verbose=False)
        azv, _xv, VELv, XIv, KPv = rollout(strade_val, rete_xi, rete_forza, norm,
                                           auto, cfg, device, beta=1.0)
        imposta_validazione(*finestre_da_rollout(azv, VELv, XIv, KPv, auto, cfg)[:4])
        print(f"    Validazione ricalibrata su rollout closed-loop di {len(strade_val)} strade "
              f"di validazione (la vecchia era sulla distribuzione passiva)")

    az_b, v_b, xi_b, f_b, kp_b, zs_b = dati_base
    acc_az, acc_v, acc_xi, acc_f = [az_b], [v_b], [xi_b], [f_b]
    acc_kp, acc_zs = [kp_b], [zs_b]

    for it in range(1, int(cfg.dagger_iterazioni) + 1):
        # beta: 1 al primo giro (guida l'esperto), poi decade -> guida la rete
        beta = float(cfg.dagger_beta ** (it - 1)) if it > 1 else 1.0
        az_cl, xi_cl, VEL, XI_E, KP_E = rollout(strade, rete_xi, rete_forza, norm,
                                                auto, cfg, device, beta=beta)
        scarto = float(np.mean(np.abs(xi_cl - XI_E)))
        rms_cl = float(np.sqrt(np.mean(az_cl ** 2)))
        rms_tr = float(np.sqrt(np.mean(np.concatenate(acc_az) ** 2)))
        nf = finestre_da_rollout(az_cl, VEL, XI_E, KP_E, auto, cfg)
        for lista, arr in zip((acc_az, acc_v, acc_xi, acc_f, acc_kp, acc_zs), nf):
            lista.append(arr)

        AZ = np.concatenate(acc_az); V = np.concatenate(acc_v)
        XI = np.concatenate(acc_xi); FZ = np.concatenate(acc_f)
        KP = np.concatenate(acc_kp); ZS = np.concatenate(acc_zs)
        print(f"    giro {it}/{cfg.dagger_iterazioni}  beta={beta:.2f}"
              f"  |  RMS a_z visitata {rms_cl:.3f} contro {rms_tr:.3f} nel dataset"
              f"  |  |xi_applicato - xi*| = {scarto:.4f}"
              f"  |  campioni {len(az_b)} -> {len(AZ)}")
        addestra(AZ, V, XI, FZ, int(cfg.dagger_epoche), kp_np=KP, zs_np=ZS)

    return rete_xi, rete_forza, (np.concatenate(acc_az), np.concatenate(acc_v),
                                 np.concatenate(acc_xi), np.concatenate(acc_f),
                                 np.concatenate(acc_kp), np.concatenate(acc_zs))
