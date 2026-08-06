# -*- coding: utf-8 -*-
"""
controllo.py — GENERAZIONE DELLE ETICHETTE: cosa le reti devono imparare a riprodurre.

Per ogni traccia produce, istante per istante, il controllo di riferimento e le grandezze che
servono a valutarlo. La legge di riferimento e' scelta da cfg.metodo_xi:

  "ottimo"  (predefinita)  la coppia (xi*, kp*) che risolve il problema di progetto vincolato
                           — minimo discomfort ISO 2631 senza staccare la ruota, toccare il
                           fine corsa o saturare l'attuatore. Vedi xi_ottimo.py.
  "sqrt2"                  la regola della trasmissibilita': xi e kp schedulati su
                           r = omega/omega_n, con il crossover a r = √2.
                           ATTENZIONE: con metodo_freq="velocita" quella regola e' una
                           funzione della SOLA velocita', quindi la finestra di a_z non porta
                           nessuna informazione sul target. Resta come termine di confronto,
                           non come etichetta da cui la rete possa imparare.

COSA PRODUCE
    xi(t), kp(t)   i due comandi di riferimento
    forza(t)       F* = clip(-kp*(t) * z_s'(t)), con z_s' ricavata dall'a_z MISURATA: e' una
                   grandezza OSSERVABILE dal sensore, quindi la rete puo' davvero impararla
    vel_cassa(t)   z_s' stessa, esportata come target ausiliario della rete forza
    acc_passiva    accelerazione della cassa con damper fisso e nessun attuatore
    acc_ideale     accelerazione con il controllo di riferimento applicato allo stato VERO

IL PROFILO STRADALE
-------------------
I vincoli che decidono il controllo ottimo dipendono da deflessione del pneumatico e corsa
della sospensione, cioe' da contenuto in bassa frequenza. Se la strada e' nota (tracce
sintetiche) si usa QUELLA: ricostruirla dall'accelerazione richiede una doppia integrazione
con passa-alto, che distorce proprio quella banda e sposta il controllo ottimo di quanto vale
l'intero errore della rete. Per le tracce registrate la ricostruzione resta l'unica via, con
l'ampiezza calibrata sui dati perche' il modello passivo riproduca l'RMS di a_z misurato.
"""
import numpy as np
from fisica import (ricostruisci_strada, velocita_cassa_misurata, stima_frequenza,
                    rms_accel_passiva, passo_rk4)

RADQ2 = np.sqrt(2.0)


def xi_da_r(r, auto, cfg):
    """REGOLA √2: xi in funzione di r = omega/omega_n, interpolando fra i limiti FISICI del
    damper — gli estremi non sono scelti, sono c_min/c_crit e c_max/c_crit.

       r < √2  ->  xi verso xi_max : sotto la risonanza piu' smorzamento riduce
                                     l'amplificazione, quindi conviene irrigidire;
       r > √2  ->  xi verso xi_min : sopra la risonanza piu' smorzamento TRASMETTE piu'
                                     accelerazione, quindi conviene isolare.

    Il crossover a √2 non e' una velocita' da scegliere: e' il punto in cui la
    trasmissibilita' di un sistema a 1 GDL vale 1 indipendentemente dallo smorzamento.
    Vale pero' solo se esiste una frequenza di eccitazione dominante — vedi la nota su
    lambda_c in config.py per il motivo per cui su strade a banda larga quella premessa e'
    debole, ed e' la ragione per cui questa regola e' un termine di confronto e non
    l'etichetta predefinita."""
    z = cfg.pendenza_sched * (r / RADQ2 - 1.0)
    return auto.xi_min + (auto.xi_max - auto.xi_min) / (1.0 + np.exp(z))


def kp_da_r(r, cfg):
    """Guadagno skyhook dell'attuatore [Ns/m] secondo la stessa regola √2: basso sotto il
    crossover, alto sopra. La logica e' complementare a xi_da_r — dove il damper viene reso
    morbido per isolare, l'attuatore prende il suo posto per tenere ferma la cassa."""
    z = cfg.pendenza_sched * (r / RADQ2 - 1.0)
    return cfg.kp_basso + (cfg.kp_alto - cfg.kp_basso) / (1.0 + np.exp(-z))


class GeneratoreEtichette:
    """Genera i target di riferimento simulando il modello fisico su una traccia."""

    def __init__(self, auto, cfg):
        self.auto = auto
        self.cfg = cfg

    def genera(self, az, v, zr_dot_vero=None):
        """Ingressi: a_z (accel. cassa misurata), v (velocita' GPS) e, se disponibile, la
        VELOCITA' VERA del profilo strada z_r'.

        Quando zr_dot_vero c'e' va usato quello. La catena alternativa — doppia integrazione
        di a_z con passa-alto a 0.3 Hz e calibrazione su un singolo fattore di scala —
        distorce la forma spettrale nella banda bassa, che e' proprio quella da cui dipendono
        deflessione del pneumatico e corsa, cioe' i due vincoli che decidono xi*. Lo
        scostamento che introduce sul controllo ottimo e' dello stesso ordine dell'errore
        complessivo della rete: sarebbe rumore d'etichetta indistinguibile da imprecisione
        del modello. Per le tracce registrate la ricostruzione resta l'unica via."""
        auto, cfg = self.auto, self.cfg
        N = len(az)
        if zr_dot_vero is not None:
            zr_dot = np.asarray(zr_dot_vero, dtype=float)[:N]
            zr = np.cumsum(zr_dot) * cfg.passo_t          # solo per il grafico
        else:
            zr, zr_dot = ricostruisci_strada(az, cfg)
            # CALIBRAZIONE ampiezza strada DAI DATI (niente numero magico): scalo z_r finche'
            # il modello passivo riproduce l'RMS dell'a_z MISURATO -> forze in Newton reali.
            if cfg.calibra_strada:
                rms_misurato = np.sqrt(np.mean(az ** 2))
                rms_passivo = rms_accel_passiva(zr_dot, auto, cfg)
                fattore = np.clip(rms_misurato / (rms_passivo + 1e-9), 0.2, 20.0)
                zr = zr * fattore
                zr_dot = zr_dot * fattore

        r = stima_frequenza(az, v, cfg, auto)           # r = omega/omega_n (dalla velocita')
        if getattr(cfg, "metodo_xi", "sqrt2") == "ottimo":
            # (xi*, kp*) = la COPPIA che minimizza il comfort ISO restando dentro i limiti
            # di distacco ruota, fine corsa e forza attuatore (xi_ottimo.py). I due canali
            # si ottimizzano INSIEME: agiscono sulla stessa massa, e schedularne uno con
            # una regola e l'altro con un'altra produce un sistema incoerente.
            from xi_ottimo import ottimo
            xi, kp = ottimo(az, v, zr_dot, auto, cfg)
        else:
            # regola √2 su ENTRAMBI i canali (baseline storica, per il confronto).
            # NB: con metodo_freq="velocita" xi e' funzione della sola velocita' ->
            # etichetta oracolo, la rete non puo' imparare nulla da a_z.
            xi = xi_da_r(r, auto, cfg)
            kp = kp_da_r(r, cfg)
        c = auto.c_da_xi(xi)                            # da xi adimensionale a c [Ns/m]
        # ETICHETTA forza: skyhook sulla velocita' cassa MISURATA -> OSSERVABILE da a_z
        # (dipende solo da zs_dot, non da zu_dot: la rete la puo' davvero imparare).
        vel_cassa_mis = velocita_cassa_misurata(az, cfg)
        forza_lab = np.clip(-kp * vel_cassa_mis, -cfg.forza_max, cfg.forza_max)

        acc_passiva = np.zeros(N); acc_ideale = np.zeros(N)
        x = np.zeros(4); xp = np.zeros(4)
        h = cfg.passo_t / cfg.n_sottopassi
        for k in range(N):
            vel_rel = x[1] - x[3]
            forza_id = np.clip(-kp[k] * x[1], -cfg.forza_max, cfg.forza_max)   # riferimento (stato vero)
            acc_ideale[k] = (-auto.rigid_molla * x[0] - c[k] * vel_rel + forza_id) / auto.massa_cassa
            acc_passiva[k] = (-auto.rigid_molla * xp[0] - auto.smorz_nom * (xp[1] - xp[3])) / auto.massa_cassa
            zr0 = zr_dot[k]; zr1 = zr_dot[k + 1] if k + 1 < N else zr_dot[k]
            for _ in range(cfg.n_sottopassi):
                x = passo_rk4(x, c[k], forza_id, zr0, zr1, auto, h)
                xp = passo_rk4(xp, auto.smorz_nom, 0.0, zr0, zr1, auto, h)

        # Si esportano anche i due FATTORI di cui la forza e' il prodotto, F* = clip(-kp*·z_s'):
        # la rete forza li predice separatamente e li ricompone (vedi reti.ReteForza), quindi
        # le servono come target ausiliari. Sono gia' calcolati qui.
        return dict(xi=xi.astype(np.float32), forza=forza_lab.astype(np.float32),
                    kp=np.asarray(kp, np.float32), vel_cassa=vel_cassa_mis.astype(np.float32),
                    r=r.astype(np.float32), acc_passiva=acc_passiva, acc_ideale=acc_ideale,
                    strada=zr, N=N)
