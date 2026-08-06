# -*- coding: utf-8 -*-
"""
simulazione.py — CLOSED-LOOP: il controllore in funzione.

Qui il controllo viene davvero applicato e la dinamica integrata passo per passo. E' la
differenza fra valutare le reti su un dataset e valutarle come CONTROLLORE: nel dataset
l'accelerazione in ingresso e' data, in closed-loop e' PRODOTTA dalle decisioni precedenti
della rete stessa. L'anello e'

    rete -> (xi, F) -> dinamica del quarter-car -> a_z -> rete

e si chiude a ogni campione. E' l'unico posto dove si vede se il controllore e' stabile, e
l'unico dove conta la distribuzione degli stati che la rete visita davvero.

FUNZIONI
  simula_closed_loop        una strada per volta, con tutte le grandezze per l'animazione e
                            il bilancio energetico. Puo' applicare anche un controllore
                            ANALITICO (xi e kp dati) invece delle reti, cosi' l'ottimo e la
                            regola √2 si valutano sulla stessa identica strada.
  simula_closed_loop_batch  molte strade in un solo batch: una forward per passo temporale
                            invece di una per strada. Usata dalla diagnostica e, tramite
                            aggregazione.rollout, dai giri di DAgger.
  strada_demo (in fisica)   profilo dimostrativo per l'animazione. Severo: equivale a una
                            classe E ISO 8608.
  traiettorie_demo_*        preparano i dati per l'animazione, scartando il warm-up.

DUE DETTAGLI CHE CONTANO
  WARM-UP. Per i primi seq_len campioni la finestra della rete e' riempita di zeri: la rete
  decide su un ingresso che non esiste. Nell'animazione e nelle metriche quel tratto va
  scartato, altrimenti il controllore sembra sbagliato per il primo secondo.
  CPU. La versione a strada singola gira su CPU anche quando c'e' la GPU: e' sequenziale con
  batch 1, e li' l'overhead di lancio e sincronizzazione supera il calcolo.

CHIAVI RESTITUITE: strada = z_r; quota_cassa_ml / quota_ruota_ml (ramo controllato),
quota_cassa_pas / quota_ruota_pas (ramo passivo, integrato in parallelo sulla stessa strada);
forza; xi; acc_cassa_ml; acc_cassa_pas; defl_gomma_* (tenuta di strada); le energie;
x_pos (avanzamento longitudinale); N.
"""
import numpy as np
import torch

from fisica import passo_rk4, ricostruisci_strada, rms_accel_passiva, strada_demo
from reti import norm_a_xi, norm_a_forza


def bilancio_energetico(traj, cfg):
    """Bilancio energetico dell'anello, in JOULE integrati sul tratto simulato.

    LE TRE VOCI MISURATE (dalla simulazione, non da ipotesi)
        E_damp   = integrale di c*v_rel^2       lavoro dell'ammortizzatore -> CALORE nell'olio
        E_inj    = integrale di max(0, F*v_rel) l'attuatore SPINGE nel verso del moto: prende
                                                 energia dalla batteria. E' il COSTO.
        E_abs    = integrale di max(0,-F*v_rel) l'attuatore FRENA il moto: l'energia arriva a
                                                 lui, quindi e' recuperabile come in una frenata
                                                 rigenerativa.

    COSA E' DAVVERO RECUPERABILE — qui stava l'errore
        L'ammortizzatore di questo progetto e' idraulico: il suo lavoro finisce in calore
        nell'olio e non torna piu'. Metterlo fra i recuperi (recuperabile = eta*(E_damp+E_abs))
        gonfia il bilancio e fa concludere che la sospensione si auto-alimenta, che e'
        falso per questo hardware. Solo l'attuatore, essendo elettrico, puo' funzionare da
        generatore quando frena.
        Il caso col damper elettromagnetico esiste e resta calcolabile, ma va DICHIARATO con
        cfg.damper_rigenerativo = True: e' un'ipotesi sull'hardware, non un dettaglio di conto.

    CONVENZIONE DEI SEGNI — dichiarata, perche' e' facile leggerla al rovescio
        Le tre voci misurate sono MODULI, sempre >= 0: il verso e' nel nome, non nel segno.
        Il bilancio invece e' ELETTRICO e ha segno, dal punto di vista della BATTERIA:
            addebito (< 0)  l'energia che esce dalla batteria per alimentare l'attuatore
            accredito (> 0) l'energia che rientra in batteria dall'attuatore che frena
            netto = accredito - addebito;  netto > 0 = il sistema si auto-alimenta.
        E_damp NON entra nel bilancio elettrico (con damper idraulico): e' energia MECCANICA
        che lascia il sistema come calore, un conto diverso. Sta nel grafico come contesto,
        non come voce da sommare.

    DUE RENDIMENTI, NON UNO — la seconda cosa che era sbagliata
        Il conto applicava eta solo al RECUPERO, lasciando l'iniezione a rendimento unitario:
        cioe' assumeva un attuatore perfetto quando spinge e imperfetto quando frena. Non e'
        coerente. L'energia che la batteria deve fornire per consegnare E_inj meccanici e'
        E_inj / eta_attuatore, ed e' MAGGIORE di E_inj.
        Con cfg.eta_attuatore = 1.0 (predefinito, ipotesi ottimistica dichiarata) il costo
        stampato e' un LIMITE INFERIORE. Il campo 'netto_con_perdite_attuatore' riporta lo
        stesso bilancio col rendimento del generatore applicato anche in trazione, ed e' la
        cifra da guardare prima di dire che il sistema si auto-alimenta: il SEGNO del netto
        dipende da quell'ipotesi, non solo il suo valore.

    ATTENDIBILITA' NUMERICA
        Le tre voci vengono da una quadratura col trapezio dentro i sottopassi RK4 (vedi
        _closed_loop). Accumularle una volta per campione di controllo, come si faceva, dava
        un errore del 54% sul netto meccanico, e l'errore NON scendeva affittando i
        sottopassi perche' nasceva dal campionare la potenza a 100 Hz su una dinamica di
        ruota a 11.9 Hz. Il netto e' una piccola differenza fra due termini grandi e opposti,
        quindi eredita l'errore relativo amplificato: con la quadratura vecchia era piu'
        grande del netto stesso, cioe' il verso del bilancio non era determinato.

    Restituisce un dict con le voci misurate, il bilancio con segno e la spiegazione testuale
    delle ipotesi, cosi' stampa e figura raccontano la stessa cosa."""
    eta = cfg.eta_generatore
    eta_att = float(getattr(cfg, "eta_attuatore", 1.0))
    E_damp = float(traj["E_damp"]); E_inj = float(traj["E_att_inj"])
    E_abs = float(traj["E_att_rec"])
    damper_recuperabile = bool(getattr(cfg, "damper_rigenerativo", False))

    meccanica = E_abs + (E_damp if damper_recuperabile else 0.0)
    accredito = eta * meccanica                 # entra in batteria  (>= 0)
    addebito = E_inj / max(eta_att, 1e-9)       # esce dalla batteria (>= 0)
    netto = accredito - addebito
    # sensibilita' all'ipotesi sul rendimento in trazione: stesso conto con eta anche in spinta
    netto_perdite = eta * meccanica - E_inj / max(eta, 1e-9)
    return dict(
        E_damp=E_damp, E_inj=E_inj, E_abs=E_abs, eta=eta, eta_attuatore=eta_att,
        damper_recuperabile=damper_recuperabile,
        meccanica_disponibile=meccanica,
        addebito=addebito, accredito=accredito,
        recuperabile=accredito,                 # nome storico, tenuto per compatibilita'
        netto=netto, netto_con_perdite_attuatore=netto_perdite,
        formula=("eta*(ATTUATORE assorbita + DAMPER dissipata)" if damper_recuperabile
                 else "eta*(ATTUATORE assorbita)"),
        ipotesi=("damper ELETTROMAGNETICO dichiarato (cfg.damper_rigenerativo=True): il calore "
                 "del damper e' contato come recuperabile" if damper_recuperabile else
                 "damper IDRAULICO: il suo calore NON e' recuperabile ed e' escluso dal recupero"),
        ipotesi_attuatore=(f"attuatore ideale in trazione (eta_attuatore = {eta_att:.0%}): il costo"
                           f" e' un LIMITE INFERIORE" if eta_att >= 0.999 else
                           f"perdite in trazione incluse (eta_attuatore = {eta_att:.0%})"),
        verdetto=("ATTIVO (rientra in batteria piu' di quanto esce)" if netto > 0
                  else "COSTO NETTO (esce dalla batteria piu' di quanto rientra)"),
        verdetto_breve=("ATTIVO: si auto-alimenta" if netto > 0 else "COSTO NETTO per la batteria"),
    )


def _device_di(rete):
    """Device su cui vivono i pesi di una rete. Si legge dal primo parametro invece di
    fidarsi dell'argomento 'device' passato in giro: e' l'unica fonte non falsificabile."""
    try:
        return next(rete.parameters()).device
    except StopIteration:                      # rete senza parametri: nulla da spostare
        return None


def simula_closed_loop(strada, strada_dot, vel, rete_xi, rete_forza, norm, auto, cfg, device,
                       xi_rif=None, kp_rif=None, usa_riferimento=False):
    """Se xi_rif/kp_rif sono dati vengono riportati nel risultato per il confronto; con
    usa_riferimento=True il controllo viene preso DA LORO invece che dalle reti (cosi'
    l'animazione puo' mostrare l'esperto e il ML sulla stessa identica strada).

    ESECUZIONE SU CPU (cfg.closed_loop_su_cpu). Questa simulazione e' sequenziale: a ogni
    passo fa UNA forward con batch 1 e ne legge subito il risultato. Su GPU ogni forward
    paga il lancio del kernel e ogni lettura forza una sincronizzazione device->host: con
    reti piccole e batch 1 quell'overhead e' molte volte il calcolo utile, e la CPU risulta
    piu' veloce. Le reti si spostano una volta sola, non a ogni passo.

    IL DEVIATORE VA RIMESSO A POSTO. nn.Module.to() sposta i pesi IN PLACE e restituisce lo
    stesso oggetto: riassegnarlo a una variabile locale non protegge il chiamante, le reti
    restano dove le si e' messe anche dopo il return. Senza il ripristino, chiunque chiami
    dopo (la diagnostica) manda tensori sul device originale a pesi rimasti su CPU e ottiene
    'Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the same'.
    Il finally garantisce il ripristino anche se la simulazione solleva."""
    su_cpu = getattr(cfg, "closed_loop_su_cpu", True) and str(device) != "cpu"
    if not su_cpu:
        return _closed_loop(strada, strada_dot, vel, rete_xi, rete_forza, norm, auto, cfg,
                            device, xi_rif, kp_rif, usa_riferimento)
    # con usa_riferimento=True le reti non vengono nemmeno interrogate: non si toccano.
    sposta = not usa_riferimento
    origine = _device_di(rete_xi) if sposta else None
    try:
        if sposta:
            rete_xi.to("cpu"); rete_forza.to("cpu")
        return _closed_loop(strada, strada_dot, vel, rete_xi, rete_forza, norm, auto, cfg,
                            torch.device("cpu"), xi_rif, kp_rif, usa_riferimento)
    finally:
        if sposta and origine is not None and str(origine) != "cpu":
            rete_xi.to(origine); rete_forza.to(origine)


def _closed_loop(strada, strada_dot, vel, rete_xi, rete_forza, norm, auto, cfg, device,
                 xi_rif=None, kp_rif=None, usa_riferimento=False):
    """Integra in CLOSED-LOOP il ramo controllato (ML) e in parallelo il ramo passivo (c_nom)
    sullo STESSO profilo strada. La rete decide xi e forza dalla finestra passata di a_z
    della cassa controllata. All'avvio la finestra e' riempita con zero (warm-up)."""
    N = len(strada); seq = cfg.seq_len; h = cfg.passo_t / cfg.n_sottopassi
    quota_cassa_ml = np.zeros(N); quota_ruota_ml = np.zeros(N)
    quota_cassa_pas = np.zeros(N); quota_ruota_pas = np.zeros(N)
    forza = np.zeros(N); xi = np.zeros(N, np.float32)
    acc_cassa_ml = np.zeros(N, np.float32); acc_cassa_pas = np.zeros(N)
    defl_gomma_ml = np.zeros(N); defl_gomma_pas = np.zeros(N)   # z_u - z_r (tenuta di strada)
    E_att_inj = 0.0; E_att_rec = 0.0; E_damp = 0.0             # energie [J]
    x = np.zeros(4); xp = np.zeros(4)
    rete_xi.eval(); rete_forza.eval()
    finestra = np.zeros(seq, np.float32)
    with torch.no_grad():
        for k in range(N):
            # finestra causale = ultimi seq campioni di accelerazione cassa CONTROLLATA
            if k >= seq:
                w = acc_cassa_ml[k - seq:k]
            else:
                finestra[:] = 0.0
                if k > 0:
                    finestra[seq - k:] = acc_cassa_ml[:k]
                w = finestra
            # NORMALIZZA con le statistiche del training, poi predice
            if usa_riferimento and xi_rif is not None:
                # ramo ANALITICO: xi e kp dati (ottimo o regola √2); la forza skyhook usa
                # lo STATO VERO x[1], perche' questi controllori non stimano nulla dal sensore
                xi_k = float(xi_rif[k])
                forza_k = float(np.clip(-float(kp_rif[k]) * x[1], -cfg.forza_max, cfg.forza_max))
            else:
                az_t = torch.as_tensor(norm.na(w), dtype=torch.float32, device=device).view(1, 1, seq)
                v_t = torch.tensor([[norm.nv(float(vel[k]))]], dtype=torch.float32, device=device)
                xi_k = norm_a_xi(float(rete_xi(az_t, v_t)), auto)
                forza_k = float(np.clip(norm_a_forza(float(rete_forza(az_t, v_t)), cfg),
                                        -cfg.forza_max, cfg.forza_max))
            c = auto.c_da_xi(xi_k)
            xi[k] = xi_k; forza[k] = forza_k

            quota_ruota_ml[k] = x[2] + strada[k]; quota_cassa_ml[k] = x[0] + quota_ruota_ml[k]
            quota_ruota_pas[k] = xp[2] + strada[k]; quota_cassa_pas[k] = xp[0] + quota_ruota_pas[k]
            acc_cassa_ml[k] = (-auto.rigid_molla * x[0] - c * (x[1] - x[3]) + forza_k) / auto.massa_cassa
            acc_cassa_pas[k] = (-auto.rigid_molla * xp[0] - auto.smorz_nom * (xp[1] - xp[3])) / auto.massa_cassa
            vrel = x[1] - x[3]                                  # velocita' relativa sospensione
            defl_gomma_ml[k] = x[2]; defl_gomma_pas[k] = xp[2]  # deflessione pneumatico

            zr0 = strada_dot[k]; zr1 = strada_dot[k + 1] if k + 1 < N else strada_dot[k]
            for _ in range(cfg.n_sottopassi):
                x_pre = x
                x = passo_rk4(x, c, forza_k, zr0, zr1, auto, h)
                xp = passo_rk4(xp, auto.smorz_nom, 0.0, zr0, zr1, auto, h)
                # ENERGIE: quadratura col TRAPEZIO, DENTRO il sottopasso.
                # Prima erano accumulate una volta per campione (Euler a dt=0.01) usando la
                # v_rel di inizio intervallo. La dinamica della ruota sta a 11.9 Hz, cioe' 8
                # campioni per periodo: il bilancio di potenza dV/dt = -c*v_rel^2 + F*v_rel
                # - k_t*x3*z_r' non chiudeva per il 10%, e l'energia dell'attuatore — che e'
                # una piccola differenza fra due termini grandi e opposti — sbagliava del 60%.
                # c e F sono costanti sul campione (sono i comandi tenuti, ZOH): varia solo v_rel.
                vr0 = x_pre[1] - x_pre[3]; vr1 = x[1] - x[3]
                pot_att = 0.5 * forza_k * (vr0 + vr1)            # potenza attuatore = F * v_rel
                E_att_inj += max(0.0, pot_att) * h               # INIETTATA (esce dalla batteria)
                E_att_rec += max(0.0, -pot_att) * h              # ASSORBITA (recuperabile)
                E_damp += 0.5 * c * (vr0 * vr0 + vr1 * vr1) * h  # dissipata dal DAMPER (calore)

    x_pos = np.cumsum(np.maximum(vel, 0.0) * cfg.passo_t)
    return dict(strada=strada, quota_cassa_ml=quota_cassa_ml, quota_ruota_ml=quota_ruota_ml,
                quota_cassa_pas=quota_cassa_pas, quota_ruota_pas=quota_ruota_pas,
                forza=forza, xi=xi, acc_cassa_ml=acc_cassa_ml, acc_cassa_pas=acc_cassa_pas,
                defl_gomma_ml=defl_gomma_ml, defl_gomma_pas=defl_gomma_pas,
                xi_rif=(np.asarray(xi_rif, np.float32) if xi_rif is not None else None),
                E_att_inj=E_att_inj, E_att_rec=E_att_rec, E_damp=E_damp,
                vel=np.asarray(vel, float), x_pos=x_pos, N=N)


def _taglia(traj, n0):
    """Scarta i primi n0 campioni da tutte le serie temporali del risultato.

    Serve per il WARM-UP: nei primi seq_len passi la finestra di a_z e' riempita di zeri,
    quindi la rete decide su un ingresso che non esiste. Mostrarlo nell'animazione faceva
    sembrare il controllore sbagliato per il primo secondo su otto."""
    N = traj["N"]
    fuori = {"N", "E_att_inj", "E_att_rec", "E_damp"}
    out = {}
    for k, v in traj.items():
        if k in fuori or v is None:
            out[k] = v
        elif hasattr(v, "__len__") and len(v) == N:
            out[k] = np.asarray(v)[n0:]
        else:
            out[k] = v
    out["N"] = N - n0
    out["x_pos"] = out["x_pos"] - out["x_pos"][0]
    return out


def traiettorie_demo_ml(rete_xi, rete_forza, norm, auto, cfg, device, usa_esperto=False):
    """Animazione su strada demo. Il controllo viene dal ML, oppure — con usa_esperto=True —
    dall'ottimo (xi*, kp*): stessa identica strada, cosi' il confronto e' pulito.

    In entrambi i casi xi* viene calcolato e allegato al risultato, cosi' l'animazione puo'
    mostrare accanto al valore comandato quello che sarebbe stato ottimo."""
    scarto = cfg.seq_len                                   # warm-up: finestra ancora a zeri
    secondi = cfg.anim_secondi + scarto / cfg.freq_campion
    zr, zr_dot, vel = strada_demo(auto, cfg, secondi=secondi)

    xi_rif = kp_rif = None
    if getattr(cfg, "metodo_xi", "sqrt2") == "ottimo":
        from xi_ottimo import ottimo
        from fisica import accel_passiva_serie
        xi_rif, kp_rif = ottimo(accel_passiva_serie(zr_dot, auto, cfg), vel, zr_dot, auto, cfg)

    traj = simula_closed_loop(zr, zr_dot, vel, rete_xi, rete_forza, norm, auto, cfg, device,
                              xi_rif=xi_rif, kp_rif=kp_rif, usa_riferimento=usa_esperto)
    return _taglia(traj, scarto)


def simula_closed_loop_batch(strade, rete_xi, rete_forza, norm, auto, cfg, device,
                             beta=0.0, xi_esperto=None, kp_esperto=None):
    """Closed-loop su PIU' STRADE contemporaneamente: una sola forward per passo temporale.

    PERCHE'. simula_closed_loop fa una forward con batch 1 a ogni passo. La diagnostica ne
    lanciava 25 simulazioni in sequenza (5 classi ISO x 5 velocita'): 25 x 1200 = 30.000
    forward da batch 1, ognuna con il proprio lancio di kernel e la propria sincronizzazione
    GPU->CPU. Il calcolo vero e' trascurabile, paga tutto l'overhead. Le 25 simulazioni sono
    INDIPENDENTI, quindi si possono impilare in un batch: 1200 forward da batch 25, cioe' 25
    volte meno lanci a parita' di risultato.

    'strade' = lista di dict con zr_dot e vel (stessa lunghezza). Restituisce dict di array
    (n_strade, N). Il ramo passivo si integra in parallelo, come nella versione a strada
    singola.

    MISCELAZIONE CON L'ESPERTO (beta, xi_esperto, kp_esperto). Con beta > 0 una quota del
    controllo viene dall'esperto analitico invece che dalle reti: e' cio' che serve a DAgger
    per raccogliere stati sulla distribuzione bersaglio. beta = 1 significa "guida solo
    l'esperto", beta = 0 "guida solo la rete".
    aggregazione.rollout e' un involucro su questa funzione, non una seconda copia: due
    implementazioni dello stesso closed-loop batchato sarebbero due punti dove correggere
    ogni bug, e il secondo si dimentica."""
    # Il device si legge DAI PESI, non dall'argomento: se qualcuno ha spostato le reti (la
    # simulazione a strada singola lo fa) l'argomento e' obsoleto e i tensori finirebbero su
    # un device diverso da quello dei pesi.
    device = _device_di(rete_xi) or device
    ns = len(strade)
    N = min(len(s["zr_dot"]) for s in strade)
    seq = cfg.seq_len
    ZR = np.stack([np.asarray(s["zr_dot"], float)[:N] for s in strade])
    VEL = np.stack([np.asarray(s["vel"], float)[:N] for s in strade])
    usa_esperto = beta > 0.0 and xi_esperto is not None and kp_esperto is not None
    if usa_esperto:
        XI_E = np.stack([np.asarray(a, float)[:N] for a in xi_esperto])
        KP_E = np.stack([np.asarray(a, float)[:N] for a in kp_esperto])

    h = cfg.passo_t / cfg.n_sottopassi
    x = np.zeros((ns, 4)); xp = np.zeros((ns, 4))
    az = np.zeros((ns, N), np.float32); az_p = np.zeros((ns, N), np.float32)
    xi = np.zeros((ns, N), np.float32); frz = np.zeros((ns, N), np.float32)
    gom = np.zeros((ns, N), np.float32); gom_p = np.zeros((ns, N), np.float32)

    rete_xi.eval(); rete_forza.eval()
    fin = np.zeros((ns, seq), np.float32)
    # tensori PREALLOCATI: ricrearli a ogni passo significa 1200 allocazioni e altrettanti
    # trasferimenti verso il device, che su MPS costano piu' del calcolo
    t_az = torch.zeros((ns, 1, seq), dtype=torch.float32, device=device)
    t_v = torch.zeros((ns, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        for k in range(N):
            if k >= seq:
                fin[:] = az[:, k - seq:k]
            else:
                fin[:] = 0.0
                if k > 0:
                    fin[:, seq - k:] = az[:, :k]
            t_az.copy_(torch.from_numpy(norm.na(fin)).unsqueeze(1))
            t_v.copy_(torch.from_numpy(norm.nv(VEL[:, k]).astype(np.float32)).view(-1, 1))
            xi_k = norm_a_xi(rete_xi(t_az, t_v).cpu().numpy().ravel(), auto)
            f_k = np.clip(norm_a_forza(rete_forza(t_az, t_v).cpu().numpy().ravel(), cfg),
                          -cfg.forza_max, cfg.forza_max)
            if usa_esperto:
                # l'esperto conosce lo STATO VERO, quindi la sua forza skyhook usa x[:,1]
                f_e = np.clip(-KP_E[:, k] * x[:, 1], -cfg.forza_max, cfg.forza_max)
                xi_k = beta * XI_E[:, k] + (1.0 - beta) * xi_k
                f_k = np.clip(beta * f_e + (1.0 - beta) * f_k, -cfg.forza_max, cfg.forza_max)
            c = auto.c_da_xi(xi_k)
            xi[:, k] = xi_k; frz[:, k] = f_k
            az[:, k] = (-auto.rigid_molla * x[:, 0] - c * (x[:, 1] - x[:, 3]) + f_k) / auto.massa_cassa
            az_p[:, k] = (-auto.rigid_molla * xp[:, 0]
                          - auto.smorz_nom * (xp[:, 1] - xp[:, 3])) / auto.massa_cassa
            gom[:, k] = x[:, 2]; gom_p[:, k] = xp[:, 2]
            zr0 = ZR[:, k]; zr1 = ZR[:, k + 1] if k + 1 < N else ZR[:, k]
            for _ in range(cfg.n_sottopassi):
                x = passo_rk4(x, c, f_k, zr0, zr1, auto, h)
                xp = passo_rk4(xp, np.full(ns, auto.smorz_nom), np.zeros(ns), zr0, zr1, auto, h)
    return dict(xi=xi, forza=frz, acc_cassa_ml=az, acc_cassa_pas=az_p,
                defl_gomma_ml=gom, defl_gomma_pas=gom_p, vel=VEL, N=N)


def _rk4_vett(*a, **k):        # alias storico
    return passo_rk4(*a, **k)


def traiettorie_demo_confronto(rete_xi, rete_forza, norm, auto, cfg, device, verbose=True):
    """Simula TUTTI i controllori sulla STESSA strada demo e restituisce {nome: traiettoria}.

    Sono quattro leggi diverse che passano per lo stesso simulatore, quindi il confronto e'
    pulito — cambia solo chi decide (xi, F) a ogni passo:

      "ML"        le due reti, dalla sola finestra di a_z e dalla velocita' (l'unico che
                  non conosce lo stato vero: e' un controllore realizzabile)
      "ottimo"    xi* e kp* dall'ottimizzazione vincolata; usa lo stato vero per lo
                  skyhook, quindi e' un LIMITE SUPERIORE, non un controllore implementabile
      "sqrt2"     la regola della trasmissibilita': xi = xi_da_r(r(v)), kp = kp_da_r(r(v)),
                  entrambe funzioni della sola velocita'
      (il ramo passivo con c_nom e' gia' dentro ogni traiettoria, ed e' identico per tutte)
    """
    from controllo import xi_da_r, kp_da_r
    from fisica import accel_passiva_serie, stima_frequenza
    from xi_ottimo import ottimo

    scarto = cfg.seq_len
    secondi = cfg.anim_secondi + scarto / cfg.freq_campion
    zr, zr_dot, vel = strada_demo(auto, cfg, secondi=secondi)
    az_pas = accel_passiva_serie(zr_dot, auto, cfg)

    xi_o, kp_o = ottimo(az_pas, vel, zr_dot, auto, cfg)
    r = stima_frequenza(az_pas, vel, cfg, auto)
    xi_2 = np.asarray(xi_da_r(r, auto, cfg), float)
    kp_2 = np.asarray(kp_da_r(r, cfg), float)

    def _sim(xr, kr, usa):
        t = simula_closed_loop(zr, zr_dot, vel, rete_xi, rete_forza, norm, auto, cfg,
                               device, xi_rif=xr, kp_rif=kr, usa_riferimento=usa)
        return _taglia(t, scarto)

    nomi_ordine = ["ML", "ottimo", "sqrt2"]
    trajs = {"ML": _sim(None, None, False),
             "ottimo": _sim(xi_o, kp_o, True),
             "sqrt2": _sim(xi_2, kp_2, True)}
    # il riferimento xi* serve a tutte per il confronto a schermo
    for t in trajs.values():
        t["xi_rif"] = np.asarray(xi_o[scarto:], np.float32)

    if verbose:
        # si usa xi_ottimo.metriche invece di ricalcolare RMS a mano: e' la stessa funzione
        # che valuta i controllori analitici, quindi i numeri sono comparabili per costruzione
        from xi_ottimo import metriche
        print(f"    Controllori simulati sulla stessa strada demo"
              f" ({trajs[nomi_ordine[0]]['N']*cfg.passo_t:.0f} s):")
        print(f"      {'':<9}{'xi':>13}{'comfort Wk':>12}{'gomma[mm]':>11}{'margine':>9}")
        for nome, t in trajs.items():
            m = metriche(t["acc_cassa_ml"], t["defl_gomma_ml"],
                         t["quota_cassa_ml"] - t["quota_ruota_ml"], auto, cfg)
            print(f"      {nome:<9}{t['xi'].min():.3f}-{t['xi'].max():.3f}"
                  f"{m['comfort_wk']:12.2f}{m['gomma_rms']*1000:11.2f}{m['margine_gomma']:9.2f}")
        t0 = next(iter(trajs.values()))
        m = metriche(t0["acc_cassa_pas"], t0["defl_gomma_pas"],
                     t0["quota_cassa_pas"] - t0["quota_ruota_pas"], auto, cfg)
        print(f"      {'passivo':<9}{auto.xi_da_c(auto.smorz_nom):.3f} (fisso)"
              f"{m['comfort_wk']:12.2f}{m['gomma_rms']*1000:11.2f}{m['margine_gomma']:9.2f}")
        print("      (margine > 1 = il vincolo di distacco ruota e' violato)")
    return trajs


def traiettorie_reali_ml(az, vel, rete_xi, rete_forza, norm, auto, cfg, device):
    """Animazione su traccia reale: strada ricostruita+calibrata, controllo closed-loop dal ML."""
    zr, zr_dot = ricostruisci_strada(az, cfg)
    if cfg.calibra_strada:
        f = np.clip(np.sqrt(np.mean(az ** 2)) / (rms_accel_passiva(zr_dot, auto, cfg) + 1e-9), 0.2, 20.0)
        zr *= f; zr_dot *= f
    return simula_closed_loop(zr, zr_dot, vel, rete_xi, rete_forza, norm, auto, cfg, device)
