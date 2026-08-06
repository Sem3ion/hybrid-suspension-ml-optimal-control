# -*- coding: utf-8 -*-
"""
main.py — Orchestratore: mette in fila i moduli e mostra il flusso a colpo d'occhio.

L'IDEA DEL PROGETTO IN QUATTRO RIGHE
------------------------------------
Un'auto ha due comandi di sospensione (smorzamento del damper e forza dell'attuatore) e un
solo sensore utile (l'accelerometro sulla cassa). Il controllo migliore si sa calcolare, ma
solo conoscendo il profilo stradale e il futuro. Queste reti imparano a riprodurlo
conoscendo soltanto un paio di secondi di accelerazione passata e la velocita'.

PIPELINE
--------
  1. CARICAMENTO. Tre tracce registrate (2 training, 1 validazione mai vista), riportate su
     griglia temporale uniforme, piu' le strade sintetiche ISO 8608 generate come disegno
     fattoriale classe x velocita'. Le sintetiche portano con se' il profilo stradale ESATTO.
  2. ETICHETTE. Per ogni istante di ogni traccia si risolve il problema di progetto
     (xi_ottimo.py): minimo discomfort ISO 2631 sotto i vincoli di distacco ruota, fine corsa
     e saturazione attuatore. Da qui escono xi*, kp* e la forza di riferimento.
  3. FINESTRE E NORMALIZZAZIONE. Le finestre scorrevoli diventano gli ingressi; media e
     deviazione standard vengono dai soli dati di training e si salvano su file insieme alle
     reti, cosi' un modello non puo' essere ricaricato con statistiche diverse.
  4. ADDESTRAMENTO. Due reti separate con loss e optimizer indipendenti. I parametri delle
     loss si ricavano dai dati (perdite.py), non si scelgono.
  5. AGGREGAZIONE (DAgger). Le reti guidano, si registrano gli stati che visitano davvero, si
     rietichettano con l'esperto e si riaddestra. Serve perche' i dati di addestramento
     vengono dal modello passivo mentre in closed-loop l'accelerazione e' quella che il
     controllo stesso ha prodotto — circa la meta'.
  6. VALIDAZIONE E DIAGNOSTICA. Metriche su dati mai visti, comfort e tenuta in closed-loop,
     e una batteria di verifiche FISICHE: l'R^2 non dice se il controllore si comporta bene.
  7. FIGURE E ANIMAZIONE, con il confronto fra tutti i controllori sulla stessa strada.

COSA SI STA CONFRONTANDO, SEMPRE
--------------------------------
  ML       le due reti. Vedono solo a_z passata e v: l'unico controllore realizzabile.
  ottimo   xi* e kp* dall'ottimizzazione. Conosce strada e stato vero: e' un LIMITE
           SUPERIORE, non un candidato.
  sqrt2    la regola della trasmissibilita', funzione della sola velocita'. E' il punto di
           partenza storico del progetto, tenuto come termine di paragone.
  passivo  damper fisso, nessun attuatore. Il riferimento da battere.

Tutti passano per lo stesso simulatore, sulla stessa strada: cambia solo chi decide.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error

from config import Auto, Config
from portabilita import descrivi_sistema, forza_utf8
from hardware import rileva_hardware
from dati import carica_tracce, costruisci_finestre, Normalizzatore, nome_traccia
from controllo import GeneratoreEtichette, kp_da_r, RADQ2
from reti import crea_reti, xi_a_norm, norm_a_xi, forza_a_norm, norm_a_forza
from simulazione import (traiettorie_demo_ml, traiettorie_demo_confronto,
                         traiettorie_reali_ml, simula_closed_loop, bilancio_energetico)
from strade_sintetiche import (genera_tracce_sintetiche, genera_tracce_validazione,
                              genera_tracce_aggregazione,
                               riepilogo_copertura)
from fisica import ricostruisci_strada, rms_accel_passiva
from grafica import salva_figura, salva_figura_efficienza, anima_confronto
from diagnostica import esegui_diagnostica


def main():
    t0 = time.time()
    auto, cfg = Auto(), Config()
    base = os.path.dirname(os.path.abspath(__file__))
    path_xi = os.path.join(base, "rete_xi.pt")
    path_forza = os.path.join(base, "rete_forza.pt")
    path_norm = os.path.join(base, "normalizzatore.json")

    print("=" * 90)
    legge = ("OTTIMO VINCOLATO (comfort ISO 2631 sotto i limiti hardware)"
             if cfg.metodo_xi == "ottimo" else "REGOLA DELLA TRASMISSIBILITA' √2")
    print(" SOSPENSIONE 2 GDL (PEUGEOT 207) — CONTROLLO APPRESO (ML)")
    print(f" Legge di riferimento (etichette): {legge}")
    print(f" omega_n = {auto.puls_nat_cassa:.2f} rad/s ({auto.puls_nat_cassa/(2*np.pi):.2f} Hz)"
          f"   |   crossover a r = √2 ≈ {RADQ2:.3f}")
    print(f" Sistema: {descrivi_sistema()}")   # nei log condivisi serve sapere su che macchina
    print("=" * 90)

    device, n_core = rileva_hardware(cfg)

    # 1-2) dati + etichette + finestre
    print("\n[1] Caricamento tracce e generazione etichette (parallela sui core)...")
    tracce_tr, tracce_va = carica_tracce(cfg)
    if cfg.metodo_xi == "ottimo":
        from xi_ottimo import riferimenti, verifica_wk, descrivi_riferimenti
        ar, dr, sr, fr = riferimenti(auto, cfg)
        # AUTOTEST del filtro ISO 2631 Wk contro la Tabella 3 della norma. Era una funzione
        # scritta e mai chiamata: un filtro di ponderazione sbagliato non da' errori, rende
        # solo ottimo il problema sbagliato — esattamente il genere di guasto silenzioso che
        # conviene far urlare all'avvio.
        _f, _g, _t = verifica_wk()
        _err = float(np.max(np.abs(_g - _t) / _t))
        print(f"    Filtro ISO 2631-1 Wk verificato sulla Tabella 3 della norma: "
              f"errore massimo {_err:.2%}" + ("" if _err < 0.02 else "   [!] CONTROLLARE"))
        print("    Etichette da OTTIMIZZAZIONE VINCOLATA: si cerca la coppia (xi, kp) che")
        print("    minimizza il comfort ISO 2631 Wk restando entro i limiti hardware.")
        # ogni limite stampato con la sua FORMULA e la sua provenienza: i valori qui sotto
        # usano il fattore di picco di ripiego, quello vero viene misurato per traccia
        for _r in descrivi_riferimenti(auto, cfg):
            print(_r)
        print(f"    Griglia {cfg.xi_ott_n_candidati}x{cfg.kp_ott_n_candidati}"
              f" = {cfg.xi_ott_n_candidati*cfg.kp_ott_n_candidati} candidati,"
              f" finestra {cfg.xi_ott_finestra_s:.0f} s, penalita' {cfg.xi_ott_penalita:.0f}")
    else:
        v_ind = np.sqrt(2.0) * auto.puls_nat_cassa * cfg.lambda_c_design / (2 * np.pi)
        print(f"    Regola √2: r = omega/omega_n (crossover FISSO a √2); omega = 2π·v/lambda_c, "
              f"lambda_c = {cfg.lambda_c_design:.0f} m -> r=√2 a ~{v_ind * 3.6:.0f} km/h")
    tracce_va_reali = tracce_va
    nome_val = nome_traccia(tracce_va_reali[0], "traccia reale di validazione")
    if cfg.usa_augmentation:                         # aggiunge strade sintetiche varie al TRAINING
        print(f"    Data augmentation: {cfg.n_tracce_sintetiche} strade sintetiche ISO 8608"
              f" (buche/dossi bruschi, gradini, salite/discese)")
        riepilogo_copertura(auto, cfg)
        tracce_tr = tracce_tr + genera_tracce_sintetiche(auto, cfg)
        # VALIDAZIONE anche su strade sintetiche, su celle (classe x velocita') MAI viste:
        # e' il test che smaschera la scorciatoia "a_z -> xi", perche' presenta combinazioni
        # rugosita' x velocita' assenti dal training (es. strada sconnessa a 90 km/h).
        tracce_va_sint = genera_tracce_validazione(auto, cfg)
    else:
        tracce_va_sint = []

    az_tr, v_tr, y_xi_tr, y_f_tr, r_tr, comfort_tr, kp_tr, zs_tr = costruisci_finestre(
        tracce_tr, auto, cfg, n_core)
    az_va, v_va, y_xi_va, y_f_va, r_va, comfort_va, kp_va, zs_va = costruisci_finestre(
        tracce_va_reali, auto, cfg, n_core)
    if tracce_va_sint:
        (az_vs, v_vs, y_xi_vs, y_f_vs, r_vs, comfort_vs,
         kp_vs, zs_vs) = costruisci_finestre(tracce_va_sint, auto, cfg, n_core)
    else:
        az_vs = v_vs = y_xi_vs = y_f_vs = r_vs = kp_vs = zs_vs = None; comfort_vs = []
    print(f"    Campioni TRAIN      : {len(az_tr)}  ({len(tracce_tr)} tracce = reali + sintetiche)")
    # SATURAZIONE DELL'ETICHETTA DI FORZA. F* = clip(-kp*·z_s', ±F_max): quando l'ottimo
    # chiede piu' di quanto l'attuatore possa dare, l'etichetta viene TAGLIATA e insegna alla
    # rete a stare incollata al limite. Senza questo numero, un "|F| max previsto = 1214 N su
    # 1500" sembra la rete che esagera, mentre puo' essere il riferimento che chiede troppo.
    # Misurato sulle strade sintetiche: |F*| max tocca esattamente 1500 N gia' dalla classe B
    # a 76 km/h, e su classe E a 97 km/h l'11% dei campioni e' saturo con RMS 657 N contro un
    # limite ammesso di 455 N — li' il problema di progetto e' INFATTIBILE con questo
    # attuatore, e l'etichetta e' il male minore, non l'ottimo.
    _sat = float(np.mean(np.abs(y_f_tr) >= 0.999 * cfg.forza_max))
    print(f"    Etichetta FORZA     : |F*| max {np.abs(y_f_tr).max():.0f} N su un limite di"
          f" {cfg.forza_max:.0f} N  |  satura nel {_sat:.2%} dei campioni")
    if _sat > 0.01:
        print(f"                          [!] oltre l'1%: su quei campioni l'ottimo vorrebbe piu'"
              f" forza di quanta ce n'e'.")
        print(f"                          L'etichetta li' e' un compromesso, e il picco predetto"
              f" dal ML va letto contro quello del target, non contro {cfg.forza_max:.0f} N.")
    print(f"    Campioni VALID reale: {len(az_va)}  (3a traccia registrata, mai vista)")
    if az_vs is not None:
        print(f"    Campioni VALID sint.: {len(az_vs)}  ({len(tracce_va_sint)} strade su celle "
              f"classe x velocita' DISGIUNTE dal training)")

    # 3) NORMALIZZAZIONE calcolata SOLO dal training (media/dev-std che emergono dai dati)
    norm = Normalizzatore(az_tr, v_tr)
    norm.salva(path_norm)        # viaggia con i .pt: la diagnostica non deve piu' inventarlo
    print(f"    Normalizzazione (dai dati di training): {norm}")
    print(f"    Salvata in {os.path.basename(path_norm)}")

    def su_device(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    # ingressi normalizzati + target scalati sui limiti fisici (xi->range damper, forza->F_max)
    AZ_tr = su_device(norm.na(az_tr)).unsqueeze(1); V_tr = su_device(norm.nv(v_tr)).unsqueeze(1)
    XI_tr = su_device(xi_a_norm(y_xi_tr, auto)).unsqueeze(1); F_tr = su_device(forza_a_norm(y_f_tr, cfg)).unsqueeze(1)
    AZ_va = su_device(norm.na(az_va)).unsqueeze(1); V_va = su_device(norm.nv(v_va)).unsqueeze(1)
    XI_va = su_device(xi_a_norm(y_xi_va, auto)).unsqueeze(1); F_va = su_device(forza_a_norm(y_f_va, cfg)).unsqueeze(1)
    if az_vs is not None:
        AZ_vs = su_device(norm.na(az_vs)).unsqueeze(1); V_vs = su_device(norm.nv(v_vs)).unsqueeze(1)

    # 4) DUE reti separate, loss e optimizer indipendenti
    print("\n[2] Addestramento di DUE reti separate (Loss() e Optimizer indipendenti)...")
    rete_xi, rete_forza = crea_reti(cfg, auto)
    # la rete xi vincolata deve conoscere le statistiche di v per ricostruire la
    # schedulazione fisica L(v) dal proprio ingresso normalizzato
    rete_xi.imposta_normalizzazione(norm)

    # alpha SUGGERITO DAI DATI: alpha e' un TETTO, non una statistica da centrare. Si tara
    # sul MASSIMO scarto etichetta-baseline (in logit, la scala su cui agisce la correzione)
    # con un margine, perche' la sigmoide raggiunge il bordo solo asintoticamente. Tararlo
    # su un percentile lascia fuori portata la coda — e la coda sono le strade sconnesse,
    # cioe' i casi che decidono la tenuta. Con metodo_xi="sqrt2" lo scarto e' zero per
    # costruzione e alpha non serve.
    alpha_fin = float(cfg.autorita_az)
    if cfg.xi_vincolo_fisico:
        from xi_ottimo import scarto_dalla_regola
        sc = scarto_dalla_regola(y_xi_tr, v_tr, auto, cfg)
        print(f"    Scarto etichette dal baseline '{sc['baseline']}' [logit]: mediana {sc['mediana']:.2f}"
              f" | p90 {sc['p90']:.2f} | p99 {sc['p99']:.2f} | max {sc['massimo']:.2f}")
        print(f"    -> alpha suggerito (max scarto +10%) = {sc['alpha_suggerito']:.2f}"
              f"   |   alpha finale in config = {alpha_fin:.2f}")
        if alpha_fin < sc["alpha_suggerito"]:
            print(f"    [!] alpha finale TROPPO PICCOLO: la correzione satura e la rete non"
                  f" puo' raggiungere xi*. Alzo a {sc['alpha_suggerito']:.2f}.")
            alpha_fin = sc["alpha_suggerito"]
        print(f"    Rete xi = sigmoid(L(v) + alpha*Delta(a_z,v)) con alpha da"
              f" {cfg.autorita_az_iniziale:.2f} a {alpha_fin:.2f} (curriculum)")
    else:
        print("    Rete xi LIBERA (ablation): nessun baseline fisico, nessun limite alla"
              " correzione")
    rete_xi.to(device); rete_forza.to(device)
    # lr e weight decay vengono da config, non scritti qui: sono manopole di taratura e
    # devono stare dove si guardano, insieme al motivo per cui valgono quel che valgono
    opt_xi = optim.AdamW(rete_xi.parameters(), lr=cfg.lr_xi, weight_decay=cfg.wd_xi)
    opt_forza = optim.AdamW(rete_forza.parameters(), lr=cfg.lr_forza, weight_decay=cfg.wd_forza)
    print(f"    Regolarizzazione: rete xi lr={cfg.lr_xi:.0e} wd={cfg.wd_xi:.0e}"
          f"  |  rete forza lr={cfg.lr_forza:.0e} wd={cfg.wd_forza:.0e}")

    # LOSS CON PARAMETRI RICAVATI DAI DATI (vedi perdite.py per il perche' e per la formula).
    # xi: MSE PESATA per strato. Il target non ha code — ha una massa concentrata al minimo
    #     del damper, cioe' un problema di sbilanciamento, che si aggredisce coi pesi e non
    #     con la robustezza (un Huber con delta ampio resterebbe sempre nel ramo quadratico,
    #     cioe' sarebbe MSE con un altro nome).
    # forza: Huber con delta ricavato dalla MAD del target. Qui le code sono vere e la
    #     saturazione hardware produce valori estremi legittimi che non devono dominare.
    from perdite import (delta_robusto, descrivi_scelta, pesi_per_strato,
                         MSEPesata, PerditaForza, HuberPesata)
    _f_n = forza_a_norm(y_f_tr, cfg)
    _kp_n = kp_tr / max(cfg.kp_max_ott, 1e-9)
    _zs_n = zs_tr / max(cfg.zs_max, 1e-9)
    d_f, i_f = delta_robusto(_f_n, k=2.0)
    d_zs, i_zs = delta_robusto(_zs_n, k=2.0)
    d_kp, i_kp = delta_robusto(_kp_n, k=2.0)
    print(descrivi_scelta("forza", d_f, i_f))
    print("    loss xi: MSE pesata per strato (target senza code, ma con una massa al "
          f"minimo del damper: serve il peso, non la robustezza)")
    loss_xi = MSEPesata()
    loss_forza = PerditaForza(d_f, d_zs, d_kp, peso_aux=cfg.peso_aux_forza)
    # per la VALIDAZIONE si usa solo il termine su F: le ausiliarie sono un vincolo di
    # coerenza interna, includerle renderebbe i valori non confrontabili fra configurazioni
    loss_forza_val = HuberPesata(d_f)
    due_uscite = bool(getattr(cfg, "forza_due_uscite", True))

    Ntr, Nva, bs = AZ_tr.size(0), AZ_va.size(0), cfg.batch

    # SELEZIONE DEL MODELLO: la validazione usata per il best-checkpoint unisce la traccia
    # reale mai vista E le strade sintetiche su celle (classe x velocita') disgiunte. Cosi'
    # il criterio di stop premia chi generalizza a combinazioni rugosita' x velocita' nuove,
    # non chi memorizza la firma spettrale delle tracce viste.
    if az_vs is not None:
        AZ_sel = torch.cat([AZ_va, AZ_vs]); V_sel = torch.cat([V_va, V_vs])
        XI_sel = torch.cat([XI_va, su_device(xi_a_norm(y_xi_vs, auto)).unsqueeze(1)])
        F_sel = torch.cat([F_va, su_device(forza_a_norm(y_f_vs, cfg)).unsqueeze(1)])
    else:
        AZ_sel, V_sel, XI_sel, F_sel = AZ_va, V_va, XI_va, F_va
    Nsel = AZ_sel.size(0)

    def loss_val(rete, target, criterio, AZ=None, V=None):
        """Validazione: solo la loss PRINCIPALE, senza pesi e senza ausiliarie.
        Deve misurare la qualita' della grandezza che conta, non il termine di coerenza
        interna — altrimenti i valori non sono confrontabili fra configurazioni."""
        AZ = AZ_sel if AZ is None else AZ; V = V_sel if V is None else V
        n = AZ.size(0)
        rete.eval(); s = 0.0
        with torch.no_grad():
            for i in range(0, n, bs):
                nb = min(bs, n - i)
                out = rete(AZ[i:i+bs], V[i:i+bs])
                s += criterio(out, target[i:i+bs]).item() * nb
        return s / n

    stato = {"best_xi": float("inf"), "best_f": float("inf"), "alpha": cfg.autorita_az_iniziale}

    def addestra(az_np, v_np, xi_np, f_np, epoche, etichetta="", alpha_finale=None,
                 kp_np=None, zs_np=None):
        """Un ciclo di addestramento su un dataset arbitrario. Estratto in funzione perche'
        DAgger lo richiama a ogni giro sul dataset aggregato.

        kp_np / zs_np sono i target AUSILIARI della rete forza (i due fattori di cui la
        forza e' il prodotto). Se mancano, la loss usa solo il termine principale su F."""
        AZ = su_device(norm.na(az_np)).unsqueeze(1); V = su_device(norm.nv(v_np)).unsqueeze(1)
        XI = su_device(xi_a_norm(xi_np, auto)).unsqueeze(1)
        FZ = su_device(forza_a_norm(f_np, cfg)).unsqueeze(1)
        # PESI per compensare lo sbilanciamento del target xi (47% al minimo del damper)
        W = su_device(pesi_per_strato(xi_a_norm(xi_np, auto))).unsqueeze(1)
        KP = ZS = None
        if due_uscite and kp_np is not None and zs_np is not None:
            KP = su_device(kp_np / max(cfg.kp_max_ott, 1e-9)).unsqueeze(1)
            ZS = su_device(zs_np / max(cfg.zs_max, 1e-9)).unsqueeze(1)
        n = AZ.size(0)
        af = alpha_fin if alpha_finale is None else alpha_finale
        for e in range(1, epoche + 1):
            # rampa di alpha solo nel ciclo iniziale; nei giri DAgger resta al valore pieno
            if alpha_finale is None:
                n_rampa = max(1, int(cfg.autorita_az_rampa * epoche))
                t = min(1.0, (e - 1) / n_rampa)
                stato["alpha"] = cfg.autorita_az_iniziale + t * (af - cfg.autorita_az_iniziale)
            else:
                stato["alpha"] = af
            rete_xi.imposta_autorita(stato["alpha"])
            rete_xi.train(); rete_forza.train()
            perm = torch.randperm(n, device=device)
            tl_xi = tl_f = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i+bs]; az_b = AZ[idx]; v_b = V[idx]

                opt_xi.zero_grad()
                lx = loss_xi(rete_xi(az_b, v_b), XI[idx], W[idx])
                lx.backward(); opt_xi.step()

                opt_forza.zero_grad()
                if KP is not None:
                    # la rete restituisce i due fattori in unita' fisiche: si normalizzano
                    # con le STESSE scale usate per i target, altrimenti i termini della
                    # loss avrebbero pesi impliciti diversi
                    zs_p, kp_p = rete_forza.fattori(az_b, v_b)
                    f_p = torch.clamp(-(kp_p * zs_p) / cfg.forza_max, -1.0, 1.0)
                    lf = loss_forza(f_p, FZ[idx],
                                    zs_p / cfg.zs_max, ZS[idx],
                                    kp_p / cfg.kp_max_ott, KP[idx])
                else:
                    lf = loss_forza(rete_forza(az_b, v_b), FZ[idx])
                lf.backward(); opt_forza.step()

                tl_xi += lx.item() * idx.size(0); tl_f += lf.item() * idx.size(0)
            tl_xi /= n; tl_f /= n
            # debug: una volta per epoca
            with torch.no_grad():
                L = rete_xi.logit_schedulazione(V[:bs])
                delta_raw = rete_xi.modulazione(AZ[:bs], V[:bs])
                delta_r = rete_xi.beta * delta_raw
                delta_clamp = torch.clamp(delta_raw, -1.0, 1.0)
                delta_c = rete_xi.beta * delta_clamp
                z_1 = L + delta_r
                z_2 = L + delta_c
                print(f"\nDEBUG EPOCA {e}")
                print(f"{'Coefficiente Alpha (beta buffer)':25s} = {rete_xi.beta.item():.4f}")
                print(f"{'L(v)':25s} min={L.min().item():8.6f} max={L.max().item():8.6f} mean={L.mean().item():8.6f}")
                print(f"{'Delta Intoccato':25s} min={delta_raw.min().item():8.6f} max={delta_raw.max().item():8.6f} mean={delta_raw.mean().item():8.6f}")
                print(f"{'beta*Delta Intoccato':25s} min={delta_r.min().item():8.6f} max={delta_r.max().item():8.6f} mean={delta_r.mean().item():8.6f}")
                print(f"{'Delta Clampato [-1,1]':25s} min={delta_clamp.min().item():8.6f} max={delta_clamp.max().item():8.6f} mean={delta_clamp.mean().item():8.6f}")
                print(f"{'beta*Delta Clampato':25s} min={delta_c.min().item():8.6f} max={delta_c.max().item():8.6f} mean={delta_c.mean().item():8.6f}")
                print(f"{'L(v) + beta*Delta':25s} min={z_1.min().item():8.6f} max={z_1.max().item():8.6f} mean={z_1.mean().item():8.6f}")
                print(f"{'L(v) + beta*Delta Clampato':25s} min={z_2.min().item():8.6f} max={z_2.max().item():8.6f} mean={z_2.mean().item():8.6f}")
            vl_xi = loss_val(rete_xi, XI_sel, loss_xi); vl_f = loss_val(rete_forza, F_sel, loss_forza_val)
            if vl_xi < stato["best_xi"]:
                stato["best_xi"] = vl_xi; torch.save(rete_xi.state_dict(), path_xi)
            if vl_f < stato["best_f"]:
                stato["best_f"] = vl_f; torch.save(rete_forza.state_dict(), path_forza)
            if e % 3 == 0 or e == 1 or epoche <= 4:
                print(f"    {etichetta}Epoca {e:02d}/{epoche}  (alpha={stato['alpha']:.2f})"
                      f"  |  RETE xi: Loss(Train)={tl_xi:.5f}  Loss(Valid)={vl_xi:.5f}"
                      f"  |  RETE F: Loss(Train)={tl_f:.5f}  Loss(Valid)={vl_f:.5f}")

    addestra(az_tr, v_tr, y_xi_tr, y_f_tr, cfg.n_epoche, kp_np=kp_tr, zs_np=zs_tr)

    # 4b) DAgger: la rete ha imparato su a_z del modello PASSIVO ma in closed-loop legge
    # l'accelerazione gia' controllata (~47% di quella). Qui si raccolgono gli stati che
    # visita davvero, si rietichettano con l'esperto e si aggrega. Vedi aggregazione.py.
    if cfg.usa_dagger and cfg.metodo_xi == "ottimo":
        from aggregazione import esegui_dagger
        rete_xi.load_state_dict(torch.load(path_xi, map_location=device))
        rete_forza.load_state_dict(torch.load(path_forza, map_location=device))

        def imposta_validazione(az_v, v_v, xi_v, f_v):
            """Sostituisce il set di selezione con stati CLOSED-LOOP e azzera i minimi.

            Senza questo, DAgger non salvava nulla: la validazione era sulla distribuzione
            passiva, che l'aggregazione peggiora per costruzione, quindi la condizione
            'vl < best' non si verificava mai e il modello finale restava quello pre-DAgger.
            Azzerare i best e' necessario anche cambiando set: i valori vecchi sono
            incommensurabili con i nuovi."""
            nonlocal AZ_sel, V_sel, XI_sel, F_sel, Nsel
            AZ_sel = su_device(norm.na(az_v)).unsqueeze(1)
            V_sel = su_device(norm.nv(v_v)).unsqueeze(1)
            XI_sel = su_device(xi_a_norm(xi_v, auto)).unsqueeze(1)
            F_sel = su_device(forza_a_norm(f_v, cfg)).unsqueeze(1)
            Nsel = AZ_sel.size(0)
            stato["best_xi"] = stato["best_f"] = float("inf")

        # ROLLOUT SU STRADE NUOVE: stesse celle del training, realizzazioni diverse (seme
        # diverso). Vedi strade_sintetiche.genera_tracce_aggregazione per il motivo.
        tracce_agg = list(tracce_tr)
        if cfg.usa_augmentation:
            nuove = genera_tracce_aggregazione(auto, cfg)
            # si tengono le 2 tracce REGISTRATE (non replicabili) e si sostituiscono le
            # sintetiche con realizzazioni fresche
            tracce_agg = [t for t in tracce_tr if len(t) < 3 or t[2] is None] + nuove
            print(f"    Rollout su {len(nuove)} strade sintetiche NUOVE (seme "
                  f"{cfg.aug_seed}+{cfg.dagger_seme_strade}, celle identiche al training) "
                  f"+ {len(tracce_agg)-len(nuove)} registrate")
        rete_xi, rete_forza, _agg = esegui_dagger(
            rete_xi, rete_forza, norm, auto, cfg, device, tracce_agg,
            (az_tr, v_tr, y_xi_tr, y_f_tr, kp_tr, zs_tr), tracce_va_sint,
            lambda a, v, x, f, ep, kp_np=None, zs_np=None: addestra(
                a, v, x, f, ep, etichetta="  ", alpha_finale=alpha_fin,
                kp_np=kp_np, zs_np=zs_np),
            imposta_validazione=imposta_validazione)

    # 5) valutazione: ricarica il best-checkpoint e valuta sui DUE set di validazione
    rete_xi.load_state_dict(torch.load(path_xi, map_location=device)); rete_xi.eval()
    rete_forza.load_state_dict(torch.load(path_forza, map_location=device)); rete_forza.eval()

    def predici(rete, AZ, V):
        out = []
        with torch.no_grad():
            for i in range(0, AZ.size(0), bs):
                out.append(rete(AZ[i:i+bs], V[i:i+bs]).cpu())
        return torch.cat(out).numpy().flatten()

    def riporta(titolo, y_xi, y_f, AZ, V):
        p_xi = norm_a_xi(predici(rete_xi, AZ, V), auto)
        p_f = norm_a_forza(predici(rete_forza, AZ, V), cfg)
        print(f"\n {titolo}")
        # AVVERTENZA su R^2: se il target e' quasi costante (su strada liscia xi* resta
        # incollato a xi_min) la varianza al denominatore va a zero e R^2 esplode a valori
        # assurdamente negativi pur con RMSE minuscolo. Li' R^2 non misura niente: si legge
        # l'RMSE, e lo diciamo esplicitamente invece di lasciare un -24 inspiegato.
        sd = float(np.std(y_xi))
        nota = "  [R^2 non informativo: target quasi costante]" if sd < 0.02 else ""
        print(f"   xi(t)    -> R^2: {r2_score(y_xi, p_xi):+.4f} | "
              f"RMSE: {np.sqrt(np.mean((y_xi - p_xi) ** 2)):.4f} | "
              f"MAE: {mean_absolute_error(y_xi, p_xi):.4f} | dev.std target: {sd:.4f}{nota}")
        print(f"   forza(t) -> R^2: {r2_score(y_f, p_f):+.4f} | "
              f"RMSE: {np.sqrt(np.mean((y_f - p_f) ** 2)):.1f} N | "
              f"MAE: {mean_absolute_error(y_f, p_f):.1f} N")
        # PICCHI DELLA FORZA. "|F| max previsto = 1214 N su un limite di 1500" da solo non
        # dice niente: puo' essere la rete che esagera, oppure il riferimento che chiede
        # davvero tanto. Servono i due picchi affiancati, e la quota di SATURAZIONE
        # dell'etichetta: F* = clip(-kp*·z_s', ±F_max), quindi quando l'ottimo chiede piu' di
        # F_max l'etichetta viene tagliata e la rete impara a stare incollata al limite. Se
        # la quota di saturazione e' alta, il picco che cresce non e' un difetto della rete:
        # e' l'attuatore troppo piccolo per quello che l'ottimizzatore vorrebbe.
        pk_t, pk_p = float(np.abs(y_f).max()), float(np.abs(p_f).max())
        sat_t = float(np.mean(np.abs(y_f) >= 0.999 * cfg.forza_max))
        sat_p = float(np.mean(np.abs(p_f) >= 0.999 * cfg.forza_max))
        alti = np.abs(y_f) > 0.5 * pk_t                    # solo la coda alta del target
        bias_pk = float(np.mean(np.abs(p_f[alti]) - np.abs(y_f[alti]))) if alti.any() else 0.0
        print(f"   picchi   -> |F| max: target {pk_t:.0f} N, ML {pk_p:.0f} N "
              f"({pk_p/max(pk_t,1e-9)-1:+.0%}) | margine ML sul limite "
              f"{100*(1-pk_p/cfg.forza_max):.0f}%")
        print(f"                saturazione a ±{cfg.forza_max:.0f} N: target {sat_t:.2%}, "
              f"ML {sat_p:.2%} | sui picchi (|F*|>50% del max) il ML sta {bias_pk:+.0f} N")
        return p_xi, p_f

    from contesto import descrivi_traccia, classe_iso_equivalente, riga_origine
    print("\n" + "=" * 90)
    print(" VALIDAZIONE — confronto fra valori APPRESI e valori CALCOLATI, su dati mai visti")
    print("=" * 90)
    print(f" xi predetto / F predetta : {riga_origine('ML')}")
    print(f" xi* / F* di riferimento  : {riga_origine('ottimo')}")
    _azv, _vv = tracce_va_reali[0][0], tracce_va_reali[0][1]
    _cl = classe_iso_equivalente(float(np.sqrt(np.mean(np.asarray(_azv) ** 2))),
                                 float(np.mean(_vv)))
    xi_pred, forza_pred = riporta(
        f"[A] REGISTRATA: {descrivi_traccia(_azv, _vv, cfg, nome_val)}"
        f"  |  rugosita' equivalente classe ISO {_cl}",
        y_xi_va, y_f_va, AZ_va, V_va)
    if az_vs is not None:
        # test decisivo: rugosita' x velocita' mai viste insieme (es. classe sconnessa a 90 km/h)
        _el = "; ".join(nome_traccia(t, "?") for t in tracce_va_sint)
        riporta(f"[B] SINTETICHE, celle classe x velocita' DISGIUNTE dal training\n"
                f"     {_el}",
                y_xi_vs, y_f_vs, AZ_vs, V_vs)
        print("     ^ [B] e' il test severo: rugosita' x velocita' mai viste insieme.")
    _pk = float(np.abs(forza_pred).max())
    print(f"\n |forza| max prevista dal ML: {_pk:.1f} N  (limite {cfg.forza_max:.0f} N,"
          f" margine {100*(1-_pk/cfg.forza_max):.0f}%)")
    if _pk > 0.9 * cfg.forza_max:
        print("    [!] margine sotto il 10%: guardare la riga 'picchi' qui sopra per capire se"
              " e' la rete che esagera o il riferimento che chiede troppo")

    # COMFORT in closed-loop. Il riferimento ottimo va calcolato SULLO STESSO SEGMENTO su
    # cui si misura il ML, altrimenti non e' un confronto: mediarlo su tutte le tracce
    # (comprese le classe E sintetiche) e accostarlo al ML misurato su un tratto urbano da'
    # numeri corretti in un accostamento privo di senso.
    # la traccia e' (a_z, v, z_r'|None, nome): si prendono i primi due elementi, non si
    # spacchetta tutto — aggiungere un campo alla tupla non deve rompere i punti d'uso
    az_val, v_val = tracce_va_reali[0][0], tracce_va_reali[0][1]
    nseg = min(len(az_val), int(cfg.comfort_secondi * cfg.freq_campion))
    traj_cl = traiettorie_reali_ml(az_val[:nseg], v_val[:nseg], rete_xi, rete_forza, norm, auto, cfg, device)
    rms_pas = float(np.sqrt(np.mean(traj_cl["acc_cassa_pas"] ** 2)))
    rms_ml = float(np.sqrt(np.mean(traj_cl["acc_cassa_ml"] ** 2)))

    from xi_ottimo import ottimo as _ott, simula_con_xi as _sim, pondera_wk as _wk
    zr_s, zrd_s = ricostruisci_strada(az_val[:nseg], cfg)
    if cfg.calibra_strada:
        _f = np.clip(np.sqrt(np.mean(az_val[:nseg] ** 2))
                     / (rms_accel_passiva(zrd_s, auto, cfg) + 1e-9), 0.2, 20.0)
        zrd_s = zrd_s * _f
    xi_id, kp_id = _ott(az_val[:nseg], v_val[:nseg], zrd_s, auto, cfg)
    az_id, _g, _c = _sim(zrd_s, v_val[:nseg], xi_id, auto, cfg, kp_serie=kp_id)
    rms_id = float(np.sqrt(np.mean(az_id ** 2)))
    print("\n COMFORT in CLOSED-LOOP — RMS accelerazione cassa")
    print(f"   dati: {nome_val}, primi {nseg/cfg.freq_campion:.0f} s"
          f"  (v {v_val[:nseg].min()*3.6:.0f}-{v_val[:nseg].max()*3.6:.0f} km/h,"
          f" rugosita' equivalente classe ISO {_cl})")
    print(f"   passiva  {rms_pas:.3f} m/s^2")
    print(f"   ML       {rms_ml:.3f} m/s^2   ({100*(1-rms_ml/rms_pas):+.1f}% vs passiva)")
    print(f"   ottimo   {rms_id:.3f} m/s^2   ({100*(1-rms_id/rms_pas):+.1f}% vs passiva)"
          f"  <- il tetto raggiungibile con le etichette")
    if rms_ml < rms_id * 0.98:
        print("   [!] il ML batte l'ottimo sull'accelerazione GREZZA: non e' un paradosso,")
        print("       l'ottimo minimizza l'accelerazione PONDERATA Wk sotto vincoli, quindi")
        print("       puo' accettare piu' RMS grezzo per tenere la ruota a terra.")
        print(f"       Comfort ponderato Wk: ML {np.sqrt(np.mean(_wk(traj_cl['acc_cassa_ml'], cfg.freq_campion)**2)):.3f}"
              f"  contro ottimo {np.sqrt(np.mean(_wk(az_id, cfg.freq_campion)**2)):.3f} m/s^2")

    # Tabella riassuntiva delle etichette di validazione.
    # NB: con metodo_xi="ottimo" NON si bin-a per r (che non guida piu' nulla) ma per
    # RUGOSITA', perche' e' da quella che dipende l'ottimo; e non si stampa kp_da_r, che
    # non e' il guadagno usato. Stampare la vecchia tabella qui sarebbe stato un errore
    # silenzioso: numeri corretti sotto un'intestazione che descrive un'altra legge.
    print("\n" + "-" * 74)
    if cfg.metodo_xi == "ottimo":
        # si usa il set SINTETICO se c'e': sulla traccia reale urbana xi* e' costante a
        # xi_min (strada liscia), quindi la tabella mostrerebbe cinque righe identiche
        _az_t, _v_t, _xi_t, _f_t = ((az_vs, v_vs, y_xi_vs, y_f_vs) if az_vs is not None
                                    else (az_va, v_va, y_xi_va, y_f_va))
        rms_fin = np.sqrt(np.mean(_az_t.astype(np.float64) ** 2, axis=1))
        print(" ETICHETTE OTTIME: xi* in funzione della RUGOSITA' (RMS a_z) e della velocita'"
              + ("   [strade sintetiche]" if az_vs is not None else "   [traccia reale]"))
        print("-" * 74)
        bordi = np.quantile(rms_fin, [0, .25, .5, .75, 1.0])
        print(f"{'RMS a_z':<14} | {'v [km/h]':<9} | {'xi* medio':<10} | {'|forza| [N]'}")
        for i in range(4):
            m = (rms_fin >= bordi[i]) & (rms_fin <= bordi[i + 1])
            if np.any(m):
                print(f"{bordi[i]:.2f}-{bordi[i+1]:<9.2f} | {_v_t[m].mean()*3.6:<9.1f} | "
                      f"{_xi_t[m].mean():<10.3f} | {np.abs(_f_t[m]).mean():.0f}")
    else:
        print(" REGOLA √2: xi in funzione del rapporto r = omega/omega_n (target validazione)")
        print("-" * 74)
        bins = [0.0, 1.0, RADQ2, 2.0, 10.0]
        etich = ["r<1", "1<r<√2", "√2<r<2", "r>2"]
        print(f"{'Range r':<10} | {'v [km/h]':<9} | {'xi medio':<9} | {'kp [Ns/m]':<10} | {'|forza| [N]'}")
        for i in range(len(bins) - 1):
            m = (r_va >= bins[i]) & (r_va < bins[i + 1])
            if np.any(m):
                print(f"{etich[i]:<10} | {v_va[m].mean()*3.6:<9.1f} | {y_xi_va[m].mean():<9.3f} | "
                      f"{kp_da_r(r_va[m].mean(), cfg):<10.0f} | {np.abs(y_f_va[m]).mean():.0f}")

    # 6) figura + animazione
    # tenuta di strada ed energia (dal closed-loop gia' calcolato)
    rh_pas = float(np.sqrt(np.mean(traj_cl["defl_gomma_pas"] ** 2)) * 1000)
    rh_ml = float(np.sqrt(np.mean(traj_cl["defl_gomma_ml"] ** 2)) * 1000)
    from xi_ottimo import riferimenti as _rif
    _a, _d, _s, _f = _rif(auto, cfg)
    print(f" TENUTA DI STRADA su {nome_val} ({nseg/cfg.freq_campion:.0f} s) — RMS deflessione"
          f" pneumatico: passiva {rh_pas:.2f} mm -> ML {rh_ml:.2f} mm"
          f"   (limite ammesso {_d*1000:.2f} mm)")
    en = bilancio_energetico(traj_cl, cfg)
    print(f"\n BILANCIO ENERGETICO — integrali su {nseg/cfg.freq_campion:.0f} s di {nome_val}")
    print(f"   IPOTESI  damper    : {en['ipotesi']}")
    print(f"   IPOTESI  attuatore : {en['ipotesi_attuatore']}")
    print("   CONVENZIONE: le voci misurate sono MODULI (>=0), il verso e' nel nome. Il"
          f" bilancio ha segno")
    print("   dal punto di vista della BATTERIA: <0 esce, >0 rientra. I joule scalano con la"
          f" durata: non sono una potenza.")
    print(f"   MECCANICA  damper dissipato in calore nell'olio"
          f"{' (contato nel recupero)' if en['damper_recuperabile'] else ' — PERSO, fuori bilancio'}"
          f" : {en['E_damp']:7.1f} J")
    print(f"   MECCANICA  attuatore, lavoro fatto sul veicolo (spinge)                :"
          f" {en['E_inj']:7.1f} J")
    print(f"   MECCANICA  attuatore, lavoro ricevuto dal veicolo (frena)              :"
          f" {en['E_abs']:7.1f} J")
    print(f"   ELETTRICO  ADDEBITO  = lavoro in spinta / eta_attuatore                : "
          f"{-en['addebito']:+7.1f} J   (esce dalla batteria)")
    print(f"   ELETTRICO  ACCREDITO = {en['formula']:<47}: {en['accredito']:+7.1f} J"
          f"   (rientra in batteria)")
    print(f"   ELETTRICO  NETTO     = accredito - addebito                            :"
          f" {en['netto']:+7.1f} J  -> {en['verdetto']}")
    print(f"   [sensibilita'] con le perdite di trazione al {en['eta']:.0%} come il generatore,"
          f" il netto sarebbe {en['netto_con_perdite_attuatore']:+.1f} J")
    if not en['damper_recuperabile']:
        # il confronto esplicito serve a mostrare quanto pesa l'ipotesi sull'hardware
        _alt = cfg.eta_generatore * (en['E_damp'] + en['E_abs']) - en['addebito']
        print(f"   [sensibilita'] con un damper ELETTROMAGNETICO al posto dell'idraulico il netto"
              f" sarebbe {_alt:+.1f} J")
    # DIAGNOSTICA FISICA: R^2 e comfort non dicono se xi segue la VELOCITA' o l'accelerazione.
    # Qui si misura direttamente xi(v, a_z), in open-loop e in closed-loop.
    esegui_diagnostica(rete_xi, rete_forza, norm, auto, cfg, device,
                       dati=(az_tr, v_tr, y_xi_tr), figura=cfg.salva_figura,
                       percorso=os.path.join(base, "diagnostica_xi.png"),
                       provenienza=(f"{len(tracce_tr)} tracce di TRAINING "
                                    f"({len(tracce_tr)-cfg.n_tracce_sintetiche} reali registrate"
                                    f" + {cfg.n_tracce_sintetiche} sintetiche ISO 8608), "
                                    f"finestre da {cfg.seq_len/cfg.freq_campion:.0f} s"))

    if cfg.salva_figura:
        salva_figura(y_xi_va, xi_pred, y_f_va, forza_pred, r_va, rms_pas, rms_ml, cfg, auto,
                     os.path.join(base, cfg.figura_file),
                     nome_traccia=nome_val, v_val=v_va,
                     secondi_comfort=nseg / cfg.freq_campion)
        salva_figura_efficienza(traj_cl, cfg, os.path.join(base, "efficienza_sospensione.png"),
                                nome_traccia=nome_val, auto=auto)
    if cfg.mostra_anim or cfg.salva_video:
        # Il vecchio if/else su cfg.anim_usa_ml era morto (i due rami identici). Ora si
        # simulano TUTTI i controllori sulla stessa strada e si commutano dal vivo con i
        # pulsanti nell'animazione: ML, ottimo (xi*, kp*) e regola √2.
        print("\n[3] Animazione: tutti i controllori sulla stessa strada demo...")
        print("    (la strada demo e' severa — equivale a una classe E ISO 8608 — quindi"
              " xi DEVE muoversi parecchio)")
        if cfg.anim_confronto:
            traj = traiettorie_demo_confronto(rete_xi, rete_forza, norm, auto, cfg, device)
            scarto_anim = float(np.mean(np.abs(traj["ML"]["xi"] - traj["ML"]["xi_rif"])))
            print(f"    scarto medio |xi_ML - xi*| sulla demo: {scarto_anim:.3f}")
        else:
            traj = traiettorie_demo_ml(rete_xi, rete_forza, norm, auto, cfg, device,
                                       usa_esperto=not cfg.anim_usa_ml)
        anima_confronto(traj, cfg, os.path.join(base, cfg.video_file), auto=auto)

    print("=" * 90)
    print(f"Completato in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    # PRIMA di qualunque stampa: su Windows la codifica di default non contiene i
    # simboli matematici del log. Vedi portabilita.py.
    forza_utf8()
    main()
