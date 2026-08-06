# -*- coding: utf-8 -*-
"""
grafica.py — Figura riassuntiva dei risultati e ANIMAZIONE di confronto (con vs senza
intervento). Non forza mai il backend 'Agg' a livello globale, cosi' la finestra
interattiva dell'animazione puo' aprirsi.
"""
import os
import numpy as np

from controllo import xi_da_r, kp_da_r, RADQ2


def salva_figura(xi_true, xi_pred, forza_true, forza_pred, r_val, rms_pas, rms_ml, cfg, auto, path,
                 nome_traccia="", v_val=None, secondi_comfort=None):
    """4 pannelli: xi(t) target vs ML, forza(t) target vs ML, xi vs r, barra comfort.

    'nome_traccia' e 'v_val' servono a scrivere in figura SU QUALI DATI sono i numeri: senza
    quell'informazione una curva xi(t) non dice se e' guida urbana o classe E a 90 km/h, e
    due pannelli della stessa figura possono riferirsi a segmenti diversi senza che si veda."""
    try:
        import matplotlib.pyplot as plt      # niente use('Agg'): non blocca l'animazione
    except Exception as e:
        print(f"    [i] matplotlib non disponibile, salto la figura ({e})")
        return
    from contesto import percentuali, titola
    legge = "xi* OTTIMO VINCOLATO" if getattr(cfg, "metodo_xi", "") == "ottimo" else "regola √2"
    # layout="constrained": i pannelli si dispongono da soli nello spazio che contesto.titola
    # lascia libero fra titolo e nota, invece di finire tagliati dai bordi
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.0), layout="constrained")
    vinfo = (f"v {np.min(v_val)*3.6:.0f}\u2192{np.max(v_val)*3.6:.0f} km/h"
             if v_val is not None and len(v_val) else "")
    titola(fig,
           "Sospensione 2 GDL Peugeot 207 — controllo APPRESO (ML) contro riferimento",
           sottotitolo=(f"DATI: {nome_traccia or 'traccia di VALIDAZIONE (mai vista in addestramento)'}"
                        + (f"   |   {vinfo}" if vinfo else "")
                        + f"   |   etichette di riferimento: {legge}"),
           nota="ROSSO / ARANCIO = calcolato (riferimento).   VERDE / BLU = predetto dalle reti.")

    n = min(800, len(xi_true)); tt = np.arange(n) * cfg.passo_t
    ax[0, 0].plot(tt, xi_true[:n], "r-", lw=1.0, label=f"xi* CALCOLATO ({legge})")
    ax[0, 0].plot(tt, xi_pred[:n], "g-", lw=1.0, alpha=0.8, label="xi APPRESO (rete)")
    ax[0, 0].set_title(f"xi(t), primi {n*cfg.passo_t:.0f} s di validazione\n"
                       f"errore medio |xi-xi*| = {np.mean(np.abs(xi_true[:n]-xi_pred[:n])):.4f}",
                       fontsize=10.5)
    ax[0, 0].set_xlabel("t [s]"); ax[0, 0].set_ylabel("xi [-]")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(tt, forza_true[:n], "r-", lw=1.0, label="F* CALCOLATA (-kp*·z_s\')")
    ax[0, 1].plot(tt, forza_pred[:n], "b-", lw=1.0, alpha=0.8, label="F APPRESA (rete)")
    ax[0, 1].axhline(cfg.forza_max, ls=":", c="k"); ax[0, 1].axhline(-cfg.forza_max, ls=":", c="k")
    # etichetta DENTRO l'area dati (va="top"): con va="bottom" finiva sopra il bordo dell'asse
    ax[0, 1].text(tt[-1], cfg.forza_max, "saturazione hardware ", fontsize=7, va="top",
                  ha="right", clip_on=True)      # clip_on: vedi nota in salva_figura_efficienza
    ax[0, 1].set_ylim(-1.18 * cfg.forza_max, 1.18 * cfg.forza_max)
    ax[0, 1].set_title(f"Forza attuatore, stessi {n*cfg.passo_t:.0f} s\n"
                       f"errore medio |F-F*| = {np.mean(np.abs(forza_true[:n]-forza_pred[:n])):.0f} N",
                       fontsize=10.5)
    ax[0, 1].set_xlabel("t [s]"); ax[0, 1].set_ylabel("F [N]")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # Pannello xi vs r. ATTENZIONE: la curva xi_da_r(r) e' la legge della regola √2, ed e'
    # il target SOLO se cfg.metodo_xi == "sqrt2". Con le etichette ottime disegnarla sopra
    # lo scatter farebbe credere che i punti debbano seguirla, mentre l'ottimo dipende dalla
    # RUGOSITA' e non da r: nella nuvola c'e' dispersione verticale, e quella e' il segnale.
    ottimo = getattr(cfg, "metodo_xi", "sqrt2") == "ottimo"
    rr = np.linspace(0.3, 4.0, 200)
    axL = ax[1, 0]
    sub = np.random.default_rng(0).choice(len(r_val), size=min(3000, len(r_val)), replace=False)
    axL.scatter(r_val[sub], xi_true[sub], s=3, alpha=0.15, c="orange", label="target (dati)")
    axL.plot(rr, xi_da_r(rr, auto, cfg), "--" if ottimo else "-", c="darkred", lw=2,
             label="regola √2 (confronto)" if ottimo else "legge xi(r)")
    axL.axvline(RADQ2, ls="--", c="k", alpha=0.7); axL.text(RADQ2 + 0.05, 0.55, "r=√2", fontsize=9)
    axL.set_xlabel("r = omega/omega_n"); axL.set_ylabel("xi", color="darkred"); axL.grid(alpha=0.3)
    axL.legend(fontsize=7, loc="center right")
    axL.set_title("xi* ottimo vs r\nla dispersione verticale = effetto della strada" if ottimo
                  else "xi e kp vs frequenza r", fontsize=10.5)
    if not ottimo:
        axR = axL.twinx(); axR.plot(rr, kp_da_r(rr, cfg), "-", c="teal", lw=2)
        axR.set_ylabel("kp attuatore [Ns/m]", color="teal")

    migliora = 100 * (1 - rms_ml / rms_pas)
    vals = [rms_pas, rms_ml]
    ax[1, 1].bar(["PASSIVA\n(c_nom fisso)", "CONTROLLATA\n(reti, closed-loop)"], vals,
                 color=["gray", "seagreen"])
    percentuali(ax[1, 1], vals, sul_totale=False, fontsize=9)
    dur = f"{secondi_comfort:.0f} s" if secondi_comfort else "segmento di validazione"
    ax[1, 1].set_title(f"COMFORT — RMS accelerazione cassa\n"
                       f"closed-loop su {dur}: {migliora:+.0f}% vs passiva", fontsize=10.5)
    ax[1, 1].set_ylabel("RMS a_z [m/s^2]"); ax[1, 1].grid(alpha=0.3, axis="y")

    fig.savefig(path, dpi=120)          # nessun tight_layout: lo spazio lo gestisce titola()
    plt.close(fig)
    print(f"    [ok] Figura salvata: {os.path.basename(path)}")


# --- geometria (metri) del quarter-car disegnato, scala realistica ---
_RAGGIO_RUOTA = 0.33
_RAGGIO_MOZZO = 0.13
_LARGH_CASSA = 1.9
_ALT_CASSA = 0.55
_LUCE_STRUT = 0.42
_QUOTA_BASE_CASSA = 2 * _RAGGIO_RUOTA + _LUCE_STRUT   # quota base cassa a riposo


def _molla_zigzag(xc, y0, y1, spire=6, largh=0.14):
    ys = np.linspace(y0, y1, 2 * spire + 2)
    xs = np.full_like(ys, xc); xs[1:-1:2] = xc - largh / 2; xs[2:-1:2] = xc + largh / 2
    return xs, ys


#: dimensioni dei caratteri dell'animazione, in un posto solo (prima erano sparse nel codice
#: e non c'era modo di sapere se due scritte avessero la stessa taglia)
_FONT = dict(titolo_pannello=11, dati=8.0, controlli=8.5, intestazione_controlli=9,
             assi=9, nota=7.5)


def _pannello(ax, ax_dati, colore, titolo):
    """Disegna un quarter-car su 'ax' e prepara il blocco dati su 'ax_dati', che e' un asse
    SEPARATO accanto al disegno.

    PERCHE' IL TESTO STA FUORI. Prima il riquadro con i dati cinematici era un ax.text in
    coordinate d'assi (0.012, 0.965), cioe' dentro l'area di disegno: la cassa passa proprio
    da quella zona e il testo le finiva sopra. Nessun aggiustamento della posizione lo
    risolve, perche' l'auto si muove in verticale e il riquadro e' alto sei righe. L'unica
    soluzione stabile e' dare al testo il suo spazio."""
    from matplotlib.patches import Circle, Rectangle, Polygon
    asfalto = Polygon([[0, 0]], closed=True, fc="0.78", ec="0.35", lw=1.0, zorder=1)
    strada, = ax.plot([], [], "-", c="0.15", lw=2.5, zorder=2)
    ruota = Circle((0, 0), _RAGGIO_RUOTA, fc="0.15", ec="black", lw=1.5, zorder=5)
    mozzo = Circle((0, 0), _RAGGIO_MOZZO, fc="0.7", ec="black", lw=1.0, zorder=6)
    cassa = Rectangle((0, 0), _LARGH_CASSA, _ALT_CASSA, fc=colore, ec="black", lw=1.8, alpha=0.92, zorder=5)
    molla, = ax.plot([], [], "-", c="crimson", lw=2.0, zorder=4)
    ammort, = ax.plot([], [], "-", c="royalblue", lw=5.0, solid_capstyle="butt", zorder=4)
    ax.axhline(_QUOTA_BASE_CASSA + _ALT_CASSA / 2, ls=":", c="green", alpha=0.5, zorder=3)  # quota neutra
    for p in (asfalto, ruota, mozzo, cassa):
        ax.add_patch(p)
    ax.set_title(titolo, fontsize=_FONT["titolo_pannello"], fontweight="bold")
    ax.set_ylabel("quota [m]", fontsize=_FONT["assi"])
    ax.tick_params(labelsize=_FONT["assi"] - 1)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, ls="--", alpha=0.25)

    ax_dati.set_axis_off()
    txt = ax_dati.text(0.0, 1.0, "", transform=ax_dati.transAxes, va="top", ha="left",
                       fontsize=_FONT["dati"], family="monospace", linespacing=1.45,
                       bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.6"))
    return dict(asfalto=asfalto, strada=strada, ruota=ruota, mozzo=mozzo, cassa=cassa,
                molla=molla, ammort=ammort, txt=txt)


#: etichette CORTE: compaiono sia nei pulsanti del selettore sia come titolo del pannello e
#: come prima riga del blocco dati, quindi devono stare in ~26 caratteri. Quella dell'ottimo
#: arrivava a 36 e sbordava dal riquadro dei comandi sul disegno.
_ETICHETTE = {"ML": "ML (le due reti)",
              "ottimo": "OTTIMO (limite superiore)",
              "sqrt2": "REGOLA √2 (solo v)"}


def anima_confronto(traj, cfg, path_video=None, auto=None):
    """Due pannelli impilati: SOPRA il controllore selezionato, SOTTO il passivo.

    Se 'traj' e' un DIZIONARIO {nome: traiettoria} compaiono dei pulsanti per commutare il
    controllore DAL VIVO. Le traiettorie sono tutte precalcolate sulla STESSA strada, quindi
    il pulsante non ricalcola nulla: scambia solo quale serie viene disegnata — l'unico modo
    di farlo fluido, visto che un passo di RK4 con forward della rete non gira in tempo reale.

    La ruota e' disegnata alla sua quota VERA z_u: lo stacco visibile dall'asfalto e' la
    deflessione del pneumatico (amplificata da anim_gain), cioe' la tenuta di strada."""
    try:
        import matplotlib
        if not cfg.mostra_anim:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.widgets import RadioButtons
    except Exception as e:
        print(f"    [i] animazione non disponibile ({e})"); return

    if isinstance(traj, dict) and "N" not in traj:
        trajs = {k: v for k, v in traj.items() if v is not None}
    else:
        trajs = {"ML": traj}
    nomi = list(trajs)
    corrente = {"nome": nomi[0]}
    traj = trajs[nomi[0]]

    # W = mezza finestra visibile [m]. Con aspetto EQUAL il rapporto fra span x e span y
    # decide la forma del pannello: con W=4.5 lo span era 9 m contro 2.2 m di quota, cioe'
    # pannelli 4:1 schiacciati che lasciavano meta' figura vuota. Con W=3 l'auto si vede
    # anche piu' grande, che in un'animazione e' cio' che serve.
    g = cfg.anim_gain; W = 3.0
    # --- GRIGLIA DEI FOTOGRAMMI, piu' fitta dei campioni ---------------------------------
    # Il rallentatore non si fa mostrando MENO campioni: cosi' l'immagine scatta. Si fa
    # generando piu' fotogrammi dei campioni e INTERPOLANDO le grandezze fra un campione e il
    # successivo. 'pos' e' la posizione FRAZIONARIA dentro la serie originale.
    rall = max(1.0, float(getattr(cfg, "anim_rallenta", 1.0)))
    n_frame = max(2, int(round(traj["N"] / cfg.freq_campion * rall * cfg.anim_fps)))
    pos = np.linspace(0.0, traj["N"] - 1.0, n_frame)
    frame_idx = range(n_frame)
    T_FRAME = pos * cfg.passo_t                  # tempo SIMULATO di ogni fotogramma [s]

    def ric(a):
        """Ricampiona una serie sulla griglia dei fotogrammi (interpolazione lineare)."""
        a = np.asarray(a, dtype=float)
        return np.interp(pos, np.arange(len(a)), a)

    # la STRADA resta a piena risoluzione: e' disegnata per intero dentro la finestra, non
    # letta a un indice, quindi interpolarla non servirebbe e perderebbe dettaglio
    X_STR = np.asarray(traj["x_pos"], float); Y_STR = np.asarray(traj["strada"], float) * g
    x_pos = ric(traj["x_pos"]); vel_f = ric(traj["vel"])
    cassa_pas_y = _QUOTA_BASE_CASSA + ric(traj["quota_cassa_pas"]) * g
    # serie precalcolate per ogni controllore (la strada e il ramo passivo sono comuni)
    # c_nom serve solo per etichettare il pannello passivo
    auto_c_nom = float(getattr(auto, "smorz_nom", 1500.0))
    xi_pas = auto_c_nom / float(getattr(auto, "smorz_critico", 4891.0))
    dt_ = cfg.passo_t

    def _cinematica(t, suff):
        """Serie cinematiche derivate, per mostrarle in tempo reale nell'animazione.
        Corsa sospensione = z_s - z_u, velocita' di cassa = d(z_s)/dt, jerk = d(a_z)/dt."""
        zs = np.asarray(t["quota_cassa_" + suff], float)
        zu = np.asarray(t["quota_ruota_" + suff], float)
        acc = np.asarray(t["acc_cassa_" + suff], float)
        return dict(corsa=zs - zu, vel_cassa=np.gradient(zs, dt_),
                    vel_ruota=np.gradient(zu, dt_), jerk=np.gradient(acc, dt_))

    SER = {}
    for n, t in trajs.items():
        d = dict(cassa=_QUOTA_BASE_CASSA + t["quota_cassa_ml"] * g,
                 ruota=t["quota_ruota_ml"] * g + _RAGGIO_RUOTA,
                 xi=t["xi"], forza=t["forza"], acc=t["acc_cassa_ml"],
                 gomma=t["defl_gomma_ml"], quota=t["quota_cassa_ml"])
        d.update(_cinematica(t, "ml"))
        # le RMS cumulate si calcolano a piena risoluzione e POI si ricampionano: farlo
        # sui fotogrammi interpolati falserebbe la media
        d["rms_cum"] = np.sqrt(np.cumsum(np.asarray(d["acc"], float) ** 2)
                               / np.arange(1, len(d["acc"]) + 1))
        SER[n] = {kk: ric(vv) for kk, vv in d.items()}
    PAS = {kk: ric(vv) for kk, vv in _cinematica(traj, "pas").items()}
    PAS["quota"] = ric(traj["quota_cassa_pas"]); PAS["acc"] = ric(traj["acc_cassa_pas"])
    PAS["gomma"] = ric(traj["defl_gomma_pas"])
    PAS["rms_cum"] = ric(np.sqrt(np.cumsum(np.asarray(traj["acc_cassa_pas"], float) ** 2)
                                 / np.arange(1, traj["N"] + 1)))
    ruota_pas_y = ric(traj["quota_ruota_pas"]) * g + _RAGGIO_RUOTA

    # classe ISO equivalente della strada demo: da' un metro immediato ai numeri
    from contesto import classe_iso_equivalente
    rms_pas_tot = float(np.sqrt(np.mean(traj["acc_cassa_pas"] ** 2)))
    cl_eq = classe_iso_equivalente(rms_pas_tot, float(np.mean(traj["vel"])))

    multi = len(nomi) > 1
    dur = traj["N"] * cfg.passo_t

    # LAYOUT A GRIGLIA, non a coordinate fisse. Prima i comandi e i riquadri stavano in
    # fig.add_axes / fig.text con le y scritte a mano (0.42, 0.30, 0.11): finivano sopra le
    # etichette degli assi, e il riquadro dei dati stava DENTRO l'area di disegno, dove passa
    # la cassa. Qui ogni cosa ha la sua cella:
    #     colonna 0  comandi (selettore + riepilogo + nota)
    #     colonna 1  i due disegni del quarter-car
    #     colonna 2  i dati istantanei, uno per disegno
    fig = plt.figure(figsize=(14.4 if multi else 11.8, 7.8))
    larghezze = [0.95, 3.4, 1.05] if multi else [0.02, 3.4, 1.05]
    gs = fig.add_gridspec(2, 3, width_ratios=larghezze,
                          left=0.035, right=0.995, top=0.875, bottom=0.085,
                          wspace=0.12, hspace=0.24)
    axT = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1], sharex=axT)
    axTd = fig.add_subplot(gs[0, 2])
    axBd = fig.add_subplot(gs[1, 2])

    fig.text(0.5, 0.962, "Peugeot 207 — confronto fra controllori sulla STESSA strada demo",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.918,
             f"strada demo sintetica (dossi periodici ogni 7 m + buche) — {dur:.0f} s, "
             f"v {traj['vel'].min()*3.6:.0f}→{traj['vel'].max()*3.6:.0f} km/h, "
             f"rugosita' equivalente classe ISO {cl_eq}   |   "
             f"movimento verticale amplificato x{g:.0f} per renderlo visibile",
             ha="center", va="center", fontsize=8.5, style="italic", color="0.3")

    A = _pannello(axT, axTd, "seagreen", _ETICHETTE.get(nomi[0], nomi[0]))
    B = _pannello(axB, axBd, "0.55", "PASSIVO — riferimento comune (c_nom fisso)")
    for ax in (axT, axB):
        ax.set_ylim(-0.3, _QUOTA_BASE_CASSA + _ALT_CASSA + 0.30)
    axB.set_xlabel("posizione strada [m]", fontsize=_FONT["assi"])
    plt.setp(axT.get_xticklabels(), visible=False)      # asse x condiviso: una sola scala
    ybot = -0.3

    # --- selettore del controllore, attivo durante l'animazione ---
    radio = None
    if multi:
        ax_c = fig.add_subplot(gs[:, 0]); ax_c.set_axis_off()
        # il selettore vive in un asse ANNIDATO nella cella dei comandi, cosi' la sua
        # posizione segue la griglia invece di essere fissata in coordinate di figura
        ax_r = ax_c.inset_axes([0.0, 0.70, 1.0, 0.26], facecolor="0.94")
        ax_r.set_title("controllore", fontsize=_FONT["intestazione_controlli"], fontweight="bold")
        radio = RadioButtons(ax_r, [_ETICHETTE.get(n, n) for n in nomi], active=0)
        for t in radio.labels:
            t.set_fontsize(_FONT["controlli"])
        etich2nome = {_ETICHETTE.get(n, n): n for n in nomi}

        def cambia(lab):
            corrente["nome"] = etich2nome[lab]
            axT.set_title(lab, fontsize=_FONT["titolo_pannello"], fontweight="bold")

        radio.on_clicked(cambia)
        # riepilogo statico: cosa aspettarsi da ciascuno, senza leggere il terminale
        righe_r = ["MEDIE SU TUTTA LA DEMO", ""]
        righe_r += [f"{n:<8}{np.sqrt(np.mean(SER[n]['acc']**2)):5.2f} "
                    f"{np.sqrt(np.mean(SER[n]['gomma']**2))*1000:5.2f}" for n in nomi]
        righe_r.append(f"{'passivo':<8}{np.sqrt(np.mean(traj['acc_cassa_pas']**2)):5.2f} "
                       f"{np.sqrt(np.mean(traj['defl_gomma_pas']**2))*1000:5.2f}")
        righe_r += ["", "col.1 RMS a_z [m/s^2]", "col.2 RMS gomma [mm]"]
        ax_c.text(0.0, 0.62, "\n".join(righe_r), transform=ax_c.transAxes, va="top",
                  fontsize=_FONT["nota"], family="monospace", linespacing=1.4,
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))
        ax_c.text(0.0, 0.24, f"lo stacco fra gomma e\nasfalto e' la deflessione\n"
                             f"del pneumatico (x{g:.0f}):\ne' la TENUTA DI STRADA",
                  transform=ax_c.transAxes, va="top", fontsize=_FONT["nota"], color="0.25",
                  linespacing=1.5)

    def disegna(P, k, cassa_y, ruota_y):
        xc = x_pos[k]; mask = (X_STR >= xc - W) & (X_STR <= xc + W)
        rx = X_STR[mask]; ry = Y_STR[mask]
        P["strada"].set_data(rx, ry)
        P["asfalto"].set_xy(np.column_stack([np.r_[rx, rx[-1], rx[0]], np.r_[ry, ybot, ybot]]))
        centro_ruota = ruota_y[k]                  # quota VERA della ruota (z_u), non l'asfalto
        P["ruota"].center = (xc, centro_ruota); P["mozzo"].center = (xc, centro_ruota)
        P["cassa"].set_xy((xc - _LARGH_CASSA / 2, cassa_y[k]))
        top_ruota = centro_ruota + _RAGGIO_RUOTA
        sx, sy = _molla_zigzag(xc - 0.28, top_ruota, cassa_y[k]); P["molla"].set_data(sx, sy)
        P["ammort"].set_data([xc + 0.28, xc + 0.28], [top_ruota, cassa_y[k]])

    xi_rif = None if traj.get("xi_rif") is None else ric(traj["xi_rif"])

    _LARG = 26          # larghezza del blocco dati in caratteri monospace

    def _blocco(k, intestazione, xi_txt, forza, zs, zs_p, az, zu_p, corsa, gomma, rms_cum):
        """Blocco dati in colonna STRETTA, non in righe larghe.

        Il formato precedente metteva tre grandezze per riga e arrivava a ~78 caratteri: in
        una colonna larga un quarto di figura veniva tagliato a destra. Incolonnare
        etichetta e valore sta in 26 caratteri e si legge anche meglio, perche' i numeri
        sono allineati fra un fotogramma e il successivo e si vede quale si muove."""
        def r(etichetta, valore, unita=""):
            return f"  {etichetta:<7}{valore:>9}{' ' + unita if unita else ''}"
        return "\n".join([
            intestazione[:_LARG],
            "-" * _LARG,
            r("t", f"{T_FRAME[k]:.2f}", "s"),
            r("v", f"{vel_f[k]*3.6:.0f}", "km/h"),
            r("x", f"{x_pos[k]:.1f}", "m"),
            "COMANDO",
            r("xi", xi_txt),
            r("F", f"{forza:+.0f}", "N"),
            "CASSA",
            r("z_s", f"{zs*1000:+.0f}", "mm"),
            r("z_s'", f"{zs_p:+.2f}", "m/s"),
            r("a_z", f"{az:+.2f}", "m/s2"),
            "RUOTA",
            r("z_u'", f"{zu_p:+.2f}", "m/s"),
            r("corsa", f"{corsa*1000:+.0f}", "mm"),
            r("gomma", f"{gomma*1000:+.0f}", "mm"),
            "DA INIZIO CORSA",
            r("RMS a_z", f"{rms_cum:.2f}", "m/s2"),
        ])

    def aggiorna(k):
        S = SER[corrente["nome"]]
        disegna(A, k, S["cassa"], S["ruota"]); disegna(B, k, cassa_pas_y, ruota_pas_y)
        # xi* accanto al xi comandato: si vede subito quanto ci si discosta dall'ottimo
        xi_txt = (f"{S['xi'][k]:.3f}" + (f" (*{xi_rif[k]:.2f})" if xi_rif is not None else ""))
        # CINEMATICA in tempo reale: senza a_z, velocita' di cassa e corsa non si capisce
        # se la cassa sta salendo, scendendo o e' ferma — la quota da sola non lo dice.
        A["txt"].set_text(_blocco(
            k, _ETICHETTE.get(corrente["nome"], corrente["nome"]), xi_txt, S["forza"][k],
            S["quota"][k], S["vel_cassa"][k], S["acc"][k], S["vel_ruota"][k],
            S["corsa"][k], S["gomma"][k], S["rms_cum"][k]))
        B["txt"].set_text(_blocco(
            k, f"PASSIVO  c={auto_c_nom:.0f} Ns/m", f"{xi_pas:.3f} (fisso)", 0.0,
            PAS["quota"][k], PAS["vel_cassa"][k], PAS["acc"][k],
            PAS["vel_ruota"][k], PAS["corsa"][k], PAS["gomma"][k], PAS["rms_cum"][k]))
        xc = x_pos[k]; axT.set_xlim(xc - W, xc + W); axB.set_xlim(xc - W, xc + W)
        return ()

    # blit=False obbligatorio: cambiando controllore va ridisegnato tutto, non solo gli
    # artisti restituiti. E il riferimento a 'radio' va tenuto vivo o il garbage collector
    # lo raccoglie e i pulsanti smettono di rispondere.
    anim = animation.FuncAnimation(fig, aggiorna, frames=frame_idx,
                                   interval=1000 / cfg.anim_fps, blit=False, repeat=True)
    anim._radio = radio
    if path_video and cfg.salva_video:
        try:
            from matplotlib.animation import FFMpegWriter
            anim.save(path_video, writer=FFMpegWriter(fps=cfg.anim_fps, bitrate=2600), dpi=120)
            print(f"    [ok] Video salvato: {os.path.basename(path_video)}")
        except Exception as e:
            from portabilita import suggerimento_ffmpeg
            print(f"    [i] MP4 non salvato: manca ffmpeg. Per installarlo: "
                  f"{suggerimento_ffmpeg()}\n        (l'animazione a schermo funziona comunque) {type(e).__name__}")
    if cfg.mostra_anim:
        from portabilita import backend_grafico_interattivo
        if backend_grafico_interattivo():
            plt.show()
        else:
            print("    [i] Nessun backend grafico interattivo (tipico su Linux senza server X "
                  "o senza python3-tk/PyQt): la finestra non si apre, ma il video resta salvato "
                  "se cfg.salva_video e' True.")
            plt.close(fig)
    else:
        plt.close(fig)
    return anim


def salva_figura_efficienza(traj, cfg, path, nome_traccia="", auto=None):
    """Tenuta di strada + BILANCIO ENERGETICO esplicito: chi DISSIPA, chi INIETTA, chi RECUPERA.

    Le energie sono INTEGRALI sul segmento simulato, quindi dipendono dalla sua durata e
    dalla strada: senza dirlo il numero in joule non e' interpretabile ne' confrontabile."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    [i] matplotlib non disponibile, salto la figura efficienza ({e})"); return
    rh_ml = np.sqrt(np.mean(traj["defl_gomma_ml"] ** 2)) * 1000.0
    rh_pas = np.sqrt(np.mean(traj["defl_gomma_pas"] ** 2)) * 1000.0
    # unica fonte del bilancio: la stessa funzione che alimenta la stampa di main.py, cosi'
    # figura e log non possono raccontare due storie diverse
    from simulazione import bilancio_energetico
    en = bilancio_energetico(traj, cfg)
    E_damp, E_inj, E_abs = en["E_damp"], en["E_inj"], en["E_abs"]
    eta, netto = en["eta"], en["netto"]
    from contesto import percentuali, classe_iso_equivalente, titola
    dur = traj["N"] * cfg.passo_t
    v_med = float(np.mean(traj["vel"]))
    rms_pas_az = float(np.sqrt(np.mean(traj["acc_cassa_pas"] ** 2)))
    cl_eq = classe_iso_equivalente(rms_pas_az, v_med)
    d_max = ((getattr(auto, "massa_cassa", 260.) + getattr(auto, "massa_ruota", 38.))
             * 9.80665 / getattr(auto, "rigid_gomma", 190000.)) * 1000.0

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.6), layout="constrained")

    vals = [rh_pas, rh_ml]
    ax[0].bar(["PASSIVA\n(c_nom fisso)", "CONTROLLATA\n(reti)"], vals, color=["gray", "seagreen"])
    percentuali(ax[0], vals, sul_totale=False, fontsize=9)
    # IL LIMITE NON SI DISEGNA come linea: sta a 5.13 mm contro barre da 0.24 e 0.08 mm, quindi
    # una axhline schiaccerebbe le barre in due strisce illeggibili per mostrare un vincolo
    # lontanissimo. Il margine e' piu' informativo scritto come numero.
    _margine = (d_max / 3.0) / max(rh_ml, 1e-9)
    ax[0].set_title(f"TENUTA DI STRADA — RMS deflessione pneumatico (z_u - z_r)\n"
                    f"limite ammesso {d_max/3:.2f} mm (distacco ruota): margine {_margine:.0f}x",
                    fontsize=11)
    ax[0].set_ylabel("RMS deflessione [mm]"); ax[0].grid(alpha=0.3, axis="y")
    # il damper idraulico e' una PERDITA: la barra resta (dice quanta energia se ne va) ma
    # e' colorata e etichettata come persa, non come recupero
    _persa = not en["damper_recuperabile"]
    # SEGNI COERENTI CON IL BILANCIO ELETTRICO, visto dalla batteria:
    #   addebito  < 0  esce dalla batteria      accredito > 0  rientra in batteria
    # Le prime due barre sono MECCANICHE (lavoro), non voci del bilancio: restano positive e
    # grigie, perche' mescolarle col conto elettrico e' l'errore che faceva sembrare tutto un
    # guadagno (prima ogni barra era formattata con un "+" davanti, damper e costo compresi).
    etich = ["MECCANICA\ndamper in calore\n" + ("(nel recupero)" if not _persa else "PERSA"),
             "MECCANICA\nattuatore\nin spinta",
             "MECCANICA\nattuatore\nin frenata",
             "ELETTRICO\naddebito\n(spinta/eta_att)",
             f"ELETTRICO\naccredito\n(eta={eta:.0%})",
             "ELETTRICO\nNETTO"]
    val = [E_damp, E_inj, E_abs, -en["addebito"], en["accredito"], netto]
    col = ["0.6" if _persa else "orangered", "0.75", "0.75", "crimson", "seagreen",
           "royalblue" if netto >= 0 else "crimson"]
    ax[1].bar(etich, val, color=col); ax[1].axhline(0, color="k", lw=0.8)
    # Il SEGNO si scrive solo dove ha significato: le voci meccaniche sono moduli (il verso e'
    # nel nome), le voci elettriche hanno segno rispetto alla batteria.
    for i, (barra, v) in enumerate(zip(ax[1].patches, val)):
        testo = f"{v:.1f} J" if i < 3 else f"{v:+.1f} J"
        ax[1].annotate(testo, (barra.get_x() + barra.get_width() / 2, v),
                       textcoords="offset points", xytext=(0, 5 if v >= 0 else -14),
                       ha="center", fontsize=9, fontweight="bold")
    # spazio verticale per le etichette sopra e sotto le barre, in entrambi i versi
    _y0, _y1 = ax[1].get_ylim(); _r = 0.16 * (_y1 - _y0)
    ax[1].set_ylim(_y0 - _r, _y1 + _r)
    ax[1].tick_params(axis="x", labelsize=8.5)
    ax[1].set_title(f"ENERGIA su {dur:.0f} s — bilancio netto {netto:+.1f} J\n"
                    f"{en['verdetto_breve']}", fontsize=11)
    ax[1].set_ylabel("Energia [J]   (elettrico: <0 esce, >0 rientra)")
    ax[1].grid(alpha=0.3, axis="y")
    _alt = eta * (E_damp + E_abs) - E_inj      # scenario damper elettromagnetico, per confronto
    titola(fig, "Tenuta di strada e BILANCIO ENERGETICO — controllo APPRESO in closed-loop",
           sottotitolo=(f"DATI: {nome_traccia or 'traccia reale di validazione'}   |   {dur:.0f} s"
                        f"   |   v media {v_med*3.6:.0f} km/h   |   rugosita' equivalente classe"
                        f" ISO {cl_eq}   |   le energie sono INTEGRALI su questi {dur:.0f} s"
                        f" (scalano con la durata, non sono una potenza)"),
           nota=(f"IPOTESI: {en['ipotesi']}; {en['ipotesi_attuatore']}."
                 + (f" Con un damper elettromagnetico il netto sarebbe {_alt:+.1f} J." if _persa else "")
                 + f" Con le perdite in trazione al {eta:.0%} come il generatore sarebbe"
                 f" {en['netto_con_perdite_attuatore']:+.1f} J."
                 "   —   CONVENZIONE: le voci MECCANICHE sono moduli di lavoro (il verso sta nel"
                 " nome); le voci ELETTRICHE hanno segno rispetto alla batteria, addebito < 0 e"
                 " accredito > 0, e il NETTO e' la loro somma. Il calore del damper idraulico non"
                 " entra nel bilancio elettrico: e' energia meccanica che lascia il sistema."
                 f"   —   TENUTA: piu' bassa = piu' aderenza; limite {d_max/3:.2f} mm."))
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"    [ok] Figura efficienza salvata: {os.path.basename(path)}")
