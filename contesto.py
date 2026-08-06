# -*- coding: utf-8 -*-
"""
contesto.py — Etichette di PROVENIENZA per stampe, report e figure.

PERCHE' ESISTE
--------------
Ogni numero di questo progetto dipende da tre cose che non si vedono guardando il numero:

  1. SU QUALE STRADA e' stato misurato (traccia reale registrata? sintetica ISO di quale
     classe? strada demo? griglia di prova?) e per quanti secondi;
  2. CHI COMANDAVA (le reti? l'ottimo analitico? la regola √2? nessuno, cioe' passivo?);
  3. SE E' APPRESO O CALCOLATO — un xi predetto dalla rete e un xi* risolto per
     ottimizzazione hanno lo stesso nome e la stessa unita', ma uno e' una stima e l'altro
     e' un riferimento.

Senza dirlo si finisce a confrontare cose non confrontabili. E' successo davvero: il
comfort del ML misurato su 60 s di guida urbana era stampato accanto a un "riferimento
ideale" che era la media su 30 tracce comprese le classe E sintetiche. Numeri giusti,
accostamento privo di senso, e sembrava che il ML battesse l'ottimo.

Queste funzioni producono stringhe brevi e uniformi da mettere nei titoli delle figure,
nelle intestazioni delle tabelle e nel report testuale.
"""
import numpy as np

#: come si chiama ogni sorgente di controllo, e se e' appresa o calcolata
ORIGINE = {
    "ML": "APPRESO dalle reti (solo a_z passata + v)",
    "ottimo": "CALCOLATO per ottimizzazione (conosce la strada e lo stato vero)",
    "sqrt2": "CALCOLATO dalla regola √2 (funzione della sola v)",
    "passivo": "NESSUN controllo (c_nom fisso)",
    "target": "CALCOLATO: etichetta di riferimento xi*",
}


def descrivi_traccia(az, v, cfg, nome="", classe=None):
    """Riga di provenienza per una traccia: nome, durata, range di velocita', rugosita'."""
    az = np.asarray(az, dtype=float); v = np.asarray(v, dtype=float)
    dur = len(az) / cfg.freq_campion
    rms = float(np.sqrt(np.mean(az ** 2)))
    parti = [nome] if nome else []
    if classe:
        parti.append(f"classe ISO {classe}")
    parti.append(f"{dur:.0f} s")
    parti.append(f"v {v.min()*3.6:.0f}-{v.max()*3.6:.0f} km/h")
    parti.append(f"RMS a_z {rms:.2f} m/s^2")
    return "  |  ".join(parti)


def classe_iso_equivalente(rms_az, v_media):
    """A quale classe ISO 8608 corrisponde, approssimativamente, un RMS di a_z misurato.

    Serve a dare un metro immediato ai numeri: "RMS 5.6 m/s^2" non dice niente, "classe E
    equivalente" dice che siamo su fondo molto sconnesso. La stima usa
    RMS ~ sqrt(Gd0 * v) del modello passivo, calibrata sui riferimenti a 58 km/h."""
    # RMS di riferimento del modello passivo a 16 m/s per le cinque classi
    rif = {"A": 0.34, "B": 0.68, "C": 1.36, "D": 2.72, "E": 5.44}
    scala = np.sqrt(max(v_media, 1.0) / 16.0)
    dist = {k: abs(np.log((rms_az + 1e-9) / (r * scala))) for k, r in rif.items()}
    return min(dist, key=dist.get)


def riga_origine(chi):
    """Frase che dice se una serie e' appresa o calcolata, e con quali informazioni."""
    return ORIGINE.get(chi, chi)


def intestazione(titolo, traccia="", chi=None, extra=""):
    """Blocco di intestazione uniforme per le tabelle stampate e il report."""
    righe = ["=" * 96, f" {titolo}", "=" * 96]
    if traccia:
        righe.append(f" DATI      : {traccia}")
    if chi:
        righe.append(f" CONTROLLO : {riga_origine(chi)}")
    if extra:
        righe.append(f" NOTA      : {extra}")
    return "\n".join(righe)


def titola(fig, titolo, sottotitolo="", nota="", dim_titolo=13, larghezza_nota=175):
    """Titolo + riga di provenienza + nota a pie' di figura, con lo spazio RISERVATO.

    PERCHE' NON BASTA fig.suptitle + fig.text
    -----------------------------------------
    Il difetto che questa funzione elimina: mettere il titolo con suptitle e la riga di
    provenienza con fig.text(0.5, 0.96, ...) piazza due testi alla STESSA altezza, e si
    sovrappongono. Poi tight_layout(rect=...) ridimensiona i pannelli senza sapere che quei
    testi esistono, quindi i titoli dei pannelli finiscono tagliati ai bordi.

    Qui si fa il contrario: si scrivono i testi, si misura quanto spazio verticale occupano
    (in frazione di figura, dalla dimensione del font e dal numero di righe) e si dice al
    motore di layout CONSTRAINED di disegnare i pannelli solo nella fascia centrale che
    resta. Il motore rispetta quel rettangolo e sistema da se' assi, colorbar e legende.

    La figura deve essere creata con layout="constrained" (pyplot) oppure
    Figure(..., layout="constrained").

    'nota' viene mandata a capo automaticamente a larghezza_nota caratteri: una nota lunga su
    una riga sola esce dai bordi della figura a sinistra e a destra."""
    import textwrap
    alt_pollici = fig.get_size_inches()[1]

    def frazione(righe, punti):
        """Altezza di un blocco di testo in frazione di figura (1 pt = 1/72 di pollice),
        con un 60% di interlinea."""
        return righe * punti * 1.6 / 72.0 / alt_pollici

    y = 1.0
    h_tit = frazione(titolo.count("\n") + 1, dim_titolo)
    y -= h_tit * 0.72
    fig.text(0.5, y, titolo, ha="center", va="center", fontsize=dim_titolo, fontweight="bold")
    alto = h_tit

    if sottotitolo:
        righe_sub = textwrap.wrap(sottotitolo, larghezza_nota) or [""]
        dim_sub = 8.5
        h_sub = frazione(len(righe_sub), dim_sub)
        y -= h_tit * 0.28 + h_sub * 0.5
        fig.text(0.5, y, "\n".join(righe_sub), ha="center", va="center",
                 fontsize=dim_sub, style="italic", color="0.3", linespacing=1.5)
        alto += h_sub

    basso = 0.0
    if nota:
        righe_nota = textwrap.wrap(nota, larghezza_nota) or [""]
        dim_nota = 7.5
        h_nota = frazione(len(righe_nota), dim_nota)
        fig.text(0.5, h_nota * 0.5, "\n".join(righe_nota), ha="center", va="center",
                 fontsize=dim_nota, color="0.2", linespacing=1.5)
        basso = h_nota

    # margine di respiro fra le fasce di testo e i pannelli
    aria = 0.012
    motore = fig.get_layout_engine()
    if motore is not None:
        # ATTENZIONE al formato: il rect del constrained layout e' (left, bottom, LARGHEZZA,
        # ALTEZZA), non (x0, y0, x1, y1). Passando 1.0-alto come quarto valore la fascia in
        # alto NON viene riservata (il rettangolo finisce comunque a ~1.0) e il titolo si
        # sovrappone ai titoli dei pannelli: e' esattamente il difetto che si vedeva.
        y0 = basso + aria
        altezza = max(0.2, 1.0 - alto - basso - 2 * aria)
        motore.set(rect=(0.0, y0, 1.0, altezza))
    else:                                      # nessun motore: si ripiega su subplots_adjust
        fig.subplots_adjust(top=1.0 - alto - aria, bottom=basso + aria)
    return fig


def percentuali(ax, valori, etichette=None, fmt="{:.0f}%", sul_totale=True, fontsize=8):
    """Scrive la percentuale sopra ogni barra di un istogramma/barplot.

    Senza le percentuali un barplot dice solo "questa e' piu' alta di quella": leggere
    "62% del totale" oppure "+38% rispetto al riferimento" e' un'informazione diversa.
    Con sul_totale=True le percentuali sono rispetto alla somma, altrimenti rispetto al
    primo valore (utile per i confronti passivo -> controllato)."""
    valori = np.asarray(valori, dtype=float)
    base = np.sum(np.abs(valori)) if sul_totale else abs(valori[0])
    base = base if base > 1e-12 else 1.0
    for i, (barra, val) in enumerate(zip(ax.patches, valori)):
        if sul_totale:
            testo = fmt.format(100.0 * abs(val) / base)
        else:
            testo = "riferimento" if i == 0 else fmt.format(100.0 * (val / valori[0] - 1.0))
            if i > 0:
                testo = ("+" if val > valori[0] else "") + testo
        y = barra.get_height()
        ax.annotate(testo, (barra.get_x() + barra.get_width() / 2, y),
                    textcoords="offset points", xytext=(0, 4 if y >= 0 else -12),
                    ha="center", fontsize=fontsize, fontweight="bold")
    # SPAZIO PER LE ETICHETTE: l'annotazione sopra la barra piu' alta e' disegnata FUORI
    # dall'area dati e viene tagliata dal bordo dell'asse. Si allarga il limite verticale del
    # 15% invece di sperare che ci stia.
    y0, y1 = ax.get_ylim()
    respiro = 0.15 * max(abs(y1 - y0), 1e-12)
    ax.set_ylim(y0 - (respiro if min(valori) < 0 else 0.0), y1 + respiro)
    return ax
