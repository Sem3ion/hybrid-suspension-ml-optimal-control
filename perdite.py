# -*- coding: utf-8 -*-
"""
perdite.py — Funzioni di costo per l'addestramento, con i parametri RICAVATI DAI DATI.

COSA VUOL DIRE "TARATA SUI DATI"
--------------------------------
La HuberLoss ha un solo parametro, delta: il punto in cui passa da quadratica a lineare.
Sotto delta si comporta come MSE (penalizza il quadrato, quindi insegue anche i piccoli
errori); sopra delta diventa lineare, cioe' smette di farsi trascinare dai valori estremi.
Serve quindi a non lasciare che una manciata di campioni fuori scala domini il gradiente.

Il punto e' che delta ha le stesse unita' del target, e va messo dove finisce il "corpo"
della distribuzione e cominciano le code. Se lo si sceglie a occhio si sbaglia in uno dei
due modi, e nessuno dei due si vede guardando la loss:

  delta troppo GRANDE -> non si entra mai nel ramo lineare, e Huber e' MSE con un altro
      nome. E' esattamente cio' che succedeva a xi: errore tipico 0.073 in unita'
      normalizzate contro delta = 0.25, cioe' sempre nel ramo quadratico. Inerte.
  delta troppo PICCOLO -> quasi tutto finisce nel ramo lineare, il gradiente diventa un
      segno costante e la rete smette di rifinire gli errori piccoli.

Qui delta si calcola dal target con una stima ROBUSTA della sua dispersione:

    delta = k * 1.4826 * MAD(target),      MAD = mediana(|target - mediana(target)|)

1.4826 converte la MAD nella deviazione standard equivalente per un segnale gaussiano
(e' la stessa costante usata nel filtro di Hampel in fisica.py). Si usa la MAD e non la
deviazione standard proprio perche' la distribuzione HA le code: la deviazione standard ne
sarebbe gonfiata, e delta finirebbe troppo in alto — cioe' nel caso "inerte" di sopra.
Con k = 2 il ramo quadratico copre circa il corpo della distribuzione e le code restano
fuori.

PERCHE' xi USA MSE E LA FORZA USA HUBER
---------------------------------------
Misurato sulle etichette:

    target                curtosi in eccesso   saturazioni   massa in un punto
    xi normalizzato               ~0                 -        47% al minimo del damper
    forza normalizzata          26.2              0.6%        (dipende da kp_min)

La forza e' il prodotto -kp*(t) * z_s'(t): eredita le code di entrambi i fattori piu' la
saturazione hardware. E' il caso per cui Huber esiste.
xi invece non ha code, ha un problema diverso — una massa concentrata su un bordo. Quello
non si aggredisce con Huber (che non cambia nulla su errori piccoli) ma con i PESI, se si
vuole: vedi pesi_per_strato.
"""
import numpy as np
import torch
import torch.nn as nn


def delta_robusto(target, k=2.0, minimo=1e-3):
    """delta per HuberLoss ricavato dal target: k volte la sigma robusta stimata dalla MAD.

    Restituisce anche le statistiche usate, cosi' la scelta finisce nel log e non resta
    un numero comparso dal nulla."""
    t = np.asarray(target, dtype=np.float64).ravel()
    mediana = float(np.median(t))
    mad = float(np.median(np.abs(t - mediana)))
    sigma = 1.4826 * mad
    d = max(float(k * sigma), float(minimo))
    quota_lineare = float(np.mean(np.abs(t - mediana) > d))
    return d, dict(mediana=mediana, mad=mad, sigma_robusta=sigma,
                   curtosi_eccesso=float(_curtosi(t)),
                   quota_oltre_delta=quota_lineare, k=k)


def _curtosi(t):
    """Curtosi in eccesso (0 = gaussiana). Alta = code pesanti, cioe' Huber ha senso."""
    t = np.asarray(t, dtype=np.float64)
    s = t.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((t - t.mean()) / s) ** 4) - 3.0)


def descrivi_scelta(nome, delta, info):
    """Riga di log che spiega da dove viene delta."""
    return (f"    loss {nome}: Huber con delta = {delta:.4f} "
            f"(= {info['k']:.0f} x 1.4826 x MAD = {info['k']:.0f} x {info['sigma_robusta']:.4f})"
            f"  |  curtosi in eccesso del target {info['curtosi_eccesso']:.1f}"
            f"  |  {info['quota_oltre_delta']:.1%} dei campioni nel ramo lineare")


def pesi_per_strato(target, n_strati=5, potenza=0.5, max_peso=8.0):
    """Pesi per campione che compensano lo SBILANCIAMENTO del target.

    Il target xi ha il 47% dei campioni esattamente al minimo del damper (strada liscia,
    niente da fare) e la coda utile — le strade sconnesse, dove vive tutto l'errore
    residuo — e' una minoranza. Una loss non pesata ottimizza quindi soprattutto il caso
    facile. Qui si divide il target in strati di uguale ampiezza e si pesa ogni campione
    come (1/frequenza_dello_strato)^potenza.

    'potenza' regola quanto si compensa: 0 = nessun peso, 1 = bilanciamento completo. Si
    usa 0.5 perche' il bilanciamento completo su uno strato con pochissimi campioni
    amplifica anche il loro rumore. 'max_peso' taglia i casi patologici.

    PERCHE' 5 STRATI: 10 E' STATO PROVATO E NON FUNZIONA — risultato negativo, tenuto
    ------------------------------------------------------------------------------------
    Gli strati sono di uguale AMPIEZZA su [xi_min, xi_max]. Con 5 il primo confine cade a
    0.209, quindi la zona di transizione bassa (xi* fra 0.11 e 0.21) finisce nello stesso
    strato della massa del 55% ferma a xi_min e ne eredita il peso basso. Ed e' proprio li'
    che vive il bias residuo del controllore: misurato sulla mappa closed-loop, l'errore
    medio con segno vale +0.004 dove xi* e' al minimo, +0.002 dove e' alto e +0.045 nella
    transizione. Ipotesi ragionevole, quindi, che bastasse infittire gli strati.
    NON E' COSI'. Con 10 strati il peso della transizione sale da 1.06 a 1.49 (+40%) ma il
    bias PEGGIORA, da +0.019 a +0.021, e la rete perde la capacita' di scendere in basso:
    sulla strada demo il xi minimo passa da 0.098 a 0.188 e il comfort Wk da 1.84 a 1.93.
    Il motivo e' che il peso e' a somma costante: dando di piu' alla transizione si toglie
    al fondo (0.625 -> 0.484), e la funzione essendo liscia si alza anche li'.

    DOVE NON STA IL BIAS — quattro interventi, nessun effetto
    ---------------------------------------------------------
        wd 1e-4,  5 strati -> +0.019
        wd 1e-3,  5 strati -> +0.017
        wd 3e-4,  5 strati -> +0.019
        wd 3e-4, 10 strati -> +0.021
    Un fattore 10 di weight decay e un fattore 2 di strati lo spostano di 0.004 in tutto.
    Non e' nella loss e non e' nella regolarizzazione. Non e' nemmeno saturazione della
    sigmoide in uscita: misurato, la rete tocca il fondo a +0.0002 da xi_min.
    L'ipotesi che resta in piedi e' che sia nell'ETICHETTA. xi* esce da un argmin su una
    griglia di soli 11 candidati distanziati 0.063, poi lisciato: il target e' una scala a
    gradini, la rete e' continua, e un'approssimazione continua di una scala sta sopra il
    gradino sul lato in salita. Il test sarebbe infittire cfg.xi_ott_n_candidati, che e' una
    modifica sul lato etichette, non sul lato apprendimento. Non e' stato provato.
    I pesi sono normalizzati a media 1, cosi' il valore della loss resta confrontabile con
    quello della versione non pesata."""
    t = np.asarray(target, dtype=np.float64).ravel()
    lo, hi = t.min(), t.max()
    if hi - lo < 1e-12:
        return np.ones_like(t, dtype=np.float32)
    bordi = np.linspace(lo, hi + 1e-12, n_strati + 1)
    idx = np.clip(np.digitize(t, bordi) - 1, 0, n_strati - 1)
    conta = np.bincount(idx, minlength=n_strati).astype(np.float64)
    freq = np.maximum(conta / conta.sum(), 1e-6)
    w = (1.0 / freq[idx]) ** potenza
    w = np.minimum(w, max_peso * w.mean())
    return (w / w.mean()).astype(np.float32)


class HuberPesata(nn.Module):
    """HuberLoss con pesi per campione (se dati) e delta fissato al momento della creazione."""

    def __init__(self, delta):
        super().__init__()
        self.delta = float(delta)
        self.base = nn.HuberLoss(delta=self.delta, reduction="none")

    def forward(self, pred, target, pesi=None):
        perdita = self.base(pred, target)
        if pesi is None:
            return perdita.mean()
        return (perdita * pesi.view_as(perdita)).mean()


class MSEPesata(nn.Module):
    """MSE con pesi per campione. Per xi: il target non ha code, quindi Huber non aggiunge
    nulla (verificato: con delta=0.25 e errori tipici 0.073 non entrava mai nel ramo
    lineare). Cio' che serve a xi sono i pesi, non la robustezza."""

    def __init__(self):
        super().__init__()
        self.base = nn.MSELoss(reduction="none")

    def forward(self, pred, target, pesi=None):
        perdita = self.base(pred, target)
        if pesi is None:
            return perdita.mean()
        return (perdita * pesi.view_as(perdita)).mean()


class PerditaForza(nn.Module):
    """Loss della rete FORZA a due uscite: principale su F, ausiliarie sui due fattori.

        L = Huber(F_pred, F*)  +  peso_aux * [ Huber(z_s'_pred, z_s'*) + Huber(kp_pred, kp*) ]

    Perche' servono le ausiliarie. La rete produce F come prodotto di due fattori: esistono
    infinite coppie (kp, z_s') il cui prodotto e' giusto ma i cui singoli valori sono
    sbagliati — per esempio kp doppio e z_s' dimezzata. Con la sola loss sul prodotto
    quell'ambiguita' non viene risolta, e le due uscite perdono il significato fisico che
    e' il motivo per cui le abbiamo separate (e con esso la possibilita' di diagnosticare
    QUALE dei due sbaglia). Supervisionare anche i fattori la elimina: i target ci sono
    entrambi, kp* dall'ottimizzatore e z_s' dal filtro di Kalman.

    Il peso delle ausiliarie resta basso (default 0.3): la grandezza che conta davvero e'
    la forza applicata, i fattori sono un vincolo di coerenza."""

    def __init__(self, delta_f, delta_zs, delta_kp, peso_aux=0.3):
        super().__init__()
        self.f = HuberPesata(delta_f)
        self.zs = HuberPesata(delta_zs)
        self.kp = HuberPesata(delta_kp)
        self.peso_aux = float(peso_aux)

    def forward(self, f_pred, f_target, zs_pred=None, zs_target=None,
                kp_pred=None, kp_target=None, pesi=None):
        perdita = self.f(f_pred, f_target, pesi)
        if zs_pred is not None and zs_target is not None:
            perdita = perdita + self.peso_aux * self.zs(zs_pred, zs_target, pesi)
        if kp_pred is not None and kp_target is not None:
            perdita = perdita + self.peso_aux * self.kp(kp_pred, kp_target, pesi)
        return perdita
