# -*- coding: utf-8 -*-
"""
reti.py — Le DUE reti neurali del controllore (ingressi gia' normalizzati).

COSA DEVONO FARE
----------------
Entrambe ricevono la stessa cosa e nient'altro: la finestra passata dell'accelerazione
verticale della cassa a_z, piu' la velocita' corrente. Da li' devono produrre i due comandi.

  Rete xi     stima quanto e' sconnessa la strada e da li' lo smorzamento ottimo del damper.
              Usa tutta la finestra (2 s) perche' l'etichetta xi* e' un RMS mobile su quella
              stessa finestra: la rete riceve esattamente l'informazione che serve a
              calcolarla, ne' meno ne' piu'.
  Rete forza  produce la forza dell'attuatore skyhook. Usa solo l'ultimo secondo
              (cfg.seq_len_forza): la velocita' di cassa si ricostruisce in quel tempo, e una
              finestra piu' lunga aggiunge solo capacita' di memorizzare.

Nessuna delle due riceve la strada, la posizione della ruota o lo stato interno. E' la
differenza fra queste reti e i controllori analitici con cui si confrontano: l'ottimo conosce
il profilo stradale e lo stato vero, la regola √2 conosce la velocita'. Le reti hanno solo
un accelerometro, che e' l'unica cosa che avrebbe un'auto vera.

--------------------------------------------------------------------------------
STRUTTURA COMUNE: TRONCO CONVOLUTIVO + FUSIONE A DUE RAMI
--------------------------------------------------------------------------------
Il tronco e' una TCN, cioe' convoluzioni 1D con dilatazione crescente: danno un campo
ricettivo ampio con pochi parametri, adatto a grandezze che dipendono dalla storia del
segnale (una velocita' e' l'integrale di un'accelerazione).

Fra i blocchi la risoluzione temporale viene ridotta tre volte con una media su finestre
adiacenti. Il motivo e' che i target sono statistiche di ENERGIA su una finestra: quanto
vibra la cassa, non in quale millisecondo. La risoluzione fine non serve a calcolarli, e
mantenerla per tutti gli strati costa tre volte le moltiplicazioni e otto volte i parametri
nella testa. Si usa la MEDIA e non il massimo perche' si sta stimando un'energia, non un
picco. La riduzione e' implementata come convoluzione depthwise a pesi fissi (MediaMobile)
e non con nn.AvgPool1d: vedi il docstring di quella classe per il motivo.

La testa fonde due rami di pari dimensione, uno per a_z e uno per v. Se l'accelerazione
entrasse con migliaia di feature e la velocita' con una sola, all'inizializzazione la
velocita' avrebbe un peso trascurabile e il gradiente imboccherebbe la strada
dell'accelerazione anche dove non e' quella giusta. Bilanciare le dimensioni non bilancia
l'informazione — se nel dataset a_z resta un predittore sufficiente la rete puo' comunque
azzerare il ramo v — ma e' un problema di condizionamento in meno.

--------------------------------------------------------------------------------
RETE xi: BASELINE FISICO + CORREZIONE APPRESA
--------------------------------------------------------------------------------
La rete non produce xi direttamente, ma una correzione a un valore di riferimento:

    xi = xi_min + (xi_max - xi_min) * sigmoid( L + alpha * tanh(f(a_z, v)) )

L e' il baseline in scala logit, f(.) la correzione appresa, alpha la sua ampiezza massima.
Due baseline sono disponibili e vanno tenuti COERENTI con l'etichetta usata:

  "comfort"  costante, corrispondente al damper piu' morbido — cioe' dove cade l'ottimo di
             comfort quando i vincoli non mordono. E' quello da usare con le etichette
             ottime: la rete impara SOLO l'irrigidimento richiesto dalla tenuta di strada,
             che e' una quantita' con un significato fisico preciso.
  "sqrt2"    L(v) = logit della regola della trasmissibilita'. Da usare solo con le etichette
             √2, di cui e' il logit esatto.

Incrociarli e' l'errore da non fare: con etichette ottime il baseline √2 spingerebbe verso
valori alti a bassa velocita' mentre il target sta al minimo, e la correzione dovrebbe
spendere tutta la propria autorita' solo per annullarlo.

alpha e' un TETTO, non una statistica da centrare: si tara sul massimo scarto fra etichette
e baseline, misurato in logit, con un margine — la sigmoide raggiunge i bordi solo
asintoticamente. Tararlo su un percentile lascia fuori portata la coda della distribuzione,
e la coda sono le strade sconnesse, cioe' i casi che decidono la tenuta. Durante
l'addestramento alpha cresce da un valore piccolo a quello pieno: all'inizio la rete resta
ancorata al baseline mentre il tronco impara a estrarre feature sensate, poi le si allarga
il margine.

Le costanti fisiche e le statistiche di normalizzazione della velocita' sono registrate come
buffer, quindi viaggiano dentro il file .pt: un modello salvato e' autosufficiente e non puo'
essere ricaricato con un normalizzatore diverso da quello con cui e' stato addestrato.

Cosa garantisce e cosa non garantisce questa struttura. Con alpha piccolo la monotonia
rispetto alla velocita' e' un'identita' algebrica, non una speranza statistica, e serve
quando l'etichetta e' una funzione della sola velocita'. Con le etichette ottime alpha va
aperto abbastanza da coprire tutto il range del damper — altrimenti la rete satura e resta
piatta sulle strade peggiori — e a quel punto il vincolo non e' piu' un prior fisico ma una
riparametrizzazione che confina l'uscita nel range realizzabile. Entrambi i modi restano nel
codice perche' il confronto fra i due e' un risultato, non un residuo.

--------------------------------------------------------------------------------
RETE FORZA: DUE USCITE CHE RICOMPONGONO L'ETICHETTA
--------------------------------------------------------------------------------
L'etichetta della forza e' il prodotto di due grandezze di natura diversa, F = clip(-kp*z_s'),
e la rete le predice separatamente per poi ricomporle. Il perche' sta nel docstring di
ReteForza: in breve, il prodotto ha code molto pesanti e chiedere a una singola regressione
di inseguirlo significa mediare fra regimi diversi, mentre i due fattori presi
separatamente hanno distribuzioni trattabili, hanno ciascuno il proprio target
supervisionato e sono ispezionabili uno per uno.

Le uscite sono confinate ai LIMITI FISICI e non a numeri scelti: xi nel range del damper
reale [c_min/c_crit, c_max/c_crit], la forza alla saturazione dell'attuatore.
"""
import math

import torch
import torch.nn as nn

RADQ2 = math.sqrt(2.0)

_ATTIVAZIONI = {
    "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
    "mish": nn.Mish, "elu": nn.ELU, "leaky": nn.LeakyReLU,
}


def _att(nome):
    return _ATTIVAZIONI.get(nome.lower(), nn.SiLU)


class FusioneDueRami(nn.Module):
    """Fusione BILANCIATA di due rami: TCN per a_z, MLP per v.

    I due rami arrivano alla fusione con la STESSA dimensionalita', quindi partono alla pari.
    Se l'accelerazione entrasse con migliaia di feature e la velocita' con una sola, la
    velocita' avrebbe all'inizializzazione un peso trascurabile e il gradiente imboccherebbe
    la strada dell'accelerazione anche dove non e' quella giusta.

    Cosa questo risolve e cosa no: bilanciare le DIMENSIONI non bilancia l'INFORMAZIONE. Se
    nel dataset a_z resta un predittore sufficiente del target, la rete puo' comunque
    azzerare il ramo v. E' un problema di condizionamento in meno, non una garanzia.

    Serve soprattutto alla rete FORZA, dove l'etichetta F = -kp(v) * z_s'(a_z) dipende
    onestamente da entrambi gli ingressi: kp e' schedulato sulla strada e sulla velocita', la
    velocita' di cassa si ricostruisce dall'accelerazione.
    """

    def __init__(self, n_flat, att, dim=64, hidden=128, dropout=0.25):
        super().__init__()
        self.ramo_az = nn.Sequential(nn.Linear(n_flat, dim), att())
        self.ramo_v = nn.Sequential(nn.Linear(1, 64), att(), nn.Linear(64, dim), att())
        self.fusione = nn.Sequential(nn.Linear(2 * dim, hidden), att(), nn.Dropout(dropout))

    def forward(self, feat_az, v_norm):
        return self.fusione(torch.cat((self.ramo_az(feat_az), self.ramo_v(v_norm)), dim=1))


class MediaMobile(nn.Module):
    """Media su finestre adiacenti con passo 'fattore' — cioe' esattamente AvgPool1d(fattore).

    PERCHE' NON nn.AvgPool1d. La copertura degli operatori del backend MPS (Apple Silicon)
    varia con la versione di torch, e un operatore mancante non degrada: fa fallire la run a
    meta'. Qui la stessa media si ottiene con una convoluzione DEPTHWISE a pesi FISSI
    1/fattore e stride 'fattore': matematicamente identica (media aritmetica di 'fattore'
    campioni adiacenti, senza sovrapposizione), ma usa solo Conv1d, che gira certamente
    perche' e' cio' di cui e' fatto tutto il resto della rete.

    I pesi sono costanti e con requires_grad=False: non sono parametri da addestrare, sono
    l'operazione stessa. Come AvgPool1d con ceil_mode=False, se la lunghezza non e' multipla
    del fattore l'ultimo residuo viene scartato."""

    def __init__(self, canali, fattore=2):
        super().__init__()
        self.fattore = int(fattore)
        self.conv = nn.Conv1d(canali, canali, self.fattore, stride=self.fattore,
                              groups=canali, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / self.fattore)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def _tronco_tcn(seq_len, att):
    """Tronco convoluzionale dilatato condiviso dalle due reti; ritorna (moduli, n_flat).

    RIDUZIONE PROGRESSIVA DELLA RISOLUZIONE TEMPORALE.
    Tre medie su finestre adiacenti portano la lunghezza da 200 a 25 campioni. Il costo di
    NON farlo: a piena risoluzione le convoluzioni valgono ~6.8 M moltiplicazioni per
    campione e il flatten produce 12800 feature, cioe' un Linear(12800, 64) da 0.82 M
    parametri — la quasi totalita' della rete, concentrata nel punto piu' esposto al
    sovradattamento. Con le riduzioni si scende a ~2.2 M moltiplicazioni e 1600 feature.

    La riduzione usa MediaMobile (convoluzione depthwise a pesi fissi) e non nn.AvgPool1d:
    identica nel risultato, ma appoggiata a Conv1d. Vedi il docstring di MediaMobile.

    Perche' non si perde informazione utile: l'etichetta xi* e' un RMS mobile sulla
    finestra, cioe' una statistica di ENERGIA. Serve l'ampiezza in banda, non il dettaglio
    temporale. La MEDIA e non il massimo, per la stessa ragione.

    Le dilatazioni sono riscalate dopo ogni pooling, cosi' il campo ricettivo in campioni
    ORIGINALI resta ampio: 1, 2, poi 4 su scala 1/2 (= 8), poi 8 su scala 1/4 (= 32)."""
    tcn = nn.Sequential(
        nn.Conv1d(1, 32, 3, padding=1, dilation=1), nn.BatchNorm1d(32), att(),
        nn.Conv1d(32, 32, 3, padding=2, dilation=2), nn.BatchNorm1d(32), att(),
        MediaMobile(32),                                          # 200 -> 100
        nn.Conv1d(32, 64, 3, padding=4, dilation=4), nn.BatchNorm1d(64), att(),
        MediaMobile(64),                                          # 100 -> 50
        nn.Conv1d(64, 64, 3, padding=8, dilation=8), nn.BatchNorm1d(64), att(),
        MediaMobile(64),                                          # 50 -> 25
        nn.Conv1d(64, 64, 3, padding=16, dilation=16), nn.BatchNorm1d(64), att(),
    )
    with torch.no_grad():                              # flatten deterministico dalla finestra fissa
        tcn.eval()
        n_flat = tcn(torch.zeros(1, 1, seq_len)).flatten(1).size(1)
        tcn.train()
    return tcn, n_flat


class ReteTCN(nn.Module):
    """TCN 1D con dilatazioni crescenti. out='sigmoid' per xi (libera), 'tanh' per la forza."""

    def __init__(self, seq_len, att=nn.SiLU, out="sigmoid", hidden=256, dim=64, usa_ultimi=None):
        super().__init__()
        # usa_ultimi: quanti campioni FINALI della finestra usare. Le due reti hanno bisogno
        # di contesti diversi: la finestra e' lunga 2 s per allineare xi* alla sua finestra di
        # costo, ma l'etichetta della forza e' F = -kp*(t) * z_s'(t) e la velocita' di cassa
        # si ricostruisce in ~1 s. Dare alla rete forza tutta la finestra peggiora il
        # rapporto dati/parametri e si paga in sovradattamento, non in accuratezza.
        self.usa_ultimi = int(usa_ultimi) if usa_ultimi else None
        eff = min(self.usa_ultimi, seq_len) if self.usa_ultimi else seq_len
        self.tcn, n_flat = _tronco_tcn(eff, att)
        self.fc = FusioneDueRami(n_flat, att, dim=dim, hidden=hidden)
        ultima = nn.Sigmoid() if out == "sigmoid" else nn.Tanh()
        self.testa = nn.Sequential(nn.Linear(hidden, 64), att(), nn.Linear(64, 1), ultima)

    def forward(self, az_norm, v_norm):
        """az_norm: (B,1,seq) normalizzata ; v_norm: (B,1). Uscita in (0,1) o (-1,1)."""
        if self.usa_ultimi:
            az_norm = az_norm[..., -self.usa_ultimi:]      # solo la coda: il presente conta
        feat = self.tcn(az_norm).flatten(1)
        return self.testa(self.fc(feat, v_norm))

    # interfaccia comune con ReteXiFisica (qui non fanno nulla: la rete libera non ha
    # ne' schedulazione da ancorare ne' autorita' da limitare)
    def imposta_normalizzazione(self, norm):
        return self

    def imposta_autorita(self, alpha):
        return self

   
class ReteXiFisica(nn.Module):
    """Rete xi con SCHEDULAZIONE IN VELOCITA' IMPOSTA e a_z solo come MODULAZIONE limitata.

        xi_norm = sigmoid( L(v) + beta * tanh(f(a_z, v)) )

    dove L(v) e' il logit esatto della regola √2. Vedi il docstring del modulo per le
    quattro proprieta' garantite per costruzione.
    """
    
    def __init__(self, seq_len, auto, cfg, att=nn.SiLU, hidden=128, beta=None, dim=64,
                 baseline=None):
        super().__init__()
        # QUALE baseline. Non e' un dettaglio: dev'essere coerente con l'ETICHETTA.
        #   "sqrt2"   : L(v) = logit della regola √2. Giusto solo se metodo_xi="sqrt2".
        #   "comfort" : L costante = ottimo di comfort a vincoli inattivi, cioe' il damper
        #               piu' morbido possibile (con eccitazione a banda larga l'ottimo di
        #               comfort cade sotto xi_min, quindi si appoggia al bordo). La rete
        #               impara SOLO l'irrigidimento richiesto dai vincoli di tenuta/corsa:
        #               "parti morbido, indurisci quando la strada lo impone".
        # Usare "sqrt2" con etichette xi* ottime e' l'errore da non fare: il baseline
        # spingerebbe verso 0.66 a bassa velocita' mentre il target e' 0.082, e servirebbe
        # alpha ~ 10 solo per annullarlo.
        self.baseline = str(baseline if baseline is not None
                            else getattr(cfg, "xi_baseline", "sqrt2"))
        self.tcn, n_flat = _tronco_tcn(seq_len, att)
        self.fc = FusioneDueRami(n_flat, att, dim=dim, hidden=hidden)
        # testa MODULANTE: uscita in (-1,1), poi scalata da beta -> autorita' limitata
        self.testa = nn.Sequential(nn.Linear(hidden, 64), att(), nn.Linear(64, 1),nn.Tanh())

        b = float(cfg.autorita_az if beta is None else beta)
        # --- costanti fisiche e statistiche di normalizzazione come BUFFER (salvate nel .pt) ---
        # SEMPRE float32 esplicito: omega_n arriva da numpy come float64 e un buffer float64
        # fa fallire il trasferimento su MPS (Apple Silicon non supporta la doppia precisione).
        def _b32(nome, valore):
            self.register_buffer(nome, torch.tensor(float(valore), dtype=torch.float32))

        _b32("beta", b)
        # r(v) = k_r * v, con k_r = 2*pi/(lambda_c*omega_n)  [s/m]
        _b32("k_r", 2.0 * math.pi / (cfg.lambda_c_design * float(auto.puls_nat_cassa)))
        _b32("pendenza", cfg.pendenza_sched)
        _b32("r_lo", 0.15)                                  # come fisica._freq_velocita
        _b32("r_hi", cfg.r_max)
        _b32("media_v", 0.0)
        _b32("std_v", 1.0)
        # logit del baseline "comfort": xi_min piu' un margine NUMERICO (il logit diverge
        # sui bordi, quindi il baseline non puo' stare esattamente su xi_min).
        u0 = float(getattr(cfg, "xi_margine_logit", 0.03))
        _b32("logit_base", math.log(u0 / (1.0 - u0)))

    def imposta_normalizzazione(self, norm):
        """Inietta le statistiche di v del training: servono per ricostruire v [m/s] dalla
        v normalizzata e valutare L(v). Da chiamare UNA volta, prima dell'addestramento;
        dopo un load_state_dict i valori arrivano dai buffer salvati."""
        with torch.no_grad():
            self.media_v.fill_(float(norm.media_v))
            self.std_v.fill_(float(norm.std_v))
        return self

    def imposta_autorita(self, alpha):
        """Cambia alpha (ampiezza concessa alla correzione appresa) durante l'addestramento.

        CURRICULUM: si parte con alpha piccolo, cosi' la rete resta ancorata al baseline
        fisico mentre il tronco convoluzionale impara a estrarre feature sensate, e si
        allarga poi il margine perche' possa raggiungere xi*. Partire subito con alpha
        grande equivale ad addestrare la rete libera, con il rischio di scorciatoia."""
        with torch.no_grad():
            self.beta.fill_(float(alpha))
        return self

    @property
    def autorita(self):
        return float(self.beta)

    def logit_schedulazione(self, v_norm):
        """Il baseline in scala logit, attorno a cui la rete impara la correzione.

        "sqrt2"  : L(v) = -pendenza*(r(v)/√2 - 1), decrescente in v (regola classica).
        "comfort": costante = ottimo di comfort a vincoli inattivi (damper morbido).
                   Non dipende da v perche' con eccitazione a banda larga il comfort
                   non ha una velocita' preferita: quello che cambia il punto di lavoro
                   e' la RUGOSITA', ed e' proprio cio' che la rete deve stimare da a_z."""
        if self.baseline == "comfort":
            return self.logit_base.expand_as(v_norm)
        v = v_norm * self.std_v + self.media_v                       # de-normalizza -> [m/s]
        r = torch.clamp(self.k_r * v, min=float(self.r_lo), max=float(self.r_hi))
        return -self.pendenza * (r / RADQ2 - 1.0)

    def modulazione(self, az_norm, v_norm):
        feat = self.tcn(az_norm).flatten(1)
        x = self.testa(self.fc(feat, v_norm))
        return torch.clamp(x,-1.0,+1.0)               #originale senza clamp
    

    def forward(self, az_norm, v_norm):
        return torch.sigmoid(self.logit_schedulazione(v_norm) + self.beta*(self.modulazione(az_norm, v_norm)))


class ReteForza(nn.Module):
    """Rete forza a DUE USCITE, strutturata come l'etichetta:  F* = clip(-kp* * z_s').

    PERCHE' DUE USCITE E NON UNA REGRESSIONE SU F
    ---------------------------------------------
    L'etichetta della forza e' il PRODOTTO di due grandezze di natura completamente diversa:

      z_s'  velocita' verticale della cassa. E' un problema di STIMA DI STATO: si ottiene
            integrando a_z (in pratica un filtro di Kalman). Segnale liscio, a media nulla,
            distribuzione ben condizionata: RMS ~0.37 m/s.
      kp*   guadagno skyhook ottimo. E' un problema di SCHEDULAZIONE, come xi: dipende dalla
            rugosita' della strada. Ed e' quasi BINARIO — su fondo sconnesso il vincolo di
            forza morde e l'ottimo SPEGNE l'attuatore: misurato kp*=0 nel 35-60% dei campioni
            su classe D/E.

    Il prodotto eredita il peggio di entrambi: code molto pesanti (curtosi in eccesso fra 8
    e 26 secondo quanto in basso si lascia scendere kp) piu' la saturazione hardware.
    Chiedere a una singola uscita la regressione su quella distribuzione significa mediare
    fra regimi diversi, e il risultato e' mediocre in entrambi.

    Qui i due fattori vengono predetti SEPARATAMENTE e ricomposti:

        F = clip( -kp_pred * zs_pred ,  +-F_max )

    Tre vantaggi concreti:
      1. ognuna delle due uscite ha una distribuzione trattabile, e ognuna ha il PROPRIO
         target supervisionato (li abbiamo entrambi: kp* dall'ottimizzatore, z_s' dal Kalman);
      2. le due uscite sono ISPEZIONABILI — si puo' verificare separatamente se sbaglia la
         stima di stato o la schedulazione, cosa impossibile con un'uscita sola;
      3. la saturazione resta FUORI dalla rete, dove appartiene: e' un limite hardware, non
         una cosa da imparare.

    kp e' vincolata a [kp_min, kp_max] via Sigmoid, z_s' a +-zs_max via Tanh: entrambi
    limiti che vengono dai dati o dall'hardware, non scelte arbitrarie."""

    def __init__(self, seq_len, cfg, att=nn.SiLU, hidden=256, dim=64, usa_ultimi=None,
                 zs_max=2.0):
        super().__init__()
        self.usa_ultimi = int(usa_ultimi) if usa_ultimi else None
        eff = min(self.usa_ultimi, seq_len) if self.usa_ultimi else seq_len
        self.tcn, n_flat = _tronco_tcn(eff, att)
        self.fc = FusioneDueRami(n_flat, att, dim=dim, hidden=hidden)
        # due teste distinte: stima di stato e schedulazione sono compiti diversi
        self.testa_zs = nn.Sequential(nn.Linear(hidden, 128),  att(), nn.Linear(128, 128), nn.LeakyReLU(0.25), nn.Linear(128, 1),nn.Tanh())
        self.testa_kp = nn.Sequential(nn.Linear(hidden, 64), att(), nn.Linear(64, 1), nn.Sigmoid())
        self.register_buffer("kp_min", torch.tensor(float(cfg.kp_min_ott)))
        self.register_buffer("kp_max", torch.tensor(float(cfg.kp_max_ott)))
        self.register_buffer("zs_max", torch.tensor(float(zs_max)))
        self.register_buffer("f_max", torch.tensor(float(cfg.forza_max)))

    def fattori(self, az_norm, v_norm):
        """Restituisce (z_s' [m/s], kp [Ns/m]) in unita' FISICHE. Utile per la diagnostica:
        permette di capire QUALE dei due fattori sbaglia."""
        if self.usa_ultimi:
            az_norm = az_norm[..., -self.usa_ultimi:]
        h = self.fc(self.tcn(az_norm).flatten(1), v_norm)
        zs = self.testa_zs(h) * self.zs_max
        kp = self.kp_min + self.testa_kp(h) * (self.kp_max - self.kp_min)
        return zs, kp

    def forward(self, az_norm, v_norm):
        """Uscita NORMALIZZATA in [-1, 1] come la rete monolitica, cosi' resta
        interscambiabile: F/F_max = clip(-kp*z_s'/F_max)."""
        zs, kp = self.fattori(az_norm, v_norm)
        return torch.clamp(-(kp * zs) / self.f_max, -1.0, 1.0)


def crea_reti(cfg, auto=None):
    """Due reti separate: xi (vincolata alla fisica se cfg.xi_vincolo_fisico) e forza (TCN, Tanh)."""
    if getattr(cfg, "xi_vincolo_fisico", False):
        if auto is None:
            raise ValueError("crea_reti: con xi_vincolo_fisico=True serve anche 'auto' "
                             "(omega_n e i limiti del damper entrano nella schedulazione).")
        rete_xi = ReteXiFisica(cfg.seq_len, auto, cfg, _att(cfg.att_xi))
    else:
        rete_xi = ReteTCN(cfg.seq_len, _att(cfg.att_xi), out="sigmoid")
    if getattr(cfg, "forza_due_uscite", True):
        rete_forza = ReteForza(cfg.seq_len, cfg, _att(cfg.att_forza),
                               usa_ultimi=getattr(cfg, "seq_len_forza", None),
                               zs_max=float(getattr(cfg, "zs_max", 2.0)))
    else:
        rete_forza = ReteTCN(cfg.seq_len, _att(cfg.att_forza), out="tanh",
                             usa_ultimi=getattr(cfg, "seq_len_forza", None))
    return rete_xi, rete_forza


# --- Scalatura dei TARGET tramite limiti FISICI (documentata, non arbitraria) ---
def limiti_xi(auto):
    # Stessi estremi usati dallo scheduling xi_da_r: derivati dal damper reale, non scelti.
    return auto.xi_min, auto.xi_max


def xi_a_norm(xi, auto):
    lo, hi = limiti_xi(auto); return (xi - lo) / (hi - lo)


def norm_a_xi(u, auto):
    lo, hi = limiti_xi(auto); return lo + u * (hi - lo)


def forza_a_norm(forza, cfg):
    return forza / cfg.forza_max


def norm_a_forza(u, cfg):
    return u * cfg.forza_max
