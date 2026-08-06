# -*- coding: utf-8 -*-
"""
dati.py — Caricamento tracce reali, costruzione finestre per il ML e NORMALIZZAZIONE
degli ingressi calcolata SOLO dai dati di training.

Perche' cosi': normalizzare "a mano" (es. dividere per 30) e' arbitrario. Il modo
corretto e' standardizzare con MEDIA e DEVIAZIONE STANDARD che EMERGONO dal training,
e riusare le stesse statistiche su validazione e in inferenza.
"""
import os
import glob
import pickle
import numpy as np

from fisica import rimuovi_offset, despica_hampel


def _uniforma(t, x, cfg, nome="", tolleranza=0.02):
    """Riporta (t, x) su griglia uniforme a cfg.freq_campion se la cadenza vera non coincide.

    Restituisce (t_uniforme, x_ricampionato). Se la cadenza misurata e' gia' quella attesa
    entro 'tolleranza' e il campionamento e' regolare, non tocca nulla.

    Il jitter si misura come deviazione standard degli intervalli rapportata al passo
    medio: un IMU da smartphone raramente e' perfettamente regolare, e un 5% di jitter
    basta a spostare la banda dei filtri."""
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return t, np.asarray(x, dtype=float)
    dt = np.diff(t)
    dt_medio = float(np.median(dt))
    fs_vera = 1.0 / dt_medio if dt_medio > 0 else float("nan")
    jitter = float(np.std(dt) / dt_medio) if dt_medio > 0 else float("inf")

    scarto = abs(fs_vera - cfg.freq_campion) / cfg.freq_campion
    if scarto <= tolleranza and jitter <= 0.05:
        return t, np.asarray(x, dtype=float)

    n = int(np.floor((t[-1] - t[0]) * cfg.freq_campion)) + 1
    t_u = t[0] + np.arange(n) / cfg.freq_campion
    x_u = np.interp(t_u, t, np.asarray(x, dtype=float))
    print(f"    [!] {nome}: cadenza reale {fs_vera:.1f} Hz (jitter {jitter:.1%}) contro "
          f"cfg.freq_campion = {cfg.freq_campion:.0f} Hz")
    print(f"        ricampionata su griglia uniforme: {len(t)} -> {n} campioni. "
          f"Senza questo, filtri e integrazioni lavorerebbero alla frequenza sbagliata.")
    return t_u, x_u


def carica_tracce(cfg):
    """Scarica il dataset road-quality (kagglehub) e lo divide in 2 tracce di TRAINING e
    1 di VALIDAZIONE (mai vista). Ogni traccia e' una coppia (a_z, v). Se il dataset non
    c'e', genera 3 tracce sintetiche distinte per far girare comunque il codice."""
    try:
        import kagglehub
        cartella = kagglehub.dataset_download("nickkotarelas/road-quality-dataset")
        files = sorted(glob.glob(os.path.join(cartella, "**", "*.pkl"), recursive=True))
    except Exception:
        files = []

    train, val = [], []
    if len(files) >= 3:
        print(f"  [ok] {len(files)} tracce reali -> {len(files) - 1} train / 1 validazione "
              f"({os.path.basename(files[-1])})")
        for gruppo, dest in ((files[:-1], train), ([files[-1]], val)):
            for pkl in gruppo:
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                imu = d["imu"]
                t_imu = np.array(imu["time"]["rel"], dtype=float)
                # sceglie l'asse accelerometrico con media assoluta maggiore (il verticale)
                medie = {k: abs(np.mean(vv)) for k, vv in imu["accel"].items()}
                asse = max(medie, key=medie.get)
                az_grezza = np.array(imu["accel"][asse], dtype=float)

                # RICAMPIONAMENTO su griglia UNIFORME a cfg.freq_campion.
                # Tutto il resto della pipeline (filtri di Butterworth, doppia integrazione,
                # ponderazione ISO 2631, RK4) assume passo costante 1/freq_campion. Se l'IMU
                # registra a una cadenza diversa o irregolare — cosa normale su uno
                # smartphone — quelle ipotesi saltano in silenzio: i filtri tagliano alla
                # frequenza sbagliata e le integrazioni sbagliano scala. Qui si misura la
                # cadenza vera e, se serve, si riporta il segnale su griglia regolare.
                t_imu, az_grezza = _uniforma(t_imu, az_grezza, cfg,
                                             nome=os.path.basename(pkl))

                az = rimuovi_offset(az_grezza, cfg.freq_campion)
                if cfg.despica:                     # toglie i picchi anomali del sensore (Hampel)
                    az = despica_hampel(az, cfg.hampel_finestra, cfg.hampel_sigma)
                t_gps = np.array(d["gps"]["time"]["rel"], dtype=float)
                v_gps = np.array(d["gps"]["speed"], dtype=float)
                v = np.interp(t_imu, t_gps, v_gps)     # velocita' allineata all'IMU
                # None = strada NON nota (va ricostruita); poi il nome del file, cosi' ogni
                # numero stampato o messo in figura si puo' ricondurre alla sua origine
                dest.append((az, v, None, os.path.basename(pkl)))
    else:
        print("  [!] Dataset reale assente: genero 3 tracce sintetiche (2 train / 1 val).")
        for i in range(3):
            rng = np.random.default_rng(42 + i)
            N = 9000
            t = np.linspace(0, 90, N)
            v0 = [8.0, 15.0, 26.0][i]
            v = np.clip(v0 + 7.0 * np.sin(2 * np.pi * (0.01 + 0.004 * i) * t) + rng.normal(0, 0.2, N), 3.0, 33.0)
            f_strada = 0.6 + 0.9 * v / 10.0
            az = np.sin(2 * np.pi * f_strada * t) * (0.7 + 0.05 * v) + rng.normal(0, 0.15, N)
            (train if i < 2 else val).append((az, v, None, f"sintetica_fallback_{i+1}"))
    return train, val


def _lavora_una_traccia(args):
    """Genera le etichette di UNA traccia (funzione a livello di modulo per il multiprocessing)."""
    auto, cfg, az, v, zr_dot = args
    from controllo import GeneratoreEtichette
    return GeneratoreEtichette(auto, cfg).genera(az, v, zr_dot_vero=zr_dot)


def _spacchetta(t):
    """Una traccia e' (a_z, v[, z_r'[, nome]]).

    z_r' presente = strada VERA nota (tracce sintetiche): va usata quella, perche'
    ricostruirla da a_z sposta xi* di 0.064 in media. 'nome' serve alla tracciabilita':
    ogni numero stampato deve poter essere ricondotto alla traccia da cui viene."""
    return (t[0], t[1], t[2] if len(t) > 2 else None)


def nome_traccia(t, default=""):
    """Nome leggibile di una traccia, se lo porta con se'."""
    return t[3] if len(t) > 3 else default


def costruisci_finestre(tracce, auto, cfg, n_worker=0):
    """Per ogni traccia genera le etichette (in parallelo sui core, le tracce sono
    indipendenti) e taglia le finestre scorrevoli per il ML:
        finestra_az[i] = ultimi seq_len campioni di a_z (l'ingresso temporale)
        vel[i]         = velocita' corrente
        target_xi[i], target_forza[i] = etichette all'istante i
        target_kp[i], target_zs[i]    = i due FATTORI della forza (F* = clip(-kp* * z_s')),
                                        usati come target ausiliari dalla rete forza
        rapporto_r[i]  = frequenza r all'istante i (solo per le tabelle)
    Restituisce array numpy + statistiche comfort (rms passiva, rms ideale) per traccia."""
    if n_worker and n_worker > 1 and len(tracce) > 1:
        try:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(min(n_worker, len(tracce))) as pool:
                uscite = pool.map(_lavora_una_traccia,
                                  [(auto, cfg) + _spacchetta(t) for t in tracce])
        except Exception as e:
            print(f"    [i] parallelo non riuscito ({e}); uso seriale")
            from controllo import GeneratoreEtichette
            gen = GeneratoreEtichette(auto, cfg)
            uscite = [gen.genera(*_spacchetta(t)) for t in tracce]
    else:
        from controllo import GeneratoreEtichette
        gen = GeneratoreEtichette(auto, cfg)
        uscite = [gen.genera(*_spacchetta(t)) for t in tracce]

    finestra_az, vel, target_xi, target_forza, rapporto_r = [], [], [], [], []
    target_kp, target_zs = [], []          # fattori della forza: F* = clip(-kp* * z_s')
    comfort = []
    for t, out in zip(tracce, uscite):
        az, v, _zr = _spacchetta(t)
        comfort.append((np.sqrt(np.mean(out["acc_passiva"] ** 2)),
                        np.sqrt(np.mean(out["acc_ideale"] ** 2))))
        passo = max(1, int(getattr(cfg, "passo_finestre", 1)))
        for i in range(cfg.seq_len, out["N"], passo):
            if v[i] < 2.0:                    # scarta l'auto quasi ferma
                continue
            finestra_az.append(az[i - cfg.seq_len:i])
            vel.append(v[i])
            target_xi.append(out["xi"][i])
            target_forza.append(out["forza"][i])
            target_kp.append(out["kp"][i])
            target_zs.append(out["vel_cassa"][i])
            rapporto_r.append(out["r"][i])
    return (np.array(finestra_az, dtype=np.float32), np.array(vel, dtype=np.float32),
            np.array(target_xi, dtype=np.float32), np.array(target_forza, dtype=np.float32),
            np.array(rapporto_r, dtype=np.float32), comfort,
            np.array(target_kp, dtype=np.float32), np.array(target_zs, dtype=np.float32))


class Normalizzatore:
    """Standardizza gli ingressi delle reti (a_z e v) con MEDIA e DEV.STD calcolate
    SOLO dal training. I numeri di normalizzazione EMERGONO dai dati, non sono fissati a
    mano; le stesse statistiche si applicano poi a validazione e closed-loop.

    IMPORTANTE: si SALVA su file insieme alle reti (salva/carica). Ricostruirlo "a mano"
    con numeri copiati a occhio falsa ogni diagnostica closed-loop, perche' sposta la
    finestra di a_z rispetto a quella su cui la rete e' stata addestrata."""

    def __init__(self, finestra_az_train, vel_train):
        self.media_az = float(np.mean(finestra_az_train))
        self.std_az = float(np.std(finestra_az_train) + 1e-8)
        self.media_v = float(np.mean(vel_train))
        self.std_v = float(np.std(vel_train) + 1e-8)

    def na(self, az):
        """Normalizza l'accelerazione a_z (finestra o valore)."""
        return (az - self.media_az) / self.std_az

    def nv(self, v):
        """Normalizza la velocita' v."""
        return (v - self.media_v) / self.std_v

    # --- persistenza (le statistiche devono viaggiare con i .pt) ---
    def salva(self, percorso):
        import json
        from portabilita import apri_testo
        with apri_testo(percorso, "w") as f:
            json.dump({"media_az": self.media_az, "std_az": self.std_az,
                       "media_v": self.media_v, "std_v": self.std_v}, f, indent=2)
        return percorso

    @classmethod
    def da_valori(cls, media_az, std_az, media_v, std_v):
        obj = cls.__new__(cls)
        obj.media_az = float(media_az); obj.std_az = float(std_az)
        obj.media_v = float(media_v); obj.std_v = float(std_v)
        return obj

    @classmethod
    def carica(cls, percorso):
        import json
        from portabilita import apri_testo
        with apri_testo(percorso) as f:
            d = json.load(f)
        return cls.da_valori(d["media_az"], d["std_az"], d["media_v"], d["std_v"])

    def __repr__(self):
        return (f"Normalizzatore(a_z: media={self.media_az:+.4f} std={self.std_az:.4f} | "
                f"v: media={self.media_v:.3f} std={self.std_v:.3f})")
