# -*- coding: utf-8 -*-
"""
hardware.py — Identifica il PC: sceglie il device di calcolo (GPU MPS su Apple Silicon,
altrimenti CUDA, altrimenti CPU) e il numero di PERFORMANCE core (salta gli efficiency
core su Apple Silicon). Serve a usare GPU per l'addestramento e i core per la fisica.
"""
import os
import torch


def rileva_hardware(cfg):
    """Restituisce (device, n_core) e imposta i thread torch per la CPU."""
    if cfg.usa_gpu and torch.cuda.is_available():
        device = torch.device("cuda"); nome = "CUDA: " + torch.cuda.get_device_name(0)
    elif cfg.usa_gpu and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        device = torch.device("mps"); nome = "Apple Silicon GPU (MPS)"
    else:
        device = torch.device("cpu"); nome = "CPU"

    perf = eff = None
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import subprocess
        def leggi(chiave):
            try:
                return int(subprocess.check_output(["sysctl", "-n", chiave]).decode().strip())
            except Exception:
                return None
        perf = leggi("hw.perflevel0.logicalcpu")   # core "performance"
        eff = leggi("hw.perflevel1.logicalcpu")    # core "efficiency" (li saltiamo)

    totali = os.cpu_count() or 1
    n_core = perf if perf else totali
    if cfg.n_core_forza > 0:
        n_core = cfg.n_core_forza
    torch.set_num_threads(max(1, n_core))

    print(f"    HW: {nome}")
    if perf:
        print(f"    Core: {totali} totali -> uso {n_core} performance (salto {eff} efficiency)")
    else:
        print(f"    Core: uso {n_core}")

    device = verifica_operatori(device, cfg)
    return device, n_core


def verifica_operatori(device, cfg):
    """Prova SUBITO gli operatori che la rete usera' davvero, su questo device.

    Perche' esiste. La copertura del backend MPS varia con la versione di torch, e un
    operatore mancante non degrada in silenzio: solleva un'eccezione. Scoprirlo dopo venti
    minuti di generazione etichette e' inutilmente costoso. Qui si costruisce una rete
    minuscola con gli stessi strati (Conv1d dilatate, BatchNorm1d, la riduzione depthwise,
    Linear, sigmoid/tanh), si fa una forward e una backward, e se qualcosa manca si ricade
    su CPU dicendolo chiaramente invece di far fallire la run a meta'."""
    import torch.nn as nn
    from reti import MediaMobile
    prova = nn.Sequential(
        nn.Conv1d(1, 8, 3, padding=2, dilation=2), nn.BatchNorm1d(8), nn.SiLU(),
        MediaMobile(8), nn.Flatten(), nn.Linear(8 * 8, 1), nn.Sigmoid(),
    )
    try:
        prova = prova.to(device)
        x = torch.zeros(4, 1, 16, device=device, requires_grad=True)
        y = prova(x).sum()
        y.backward()                              # anche il passo indietro, non solo avanti
        _ = float(y.detach().cpu())
        print(f"    Operatori verificati su {device} (conv dilatate, batchnorm, "
              f"riduzione depthwise, forward+backward)")
        return device
    except Exception as e:
        print(f"    [!] {device} non supporta tutti gli operatori richiesti: {type(e).__name__}: "
              f"{str(e)[:160]}")
        print("        Ripiego su CPU. Per forzarlo sempre: usa_gpu = False in config.")
        torch.set_num_threads(max(1, torch.get_num_threads()))
        return torch.device("cpu")
