# -*- coding: utf-8 -*-
"""
portabilita.py — Rende il progetto eseguibile su Windows, macOS e Linux senza modifiche.

Il codice e' scritto su macOS e usa simboli matematici nelle stampe (√, ≈, σ, π, ·, —) perche'
rendono le formule leggibili nel log. Su Windows quei simboli sono il problema: la codifica
predefinita per file e pipe e' cp1252, che NON contiene √ ≈ σ π ∝. Il risultato non e' un
carattere sbagliato, e' un'eccezione UnicodeEncodeError che interrompe la run — tipicamente
proprio mentre si scrive il report di diagnostica, cioe' dopo tutto il lavoro.

Quando si manifesta su Windows:
  - sempre, scrivendo diagnostica_report.txt (era open() senza encoding);
  - stampando a schermo se l'output e' REDIRETTO (python main.py > log.txt) o se il terminale
    e' in code page legacy. Su una console Windows moderna print() passa per l'API Unicode e
    funziona, ma non si puo' contare su quello.

Le altre insidie di portabilita' del progetto sono gia' gestite altrove e sono elencate qui
per non doverle ricercare:
  - PERCORSI: sempre os.path.join e BASE ricavata da __file__, mai separatori scritti a mano.
  - MULTIPROCESSING: dati.py usa get_context("spawn"), che e' l'unico metodo su Windows, la
    funzione worker sta a livello di modulo (requisito di spawn) e main.py ha la guardia
    if __name__ == "__main__" senza la quale spawn rilancerebbe il programma a ogni processo.
  - CORE PERFORMANCE: la lettura via sysctl e' dentro un ramo platform.system() == "Darwin".
  - GPU: hardware.py prova CUDA (Windows/Linux con NVIDIA), poi MPS (Apple), poi CPU, e
    verifica gli operatori prima di iniziare.
"""
import os
import platform
import sys


def forza_utf8():
    """Mette stdout/stderr in UTF-8 con errors='replace'.

    Da chiamare come PRIMA istruzione di un entry point (main.py, diagnostica.py). Non tocca
    nulla se lo stream non e' riconfigurabile (per esempio se e' gia' stato sostituito).
    errors='replace' e' voluto: se anche cosi' un carattere non passasse, si vede un '?' nel
    log invece di perdere una run di venti minuti."""
    for flusso in (sys.stdout, sys.stderr):
        riconfigura = getattr(flusso, "reconfigure", None)
        if riconfigura is not None:
            try:
                riconfigura(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass                      # stream non riconfigurabile: si prosegue comunque


def apri_testo(percorso, modo="r"):
    """open() per file di TESTO, con encoding UTF-8 esplicito.

    Da usare per tutti i file di testo del progetto (report, json). Senza encoding esplicito
    Python usa quello della locale: UTF-8 su macOS e Linux, cp1252 su Windows in italiano.
    Lo stesso file scritto su un sistema e riletto sull'altro non tornerebbe."""
    if "b" in modo:
        raise ValueError("apri_testo e' per file di testo; per i binari usa open(..., 'rb')")
    return open(percorso, modo, encoding="utf-8")


def suggerimento_ffmpeg():
    """Comando di installazione di ffmpeg per il sistema su cui si sta girando.

    ffmpeg non e' una dipendenza Python: matplotlib lo cerca come eseguibile esterno. Se manca
    l'animazione a schermo funziona comunque, non si salva solo l'MP4."""
    sistema = platform.system()
    if sistema == "Darwin":
        return "brew install ffmpeg"
    if sistema == "Windows":
        return "winget install ffmpeg   (oppure: choco install ffmpeg)"
    return "sudo apt install ffmpeg"


def backend_grafico_interattivo():
    """True se matplotlib ha un backend capace di aprire una finestra.

    Su Linux senza server grafico (o senza python3-tk / PyQt installato) matplotlib ripiega
    da solo su Agg: plt.show() non solleva nessun errore, semplicemente non succede niente e
    la finestra non si apre mai. Chi guarda la console non ha modo di distinguere "sta ancora
    calcolando" da "e' gia' finito e non mostrera' nulla" senza questo controllo."""
    import matplotlib
    return matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "template")


def descrivi_sistema():
    """Riga di intestazione con sistema, architettura e versione di Python.

    Serve nei log condivisi: quando un compagno di corso segnala un comportamento diverso, la
    prima domanda e' sempre su quale macchina girava."""
    return (f"{platform.system()} {platform.release()} / {platform.machine()} / "
            f"Python {sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro} / codifica stampe "
            f"{getattr(sys.stdout, 'encoding', 'ignota')}")
