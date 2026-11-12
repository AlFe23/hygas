#!/usr/bin/env python3
"""Save an Earthdata token privately for future HyGAS download sessions."""
from __future__ import annotations

import argparse
import getpass
from pathlib import Path
import sys
import warnings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.downloads.credentials import DEFAULT_TOKEN_FILE, read_token, save_token


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Replace the existing token, e.g. after expiration")
    parser.add_argument("--check", action="store_true", help="Check local file configuration without showing the token or contacting NASA")
    args = parser.parse_args(argv)
    try:
        if args.check:
            read_token(DEFAULT_TOKEN_FILE)
            print(f"Token locale configurato: {DEFAULT_TOKEN_FILE}")
            print("Controllo del file completato; la validità NASA viene verificata al login.")
            return 0
        if DEFAULT_TOKEN_FILE.exists() and not args.replace:
            print("File già presente. Usa --check per controllarlo o --replace per aggiornare il token.", file=sys.stderr)
            return 2
        # Refuse terminals that cannot hide input instead of falling back to echoed input.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            token = getpass.getpass("Incolla il token Earthdata e premi Invio (input nascosto): ")
        path = save_token(token, DEFAULT_TOKEN_FILE, replace=args.replace)
        print(f"Token salvato in {path}")
        if sys.platform.startswith("linux"):
            print("Permessi del file: 600 (lettura/scrittura solo per il tuo utente).")
        print("I downloader HyGAS lo leggeranno automaticamente anche nei nuovi terminali.")
        print("Una variabile EARTHDATA_TOKEN già esportata ha precedenza sul file.")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("Configurazione annullata.", file=sys.stderr)
        return 2
    except getpass.GetPassWarning:
        print("Esegui questo comando da un terminale interattivo che supporti input nascosto.", file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print("Configurazione non riuscita: controlla file, permessi e che il token sia una singola riga non vuota.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
