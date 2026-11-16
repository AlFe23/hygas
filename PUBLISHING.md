# Pubblicare e aggiornare il sito HyGAS

Il progetto è un sito statico: HTML, CSS, JavaScript, figure e due poster PDF. Non richiede build sul server.

## Destinazione

- Repository: `AlFe23/hygas`, la stessa del software scientifico.
- Branch di pubblicazione: `gh-pages`.
- Cartella: `/(root)`.
- Modalità Pages: **Deploy from a branch**.
- Indirizzo standard atteso, dopo l'attivazione: `https://alfe23.github.io/hygas/`.

La schermata **Settings → Pages** e la risposta effettiva dell'indirizzo pubblico confermano l'avvenuta pubblicazione. La sola presenza dei file nella repository non attiva automaticamente un sito Pages.

## Primo caricamento

La cartella del sito è una copia di lavoro Git del branch dedicato `gh-pages`, con lo stesso remote del software scientifico. Questo branch contiene le pagine del sito; `main`, `develop` e i branch delle pubblicazioni conservano il codice e gli snapshot scientifici. Esegui i comandi Git dalla cartella `hygas-website/` e verifica il percorso prima di aggiungere file.

```bash
git status
git remote -v
git push -u origin gh-pages
```

Il remote `origin` punta a `git@github.com:AlFe23/hygas.git`. Il nome della cartella locale `hygas-website` distingue questa copia di lavoro da quella dei branch scientifici.

## Attivazione di Pages

1. Apri la repository su GitHub.
2. Vai in **Settings → Pages**.
3. In **Build and deployment**, scegli **Deploy from a branch**.
4. Scegli **gh-pages** e **/(root)**, poi **Save**.
5. Attendi il completamento della pubblicazione e usa il collegamento fornito da GitHub.

La cartella radice contiene `index.html` e `.nojekyll`. Tutti i percorsi dei contenuti sono relativi, compatibili con il percorso del progetto GitHub Pages.

## Aggiornamenti

Per aggiornare i casi, modifica `data/cases.json`, aggiungi gli asset con didascalia e fonte, poi rigenera il catalogo. Per modifiche ai soli testi o allo stile, modifica direttamente i rispettivi file.

```bash
python tools/render_catalog.py
python tools/check_site.py
git status
```

Dopo la revisione, aggiungi i file modificati, crea un commit e invialo:

```bash
git add index.html case-studies.html data assets
git commit -m "Update HyGAS website content"
git push origin gh-pages
```

Adatta l'elenco dei file alle modifiche effettive. Il comando di esempio non include automaticamente le altre guide o la documentazione.

Per un controllo locale apri `index.html` oppure esegui `python -m http.server 8000 --bind 127.0.0.1` e visita `http://localhost:8000/`. I report generati dai controlli sono ignorati da Git.

## Risorse di riferimento

- [Configurare la sorgente di pubblicazione](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
- [Consultare gli esiti delle esecuzioni](https://docs.github.com/en/actions/how-tos/monitor-workflows/view-workflow-run-history).

Le Actions per pubblicare nuovi prodotti scientifici potranno usare lo stesso catalogo JSON. Questa prima pubblicazione usa la modalità standard di GitHub Pages.
