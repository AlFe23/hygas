# HyGAS website

Prima versione locale, 16 novembre 2026. Il sito è in inglese e si rivolge agli utenti del software HyGAS.

## Aprire il sito

Apri **`index.html`** nel browser. Le pagine, i filtri dei casi, il selettore delle figure e l'ingrandimento delle immagini funzionano anche senza un server; la copia negli appunti dipende dalle autorizzazioni del browser e offre una selezione manuale come alternativa.

Per provarlo tramite HTTP, apri un terminale **in questa cartella** ed esegui:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Poi visita **http://localhost:8000/**. Ferma il server con `Ctrl+C`. Il sito non richiede l'ambiente scientifico HyGAS, Node, pacchetti npm, una build o una connessione per i propri asset. I collegamenti a GitHub, DOI e sorgenti dati richiedono rete.

I comandi scientifici mostrati nelle guide vanno eseguiti nella repository del software HyGAS, dopo aver installato il suo ambiente e ottenuto gli input. Il sito visualizza risultati preparati.

## Pagine e funzionalità

- `index.html`: presentazione del software, percorsi iniziali, workflow e anteprime dei casi.
- `get-started.html`: installazione, checklist EnMAP, comando completo e lettura dei file prodotti.
- `user-guide.html`: prodotti per sensore, modalità matched filter, SNR, output, incertezza e troubleshooting.
- `case-studies.html`: quattro casi, ricerca, filtro per sensore, schede delle figure e notebook della pubblicazione.
- `science.html`: metodo, paper, citazione copiabile, poster PDF, autori e crediti.

Le pagine sono HTML statico completo. JavaScript aggiunge menu mobile, copia del codice, filtri, tab e dialoghi per ingrandire le figure. Il testo, i collegamenti e tutte le figure dei casi rimangono consultabili con JavaScript disattivato. I percorsi sono relativi alla cartella del sito, adatti anche a un progetto GitHub Pages sotto `/nome-repository/`.

## Aggiornare i contenuti

Modifica direttamente HTML, `assets/css/styles.css` e `assets/js/site.js`. Header e footer sono presenti in ciascuna delle cinque pagine: mantienili coerenti quando cambi la navigazione.

Il catalogo dei casi e le tre anteprime in homepage sono generati da **`data/cases.json`**. Dopo aver modificato quel file o aggiunto immagini, esegui dalla cartella del sito:

```bash
python tools/render_catalog.py
python tools/check_site.py
```

Il primo comando aggiorna soltanto i blocchi marcati `FEATURED_CASES` e `CASE_CATALOG` nei due HTML. Non rimuovere quelle coppie di commenti. Gli altri contenuti delle pagine rimangono invariati. Il sito continua a funzionare come semplice HTML dopo la generazione; il browser non deve caricare il JSON tramite una richiesta separata.

I campi di un caso comprendono identificativo, località, acquisizioni, sensori, percorso scientifico, figure con unità e crediti, notebook e riferimento al paper. `date` può essere `null` per un confronto fra più date; `dateLabel` esplicita le acquisizioni.

## Fonti e accuratezza scientifica

- I notebook dei casi sono collegati al commit della pubblicazione `5b98cdcdce52c7d4d27f6260410ceae48d9bb407`.
- Le guide operative seguono il branch `develop` ispezionato durante la preparazione del sito. Aggiornale se cambiano CLI o requisiti.
- Le figure conservano pixel, etichette, unità e scale originali. I pannelli storici in ppb non sono rietichettati in ppm·m.
- La CLI produce enhancement e incertezza strumentale. Il contributo del background e l'analisi IME/flusso appartengono ai notebook downstream.
- CO₂ e retrieval EMIT Level-1B sono identificati come sperimentali.
- Le fonti delle figure, i crediti e gli hash sono in `ASSET_SOURCES.md` e `data/asset-sources.json`. I due poster PDF sono copie delle versioni fornite; il paper completo è raggiungibile dal DOI.

## Verifica

`tools/check_site.py` usa la libreria standard di Python per controllare struttura HTML, ID, link e ancore locali, riferimenti ARIA, immagini, catalogo e integrità degli asset. Il report di questa consegna è in `verification.json`. I controlli automatici non sostituiscono la prova visiva e delle interazioni in un browser.

La consegna è stata verificata anche con un server HTTP locale temporaneo, sia alla radice sia sotto `/example-project/`: tutte le cinque pagine e le 26 destinazioni locali distinte rispondono correttamente. Il risultato è in `http-verification.json`; i server di prova sono stati chiusi. La sintassi JavaScript è stata controllata con Node. Nell'ambiente di esecuzione non è disponibile un browser per la verifica visiva.

## Pubblicazione con GitHub Pages

Il sito usa la repository del progetto `AlFe23/hygas`, con i file HTML nella radice del branch dedicato `gh-pages`. Dopo il primo caricamento, in **Settings → Pages** seleziona **Deploy from a branch**, **gh-pages** e **/(root)**, quindi salva. `.nojekyll` permette di servire direttamente i file statici. L'indirizzo standard sarà `https://alfe23.github.io/hygas/`; la schermata Pages ne conferma l'attivazione.

Questa cartella è una copia di lavoro Git del solo branch `gh-pages`, collegata alla stessa repository remota del software. Per pubblicare gli aggiornamenti, modifica e verifica i file qui, registra un commit e invialo a `gh-pages`. GitHub Pages aggiornerà il sito dalla nuova versione. I branch scientifici `main`, `develop` e quello del paper conservano i propri contenuti. Le automazioni per aggiungere prodotti scientifici sono una fase separata e potranno essere introdotte in seguito.

I report `verification.json` e `http-verification.json` sono locali e ignorati da Git. Le pagine e gli asset non dipendono da questi report. I dettagli operativi sono in `PUBLISHING.md`.

La licenza GPL-3.0 del software HyGAS e le attribuzioni delle pubblicazioni, delle figure e dei dati sono distinte. Conserva i crediti già presenti durante le modifiche.
