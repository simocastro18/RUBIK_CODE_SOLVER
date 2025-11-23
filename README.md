# RUBIK_CODE_SOLVER

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Kociemba](https://img.shields.io/badge/Algorithm-Kociemba-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

Un risolutore automatico per il Cubo di Rubik con interfaccia web, basato sull’algoritmo Kociemba tramite la libreria Python *kociemba*.

## Descrizione

RUBIK_CODE_SOLVER permette di inserire la configurazione del Cubo di Rubik e ottenere automaticamente una sequenza di mosse per risolverlo.  
Il backend elabora la configurazione tramite l’algoritmo Kociemba, mentre l’interfaccia web consente una gestione semplice e intuitiva.

## Caratteristiche

- Inserimento manuale o programmato dello stato del cubo  
- Risoluzione automatica basata sulla libreria **kociemba**  
- Interfaccia web user-friendly  
- Soluzioni rapide e generalmente ottimizzate  
- Struttura chiara e facilmente estendibile

## Tecnologie utilizzate

- **Python 3.x**
- **kociemba** – libreria ufficiale dell’algoritmo Two-Phase di Herbert Kociemba
- Framework web Python (es. Flask)
- HTML / CSS / JavaScript per l'interfaccia grafica
- Tutte le dipendenze sono presenti nel file `requirements.txt`

## Algoritmo di risoluzione

Il progetto utilizza direttamente la libreria **kociemba**, una delle implementazioni più performanti per la risoluzione computazionale del Cubo di Rubik.

**Two-Phase Algorithm:**
1. *Phase 1:* riduce lo stato al sotto-spazio G1  
2. *Phase 2:* calcola la soluzione finale ottimizzata  

Questo approccio produce soluzioni tipicamente brevi (circa 20–22 mosse HTM).

## Installazione

```bash
git clone https://github.com/simocastro18/RUBIK_CODE_SOLVER.git
cd RUBIK_CODE_SOLVER
pip install -r requirements.txt
python app.py
