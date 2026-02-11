# Monte Carlo — Supply Chain (ABC)

UI Streamlit pour paramétrer et lancer des simulations Monte Carlo à partir des notebooks:
- `MonteCarlo_ABC_SupplyChain_AnnualUniform_OptionB.ipynb` (stocks matières)
- `abc_ramp_simulation_updated.ipynb` (ramp capacité + marché)

## Pré-requis
- Python 3.12+ recommandé (Python 3.13 fonctionne si tes dépendances le supportent).

## Installation (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Lancer l’UI
```powershell
streamlit run app.py
```

## Tests
```powershell
pip install -r requirements-dev.txt
pytest
```

## Notes modèle
- Modèle mensuel avec politique \((R,S)\) : réceptions \(\rightarrow\) revue/commande \(\rightarrow\) consommation (stock fin de mois).
- Lead times en mois (fractionnaire autorisé). Convention : commande en début de mois, et si \(LT < 1\) mois alors la livraison arrive dans le même mois (décalage 0).
- La production annuelle est tirée selon une loi **Uniforme** entre `annual_min` et `annual_max` (Option B: tirage par année).
