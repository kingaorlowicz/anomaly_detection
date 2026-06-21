# Visual Anomaly Detection Prototype

Prototyp porownuje klasyczna detekcje anomalii na cechach numerycznych z
modelem wizualnym klasyfikujacym wygenerowane obrazy wykresow.

## Uruchomienie w Jupyter Notebook

Najwygodniejsza sciezka testowa jest w pliku:

```text
notebooks/run_full_pipeline.ipynb
```

Uruchom Jupyter z katalogu projektu:

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook
```

Potem otworz `notebooks/run_full_pipeline.ipynb` i wykonuj komorki od gory do
dolu. Notebook tworzy osobny testowy katalog `dataset_notebook_demo`, trenuje
baseline numeryczny, a jesli jest zainstalowany PyTorch, trenuje tez model CNN.

## Szybki start z terminala

```bash
python data_generator.py --output-dir dataset --overwrite
python numerical_baseline.py --dataset-dir dataset --model-out artifacts/numerical_baseline.pkl
python visual_anomaly_detector.py --dataset-dir dataset --model-out artifacts/visual_detector.pt --epochs 8
python evaluation.py --dataset-dir dataset --train-missing
```

Domyslnie generator balansuje klasy osobno w kazdym splicie. Oznacza to, ze
`dataset/train`, `dataset/val` i `dataset/test` maja po 50% obrazow `normal`
oraz 50% obrazow `anomaly`. Szczegoly sa zapisywane w
`dataset/balance_summary.csv`.

Jesli chcesz wymusic konkretna liczbe obrazow na klase w kazdym splicie:

```bash
python data_generator.py --windows-per-class-per-split 500 --overwrite
```

Jesli chcesz wylaczyc balansowanie i zapisac wszystkie okna:

```bash
python data_generator.py --no-balance-classes --overwrite
```

W srodowisku bez Matplotlib generator moze uzyc fallbacku:

```bash
python data_generator.py --plot-backend pil --overwrite
```

`robust_threshold` jest lekkim fallbackiem numerycznym bez scikit-learn:

```bash
python numerical_baseline.py --model-type robust_threshold
```

## Jak dziala projekt

1. `data_generator.py` tworzy syntetyczne szeregi czasowe: trend, sezonowosc i
   szum. Nastepnie wstrzykuje znane anomalie: skoki, spadki, anomalie
   kontekstowe oraz zmiany wariancji.
2. Te same dane sa zapisywane dwojako: jako tabela `numeric_data.csv` z flaga
   `is_anomaly` oraz jako obrazy PNG wykresow w katalogach
   `train/val/test/normal/anomaly`.
3. Przed renderowaniem obrazow generator zbiera pule okien i losuje rowna
   liczbe okien normalnych i anomalnych w kazdym splicie. Dzieki temu trening,
   walidacja i test nie sa zaburzone przez niezbalansowane klasy.
4. `numerical_baseline.py` nie widzi obrazow. Dzieli szereg na okna i liczy
   cechy statystyczne, np. srednia, odchylenie, zakres, IQR, trend i skoki.
   Na tych cechach trenuje klasyczny detektor.
5. `visual_anomaly_detector.py` nie widzi wartosci liczbowych. Dostaje tylko
   obraz wykresu i klasyfikuje go jako `normal` albo `anomaly`.
6. `evaluation.py` porownuje oba podejscia na tych samych oknach testowych,
   liczy Accuracy, Precision, Recall, F1 i ROC-AUC, a potem zapisuje przypadki,
   w ktorych jedna metoda wygrala z druga.

Generator zapisuje:

- `dataset/numeric_data.csv` - pelne szeregi czasowe z flaga `is_anomaly`.
- `dataset/window_metadata.csv` - etykiety okien i sciezki do obrazow.
- `dataset/balance_summary.csv` - ile okien bylo dostepnych i ile wybrano.
- `dataset/{train,val,test}/{normal,anomaly}` - obrazy PNG wykresow.
