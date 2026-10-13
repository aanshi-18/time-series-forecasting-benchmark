# time-series-forecasting-benchmark
Time-Series Forecasting Benchmark
This project compares statistical models and neural networks on two forecasting problems: monthly champagne sales and hourly bike-sharing demand. The aim is to see how their performance changes across datasets and forecast horizons, rather than assume that a more complex model will always work better.
[Read the whitepaper](paper/whitepaper.pdf) 
What is compared?
The benchmark includes Seasonal Naive, SARIMA, ARIMAX, LSTM, GRU, CNN-LSTM, and two Transformer encoders: one without positional information and one with learned positional embeddings. Every configuration is evaluated on both datasets.
- Champagne sales: 105 monthly observations, with four 12-month test blocks. Horizons are 1, 3, 6 and 12 months.
- Bike sharing: 17,379 records, with five 168-record test blocks. Horizons are 1, 6, 24 and 48 records. Some hours are absent, so these are record-based horizons rather than exact elapsed-hour intervals.
The evaluation uses chronological splits and expanding neural training histories. A separate neural model is trained for each horizon, using seeds 42, 43 and 44. Errors are reported in the original target units.
Main results
These are the configurations with the lowest mean absolute error (MAE) in the reported evaluation:
Dataset	Horizon	Lowest mean MAE	Model
Champagne	1 month	635.74	SARIMA
Champagne	3, 6 and 12 months	641.46 at each horizon	Seasonal Naive
Bike sharing	1 record	31.75	GRU
Bike sharing	6 records	52.52	GRU
Bike sharing	24 records	67.10	LSTM
Bike sharing	48 records	74.90	Seasonal Naive


Lower MAE means a smaller average absolute error. The table shows means; the full results include neural seed variability, other metrics and per-fold errors.
Adding positional embeddings reduced mean MAE for the tested Transformer at every evaluated horizon on both datasets. GRU also had fewer parameters than LSTM. Neither observation establishes a universal architecture ranking or a measured runtime advantage.
Important evaluation limitations
The multi-step evaluation has a timing issue. Each fold fits a model through the training boundary, but the first h - 1 test targets at horizon h have forecast origins before that boundary. Their input windows end at the right origin, yet fitting can use observations that were not available then. Test targets are excluded from fitting. One-step results avoid this particular issue, and its effect on the multi-step scores was not measured.
Other differences matter too:
- Bike SARIMA and ARIMAX use only the latest 720 training records, while neural models use much longer histories.
- Bike ARIMAX receives actual future covariates, including weather. It is an oracle benchmark, not a deployable historical-only forecast.
- Champagne SARIMA failed to converge in its first fold; some bike neural runs reached the epoch limit.
- Hyperparameters were not systematically tuned. The DM-type comparisons are exploratory and are not adjusted for multiple comparisons.
The paper discusses these limitations. Results describe the tested configurations under this setup; they should not be read as general model-selection rules.
Repository contents
run_walkforward.py       Experiment and result-export code
results/summary.txt      Reported metrics and diagnostics
results/metrics_long.csv Per-seed and per-fold metrics
paper/whitepaper.pdf     Paper for reading
paper/whitepaper.tex     Editable LaTeX source
Running the experiment
The script requires Python, NumPy, pandas, TensorFlow, statsmodels and SciPy. The recorded run used Python 3.13.15, TensorFlow 2.20.0 and statsmodels 0.15.0. Training the neural models can take several hours; a GPU is recommended.
Place the champagne CSV and hour.csv (or its ZIP) in a data/ folder, then run:
python run_walkforward.py --data-dir data --out-dir results
If interrupted, resume using the same settings and output folder:
python run_walkforward.py --data-dir data --out-dir results --resume
Use a fresh output folder if results/results.json already exists from a different experiment. Only resume checkpoints created with matching settings. The --quick and --dry-run options are diagnostic modes; their outputs are not the paper's results.
Data sources
- UCI Bike Sharing dataset
- Time Series Data Library, which distributes the Perrin Frères champagne series. The exact primary collection source and sales units still need confirmation; the paper therefore uses source-file sales units.
Raw datasets and partial checkpoints are not included in this repository.
