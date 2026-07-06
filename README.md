# Bitcoin Halving Cycle Research

> Exploratory research on the temporal structure of Bitcoin halving cycles: is there a **universal cycle shape** φ(t) once each cycle is rescaled by its length L(n) and its amplitude A(n)?

**[Versión en español →](README.es.md)**

![Model overview](figures/modelo_avanzado_latest.png)

## The model

Each halving cycle `n` (1 = 2012, 2 = 2016, 3 = 2020, 4 = 2024) is treated as an instance of the same underlying curve:

```
Y_n(d) = A(n) · φ(d / L(n)) + ε
```

| Term | Meaning | Behavior |
|---|---|---|
| `Y_n(d)` | log of price indexed to the halving day (`halving = 100x`) | observed |
| `L(n)` | cycle length in days | grows with n, converges to the protocol limit L∞ = 210,000 blocks × 10 min ≈ 1,458 days |
| `A(n)` | cycle amplitude in log space | decays with n |
| `φ(t)` | universal normalized shape on t ∈ [0,1], φ(0)=0, max φ = 1 | shared by all cycles |

If L(n) and A(n) are modeled correctly, the four normalized curves should collapse onto a single shape (Panel A of the figure). Censoring is handled inside the model: the ongoing 2024 cycle is right-truncated, and its amplitude is estimated from the observed portion rather than from a (biased) partial maximum.

## Methodology

- **Data** — CoinMetrics community API (2012–2014) spliced with Yahoo Finance (2014–today), weekly closes.
- **Shape φ** — normalized curves are bin-averaged over a common time grid (80 bins in t, first within each cycle, then across cycles), then fitted with a cubic smoothing spline whose penalty λ is chosen by Generalized Cross-Validation (`scipy.interpolate.make_smoothing_spline`).
- **Amplitudes A(n)** — alternating least squares: given φ, each A(n) is a regression through the origin `A = Σφ·Y / Σφ²`; given the A(n), φ is refitted. This resolves the right-censoring of the 2024 cycle naturally.
- **Uncertainty** — circular block bootstrap of the residuals (8-week blocks), with a full model refit on every replicate (N = 1,000). The 80% band is conditional on the model being correct.
- **Amplitude decay** — power-law and exponential fits of A(n)·L(n) are reported as *separate scenarios*: with only 3 complete cycles they are indistinguishable in-sample and diverge out-of-sample.

## Out-of-sample validation

The model is trained on cycles 2012 + 2016 and predicts the **full 2020 cycle**. The amplitude is fitted by regression for every candidate, so the comparison isolates the quality of the *shape* — which is what the model claims to contribute:

| Candidate shape | RMSE | MAE | R² | r |
|---|---|---|---|---|
| **Model φ̂** | **0.2447** | 0.1829 | 0.4208 | 0.6805 |
| Straight line φ(t) = t | 0.3368 | — | — | — |
| Rescaled 2016 cycle | 0.3114 | — | — | — |

## Key empirical findings (as of July 2026, day d+808 of the 2024 cycle)

- Cycles 2012–2020 collapse reasonably well onto a single normalized shape.
- Estimated amplitudes: **A(n) = 4.64, 3.00, 1.93, 0.44**. The historical decay pattern (power-law / exponential fits on the first three cycles) predicted A(4) ≈ 1.25–1.69; the observed data give **0.44** — the 2024 cycle is far weaker than the pattern implied, which strains the universality hypothesis it was designed to test.
- Under the exponential scenario, the cycle "area" A(n)·L(n) falls below 1% of its initial value around cycle ~13; under the power-law scenario it decays slowly and never reaches zero. The data cannot distinguish between the two.

## Running it

```bash
pip install -r requirements.txt
python modelo_avanzado.py
```

The script downloads the data, fits the model, prints a full console report and writes the four-panel figure to `figures/`. `base.py` contains the simple historical-average baseline the model is compared against.

## Limitations

- Only **3 complete halving cycles** exist (plus one partial). The sample size is extremely small and every extrapolation should be read accordingly.
- Halvings may not be the causal driver of the observed cycles — macro liquidity cycles, adoption waves and reflexivity are plausible confounders.
- The parameters L(n), A(n) may not be stable in future cycles; the 2024 cycle already deviates strongly from the historical amplitude pattern.
- The bootstrap band is conditional on the model specification and understates total uncertainty.
- The model must be considered **exploratory**, not predictive.

## Disclaimer

This project is exploratory research on the temporal structure of Bitcoin halving cycles. It does **not** claim to predict future prices and does **not** constitute financial advice.

Part of the mathematical formulation and the modeling process was developed iteratively with the assistance of AI models (Claude, by Anthropic). The author understands the conceptual logic, the hypotheses and the validation procedures implemented, while some of the more advanced statistical formulations remain under study and refinement. The design notes used during that AI-assisted process are preserved in [`CLAUDE.md`](CLAUDE.md) for transparency.

## License

[MIT](LICENSE)
