# Analyst Recommendations → Portfolio

Point-in-time multi-agent research system | Langflow 1.12.0 | GPT-5.6 Luna

## Research question and implementation

Can a fundamental, technical and news analyst team explain analyst recommendations, improve Buy/Hold/Sell calls, and feed an investable Markowitz portfolio? This implementation combines the role-based recommendation design of Multi-Agent-Analyst with the screening-to-portfolio design of portfolio-construction. It is a new Langflow implementation, not a verbatim merge of their CrewAI code.

The actual flow has Chat Input, dated evidence, three specialist nodes, a two-pass manager, Markowitz construction, research formatting and Chat Output. Every specialist and both manager passes use the requested `gpt-5.6-luna`; numeric data preparation and optimization run in Python. External browsing and model tools are disabled during financial inference. The agent can cite only supplied evidence IDs.

## Development, freezing and holdout

The model knowledge cutoff is **2026-02-16**. Only `open/` is used for fitting and selection. The frozen ridge return prior has **897** stock-date observations across **13** development decisions, ending with labels on **2026-02-04**. The last three development dates are a chronological validation block; earlier fitting rows are purged unless their outcome horizon matures before validation starts. Ridge alpha candidates are 10, 100 and 1000; selected alpha is **100**. Medians, scaling and coefficients are then fitted on the full allowed development period. Strategy hyperparameters and prompts are fixed before inspecting holdout performance. GPT foundation-model weights are unchanged; development results are not claimed as out-of-sample evidence.

Frozen artifact SHA-256: `d1a1a8f82abe529544dd4a0e933eee8426a2e24b4077e15570b2a9e7eb5c260d`. Source parquet hashes, all coefficients, configuration and training dates are saved in `artifacts/frozen_model.json`. Test reads do not fit parameters or change prompts in response to outcomes. Individual API responses are cached using a hash of model, exact instructions, schema and dated evidence; retries after structural/citation validation failure are logged, rather than replaced by fabricated ratings.

**Main test dates:** 2026-03-02, 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-03, 2026-09-01. The fixed supplied IT sector has 70 eligible stocks at the first test date. No Health Care outcomes are used in calibration. The `test_conditional` October date is excluded from headline results because the source report marks it conditional. Separate CLI switches support conditional and transfer runs, without relabeling them as the main test.

## Information available at a decision

Fundamentals must have `available_date < decision_date`; news must have `usable_date < decision_date`, a usable original text version, and no analyst-action flag. News uses the last 90 calendar days and up to eight most recent events. Fundamentals older than 180 days and valuations older than seven days are omitted. Technical features use prices through the decision close; explanation-only mode uses the previous session close. The three specialists and independent manager do not receive ratings, price targets, EPS estimates/revisions, ticker symbols, or future returns. IDs are gvkeys; company names can still appear inside news, so anonymization is incomplete.

Only the second manager sees IBES context: consensus definition B, 12-month maximum age measured from the **usable date**, at least three brokers. Confirmation-date aging is avoided. Current FY1 EPS snapshots must be at most 35 days old; small-base revisions/surprises are excluded. Stale source coverage is visible and missing consensus is not treated as Hold. Public current identities and a fixed 2026-03-01 universe follow the supplied files; this is not a historically reconstructed constituent backtest.

## Recommendation evaluation

Entry is the close of the first session after the decision; exit is the close 21 sessions after entry. Returns exclude the entry day's return. Primary success is sector relative: Buy succeeds above zero, Sell below zero, Hold inside ±1%. The sector return is the equal-weight mean of valid outcomes in the same eligible universe. Three-way classification instead uses a ±1% Hold band. Directional hit rate excludes Hold. Matched comparisons restrict both methods to stock-dates with an available IBES consensus. Missing outcome horizons are retained as invalid records, not silently treated as zero.

|Method|Valid calls|Hit rate|Directional hit|Matched hit|3-way accuracy|
|---|---|---|---|---|---|
|Independent agent|490|39.59%|50.26%|40.04%|36.94%|
|Agent with IBES|490|38.98%|50.68%|39.41%|36.53%|
|IBES consensus|472|37.08%|47.35%|37.08%|34.11%|

- Independent: paired hit-rate difference vs IBES +2.97%; date-block bootstrap 95% interval [-4.02%, +9.79%]. Wins / ties / losses: 18.43% / 66.10% / 15.47%.
- With IBES: paired hit-rate difference vs IBES +2.33%; date-block bootstrap 95% interval [-4.86%, +9.81%]. Wins / ties / losses: 18.64% / 65.04% / 16.31%.

The confidence interval resamples whole decision-date blocks, preserving each block's matched-call count, with 2,000 draws and seed 2026. Seven date clusters and overlapping call horizons limit inference; these are descriptive uncertainty estimates, not a claim of statistical dominance. Raw-return hits, labels, matched availability and realized sector-relative returns are also saved per stock-date. IBES is primarily Buy in this universe, so overall hit rate must be read alongside class counts and directional coverage.

## Recommendations feed portfolio construction

Only Buy/Strong Buy calls enter each agent portfolio. Expected 21-session excess return is `0.5 × open-trained prior + 0.0125 × manager score × confidence`, with score in −2, −1, 0, 1, 2. IBES uses the same trained prior plus a fixed +1 Buy conviction; thus its benchmark isolates analyst selection and a specified signal conversion rather than claiming that IBES supplied expected 21-day returns. The objective is `0.5 × 5 × w'Σw − μ'w`. Covariance uses 252 past sessions and Ledoit–Wolf shrinkage, annualized consistently with μ. A candidate needs at least 200 observed historical returns; remaining historical missing values use its historical mean. Weights are long only, capped at 10%. When fewer than ten stocks qualify, residual capital remains in cash. No leverage or shorts are used.

The same optimizer constructs an independent-agent portfolio, an agent-with-IBES portfolio and an IBES-Buy portfolio. Equal weight uses the same full eligible universe. Minimum variance and maximum Sharpe are predefined sensitivity controls; no best-performing variant is selected after testing.

Portfolio returns follow one contiguous daily path: rebalance at each next entry close, drift weights between rebalances, and liquidate at the last horizon exit. Monthly call horizons can overlap; their full 21-day returns are used only for recommendation evaluation, not concatenated into portfolio returns. Each dollar of stock purchases or sales costs 10 bps, including initial funding and final liquidation. A missing held-security return invalidates the portfolio path rather than inventing a delisting payout.

|Portfolio|Net return|Sharpe: observed RF|Sharpe: zero-RF proxy|Max drawdown|RF sessions|
|---|---|---|---|---|---|
|Independent agent · MV|73.26%|2.288|2.322|-18.62%|147|
|Universe equal weight|43.32%|2.276|2.433|-12.03%|147|
|IBES Buy calls · MV|59.73%|2.075|2.082|-18.14%|147|
|Agent + IBES · Max Sharpe|90.43%|2.812|2.863|-12.49%|147|
|Agent + IBES · MV|79.34%|2.412|2.460|-17.08%|147|
|Agent + IBES · Min variance|15.07%|0.988|1.193|-11.98%|147|

![Net portfolio wealth](report_assets/wealth.png)

![Rolling Sharpe](report_assets/rolling_sharpe.png)

Observed-RF Sharpe uses only dates with a reported risk-free return and the same dates for all strategies. The full-period zero-RF proxy assumes a zero cash rate when RF is absent; it is labeled separately and must not be confused with measured full-period excess-return Sharpe. The 63-session rolling Sharpe requires 63 observed RF values; the line stops when coverage is missing.

## Evidence-based explanation examples

These are the first four stocks by gvkey on the first test date, not selected for successful outcomes. An explanation is a plausible inference from recorded evidence, not knowledge of an analyst's private rationale.

**AMD · Buy · confirm**

Consensus Buy is reasonable given 34% revenue growth, sharply higher earnings, cash generation, and a major partnership. I confirm Buy, but do not upgrade: 74.7x earnings, negative short-term momentum, and a 25.9% drawdown are material counterevidence. Consensus is usable as of 2026-03-02.

Evidence: IBES:001161:2026-03-02, F:001161:rev_yoy, F:001161:ni_yoy, V:001161:price_to_earnings_ttm, T:001161:momentum_21, T:001161:drawdown_63.
**SWKS · Sell · downgrade**

Consensus Buy has support from strong FCF, net cash, and very positive FY1 revisions, but current evidence favors downgrading to Sell: revenue and earnings are declining, guidance anticipates a roughly 20% sequential Mobile decline, and longer momentum is negative. Valuation is attractive but insufficient near term.

Evidence: IBES:001327:2026-03-02, F:001327:rev_yoy, F:001327:ni_yoy, N:1982436898, T:001327:momentum_63, V:001327:fcf_yield_ttm.
**ADI · Strong Buy · upgrade**

Consensus Buy is reasonable and I upgrade to Strong Buy: fresh results show 30% revenue growth, 112% earnings growth, expanding margins, strong FCF, and positive revisions and momentum. Rich valuation and positive net debt are material counterevidence, but do not outweigh the current operating and technical evidence.

Evidence: IBES:001632:2026-03-02, F:001632:rev_yoy, F:001632:ni_yoy, F:001632:op_margin_chg_yoy, T:001632:momentum_63, V:001632:price_to_earnings_ttm.
**AAPL · Buy · confirm**

Consensus Buy is reasonable given sequential acceleration, high margins, positive revisions, buybacks, and long-term momentum. I confirm Buy rather than upgrade: 76.7% liabilities, 44.1x book value, negative 63-session momentum, and very low turnover temper the outlook. Consensus is usable as of 2026-03-02.

Evidence: IBES:001690:2026-03-02, F:001690:rev_qoq, F:001690:liab_to_assets, V:001690:price_to_book, T:001690:momentum_63, N:1981923517.

## Limits and remaining validation

- Current-vintage Compustat data can include restatements or split adjustments. Strict row-date filtering cannot prove original data-vintage availability. Prebuilt consensus and targets likewise require upstream PIT validation; source build rules include provisional IBES links and broker-day tie resolution.
- Later-modified news records have no original text in these files. Omitting them avoids revised-text leakage but creates retrospective coverage selection; the original information set cannot be reconstructed completely.
- IBES updates end in August 2026. Consensus B may carry still-live prior opinions into later decisions; missing fresh A snapshots and EPS coverage must not be interpreted as new analyst information.
- One Health Care stock (HOLX) stops trading in the supplied sample. The transfer evaluator explicitly invalidates any portfolio path holding a missing return; it does not use future lifetime metadata to avoid that stock or invent acquisition proceeds.
- Only seven main decision dates are available after the model cutoff. Model knowledge cutoffs reduce one form of leakage but do not prove the absence of memorized/revised source data. Results do not justify tuning on this test set.
- Human explanation quality has not been scored. `human_evaluation_template.csv` contains 12 fixed-seed sampled cases and blank reviewer fields for factual accuracy, temporal fidelity, balance and clarity. No human ratings are fabricated.
- Scope is an academic historical backtest, with simplified fees, close execution, no market-impact model and no broker execution connection.

API ledger currently contains 262 recorded requests/retries, 6,611,925 input tokens and 353,988 output tokens. The access smoke test and the first rejected pre-ledger batch are excluded from this count; the provider's account usage is authoritative. No key is embedded in this report or flow export.

## Sources and implementation references

- Assignment: user-supplied Project 1 PDF, pages 1–3; data dictionary and build report provided alongside the parquet files.
- [Multi-Agent-Analyst](https://github.com/AgenticsFintekColumbia/Multi-Agent-Analyst): specialist reports and manager synthesis; separate explanation and recommendation modes.
- [portfolio-construction](https://github.com/AgenticsFintekColumbia/portfolio-construction): signals drive stock selection and out-of-sample variance, mean–variance and Sharpe objectives.
- [Official GPT-5.6 Luna documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna): exact model identifier and documented 16 February 2026 cutoff.
- [Langflow flow import/export](https://docs.langflow.org/concepts-flows-import) and [custom components](https://docs.langflow.org/components-custom-components): portable node/edge JSON and Python component interfaces.
