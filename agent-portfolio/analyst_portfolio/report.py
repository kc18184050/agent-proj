import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import save

LABELS={'blind_mean_variance':'Independent agent · MV','with_mean_variance':'Agent + IBES · MV',
        'ibes_mean_variance':'IBES Buy calls · MV','ridge_mean_variance':'Ridge-only · MV','equal_weight':'Universe equal weight',
        'with_minimum_variance':'Agent + IBES · Min variance','with_maximum_sharpe':'Agent + IBES · Max Sharpe'}
CALL_LABELS={'ridge':'Ridge-only','blind':'Independent agent','with':'Agent with IBES','ibes':'IBES consensus'}

def percent(x):return 'N/A' if x is None else f'{x:.2%}'
def number(x):return 'N/A' if x is None else f'{x:.3f}'

def report(root):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(Path(root)/'artifacts/.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'artifacts/results/core_main'
    m=json.loads((out/'metrics.json').read_text());model=json.loads((root/'artifacts/frozen_model.json').read_text())
    calls=pd.read_csv(out/'call_outcomes.csv',dtype={'gvkey':str});daily=pd.read_csv(out/'daily_portfolios.csv',parse_dates=['date'])
    charts=root/'report_assets';charts.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#f6f8fa','axes.facecolor':'#ffffff','axes.labelcolor':'#25364a'})
    colors={'blind_mean_variance':'#136f63','with_mean_variance':'#285ca8','ibes_mean_variance':'#bc6c25','ridge_mean_variance':'#a43246',
            'equal_weight':'#424b54','with_minimum_variance':'#8b6fa8','with_maximum_sharpe':'#b04e77'}
    fig,ax=plt.subplots(figsize=(11,5))
    for name,d in daily.groupby('strategy',sort=False):
        color=colors[name]
        ax.plot(d.date,d.wealth,label=LABELS[name],color=color,lw=2 if 'mean_variance' in name or name=='equal_weight' else 1,alpha=.95)
    ax.set_title('Out-of-sample portfolio wealth · net of trading costs',loc='left',pad=18,fontweight='bold')
    ax.set_ylabel('Growth of $1');ax.grid(alpha=.15);ax.legend(ncol=2,frameon=False,loc='upper left');fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(charts/'wealth.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for name in ['ridge_mean_variance','blind_mean_variance','with_mean_variance','ibes_mean_variance','equal_weight']:
        d=daily[daily.strategy==name].copy();x=d.net_return-d.rf_daily
        if d.empty:continue
        rolling=np.sqrt(252)*x.rolling(63,min_periods=63).mean()/x.rolling(63,min_periods=63).std()
        ax.plot(d.date,rolling,label=LABELS[name],color=colors[name],lw=1.8)
    ax.axhline(0,color='#c9d1d9',lw=.8);ax.set_title('63-session rolling Sharpe · observed risk-free dates only',loc='left',pad=18,fontweight='bold')
    ax.grid(alpha=.15);ax.legend(ncol=2,frameon=False);fig.autofmt_xdate();fig.tight_layout();fig.savefig(charts/'rolling_sharpe.png',dpi=180);plt.close(fig)
    ctable=[];ptable=[]
    for mode,title in CALL_LABELS.items():
        if mode not in m['calls']:continue
        c=m['calls'][mode]
        matched_accuracy=calls.loc[(calls['mode']==mode)&calls.valid&calls.analyst_available,'class_correct']
        ctable.append([title,str(c['n']),percent(c['classification_accuracy']),str(len(matched_accuracy)),
                       percent(matched_accuracy.astype(float).mean()) if len(matched_accuracy) else 'N/A',percent(c['directional_hit_rate'])])
    for name,p in m['portfolios'].items():
        ptable.append([LABELS[name],percent(p['total_return']),number(p['sharpe_rf_covered']),number(p['sharpe_zero_rf_proxy']),percent(p['max_drawdown']),str(p['rf_covered_sessions'])])
    def mdtable(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(r)+'|' for r in rows])
    cheaders=['Method','Valid calls','3-way accuracy','IBES-matched calls','IBES-matched 3-way accuracy','Directional hit']
    pheaders=['Portfolio','Net return','Sharpe: observed RF','Sharpe: zero-RF proxy','Max drawdown','RF sessions']
    config=json.loads((root/'config.json').read_text())
    usage=[]
    if (root/'artifacts/api_usage.jsonl').exists():usage=[json.loads(l) for l in (root/'artifacts/api_usage.jsonl').read_text().splitlines() if l.strip()]
    input_tokens=sum(u['usage'].get('input_tokens',0) for u in usage);output_tokens=sum(u['usage'].get('output_tokens',0) for u in usage)
    improvement=[]
    for mode,comparisons in m.get('paired_classification',{}).items():
        for baseline,c in comparisons.items():
            lo,hi=c['date_block_ci95']
            improvement.append(f"- {CALL_LABELS[mode]} vs {CALL_LABELS[baseline]}: paired accuracy difference {c['accuracy_difference']:+.2%} on {c['n']} stock-dates. Date-block bootstrap 95% interval [{lo:+.2%}, {hi:+.2%}]. Wins / ties / losses: {c['wins']:.2%} / {c['ties']:.2%} / {c['losses']:.2%}.")
    # Select samples by a fixed seed, not by successful outcome.
    eligible=calls[(calls['mode']=='with')&calls.valid]
    sample=eligible.sample(n=min(12,len(eligible)),random_state=2026)[['date','gvkey','label']].copy()
    for col in ['evidence_accuracy_1to5','temporal_fidelity_1to5','balance_1to5','clarity_1to5','reviewer_notes']:sample[col]=''
    sample.to_csv(root/'human_evaluation_template.csv',index=False)
    bydate=calls[calls.valid].groupby(['date','mode']).hit.mean().unstack();bydate.to_csv(out/'hit_rate_by_date.csv')
    calls[calls.valid].groupby(['date','mode']).class_correct.mean().unstack().to_csv(out/'classification_accuracy_by_date.csv')
    matched=calls[calls.valid&calls.analyst_available]
    confusion=pd.crosstab(matched[matched['mode']=='with'].outcome_label,matched[matched['mode']=='with'].label).reindex(index=['Buy','Hold','Sell'],columns=['Buy','Hold','Sell'],fill_value=0)
    confusion.to_csv(out/'with_ibes_confusion.csv')
    first=next(iter(sorted((root/'artifacts/decisions/core').glob('*.json'))));decision=json.loads(first.read_text())
    sample_explanations=[]
    u=pd.read_parquet(Path(config['full_dir'])/'universe.parquet').set_index('gvkey')
    for r in decision['with'][:4]:
        sample_explanations.append(f"**{u.at[r['id'],'ticker_symbol']} · {r['rating_5']} · {r['action']}**\n\n{r['rationale']}\n\nEvidence: {', '.join(r['evidence_ids'])}.")
    band=f"{100 * config['hold_band']:g}%"
    ridge_description='Ridge-only uses the frozen prediction directly for expected returns and applies the same three-way cutoff to select Buy stocks. The same optimizer and trading costs apply to Ridge and the other mean–variance strategies.' if 'ridge' in m['calls'] else 'Rerun `evaluate` to include the Ridge-only baseline.'
    text=f'''# Analyst Recommendations → Portfolio

Point-in-time multi-agent research system | Langflow 1.12.0 | GPT-5.6 Luna

## Research question and implementation

Can a fundamental, technical and news analyst team explain analyst recommendations, improve Buy/Hold/Sell calls, and feed an investable Markowitz portfolio? This implementation combines the role-based recommendation design of Multi-Agent-Analyst with the screening-to-portfolio design of portfolio-construction. It is a new Langflow implementation, not a verbatim merge of their CrewAI code.

The actual flow has Chat Input, dated evidence, three specialist nodes, a two-pass manager, Markowitz construction, research formatting and Chat Output. Every specialist and both manager passes use the requested `gpt-5.6-luna`; numeric data preparation and optimization run in Python. External browsing and model tools are disabled during financial inference. The agent can cite only supplied evidence IDs.

## Development, freezing and holdout

The model knowledge cutoff is **{config['cutoff']}**. Only `open/` is used for fitting and selection. The frozen ridge return prior has **{model['rows']}** stock-date observations across **{len(model['training_dates'])}** development decisions, ending with labels on **{model['max_label_date']}**. The last three development dates are a chronological validation block; earlier fitting rows are purged unless their outcome horizon matures before validation starts. Ridge alpha candidates are 10, 100 and 1000; selected alpha is **{model['alpha']}**. Medians, scaling and coefficients are then fitted on the full allowed development period. Strategy hyperparameters and prompts are fixed before inspecting holdout performance. GPT foundation-model weights are unchanged; development results are not claimed as out-of-sample evidence.

Frozen artifact SHA-256: `{model['freeze_sha256']}`. Source parquet hashes, all coefficients, configuration and training dates are saved in `artifacts/frozen_model.json`. Test reads do not fit parameters or change prompts in response to outcomes. Individual API responses are cached using a hash of model, exact instructions, schema and dated evidence; retries after structural/citation validation failure are logged, rather than replaced by fabricated ratings.

**Main test dates:** {', '.join(m['dates'])}. The fixed supplied IT sector has 70 eligible stocks at the first test date. No Health Care outcomes are used in calibration. The `test_conditional` October date is excluded from headline results because the source report marks it conditional. Separate CLI switches support conditional and transfer runs, without relabeling them as the main test.

## Information available at a decision

Fundamentals must have `available_date < decision_date`; news must have `usable_date < decision_date`, a usable original text version, and no analyst-action flag. News uses the last 90 calendar days and up to eight most recent events. Fundamentals older than 180 days and valuations older than seven days are omitted. Technical features use prices through the decision close; explanation-only mode uses the previous session close. The three specialists and independent manager do not receive ratings, price targets, EPS estimates/revisions, ticker symbols, or future returns. IDs are gvkeys; company names can still appear inside news, so anonymization is incomplete.

Only the second manager sees IBES context: consensus definition B, 12-month maximum age measured from the **usable date**, at least three brokers. Confirmation-date aging is avoided. Current FY1 EPS snapshots must be at most 35 days old; small-base revisions/surprises are excluded. Stale source coverage is visible and missing consensus is not treated as Hold. Public current identities and a fixed 2026-03-01 universe follow the supplied files; this is not a historically reconstructed constituent backtest.

## Recommendation evaluation

Entry is the close of the first session after the decision. Exit is the close 21 sessions after entry. The sector return is the equal-weight mean of valid outcomes in the same eligible universe. The primary metric is three-way accuracy. A stock is Buy if it beats the sector by more than {band}, Sell if it trails the sector by more than {band}, and Hold within the inclusive ±{band} band. The directional hit rate checks the sign of the return for Buy and Sell calls. IBES-matched columns use only stock-dates with an available IBES consensus. Missing outcome horizons are retained as invalid records.

{mdtable(cheaders,ctable)}

{chr(10).join(improvement)}

The confidence interval resamples whole decision-date blocks, preserving each block's paired sample count, with 2,000 draws and seed 2026. Seven date clusters and overlapping call horizons limit inference. Raw-return hits, labels, consensus availability and realized sector-relative returns are also saved per stock-date.

## Recommendations feed portfolio construction

Only Buy/Strong Buy calls enter each agent portfolio. Expected 21-session excess return is `0.5 × open-trained prior + 0.0125 × manager score × confidence`, with score in −2, −1, 0, 1, 2. IBES uses the same trained prior plus a fixed +1 Buy conviction; thus its benchmark isolates analyst selection and a specified signal conversion rather than claiming that IBES supplied expected 21-day returns. The objective is `0.5 × 5 × w'Σw − μ'w`. Covariance uses 252 past sessions and Ledoit–Wolf shrinkage, annualized consistently with μ. A candidate needs at least 200 observed historical returns; remaining historical missing values use its historical mean. Weights are long only, capped at 10%. When fewer than ten stocks qualify, residual capital remains in cash. No leverage or shorts are used.

{ridge_description}

Equal weight uses the same full eligible universe. Minimum variance and maximum Sharpe are predefined sensitivity controls.

Portfolio returns follow one contiguous daily path: rebalance at each next entry close, drift weights between rebalances, and liquidate at the last horizon exit. Monthly call horizons can overlap; their full 21-day returns are used only for recommendation evaluation, not concatenated into portfolio returns. Each dollar of stock purchases or sales costs 10 bps, including initial funding and final liquidation. A missing held-security return invalidates the portfolio path rather than inventing a delisting payout.

{mdtable(pheaders,ptable)}

![Net portfolio wealth](report_assets/wealth.png)

![Rolling Sharpe](report_assets/rolling_sharpe.png)

Observed-RF Sharpe uses only dates with a reported risk-free return and the same dates for all strategies. The full-period zero-RF proxy assumes a zero cash rate when RF is absent; it is labeled separately and must not be confused with measured full-period excess-return Sharpe. The 63-session rolling Sharpe requires 63 observed RF values; the line stops when coverage is missing.

## Evidence-based explanation examples

These are the first four stocks by gvkey on the first test date, not selected for successful outcomes. An explanation is a plausible inference from recorded evidence, not knowledge of an analyst's private rationale.

{chr(10).join(sample_explanations)}

## Limits and remaining validation

- Current-vintage Compustat data can include restatements or split adjustments. Strict row-date filtering cannot prove original data-vintage availability. Prebuilt consensus and targets likewise require upstream PIT validation; source build rules include provisional IBES links and broker-day tie resolution.
- Later-modified news records have no original text in these files. Omitting them avoids revised-text leakage but creates retrospective coverage selection; the original information set cannot be reconstructed completely.
- IBES updates end in August 2026. Consensus B may carry still-live prior opinions into later decisions; missing fresh A snapshots and EPS coverage must not be interpreted as new analyst information.
- One Health Care stock (HOLX) stops trading in the supplied sample. The transfer evaluator explicitly invalidates any portfolio path holding a missing return; it does not use future lifetime metadata to avoid that stock or invent acquisition proceeds.
- Only seven main decision dates are available after the model cutoff. Model knowledge cutoffs reduce one form of leakage but do not prove the absence of memorized/revised source data. Results do not justify tuning on this test set.
- Human explanation quality has not been scored. `human_evaluation_template.csv` contains 12 fixed-seed sampled cases and blank reviewer fields for factual accuracy, temporal fidelity, balance and clarity. No human ratings are fabricated.
- Scope is an academic historical backtest, with simplified fees, close execution, no market-impact model and no broker execution connection.

API ledger currently contains {len(usage)} recorded requests/retries, {input_tokens:,} input tokens and {output_tokens:,} output tokens. The access smoke test and the first rejected pre-ledger batch are excluded from this count; the provider's account usage is authoritative. No key is embedded in this report or flow export.

## Sources and implementation references

- Assignment: user-supplied Project 1 PDF, pages 1–3; data dictionary and build report provided alongside the parquet files.
- [Multi-Agent-Analyst](https://github.com/AgenticsFintekColumbia/Multi-Agent-Analyst): specialist reports and manager synthesis; separate explanation and recommendation modes.
- [portfolio-construction](https://github.com/AgenticsFintekColumbia/portfolio-construction): signals drive stock selection and out-of-sample variance, mean–variance and Sharpe objectives.
- [Official GPT-5.6 Luna documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna): exact model identifier and documented 16 February 2026 cutoff.
- [Langflow flow import/export](https://docs.langflow.org/concepts-flows-import) and [custom components](https://docs.langflow.org/components-custom-components): portable node/edge JSON and Python component interfaces.
'''
    (root/'Research_Report.md').write_text(text)
    # Markdown is converted locally; images are embedded for a standalone report.
    import markdown
    rendered=markdown.markdown(text,extensions=['tables','fenced_code'])
    for filename in ['wealth.png','rolling_sharpe.png']:
        data=base64.b64encode((charts/filename).read_bytes()).decode()
        rendered=rendered.replace(f'report_assets/{filename}',f'data:image/png;base64,{data}')
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analyst Recommendations → Portfolio</title><style>
    body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#243449;background:#f2f5f7;margin:0}main{max-width:1100px;margin:35px auto;background:white;padding:50px 65px;border-top:7px solid #136f63}h1{font-size:39px;line-height:1.15;color:#123b3a}h2{margin-top:44px;font-size:24px;color:#136f63}p,li{max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px;margin:22px 0}th{text-align:left;background:#eaf2f1}td,th{padding:10px;border-bottom:1px solid #dbe3e9}code{font-size:12px;background:#eef2f5;padding:2px 4px;overflow-wrap:anywhere}img{max-width:100%;height:auto}a{color:#285ca8}li{margin:8px 0}@media(max-width:700px){main{padding:22px;margin:0}table{display:block;overflow-x:auto}h1{font-size:30px}}@media print{body{background:white}main{margin:0;padding:0;border:0}h2{break-after:avoid}img,table{break-inside:avoid}}
    </style><main>'''+rendered+'</main></html>'
    (root/'Research_Report.html').write_text(document)
    return {'report':str(root/'Research_Report.html'),'markdown':str(root/'Research_Report.md')}
