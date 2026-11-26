# Analyst Recommendations → Portfolio

[Download the full project source ZIP](./agent-portfolio-github.zip) · [Read the research report](./Research_Report.md)

Download and extract the ZIP, then run the commands below from the `agent-portfolio` folder. The ZIP contains the Python source code, the Langflow JSON, and an HTML report with charts.

This Langflow project uses three specialists to analyze stocks. They cover fundamentals, technical indicators, and news. The manager first makes an independent recommendation, then reviews the IBES consensus and adjusts the rating as appropriate. The project uses the Markowitz method to build portfolios.

## View the existing results

After downloading the project, open `Research_Report.html` in your browser. The report includes the charts and opens without Python, data files, or an API key. The results and the frozen model hash come from the original experiment. If you train the model again, the program saves a new frozen model using your local configuration.

## Run locally

Use **Python 3.12** and the original `open/` and `full/` Parquet datasets. Langflow analysis and the full backtest also require an OpenAI API key with access to `gpt-5.6-luna`. The project uses this model for all API calls. Obtain the datasets through the course or an authorized data provider.

### Install dependencies

Open a terminal in the project directory that contains `pyproject.toml`.

**macOS / Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[langflow,test]'
cp .env.example .env
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[langflow,test]'
Copy-Item .env.example .env
```

Open `.env` in a text editor and enter your key after `OPENAI_API_KEY=` if you plan to run the agent. If you keep the datasets in `open/` and `full/` under the project directory, set `open_dir` and `full_dir` in `config.json` accordingly. Each dataset folder should directly contain its own `decision_dates.parquet` and the other Parquet files. You can also point the settings to other locations. On Windows, you can use forward slashes, as in `C:/Users/your-name/data/open`.

### Train and start Langflow

**macOS / Linux**

```bash
.venv/bin/python -m analyst_portfolio.cli train
.venv/bin/python build_flow.py
.venv/bin/python launch_local.py
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m analyst_portfolio.cli train
.\.venv\Scripts\python.exe build_flow.py
.\.venv\Scripts\python.exe launch_local.py
```

`train` fits the Ridge model on `open/` and saves the fitted parameters. It runs without calling the model API. Run `train` again after changing the data paths because the program checks the configuration against the frozen model file.

Open Langflow and select **Upload a flow** on the project page. Import the generated `Analyst_Portfolio_Langflow.json`. It contains the same flow as `Agent_Langflow.json` and sets **Project directory** to your absolute project path. Run Langflow in the Python environment above and enable custom components. [Open local Langflow](http://127.0.0.1:7860)

Enter the following JSON in **Playground** to explain a single stock rating. The model API call may incur a charge.

```json
{"date":"2026-03-02","role":"core","explain":true,"ticker":"AAPL","rating":"Strong Buy"}
```

To build the full portfolio for the same date, enter `{"date":"2026-03-02","role":"core","explain":false}`. Explanation mode accepts `Strong Buy / Buy / Hold / Sell / Strong Sell` as input. It uses prices and technical indicators from the previous trading day. The portfolio backtest excludes explanation-only runs.

## Test and reproduce the experiment

Run the following commands from the project directory using the Python environment above. The test suite needs the datasets and a frozen model file. It runs without calling the model API.

```bash
python -m pytest tests -q
python -m analyst_portfolio.cli evaluate
python -m analyst_portfolio.cli report
```

`evaluate` calls the model and generates new backtest results. Run `report` afterward to update the local report. The original experiment uses `open/` to develop and calibrate the Ridge return estimate and includes training labels through the 2026-02-16 cutoff. It uses the base GPT model as provided. The `full/` dataset supplies decision inputs and realized returns for the test period. The main report covers seven decision dates from March through September 2026. Use `evaluate --conditional` for October and `evaluate --role transfer` for the health care sector.

Git ignores the local `.env`, raw datasets, the generated flow, and outputs under `artifacts/`. For a manual web upload, use the files from the original share package. The project builds on two reference implementations. [Multi-Agent-Analyst](https://github.com/AgenticsFintekColumbia/Multi-Agent-Analyst) · [portfolio-construction](https://github.com/AgenticsFintekColumbia/portfolio-construction)
