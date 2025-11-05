# Fundamental Stock Screening

This project provides a set of Python scripts for performing fundamental analysis of stocks. It allows you to clean financial data, calculate quality scores, and screen for undervalued companies based on various metrics.

## Project Structure

- `main.py`: The main entry point for running the analysis scripts.
- `commands/`: Contains the individual scripts for each step of the analysis, organized into subdirectories:
  - `data_cleaning/`: Scripts for cleaning and preparing the raw data.
  - `quality_scoring/`: Scripts for calculating quality scores for each company.
  - `valuation_screening/`: Scripts for screening companies based on valuation metrics.
  - `sector_composition_analysis/`: Scripts for analyzing the composition of different market sectors.
  - `quality_price/`: Scripts for analyzing the relationship between quality and price.
- `data/`: Directory for storing input data (e.g., CSV files with financial data).
- `out/`: Directory for storing the output of the analysis (e.g., reports, plots, and cleaned data).

## How to Use

The `main.py` script is used to execute the different analysis commands. You can run a specific command by providing the path to the script using the `--func` argument.

### Example

To run the `csv_overview.py` script to get an overview of a CSV file, you would use the following command:

```bash
python main.py --func commands/data_cleaning/csv_overview.py --input_data data/original/us_stock_valuation.csv --out_name overview_report.txt
```

This will execute the `run` function in `commands/data_cleaning/csv_overview.py`, passing the specified input and output file names as arguments. The output report will be saved in the `out/` directory.

### Available Commands

You can explore the `commands/` directory to see the available analysis scripts. Each script is designed to be run as a standalone command using `main.py`.

## Data

The data folder is slighly misleading. It contains data from the primary csv that is used to generate insights (including benchmark tables etc...). Many of the visualizations have already been created. To locate this, navigate to out. Here you will find various categories of types of data, including summary reports for intermediary steps such as datacleaning. The summarized data for QARP is located inside out/quality_price
