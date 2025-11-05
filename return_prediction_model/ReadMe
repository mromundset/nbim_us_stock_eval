# Return Prediction Model

This project contains a machine learning model for predicting stock returns based on fundamental data.

## Project Structure

- `main.py`: The main script for training the model and running other commands.
- `visualize.py`: A script for generating visualizations from the model results.
- `data/`: Directory for storing input data.
- `figures/`: Directory for storing generated plots.
- `results/`: Directory for storing model results.

## How to Use

The `main.py` script is the entry point for various commands, such as cleaning data and training the model. You can specify the command and its parameters using command-line arguments.

### Example

To train the model, you can run the following command:

```bash
python main.py --command train_model --input data/us_stock_valuation_clean_sorted_zero_handled.csv --output results/model_results.json --params model_type=gbdt
```

This will train a gradient boosting model (`gbdt`) on the specified input data and save the results to `results/model_results.json`.

### Visualizations

The `visualize.py` script can be used to generate plots from the model evaluation results.

```bash
python visualize.py --command run_all --input results/model_runs.csv --output figures
```

This will generate all available plots and save them in the `figures/` directory.

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- scipy
- lightgbm
- matplotlib

You can install them using pip:

```bash
pip install -r requirements.txt
```
