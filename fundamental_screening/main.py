import argparse
import importlib.util
import inspect
import os
import sys

def load_module_from_path(path: str):
    """Dynamically load a Python module from a given file path."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module path not found: {path}")

    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Run a function from a file path (e.g., ./commands/sector_overview.py).")
    parser.add_argument('--func', required=True, help="Path to the Python file containing a 'run' function.")
    parser.add_argument('--input_data', help="Path to the input data file.")
    parser.add_argument('--out_name', help="Name for the output file.")
    parser.add_argument('--report_name', help="Filename for the report in ./out.")

    args, unknown = parser.parse_known_args()

    try:
        # Load module from path
        module = load_module_from_path(args.func)

        if hasattr(module, 'run'):
            run_func = module.run
            run_signature = inspect.signature(run_func)
            run_params = run_signature.parameters

            # Collect arguments dynamically
            kwargs = {}
            for param_name in run_params.keys():
                if hasattr(args, param_name) and getattr(args, param_name) is not None:
                    kwargs[param_name] = getattr(args, param_name)

            # Pass any unknown CLI args as **kwargs** if they match the function
            # Format: --key value → key=value
            for i in range(0, len(unknown), 2):
                if unknown[i].startswith('--'):
                    key = unknown[i][2:]
                    val = unknown[i + 1] if (i + 1) < len(unknown) else True
                    if key in run_params:
                        kwargs[key] = val

            # Run the function
            print(f"Running: {args.func} -> run(**{kwargs})")
            run_func(**kwargs)
        else:
            print(f"Error: The file '{args.func}' does not define a 'run' function.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
