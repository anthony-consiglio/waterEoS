"""
CSV export for plot data.
"""

import csv


def export_property_curves_csv(filepath, data):
    """
    Write property curves to CSV.

    data : dict from compute_property_curves / compute_multi_model_curves
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # For single-model data
        if 'x_values' in data:
            for x, y, label in zip(data['x_values'], data['y_values'], data['curve_labels']):
                writer.writerow([data['x_label'], data['y_label'], label])
                for xi, yi in zip(x, y):
                    writer.writerow([xi, yi])
                writer.writerow([])
        # For multi-model data  {model_key: single_data}
        else:
            for mk, single in data.items():
                writer.writerow([f'Model: {mk}'])
                export_property_curves_csv.__wrapped_single(writer, single)
                writer.writerow([])


def _write_single(writer, data):
    for x, y, label in zip(data['x_values'], data['y_values'], data['curve_labels']):
        writer.writerow([data['x_label'], data['y_label'], label])
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])
        writer.writerow([])


export_property_curves_csv.__wrapped_single = _write_single


def export_point_calc_csv(filepath, results_dict, T_K, P_MPa):
    """Write point calculator results to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'T = {T_K} K', f'P = {P_MPa} MPa'])
        writer.writerow([])

        models = list(results_dict.keys())
        writer.writerow(['Property'] + models)

        all_props = set()
        for props in results_dict.values():
            all_props.update(props.keys())

        for prop in sorted(all_props):
            row = [prop]
            for mk in models:
                val = results_dict[mk].get(prop)
                row.append(f'{val:.10g}' if val is not None else 'N/A')
            writer.writerow(row)
