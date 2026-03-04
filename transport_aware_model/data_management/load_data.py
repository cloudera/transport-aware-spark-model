import json

import pandas as pd


def load_to_df(filepath):
    with open(filepath) as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    return df


def load_enriched_measurement_summary(data_size="1tb"):
    return load_to_df("data/enriched_measurement_summary.json")


def load_enriched_stage_measurement_summary():
    return load_to_df("data/enriched_stage_measurement_summary.json")
