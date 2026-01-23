import numpy as np
import pandas as pd
import requests
import time
import json


def fetch_melodi_dataset(dataset_id: str, geo_code: str) -> pd.DataFrame:
    """
    Fetch filtered data from the INSEE Melodi API for a given dataset and geographic code.

    Parameters
    ----------
    dataset_id : str
        The identifier of the Melodi dataset (e.g., "DS_POPULATIONS_REFERENCE").
    geo_code : str
        The geographic filter (e.g., "DEP-75", "DEP-44").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted observations.
    """

    # Build the API URL
    url = f"https://api.insee.fr/melodi/data/{dataset_id}?GEO={geo_code}"

    # Send the request
    response = requests.get(url)

    # Convert JSON response to Python dict
    data = json.loads(response.content)

    # Extract observations
    observations = data.get("observations", [])
    extracted_rows = []

    for obs in observations:
        dimensions = obs.get("dimensions", {})
        attributes = obs.get("attributes", {})
        measures = obs["measures"]["OBS_VALUE_NIVEAU"].get("value", None)

        row = {**dimensions, **attributes, "OBS_VALUE_NIVEAU": measures}
        extracted_rows.append(row)

    return pd.DataFrame(extracted_rows)



def fetch_idf_departements(dataset_id: str) -> pd.DataFrame:
    """
    Fetch data from the INSEE Melodi API for all Île-de-France départements.

    Parameters
    ----------
    dataset_id : str
        The identifier of the Melodi dataset (e.g., "DS_POPULATIONS_REFERENCE").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted observations for all IDF départements.
    """

    idf_departements = [
        "DEP-75", "DEP-77", "DEP-78", "DEP-91",
        "DEP-92", "DEP-93", "DEP-94", "DEP-95"
    ]

    all_data = []

    for dep in idf_departements:
        df = fetch_melodi_dataset(dataset_id, dep)
        df["departement_code"] = dep
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)



def fetch_dep_dataset(dataset_id: str) -> pd.DataFrame:
    """
    Fetch data from the INSEE Melodi API for all French départements.

    Parameters
    ----------
    dataset_id : str
        The identifier of the Melodi dataset (e.g., "DS_POPULATIONS_REFERENCE").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted observations for all french départements.
    """

    all_departements = [
        "DEP-01","DEP-02","DEP-03","DEP-04","DEP-05","DEP-06","DEP-07","DEP-08","DEP-09",
        "DEP-10","DEP-11","DEP-12","DEP-13","DEP-14","DEP-15","DEP-16","DEP-17","DEP-18","DEP-19",
        "DEP-21","DEP-22","DEP-23","DEP-24","DEP-25","DEP-26","DEP-27","DEP-28","DEP-29",
        "DEP-2A","DEP-2B",
        "DEP-30","DEP-31","DEP-32","DEP-33","DEP-34","DEP-35","DEP-36","DEP-37","DEP-38","DEP-39",
        "DEP-40","DEP-41","DEP-42","DEP-43","DEP-44","DEP-45","DEP-46","DEP-47","DEP-48","DEP-49",
        "DEP-50","DEP-51","DEP-52","DEP-53","DEP-54","DEP-55","DEP-56","DEP-57","DEP-58","DEP-59",
        "DEP-60","DEP-61","DEP-62","DEP-63","DEP-64","DEP-65","DEP-66","DEP-67","DEP-68","DEP-69",
        "DEP-70","DEP-71","DEP-72","DEP-73","DEP-74","DEP-75","DEP-76","DEP-77","DEP-78","DEP-79",
        "DEP-80","DEP-81","DEP-82","DEP-83","DEP-84","DEP-85","DEP-86","DEP-87","DEP-88","DEP-89",
        "DEP-90","DEP-91","DEP-92","DEP-93","DEP-94","DEP-95",
        "DEP-971","DEP-972","DEP-973","DEP-974","DEP-976"
    ]


    all_data = []

    for dep in all_departements:
        df = fetch_melodi_dataset(dataset_id, dep)
        df["departement_code"] = dep
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)




def list_melodi_datasets() -> pd.DataFrame:
    """
    Retrieve and flatten the Melodi dataset catalog from INSEE.

    Returns
    -------
    pd.DataFrame
        A DataFrame with dataset_id, title_fr, and description_fr.
    """

    url = "https://api.insee.fr/melodi/catalog/all"
    response = requests.get(url)
    raw_data = json.loads(response.content)

    # Handle both possible structures: dict with "datasets" or direct list
    if isinstance(raw_data, dict) and "datasets" in raw_data:
        items = raw_data["datasets"]
    elif isinstance(raw_data, list):
        items = raw_data
    else:
        # Fallback: nothing usable
        return pd.DataFrame(columns=["dataset_id", "title_fr", "description_fr"])

    rows = []
    for item in items:
        # Skip anything that is not a dict
        if not isinstance(item, dict):
            continue

        dataset_id = item.get("identifier")

        # Extract French title and description safely
        title_list = item.get("title", [])
        desc_list = item.get("description", [])

        title_fr = next(
            (t.get("content") for t in title_list if isinstance(t, dict) and t.get("lang") == "fr"),
            None
        )
        description_fr = next(
            (d.get("content") for d in desc_list if isinstance(d, dict) and d.get("lang") == "fr"),
            None
        )

        rows.append({
            "dataset_id": dataset_id,
            "title_fr": title_fr,
            "description_fr": description_fr
        })

    return pd.DataFrame(rows)


import requests
import pandas as pd
import time

def fetch_indicator_for_all_departements(dataset_id: str) -> pd.DataFrame:
    """
    Extracts indicator data from the Melodi API for all départements in France.

    Parameters
    ----------
    dataset_id : str
        The Melodi dataset ID to extract (e.g. 'DS_ERFS_MENAGE_SL', 'DS_RP_EMPLOI_LR_PRINC').

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with columns: departement_code, year, measure, value.
    """

    # List of all 101 département codes (including DOMs)
    departements = [
        f"{i:02d}" for i in range(1, 96)
    ] + ["2A", "2B", "971", "972", "973", "974", "976"]

    base_url = "https://api.insee.fr/melodi/v1/data"
    rows = []

    for dep in departements:
        geo = f"DEP-{dep}"
        params = {
            "dataset": dataset_id,
            "geo": geo
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            for obs in data.get("observations", []):
                rows.append({
                    "departement_code": geo,
                    "year": obs.get("TIME_PERIOD"),
                    "measure": obs.get("MEASURE", "value"),
                    "value": obs.get("OBS_VALUE")
                })

        except Exception as e:
            print(f"Error for {geo}: {e}")
            continue

        time.sleep(1.5)  # polite delay to avoid hammering the API

    df = pd.DataFrame(rows)

    # Optional: clean and sort
    df["departement_code"] = df["departement_code"].str.upper()
    df = df.sort_values(by=["departement_code", "year"]).reset_index(drop=True)

    return df
