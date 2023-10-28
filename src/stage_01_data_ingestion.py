import pandas as pd
import numpy as np
import xmltodict
import yaml
import os
import glob


def read_yaml_file():
    path_to_yaml = "config.yaml"
    try:
        with open(path_to_yaml, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print("Error reading the config file")


def data_ingest_xml(path_xml, config):
    with open(path_xml) as file_xml:
        doc = xmltodict.parse(file_xml.read())
        news_items = doc["rss"]["channel"]["item"]
        df = pd.DataFrame(news_items)
        df.to_parquet(
            os.path.join(
                config["data_ingestion"]["local_data_file_parquet"],
                path_xml.split("/")[-1].split(".")[-2] + ".parquet",
            )
        )


if __name__ == "__main__":
    config = read_yaml_file()
    print(config["data_ingestion"]["local_data_file_xml"])
    print(config["data_ingestion"]["local_data_file_parquet"])
    path_xml = (
        "/Users/piyush/Desktop/dsml_Portfolio/LLM_project/data/xml/technology.xml"
    )

    directory_path = "data/xml/*.xml"
    file_names = sorted(glob.glob(directory_path))

    for file in file_names:
        print(file)
        data_ingest_xml(path_xml=file, config=config)
