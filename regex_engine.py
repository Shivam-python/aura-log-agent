import re
import yaml


class RegexClassifier:

    def __init__(self, config_path):

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.patterns = [
            (re.compile(item["pattern"]), item["label"])
            for item in config["patterns"]
        ]

    def classify(self, log_message):

        for pattern, label in self.patterns:

            if pattern.search(log_message):
                return label

        return None