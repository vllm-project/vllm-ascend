from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import regex as re
import requests


class MetricsAggregator:
    def __init__(self, node_urls_fn):
        self.node_urls_fn = node_urls_fn

    def fetch_metrics(self, url):
        """fetch metrics from a node"""
        try:
            resp = requests.get(url, timeout=5)
            return resp.text
        except requests.exceptions.RequestException:
            return ""

    def parse_metric_line(self, line):
        """parse a metric line"""
        if line.startswith("#") or not line.strip():
            return None

        # parse metric value: metric{labels} value
        match = re.match(r"([^{]+)(?:\{([^}]*)\})?\s+([\d\.eE+-]+)", line)
        if not match:
            return None

        name = match.group(1)
        labels_str = match.group(2) or ""
        value = float(match.group(3))

        # parse labels
        labels = {}
        if labels_str:
            for label in labels_str.split(","):
                if "=" in label:
                    k, v = label.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')

        return name, labels, value

    def aggregate_metrics(self, additional_labels=None):
        """aggregate all node metrics"""
        if additional_labels is None:
            additional_labels = {}
        all_metrics = defaultdict(list)
        node_urls = self.node_urls_fn()

        # fetch metrics from nodes
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.fetch_metrics, url): url for url in node_urls}

            for future in as_completed(futures):
                metrics_text = future.result()
                if not metrics_text:
                    continue

                for line in metrics_text.split("\n"):
                    parsed = self.parse_metric_line(line)
                    if parsed:
                        name, labels, value = parsed

                        def label_str(labels):
                            return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()) if k not in ["engine"])

                        labels.update(additional_labels)
                        all_metrics[(name, label_str(labels))].append(value)

        aggregated = {}
        for (metric_name, label), values in all_metrics.items():
            if values:
                # get sum or avg by type
                if any(
                    x in metric_name for x in ["bucket", "_count", "_sum", "_total", "tokens", "requests", "memory"]
                ):
                    # sum metrics
                    aggregated[(metric_name, label)] = sum(values)
                else:
                    # avg metrics
                    aggregated[(metric_name, label)] = sum(values) / len(values)

        return aggregated

    def format_as_prometheus(self, aggregated_metrics):
        """format as prometheus"""
        output_lines = []

        for (metric_name, label), value in aggregated_metrics.items():
            output_lines.append(f"{metric_name}{{{label}}} {value}")

        return "\n".join(output_lines)


# quick usage example
if __name__ == "__main__":
    # configure your node URLs
    nodes = [
        "http://localhost:9000/metrics",
    ]

    aggregator = MetricsAggregator(lambda: nodes)
    metrics = aggregator.aggregate_metrics({"role": "prefiler"})

    print(aggregator.format_as_prometheus(metrics))
