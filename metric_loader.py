import pandas as pd
from typing import Dict, List


class GoogleAnalyticsMetricLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.root_domain = "https://www.getclientell.com"

    def load_metrics(self) -> List[Dict]:
        """
        Reads the Google Analytics metrics CSV file and returns a list of dictionaries
        containing selected metrics for each page.

        Returns:
            List[Dict]: List of dictionaries containing metrics for each page
        """
        try:
            # Read the first few lines to skip the header comments
            with open(self.file_path, 'r') as f:
                header_position = 0
                for i, line in enumerate(f):
                    if 'Page path and screen class' in line:
                        header_position = i
                        break

            # Read CSV with the correct header row
            df = pd.read_csv(self.file_path,
                             skiprows=header_position,
                             skipinitialspace=True)

            # Clean column names by removing any leading/trailing whitespace
            df.columns = df.columns.str.strip()

            # Select only the required columns
            selected_columns = [
                'Page path and screen class',
                'Views',
                'Engagement rate',
                'Bounce rate',
                'Average session duration',
                'Views per session'
            ]

            # Verify all required columns exist
            missing_columns = [
                col for col in selected_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in CSV: {missing_columns}")

            # Append root domain to each path
            df['Page path and screen class'] = df['Page path and screen class'].apply(
                lambda x: f"{self.root_domain}{x}" if pd.notna(x) else x
            )

            # Filter columns and convert to list of dictionaries
            metrics_data = df[selected_columns].to_dict('records')

            return metrics_data

        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return []
        except Exception as e:
            print(f"Error loading metrics: {str(e)}")
            return []


# Example usage:
if __name__ == "__main__":
    loader = GoogleAnalyticsMetricLoader("google_anlaytics_metrics.csv")
    metrics = loader.load_metrics()

    # Print first few entries to verify
    for metric in metrics:
        print(metric)
