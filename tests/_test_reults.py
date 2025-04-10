import csv

from tabulate import tabulate


def parse_test_log(file_path):
    """Parse the test_log.csv file and extract test name, status, and time."""
    results = []
    with open(file_path, newline="") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            results.append(row)
    return results


def display_results(results):
    """Display the parsed results in a tabular format."""
    # Sort results by test Time
    results.sort(key=lambda x: float(x[2]))

    headers = ["Name", "Status", "Time"]
    print(tabulate(results, headers=headers, tablefmt="rounded_outline"))


if __name__ == "__main__":
    log_file_path = "tests/_test_log.csv"
    test_results = parse_test_log(log_file_path)
    display_results(test_results)
