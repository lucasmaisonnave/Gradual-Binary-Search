import json
import statistics
import csv
import os

def calculate_average_accuracy(results):
    accuracies = [values['acc,none'] for values in results.values() if 'acc,none' in values]
    return statistics.mean(accuracies) if accuracies else 0

def update_csv_with_averages(data, csv_filename):
    # Read existing CSV file
    rows = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['avg']
        rows = list(reader)

    # Calculate averages and update rows
    for i, config in enumerate(data):
        avg_accuracy = calculate_average_accuracy(config['results'])
        print(avg_accuracy)
        rows[i]['avg'] = f"{avg_accuracy:.4f}"

    # Write updated data back to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Load the JSON data
with open('./results/evaluation_results_smooth_mix.json', 'r') as file:
    data = json.load(file)

# Update CSV file with averages
csv_filename = './results/table.csv'  # Replace with your actual CSV filename
update_csv_with_averages(data, csv_filename)

print(f"CSV file '{csv_filename}' has been updated with average accuracies.")