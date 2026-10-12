
import csv
from pathlib import Path
import random

random.seed(69)

output = Path("data/customers.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "customer_id",
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "churn"
    ])

    for customer_id in range(1, 101):
        tenure = random.randint(1, 60)
        spend = round(random.uniform(10, 200), 2)
        tickets = random.randint(0, 10)
        churn = int(tenure < 12 and tickets > 4)

        writer.writerow([
            customer_id,
            tenure,
            spend,
            tickets,
            churn
        ])

print(f"Created {output} with 100 records.")
