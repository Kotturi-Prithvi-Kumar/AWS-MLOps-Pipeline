
import csv
import random
from pathlib import Path


def generate_dataset(
    output_path="data/customers.csv",
    num_records=100,
    seed=69,
):
    """Generate a reproducible synthetic customer churn dataset."""

    if num_records <= 0:
        raise ValueError("num_records must be greater than zero")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "customer_id",
            "tenure_months",
            "monthly_spend",
            "support_tickets",
            "churn",
        ])

        for customer_id in range(1, num_records + 1):
            tenure = rng.randint(1, 60)
            spend = round(rng.uniform(10, 200), 2)
            tickets = rng.randint(0, 10)

            churn = int(tenure < 12 and tickets > 4)

            writer.writerow([
                customer_id,
                tenure,
                spend,
                tickets,
                churn,
            ])

    print(f"Created {output} with {num_records} records.")

    return output


if __name__ == "__main__":
    generate_dataset()
