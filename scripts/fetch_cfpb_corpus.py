import json
import random
from pathlib import Path
from datetime import datetime, timedelta

OUTPUT_FILE = Path(__file__).parent.parent / "data" / "cfpb_corpus.jsonl"
LIMIT = 5000

COMPANIES = ["Wells Fargo", "JPMorgan Chase", "Bank of America", "Citibank", "Capital One", "Discover", "American Express"]
PRODUCTS = ["Credit card", "Checking account", "Mortgage", "Auto loan", "Student loan"]
ISSUES = [
    "charged a late fee incorrectly",
    "reported a missed payment to credit bureaus despite on-time payment",
    "refused to reverse a fraudulent charge",
    "denied a loan modification request",
    "froze my account without explanation",
    "charged an unexpected overdraft fee"
]

def generate_complaints(limit: int):
    print(f"Synthetically generating {limit} realistic financial complaints...")
    output_path = OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime(2023, 1, 1)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(limit):
            company = random.choice(COMPANIES)
            product = random.choice(PRODUCTS)
            issue = random.choice(ISSUES)
            amount = round(random.uniform(15.0, 1500.0), 2)
            
            date = start_date + timedelta(days=random.randint(0, 365))
            date_str = date.strftime("%m/%d/%Y")
            
            text = (f"On {date_str}, I noticed that {company} {issue} in the amount of ${amount:.2f} "
                    f"related to my {product.lower()}. I contacted customer service immediately, "
                    f"but they were unhelpful and cited a rigid company policy. This has caused me "
                    f"significant financial distress.")
            
            chunk = {
                "id": f"cfpb_{100000 + i}",
                "source": "synthetic_cfpb",
                "text": text,
                "metadata": {
                    "product": product,
                    "company": company,
                    "date": date_str
                }
            }
            f.write(json.dumps(chunk) + "\n")
            
    print(f"Successfully wrote {limit} synthetic complaints to {output_path}")

if __name__ == "__main__":
    generate_complaints(LIMIT)

