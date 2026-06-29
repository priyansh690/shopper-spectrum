"""
Generate a realistic synthetic Online Retail dataset.
Matches the schema described in the Shopper Spectrum project.
Covers transaction dates in 2022–2023.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

# ── Product catalogue ────────────────────────────────────────────────────────
PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
    ("71053",  "WHITE METAL LANTERN"),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER"),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE"),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART"),
    ("22752",  "SET 7 BABUSHKA NESTING BOXES"),
    ("21730",  "GLASS STAR FROSTED T-LIGHT HOLDER"),
    ("22633",  "HAND WARMER UNION JACK"),
    ("22632",  "HAND WARMER RED POLKA DOT"),
    ("84879",  "ASSORTED COLOUR BIRD ORNAMENT"),
    ("47566",  "PARTY BUNTING"),
    ("20725",  "LUNCH BAG RED RETROSPOT"),
    ("22960",  "JAM MAKING SET WITH JARS"),
    ("22086",  "PAPER CHAIN KIT 50S CHRISTMAS"),
    ("85099B", "JUMBO BAG RED RETROSPOT"),
    ("23084",  "RABBIT NIGHT LIGHT"),
    ("22197",  "POPCORN HOLDER"),
    ("22693",  "GROW YOUR OWN PIZZA GARDEN"),
    ("22423",  "REGENCY CAKESTAND 3 TIER"),
    ("47566B", "TEA TIME PARTY BUNTING"),
    ("21212",  "PACK OF 72 RETROSPOT CAKE CASES"),
    ("22111",  "SCOTTIE DOG HOT WATER BOTTLE"),
    ("21977",  "PACK OF 60 PINK PAISLEY CAKE CASES"),
    ("22469",  "HEART OF WICKER SMALL"),
    ("22114",  "HOT WATER BOTTLE KEEP CALM"),
    ("20727",  "LUNCH BAG BLACK SKULL"),
    ("22383",  "LUNCH BAG SUKI DESIGN"),
    ("22386",  "JUMBO BAG PINK POLKADOT"),
    ("22457",  "NATURAL SLATE HEART CHALKBOARD"),
    ("22834",  "HAND CREAM LAVENDER BOTTLE"),
    ("23206",  "LUNCH BAG APPLE DESIGN"),
    ("84991",  "60 TEATIME FAIRY CAKE CASES"),
    ("22551",  "PLASTERS IN TIN WOODLAND ANIMALS"),
    ("21928",  "JUMBO BAG BAROQUE BLACK WHITE"),
    ("22411",  "JUMBO SHOPPER VINTAGE RED PAISLEY"),
    ("23355",  "HOT WATER BOTTLE BLUE ROSE"),
    ("22699",  "ROSES REGENCY TEACUP AND SAUCER"),
    ("22698",  "PANSIES REGENCY TEACUP AND SAUCER"),
    ("22697",  "DAISIES REGENCY TEACUP AND SAUCER"),
    ("84596F", "SMALL MARSHMALLOW PINK STICKER"),
    ("22578",  "BICYCLE PUNCTURE REPAIR KIT"),
    ("21108",  "FAIRY TALE COTTAGE NIGHT LIGHT"),
    ("21915",  "RED  HARMONICA IN BOX"),
    ("22375",  "AIRLINE BAG VINTAGE WORLD CHAMPION"),
    ("23321",  "VINTAGE SNAP CARDS"),
    ("23322",  "VINTAGE DOMINOES"),
    ("22326",  "ROUND SNACK BOXES SET OF 4 FRUITS"),
    ("22327",  "ROUND SNACK BOXES SET OF 4 WOODLAND"),
    ("22328",  "ROUND SNACK BOXES SET OF 4 ANIMALS"),
    ("22175",  "GIN AND TONIC DIET METAL SIGN"),
    ("22176",  "COCKTAIL PARTY METAL SIGN"),
    ("85099C", "JUMBO BAG STRAWBERRY"),
    ("22629",  "SPACEBOY LUNCH BOX"),
    ("22745",  "POPPY S PLAYHOUSE BEDROOM"),
    ("22748",  "POPPY S PLAYHOUSE KITCHEN"),
    ("22749",  "FELTCRAFT PRINCESS CHARLOTTE DOLL"),
    ("22771",  "CLEAR DRAWER KNOB"),
    ("22772",  "PINK DRAWER KNOB"),
    ("22941",  "CHRISTMAS LIGHTS 10 REINDEER"),
    ("21871",  "SAVE THE PLANET MUG"),
    ("21080",  "SET OF 6 SOLDIER SKITTLES"),
    ("22720",  "SET OF 3 CAKE TINS PANTRY DESIGN"),
    ("22721",  "IVORY SHOT GLASSES SET OF 6"),
    ("21843",  "CHRISTMAS HANGING CERAMIC BELL"),
    ("21089",  "CIRCULAR GIFT BOX SET OF 3"),
    ("23274",  "SET OF 4 KNITTED EASTER EGG CUPS"),
    ("22556",  "PLASTERS IN TIN CIRCUS PARADE"),
    ("22557",  "PLASTERS IN TIN VINTAGE PAISLEY"),
    ("22558",  "PLASTERS IN TIN CAMPERS"),
    ("22559",  "PLASTERS IN TIN WOODLAND ANIMALS"),
    ("22560",  "PLASTERS IN TIN RETROSPOT"),
    ("85123B", "PINK HANGING HEART T-LIGHT HOLDER"),
    ("20712",  "LUNCH BAG CARS BLUE"),
    ("23170",  "MIRRORED HEART GLASS TRAY"),
    ("22423B", "REGENCY CAKESTAND 2 TIER"),
    ("22960B", "JAM MAKING SET PRINTED"),
    ("22467",  "GINGHAM HEART DOORSTOP"),
    ("22468",  "GINGHAM HEART NAPKIN RINGS"),
    ("22659",  "LUNCH BAG WOODLAND"),
    ("22660",  "LUNCH BAG DOILEY"),
    ("22661",  "LUNCH BAG APPLE DESIGN"),
    ("22662",  "LUNCH BAG JUNGLE DESIGN"),
    ("22663",  "LUNCH BAG PINK POLKADOT"),
    ("22664",  "LUNCH BAG PINK SKULL"),
    ("22665",  "LUNCH BAG SPOTTY"),
    ("22150",  "IVORY SWEETHEART SOAP"),
    ("22151",  "ROSE SWEETHEART SOAP"),
    ("22152",  "BLUE SWEETHEART SOAP"),
    ("22730",  "ALARM CLOCK BAKELIKE IVORY"),
    ("22731",  "ALARM CLOCK BAKELIKE PINK"),
    ("22732",  "ALARM CLOCK BAKELIKE GREEN"),
    ("22733",  "ALARM CLOCK BAKELIKE RED"),
    ("23084B", "RABBIT NIGHT LIGHT PINK"),
    ("21080B", "SET OF 6 SOLDIER SKITTLES PINK"),
    ("22900",  "SET OF 6 T-LIGHTS VINTAGE STAR"),
    ("22901",  "SET OF 6 T-LIGHTS VINTAGE FLORAL"),
    ("22902",  "SET OF 6 T-LIGHTS FAIRY CAKE"),
    ("22903",  "SET OF 6 T-LIGHTS HEART"),
    ("22904",  "SET OF 6 T-LIGHTS WOODLAND"),
]

# ── Country distribution ─────────────────────────────────────────────────────
COUNTRIES = [
    ("United Kingdom", 0.82),
    ("Germany",        0.05),
    ("France",         0.04),
    ("Australia",      0.02),
    ("Netherlands",    0.02),
    ("Spain",          0.01),
    ("Sweden",         0.01),
    ("Japan",          0.01),
    ("Norway",         0.01),
    ("Portugal",       0.01),
]

COUNTRY_NAMES, COUNTRY_WEIGHTS = zip(*COUNTRIES)

# ── Date range ───────────────────────────────────────────────────────────────
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2023, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days


def random_date():
    """Return a random datetime within the 2022-2023 range."""
    day_offset = np.random.randint(0, TOTAL_DAYS)
    hour  = np.random.randint(8, 20)
    minute = np.random.randint(0, 60)
    return START_DATE + timedelta(days=int(day_offset), hours=hour, minutes=minute)


def generate_dataset(n_customers=4000, avg_invoices_per_customer=5, output_path="data/online_retail.csv"):
    """Generate a realistic synthetic e-commerce dataset."""

    print("[*] Generating synthetic Online Retail dataset ...")

    records = []
    invoice_counter = 536365

    # Assign customer types (affects RFM behavior)
    # high_value: ~10%, regular: ~30%, occasional: ~40%, at_risk: ~20%
    customer_types = np.random.choice(
        ["high_value", "regular", "occasional", "at_risk"],
        size=n_customers,
        p=[0.10, 0.30, 0.40, 0.20]
    )

    for cid_idx in range(n_customers):
        customer_id = 12000 + cid_idx
        ctype = customer_types[cid_idx]
        country = np.random.choice(COUNTRY_NAMES, p=COUNTRY_WEIGHTS)

        # Determine number of invoices based on customer type
        if ctype == "high_value":
            n_invoices = np.random.randint(15, 60)
            recency_days = np.random.randint(1, 30)        # recent
        elif ctype == "regular":
            n_invoices = np.random.randint(5, 20)
            recency_days = np.random.randint(15, 90)
        elif ctype == "occasional":
            n_invoices = np.random.randint(1, 6)
            recency_days = np.random.randint(60, 200)
        else:  # at_risk
            n_invoices = np.random.randint(1, 5)
            recency_days = np.random.randint(200, 365)

        # Last purchase date drives recency
        last_purchase = END_DATE - timedelta(days=recency_days)

        for inv_idx in range(n_invoices):
            # Distribute invoice dates; most recent invoice = last_purchase
            if inv_idx == 0:
                inv_date = last_purchase
            else:
                days_back = np.random.randint(1, min(TOTAL_DAYS, (END_DATE - START_DATE).days))
                inv_date = last_purchase - timedelta(days=int(days_back))
                inv_date = max(inv_date, START_DATE)

            invoice_no = str(invoice_counter)
            invoice_counter += 1

            # Random items per invoice (1–10)
            n_items = np.random.randint(1, 11)
            selected = random.sample(PRODUCTS, min(n_items, len(PRODUCTS)))

            for stock_code, description in selected:
                qty = np.random.randint(1, 24) if ctype == "high_value" else np.random.randint(1, 12)
                price_base = round(np.random.uniform(0.5, 15.0), 2)
                records.append({
                    "InvoiceNo":   invoice_no,
                    "StockCode":   stock_code,
                    "Description": description,
                    "Quantity":    qty,
                    "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "UnitPrice":   price_base,
                    "CustomerID":  customer_id,
                    "Country":     country,
                })

        # Add ~5% cancelled invoices
        if np.random.random() < 0.05:
            cancel_date = last_purchase - timedelta(days=np.random.randint(1, 30))
            cancel_date = max(cancel_date, START_DATE)
            cancel_no = "C" + str(invoice_counter)
            invoice_counter += 1
            stock_code, description = random.choice(PRODUCTS)
            records.append({
                "InvoiceNo":   cancel_no,
                "StockCode":   stock_code,
                "Description": description,
                "Quantity":    -np.random.randint(1, 6),
                "InvoiceDate": cancel_date.strftime("%Y-%m-%d %H:%M:%S"),
                "UnitPrice":   round(np.random.uniform(0.5, 15.0), 2),
                "CustomerID":  customer_id,
                "Country":     country,
            })

    # ~2% rows with missing CustomerID
    df = pd.DataFrame(records)
    missing_mask = np.random.random(len(df)) < 0.02
    df.loc[missing_mask, "CustomerID"] = np.nan

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'[OK] Dataset saved -> {output_path}')
    print(f'   Shape      : {df.shape}')
    print(f'   Customers  : {df["CustomerID"].nunique()} unique')
    print(f'   Products   : {df["Description"].nunique()} unique')
    print(f'   Date range : {df["InvoiceDate"].min()} -> {df["InvoiceDate"].max()}')
    return df


if __name__ == "__main__":
    generate_dataset()
