"""
Generate synthetic Vietnamese credit card fraud dataset.
Maintains the same schema as the original Kaggle dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================================================
# VIETNAMESE DATA
# =============================================================================

# Vietnamese first names (Họ)
VIETNAMESE_FIRST_NAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng",
    "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Đinh", "Trương", "Mai", "Lưu",
    "Tạ", "Hà", "Chu", "Cao", "Tô", "Lâm", "Quách", "Thái", "Kiều", "Từ"
]

# Vietnamese last names (Tên)
VIETNAMESE_LAST_NAMES = [
    "Anh", "Bình", "Chi", "Dũng", "Em", "Giang", "Hà", "Hải", "Hoa", "Hùng",
    "Hương", "Khoa", "Lan", "Linh", "Long", "Mai", "Minh", "Nam", "Nga", "Ngọc",
    "Phong", "Phúc", "Quang", "Sơn", "Tâm", "Thanh", "Thảo", "Thiên", "Thu", "Trang",
    "Tuấn", "Tùng", "Uyên", "Vân", "Việt", "Xuân", "Yến", "Đạt", "Đức", "Đăng"
]

# Vietnamese cities/provinces
VIETNAMESE_CITIES = [
    {"city": "Hà Nội", "state": "HN", "lat": 21.0285, "long": 105.8542, "city_pop": 8000000},
    {"city": "TP. Hồ Chí Minh", "state": "HCM", "lat": 10.8231, "long": 106.6297, "city_pop": 9000000},
    {"city": "Đà Nẵng", "state": "ĐN", "lat": 16.0471, "long": 108.2068, "city_pop": 1100000},
    {"city": "Hải Phòng", "state": "HP", "lat": 20.8449, "long": 106.6881, "city_pop": 2000000},
    {"city": "Cần Thơ", "state": "CT", "lat": 10.0452, "long": 105.7469, "city_pop": 1200000},
    {"city": "Nha Trang", "state": "KH", "lat": 12.2388, "long": 109.1967, "city_pop": 400000},
    {"city": "Huế", "state": "TTh", "lat": 16.4637, "long": 107.5909, "city_pop": 350000},
    {"city": "Đà Lạt", "state": "LĐ", "lat": 11.9404, "long": 108.4583, "city_pop": 250000},
    {"city": "Vinh", "state": "NA", "lat": 18.6739, "long": 105.6823, "city_pop": 300000},
    {"city": "Hải Dương", "state": "HD", "lat": 20.9375, "long": 106.3146, "city_pop": 500000},
    {"city": "Bắc Ninh", "state": "BN", "lat": 21.1861, "long": 106.0763, "city_pop": 400000},
    {"city": "Bình Dương", "state": "BD", "lat": 11.1257, "long": 106.6779, "city_pop": 2000000},
    {"city": "Đồng Nai", "state": "ĐNai", "lat": 10.9453, "long": 106.8547, "city_pop": 3000000},
    {"city": "Quảng Ninh", "state": "QN", "lat": 21.0061, "long": 107.2728, "city_pop": 1200000},
    {"city": "Thái Nguyên", "state": "TNg", "lat": 21.5942, "long": 105.8482, "city_pop": 350000},
    {"city": "Nam Định", "state": "NĐ", "lat": 20.4389, "long": 106.1621, "city_pop": 250000},
    {"city": "Thanh Hóa", "state": "TH", "lat": 19.8076, "long": 105.7764, "city_pop": 400000},
    {"city": "Nghệ An", "state": "NA", "lat": 18.6745, "long": 105.6902, "city_pop": 350000},
    {"city": "An Giang", "state": "AG", "lat": 10.3864, "long": 105.4352, "city_pop": 200000},
    {"city": "Kiên Giang", "state": "KG", "lat": 10.0095, "long": 105.0809, "city_pop": 180000},
]

# Vietnamese merchants by category
VIETNAMESE_MERCHANTS = {
    "grocery_pos": [
        "Co.opMart", "Big C", "VinMart", "Lotte Mart", "Aeon Mall",
        "Maximark", "Citimart", "Family Mart", "Circle K", "GS25"
    ],
    "grocery_net": [
        "Tiki", "Sendo", "ShopeeFood", "Lazada", "Amazon VN",
        "FreshGo", "Now", "GrabMart", "Bach Hoa Xanh", "Nest"
    ],
    "gas_transport": [
        "Petrolimex", "Saigon Petro", "PV Oil", "Cao Ngạn", "Phú Nhuận",
        "GrabCar", "Be", "Xanh SM", "FastGo", "Gojek"
    ],
    "entertainment": [
        "CGV", "Lotte Cinema", "Beta", "Galaxy Cinema", "BHD Star",
        "Netflix VN", "Spotify", "Zing MP3", "Karaoke", "Bowling"
    ],
    "food_dining": [
        "Highlands Coffee", "The Coffee House", "Starbucks", "Phở 24", "Cơm Tấm Cali",
        "KFC VN", "Lotteria", "Jollibee", "Pizza Hut", "Domino's",
        "GrabFood", "Baemin", "NowFood", "ShopeeFood", "Deliveree"
    ],
    "health_fitness": [
        "California Fitness", "Elite Fitness", "Golds Gym", "Yoga Plus", "SPA",
        "Phòng Khám", "Bệnh Viện", "Nhà Thuốc", "Long Châu", "Pharmacity"
    ],
    "home": [
        "Home Credit", "Fe Credit", "PPF", "FPT Shop", "Điện Máy Xanh",
        "Toyota", "Honda", "Yamaha", "Suzuki", "Ford VN"
    ],
    "kids_pets": [
        "Toys "R" Us", "Chicky", "Pet Mart", "Thu Cưng", "MobiFone",
        "Viettel", "Vinaphone", "School", "IELTS", "Mathnasium"
    ],
    "misc_net": [
        "ZaloPay", "MoMo", "Vietcombank", "Techcombank", "ACB",
        "VPBank", "BIDV", "Agribank", "MB Bank", "TPBank"
    ],
    "misc_pos": [
        "Vietnam Post", "Grab Express", "J&T Express", "GHN", "GHTK",
        "Bưu Điện", "Bus", "Taxi", "Metro", "VinBus"
    ],
    "personal_care": [
        "Watsons", "Guardian", "Hasaki", "The Face Shop", "Sephora",
        "L'Oréal", "Aphro", "Lush", "Yves Rocher", "Organic"
    ],
    "shopping_net": [
        "Shopee", "Lazada", "Tiki", "Sendo", "Fahasa",
        "Tiki Trading", "Amazon VN", "Ebay VN", "Nguyenkim", "Dienmaycholon"
    ],
    "shopping_pos": [
        "Vincom", "Aeon", "Lotte", "Diamond Plaza", "Saigon Centre",
        "Landmark", "Tràng Tiền", "Bitexco", "Royal City", "Times City"
    ],
    "travel": [
        "Vietnam Airlines", "Jetstar", "Bamboo Airways", "Vietravel", "Saigontourist",
        "Agoda", "Booking.com", "Airbnb", "Traveloka", "Expedia"
    ],
}

# Vietnamese street names (for realism)
VIETNAMESE_STREETS = [
    "Nguyễn Huệ", "Lê Lợi", "Trần Hưng Đạo", "Hai Bà Trưng", "Phan Đình Phùng",
    "Lý Thường Kiệt", "Võ Văn Tần", "Nguyễn Văn Cừ", "Lê Văn Sỹ", "Cách Mạng Tháng 8",
    "Võ Thị Sáu", "Nguyễn Đình Chiểu", "Lê Duẩn", "Điện Biên Phủ", "Ba Tháng Hai",
    "Nguyễn Trãi", "Hàm Nghi", "Tôn Đức Thắng", "Hoàng Văn Thụ", "Nguyễn Thị Minh Khai"
]

# Jobs (Vietnamese context)
VIETNAMESE_JOBS = [
    "Kỹ sư phần mềm", "Giáo viên", "Bác sĩ", "Dược sĩ", "Kế toán",
    "Nhân viên văn phòng", "Chuyên viên tư vấn", "Quản lý", "Giám đốc",
    "Chủ cửa hàng", "Freelancer", "Nhà báo", "Luật sư", "Kiến trúc sư",
    "Thợ điện", "Thợ cơ khí", "Tài xế", "Bán hàng", "Marketing", "HR",
    "Nhân viên ngân hàng", "Chuyên gia tài chính", "Data Scientist", "Designer"
]

# Vietnamese last names for cards (for generating cc_num suffix)
VIETNAMESE_CARD_PREFIXES = ["9704", "9705", "9706", "9707", "9708", "9709"]


def generate_cc_num():
    """Generate a realistic Vietnamese credit card number."""
    prefix = random.choice(["4", "5", "3"])  # Visa/Mastercard/JCB
    number = prefix + ''.join([str(random.randint(0, 9)) for _ in range(15)])
    return number


def generate_transaction_datetime(start_date, end_date):
    """Generate a random datetime within the range."""
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)


def generate_transaction(city_data, is_fraud, start_date, end_date):
    """Generate a single transaction."""
    # Basic info
    first_name = random.choice(VIETNAMESE_FIRST_NAMES)
    last_name = random.choice(VIETNAMESE_LAST_NAMES)
    gender = random.choice(["M", "F"])
    dob_year = random.randint(1960, 2000)
    dob_month = random.randint(1, 12)
    dob_day = random.randint(1, 28)
    dob = datetime(dob_year, dob_month, dob_day)
    
    # Location
    city_info = city_data
    city = city_info["city"]
    state = city_info["state"]
    lat = city_info["lat"] + random.uniform(-0.1, 0.1)
    long = city_info["long"] + random.uniform(-0.1, 0.1)
    city_pop = city_info["city_pop"]
    
    # Street address
    street_num = random.randint(1, 200)
    street_name = random.choice(VIETNAMESE_STREETS)
    street = f"{street_num} {street_name}"
    
    # Transaction datetime
    trans_datetime = generate_transaction_datetime(start_date, end_date)
    
    # Amount - fraud transactions tend to have different patterns
    if is_fraud:
        # Fraud transactions: more likely to be very small or very large
        if random.random() < 0.3:
            amt = random.uniform(50, 500)  # Small amount testing
        elif random.random() < 0.5:
            amt = random.uniform(5000, 50000)  # Large fraudulent amount
        else:
            amt = random.uniform(200, 2000)  # Medium fraud
    else:
        # Legitimate transactions: more normal distribution
        amt = random.uniform(20, 2000)
    
    amt = round(amt, 2)
    
    # Category and merchant
    category = random.choice(list(VIETNAMESE_MERCHANTS.keys()))
    merchant = random.choice(VIETNAMESE_MERCHANTS[category])
    
    # Job
    job = random.choice(VIETNAMESE_JOBS)
    
    # Card number
    cc_num = generate_cc_num()
    
    # Transaction number
    trans_num = ''.join([random.choice('0123456789abcdef') for _ in range(32)])
    
    # Merchant location (slightly different from cardholder for realism)
    merch_lat = lat + random.uniform(-0.05, 0.05)
    merch_long = long + random.uniform(-0.05, 0.05)
    
    # Zip code (Vietnamese format: 6 digits)
    zip_code = f"{random.randint(100000, 999999)}"
    
    # Unix time
    unix_time = int(trans_datetime.timestamp())
    
    return {
        "trans_date_trans_time": trans_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": cc_num,
        "merchant": f"fraud_{merchant}" if is_fraud else merchant,
        "category": category,
        "amt": amt,
        "first": first_name,
        "last": last_name,
        "gender": gender,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": round(lat, 4),
        "long": round(long, 4),
        "city_pop": city_pop,
        "job": job,
        "dob": dob.strftime("%Y-%m-%d"),
        "trans_num": trans_num,
        "unix_time": unix_time,
        "merch_lat": round(merch_lat, 4),
        "merch_long": round(merch_long, 4),
        "is_fraud": 1 if is_fraud else 0
    }


def generate_dataset(n_samples, fraud_rate=0.05, start_date=None, end_date=None):
    """Generate a complete dataset with specified fraud rate."""
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    if end_date is None:
        end_date = datetime(2024, 12, 31)
    
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    transactions = []
    
    # Generate legitimate transactions
    print(f"Generating {n_legit} legitimate transactions...")
    for i in range(n_legit):
        city_data = random.choice(VIETNAMESE_CITIES)
        trans = generate_transaction(city_data, is_fraud=False, 
                                     start_date=start_date, end_date=end_date)
        transactions.append(trans)
        if (i + 1) % 50000 == 0:
            print(f"  Generated {i + 1}/{n_legit} legitimate transactions...")
    
    # Generate fraud transactions
    print(f"Generating {n_fraud} fraud transactions...")
    for i in range(n_fraud):
        city_data = random.choice(VIETNAMESE_CITIES)
        trans = generate_transaction(city_data, is_fraud=True,
                                     start_date=start_date, end_date=end_date)
        transactions.append(trans)
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1}/{n_fraud} fraud transactions...")
    
    # Shuffle the dataset
    print("Shuffling dataset...")
    random.shuffle(transactions)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Reorder columns to match original schema
    columns = [
        "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
        "first", "last", "gender", "street", "city", "state", "zip",
        "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
        "merch_lat", "merch_long", "is_fraud"
    ]
    df = df[columns]
    
    return df


def main():
    """Generate Vietnamese datasets."""
    output_dir = Path(__file__).parent.parent / "dataset"
    output_dir.mkdir(exist_ok=True)
    
    # Generate training set (~500k rows, manageable for training)
    print("=" * 60)
    print("VIETNAMESE CREDIT CARD FRAUD DATASET GENERATOR")
    print("=" * 60)
    print(f"\nGenerating training set (500,000 rows)...")
    train_df = generate_dataset(
        n_samples=500000,
        fraud_rate=0.05,  # 5% fraud rate (realistic for credit cards)
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    train_path = output_dir / "fraudTrain.csv"
    train_df.to_csv(train_path, index=False)
    print(f"\n[OK] Training set saved to: {train_path}")
    print(f"  Shape: {train_df.shape}")
    print(f"  Fraud rate: {train_df['is_fraud'].mean() * 100:.2f}%")
    
    # Generate test set (~100k rows)
    print(f"\nGenerating test set (100,000 rows)...")
    test_df = generate_dataset(
        n_samples=100000,
        fraud_rate=0.05,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30)
    )
    
    test_path = output_dir / "fraudTest.csv"
    test_df.to_csv(test_path, index=False)
    print(f"\n[OK] Test set saved to: {test_path}")
    print(f"  Shape: {test_df.shape}")
    print(f"  Fraud rate: {test_df['is_fraud'].mean() * 100:.2f}%")
    
    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE DATA (First 5 rows)")
    print("=" * 60)
    print(train_df.head())
    
    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION")
    print("=" * 60)
    print(train_df["category"].value_counts())
    
    print("\n" + "=" * 60)
    print("STATE DISTRIBUTION")
    print("=" * 60)
    print(train_df["state"].value_counts())
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    from pathlib import Path
    main()
