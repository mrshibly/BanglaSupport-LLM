"""
Agentic Tool Calling module.
Defines mock APIs for order tracking, refund initiation, and return eligibility.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple

DB_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "orders.db"

def init_mock_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            status TEXT,
            location TEXT,
            delivery_date TEXT,
            amount REAL
        )
    """)
    # Insert sample order if empty
    cursor.execute("SELECT COUNT(*) FROM orders")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO orders VALUES (?, ?, ?, ?, ?)
        """, [
            ("BD1001", "ঢাকার হাবে পৌঁছায়েছে", "ঢাকা সেন্ট্রাল হাব", "2026-07-21", 1250.00),
            ("BD1002", "ডেলিভারির জন্য বের হয়েছে", "মিরপুর এরিয়া", "2026-07-20", 890.00),
            ("BD1003", "বাতিল করা হয়েছে", "N/A", "N/A", 450.00)
        ])
    conn.commit()
    conn.close()

# Initialize DB on import
init_mock_db()

def get_order_status(order_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT status, location, delivery_date FROM orders WHERE order_id = ?", (order_id.upper(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"found": True, "order_id": order_id, "status": row[0], "location": row[1], "estimated_delivery": row[2]}
    return {"found": False, "order_id": order_id, "message": "অর্ডার আইডিটি পাওয়া যায়নি।"}

def check_return_eligibility(order_id: str) -> Dict[str, Any]:
    return {"order_id": order_id, "eligible": True, "reason": "পণ্য গ্রহণের ৭ দিন সময়সীমার ভেতরে রয়েছে।"}

TOOLS = {
    "get_order_status": get_order_status,
    "check_return_eligibility": check_return_eligibility
}

def detect_tool_intent(message: str) -> Tuple[str, Dict[str, Any]]:
    import re
    # Match order numbers like BD1001 or 1001
    match = re.search(r'(BD\d{4}|\b\d{4}\b)', message, re.IGNORECASE)
    if match and ("অর্ডার" in message or "কোথায়" in message or "স্ট্যাটাস" in message):
        order_id = match.group(1)
        if not order_id.upper().startswith("BD"):
            order_id = f"BD{order_id}"
        return "get_order_status", {"order_id": order_id}
    
    if match and ("ফেরত" in message or "রিটার্ন" in message):
        order_id = match.group(1)
        return "check_return_eligibility", {"order_id": order_id}

    return None, {}
