import sqlite3

# Check root database
try:
    conn = sqlite3.connect('fashion_app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Root DB tables: {tables}")
    conn.close()
except Exception as e:
    print(f"Root DB error: {e}")

# Check instance database
try:
    conn = sqlite3.connect('instance/fashion_app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Instance DB tables: {tables}")
    conn.close()
except Exception as e:
    print(f"Instance DB error: {e}")
