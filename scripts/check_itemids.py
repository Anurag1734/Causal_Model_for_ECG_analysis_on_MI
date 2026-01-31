import duckdb

conn = duckdb.connect('data/mimic_database.duckdb', read_only=True)

print("HDL itemids:")
print(conn.execute("SELECT itemid, label FROM d_labitems WHERE label LIKE '%HDL%' ORDER BY itemid").df())

print("\nTemperature itemids:")
print(conn.execute("SELECT itemid, label FROM d_items WHERE label LIKE '%emperature%' ORDER BY itemid LIMIT 10").df())

conn.close()
