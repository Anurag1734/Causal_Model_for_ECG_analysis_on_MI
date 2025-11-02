import pandas as pd

df = pd.read_parquet('data/processed/ecg_features.parquet')
print(f'Total records: {len(df)}')
print(f'Successful: {df["extraction_success"].sum()}')
print(f'Failed: {(~df["extraction_success"]).sum()}')

print('\nSample of successful records:')
successful = df[df['extraction_success']==True]
if len(successful) > 0:
    print(successful.head(3)[['record_id', 'heart_rate', 'qrs_duration_ms', 'qt_interval_ms']])
else:
    print("NO SUCCESSFUL EXTRACTIONS!")
    
print('\nSample of failed records:')
failed = df[df['extraction_success']==False]
print(failed.head(3)[['record_id', 'extraction_success']])
