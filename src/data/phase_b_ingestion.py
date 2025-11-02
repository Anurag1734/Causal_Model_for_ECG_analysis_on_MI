"""
Phase B: MIMIC-IV and MIMIC-ECG Data Ingestion Script
======================================================

This script ingests all necessary MIMIC-IV and MIMIC-ECG data into a local DuckDB database.
It follows a methodical approach:
    1. Database setup and connection
    2. Load MIMIC-IV tables (Parquet format)
    3. Load MIMIC-ECG tables (CSV format)
    4. Create performance indices

Author: Data Engineering Team
Date: October 23, 2025
"""

import duckdb
import os


# =============================================================================
# PATH CONFIGURATION - EDIT THESE PATHS TO MATCH YOUR LOCAL SETUP
# =============================================================================

# Base path to data directory
DATA_BASE_PATH = "data/raw/"

# Path to MIMIC-IV data directory (contains hosp/ and icu/ subdirectories with CSV files)
MIMIC_IV_HOSP_PATH = f"{DATA_BASE_PATH}MIMIC-IV-2.2/hosp/"
MIMIC_IV_ICU_PATH = f"{DATA_BASE_PATH}MIMIC-IV-2.2/icu/"

# Path to MIMIC-ECG data directory (contains CSV files)
MIMIC_ECG_PATH = f"{DATA_BASE_PATH}MIMIC-IV-ECG-1.0/"


# =============================================================================
# MAIN INGESTION FUNCTION
# =============================================================================

def main():
    """
    Main function to orchestrate the complete data ingestion pipeline.
    Connects to DuckDB, loads all required tables, and creates indices.
    """
    
    # Database file path
    db_path = "mimic_database.duckdb"
    
    print("=" * 70)
    print("PHASE B: MIMIC DATA INGESTION")
    print("=" * 70)
    print(f"Database file: {db_path}")
    print(f"MIMIC-IV Hospital path: {MIMIC_IV_HOSP_PATH}")
    print(f"MIMIC-IV ICU path: {MIMIC_IV_ICU_PATH}")
    print(f"MIMIC-ECG path: {MIMIC_ECG_PATH}")
    print("=" * 70)
    print()
    
    # Connect to DuckDB database
    con = duckdb.connect(db_path)
    
    try:
        # =====================================================================
        # B.2: LOAD MIMIC-IV TABLES (CSV Format)
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.2: LOADING MIMIC-IV TABLES")
        print("=" * 70 + "\n")
        
        # Load patients table (from hosp directory)
        print("Loading 'patients' table...")
        con.execute(f"""
            CREATE TABLE patients AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}patients.csv', header=True)
        """)
        print("✓ 'patients' table loaded successfully.\n")
        
        # Load admissions table (from hosp directory)
        print("Loading 'admissions' table...")
        con.execute(f"""
            CREATE TABLE admissions AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}admissions.csv', header=True)
        """)
        print("✓ 'admissions' table loaded successfully.\n")
        
        # Load diagnoses_icd table (from hosp directory)
        print("Loading 'diagnoses_icd' table...")
        con.execute(f"""
            CREATE TABLE diagnoses_icd AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}diagnoses_icd.csv', header=True)
        """)
        print("✓ 'diagnoses_icd' table loaded successfully.\n")
        
        # Load labevents table (from hosp directory) - This is a large file
        print("Loading 'labevents' table (this may take several minutes for large files)...")
        con.execute(f"""
            CREATE TABLE labevents AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}labevents.csv', header=True)
        """)
        print("✓ 'labevents' table loaded successfully.\n")
        
        # Load d_labitems table (from hosp directory)
        print("Loading 'd_labitems' table...")
        con.execute(f"""
            CREATE TABLE d_labitems AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}d_labitems.csv', header=True)
        """)
        print("✓ 'd_labitems' table loaded successfully.\n")
        
        # Load chartevents table (from icu directory) - This is a very large file
        print("Loading 'chartevents' table (this may take several minutes for large files)...")
        con.execute(f"""
            CREATE TABLE chartevents AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_ICU_PATH}chartevents.csv', header=True)
        """)
        print("✓ 'chartevents' table loaded successfully.\n")
        
        # Load prescriptions table (from hosp directory)
        print("Loading 'prescriptions' table...")
        con.execute(f"""
            CREATE TABLE prescriptions AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}prescriptions.csv', header=True)
        """)
        print("✓ 'prescriptions' table loaded successfully.\n")
        
        print("=" * 70)
        print("MIMIC-IV TABLES LOADED: 7/7")
        print("=" * 70)
        
        # =====================================================================
        # B.3: LOAD MIMIC-ECG TABLES (CSV Format)
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.3: LOADING MIMIC-ECG TABLES")
        print("=" * 70 + "\n")
        
        # Load record_list table
        print("Loading 'record_list' table...")
        con.execute(f"""
            CREATE TABLE record_list AS 
            SELECT * FROM read_csv_auto('{MIMIC_ECG_PATH}record_list.csv', header=True)
        """)
        print("✓ 'record_list' table loaded successfully.\n")
        
        # Load machine_measurements table
        print("Loading 'machine_measurements' table...")
        con.execute(f"""
            CREATE TABLE machine_measurements AS 
            SELECT * FROM read_csv_auto('{MIMIC_ECG_PATH}machine_measurements.csv', header=True)
        """)
        print("✓ 'machine_measurements' table loaded successfully.\n")
        
        print("=" * 70)
        print("MIMIC-ECG TABLES LOADED: 2/2")
        print("=" * 70)
        
        # =====================================================================
        # B.4: CREATE INDICES FOR PERFORMANCE OPTIMIZATION
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.4: CREATING PERFORMANCE INDICES")
        print("=" * 70 + "\n")
        
        print("Creating index on labevents.subject_id...")
        con.execute("CREATE INDEX idx_labevents_subject ON labevents(subject_id)")
        print("✓ Index 'idx_labevents_subject' created.\n")
        
        print("Creating index on labevents.hadm_id...")
        con.execute("CREATE INDEX idx_labevents_hadm ON labevents(hadm_id)")
        print("✓ Index 'idx_labevents_hadm' created.\n")
        
        print("Creating index on labevents.charttime...")
        con.execute("CREATE INDEX idx_labevents_charttime ON labevents(charttime)")
        print("✓ Index 'idx_labevents_charttime' created.\n")
        
        print("Creating index on chartevents.subject_id...")
        con.execute("CREATE INDEX idx_chartevents_subject ON chartevents(subject_id)")
        print("✓ Index 'idx_chartevents_subject' created.\n")
        
        print("Creating index on chartevents.charttime...")
        con.execute("CREATE INDEX idx_chartevents_charttime ON chartevents(charttime)")
        print("✓ Index 'idx_chartevents_charttime' created.\n")
        
        print("Creating index on diagnoses_icd.hadm_id...")
        con.execute("CREATE INDEX idx_diagnoses_hadm ON diagnoses_icd(hadm_id)")
        print("✓ Index 'idx_diagnoses_hadm' created.\n")
        
        print("=" * 70)
        print("INDICES CREATED: 6/6")
        print("=" * 70)
        
        # =====================================================================
        # COMPLETION MESSAGE
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("Phase B ingestion complete. Database is ready.")
        print("=" * 70)
        print(f"\nDatabase location: {os.path.abspath(db_path)}")
        print("Total tables loaded: 9")
        print("Total indices created: 6")
        print("\nYou can now proceed to Phase C: Data Processing and Analysis")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR DURING INGESTION")
        print("=" * 70)
        print(f"An error occurred: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nPlease check:")
        print("  1. File paths are correct")
        print("  2. All required files exist")
        print("  3. Files are not corrupted")
        print("  4. You have sufficient disk space")
        print("=" * 70 + "\n")
        raise
        
    finally:
        # Always close the database connection
        con.close()
        print("Database connection closed.")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
