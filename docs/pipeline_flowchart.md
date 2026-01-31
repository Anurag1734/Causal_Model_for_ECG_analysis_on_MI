---
config:
  layout: elk
---
flowchart LR
    %% GLOBAL STYLES
    classDef script fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,stroke-dasharray: 5 5
    classDef data fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef artifact fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef model fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef validation fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef future fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,stroke-dasharray: 10 5

    %% ==========================================
    %% ZONE 1: DATA INGESTION (Phase B)
    %% ==========================================
    subgraph Phase_B["✓ Phase B: Data Ingestion (COMPLETED)"]
        direction TB
        RawFiles[("MIMIC-IV CSVs\n299,712 patients")]:::data
        RawWFDB[("MIMIC-IV-ECG\nWFDB Files\n800,000+ ECGs")]:::data
        ExtData[("PTB-XL Dataset\n(Validation Only)")]:::data
        
        subgraph Ingestion_Script["data_ingestion.py"]
            direction TB
            LoadMIMIC["Load 7 MIMIC-IV Tables:\npatients, admissions,\ndiagnoses_icd, labevents,\nd_labitems, chartevents,\nprescriptions"]:::process
            LoadECG["Load 2 MIMIC-ECG Tables:\nrecord_list,\nmachine_measurements"]:::process
            CreateIndices["Create 6 Indices:\nlabevents (subject, hadm, time)\nchartevents (subject, time)\ndiagnoses_icd (hadm)"]:::process
        end
        
        DuckDB[("Local DuckDB\nmimic_database.duckdb\n18.4 GB, 9 Tables")]:::artifact
    end

    %% ==========================================
    %% ZONE 2: COHORT DEFINITION (Phase C)
    %% ==========================================
    subgraph Phase_C["✓ Phase C: Cohort Definition (COMPLETED)"]
        direction TB
        
        subgraph Cohort_Script["cohort_labeling.py"]
            direction TB
            TropID["C.1: Identify\nTroponin Assays"]:::process
            TropThresh["C.2: Define Thresholds\n(0.10 ng/mL)"]:::process
            MIEvents["C.3: Define MI Events\n(Time-Anchored)"]:::process
            Labels["C.4: Primary Labels\n(MI vs Control)"]:::process
            Controls["C.5: Define Control Groups\n(Symptomatic)"]:::process
            Comorbid["C.6: Comorbidity Features\n(ICD-10 codes)"]:::process
            PowerAnalysis["C.8: Power Analysis\n(Sample Size)"]:::validation
        end
        
        Adjudication[("C.7: Clinician\nAdjudication\n(Protocol + Template)")]:::validation
        
        CohortMaster[("cohort_master.parquet\n259,117 ECGs\n80,316 subjects")]:::artifact
        CohortStrata[("Stratified Cohorts:\n• strict: 2,902\n• moderate: 4,954\n• broad: 8,223")]:::artifact
        PowerReport[("power_analysis_report.txt")]:::artifact
    end

    %% ==========================================
    %% ZONE 3: FEATURE ENGINEERING (Phase D)
    %% ==========================================
    subgraph Phase_D["✓ Phase D: Feature Engineering (COMPLETED)"]
        direction TB
        
        %% Track 1: Explicit Features
        subgraph Track1["Track 1: Explicit Clinical Features"]
            direction TB
            
            subgraph Feature_Script["ecg_feature_extraction.py"]
                direction TB
                PTB_Valid["D.1: PTB-XL+\nValidation\n(QRS, QT, QTc)"]:::validation
                NeuroKit["D.2: NeuroKit2\nExtraction\n(24 features)"]:::process
                Quality["D.3: Quality Control\n(Plausibility Checks)"]:::validation
            end
            
            ECGFeatsRaw[("ecg_features.parquet\n125,882 records\n24 features")]:::artifact
            ECGFeatsClean[("ecg_features_clean.parquet\n47,852 records\n(quality-filtered)")]:::artifact
            ECGFeatsFinal[("ecg_features_with_demographics.parquet\n47,852 records, 29 columns\n(FINAL DATASET)")]:::artifact
        end

        %% Track 2: Latent Features (VAE)
        subgraph Track2["Track 2: Latent Representation Learning"]
            direction TB
            
            subgraph Dataset_Module["ecg_dataset.py"]
                direction TB
                DatasetClass["ECGDataset Class:\n• Load WFDB signals\n• Per-lead normalization\n• Quality filtering"]:::script
                Splits["Train/Val/Test Splits:\n• Stratified by Label\n• 70/15/15"]:::process
            end
            
            subgraph VAE_Architecture["vae_conv1d.py"]
                direction TB
                Encoder["Conv1D Encoder:\n12 leads → 64 latent\n(3 conv layers)"]:::model
                LatentSpace["Latent Space Z\n64 dimensions\nμ, log(σ²)"]:::model
                Decoder["Conv1D Decoder:\n64 latent → 12 leads\n(3 transpose conv)"]:::model
            end
            
            subgraph Training_Script["train_vae.py"]
                direction TB
                BetaSchedule["β-Annealing:\n4 cycles × 40 epochs\n0 → 4.0 (cyclical)"]:::process
                FreeBits["Free Bits: 2.0\n(prevent collapse)"]:::process
                EarlyStopping["Early Stopping:\nPatience = 10"]:::validation
            end
            
            VAEModel[("β-VAE Model\n82.4M parameters\nβ=4.0, z_dim=64")]:::model
            LatentZ[("Latent Embeddings\n64-dim vectors\n(extraction pending)")]:::artifact
        end
    end

    %% ==========================================
    %% ZONE 4: MODELING & APP (Phase G-M) - FUTURE WORK
    %% ==========================================
    subgraph Phase_Future["⚠ Phases G-M: Causal Inference & Application (PLANNED)"]
        direction TB
        
        MergedData[("Master Dataset:\nExplicit + Latent + Demographics")]:::future
        
        %% Predictive & Causal Models
        subgraph Models["Baseline & Causal Models"]
            direction TB
            XGBoost["Baseline:\nXGBoost"]:::future
            
            subgraph Causal_Engine["Causal Inference Framework"]
                direction TB
                DAG["Causal DAG\n(Expert-driven)"]:::future
                IRM["IRM\n(Env. Robustness)"]:::future
                CATE["Causal Forest\n(Heterogeneity)"]:::future
            end
        end

        %% Application Layer
        subgraph App_Layer["Clinical Application"]
            direction TB
            Counterfactuals["Counterfactual\nGeneration"]:::future
            NegControl{"Negative\nControls"}:::future
            
            subgraph Deliverables["Deliverables"]
                direction TB
                Streamlit["Streamlit\nDashboard"]:::future
                RiskRep["Risk Reports\n(ATE/CATE)"]:::future
                SynECG["Synthetic ECGs"]:::future
            end
        end
    end

    %% ==========================================
    %% CONNECTIONS - PHASE B
    %% ==========================================
    RawFiles --> LoadMIMIC
    RawFiles --> LoadECG
    LoadMIMIC --> CreateIndices
    LoadECG --> CreateIndices
    CreateIndices --> DuckDB
    
    %% ==========================================
    %% CONNECTIONS - PHASE C
    %% ==========================================
    DuckDB --> TropID
    TropID --> TropThresh
    TropThresh --> MIEvents
    MIEvents --> Labels
    Labels --> Controls
    Controls --> Comorbid
    Comorbid --> PowerAnalysis
    PowerAnalysis --> CohortMaster
    PowerAnalysis --> PowerReport
    PowerAnalysis -.-> Adjudication
    Adjudication -.-> CohortMaster
    CohortMaster --> CohortStrata
    
    %% ==========================================
    %% CONNECTIONS - PHASE D (TRACK 1)
    %% ==========================================
    ExtData -.-> PTB_Valid
    CohortMaster --> NeuroKit
    RawWFDB -.-> NeuroKit
    PTB_Valid --> NeuroKit
    NeuroKit --> Quality
    Quality --> ECGFeatsRaw
    ECGFeatsRaw --> ECGFeatsClean
    ECGFeatsClean --> ECGFeatsFinal
    DuckDB --> ECGFeatsFinal
    
    %% ==========================================
    %% CONNECTIONS - PHASE D (TRACK 2)
    %% ==========================================
    CohortMaster --> DatasetClass
    RawWFDB --> DatasetClass
    DatasetClass --> Splits
    Splits --> Encoder
    Encoder --> LatentSpace
    LatentSpace --> Decoder
    Decoder -.-> BetaSchedule
    BetaSchedule --> FreeBits
    FreeBits --> EarlyStopping
    EarlyStopping --> VAEModel
    LatentSpace -.-> LatentZ
    
    %% ==========================================
    %% CONNECTIONS - FUTURE WORK
    %% ==========================================
    ECGFeatsFinal -.-> MergedData
    LatentZ -.-> MergedData
    MergedData -.-> XGBoost
    MergedData -.-> IRM
    MergedData -.-> CATE
    DAG -.-> CATE
    CATE -.-> RiskRep
    CATE -.-> NegControl
    LatentSpace -.-> Counterfactuals
    Counterfactuals -.-> Decoder
    Decoder -.-> SynECG
    RiskRep -.-> Streamlit
    SynECG -.-> Streamlit

    %% ==========================================
    %% ANNOTATIONS
    %% ==========================================
    
    %% Add status indicators
    Phase_B:::validation
    Phase_C:::validation
    Phase_D:::validation
    Phase_Future:::future
