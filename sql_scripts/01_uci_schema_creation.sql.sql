/******************************************************************************************
    PROJECT: SEGMENTATION PIPELINE (LOAN + UCI DATASETS)
    AUTHOR: DAVY
******************************************************************************************/

/* ---------------------------------------------------------
   1. CREATE DATABASE
--------------------------------------------------------- */

CREATE DATABASE IF NOT EXISTS segmentation_project;
USE segmentation_project;


/* ---------------------------------------------------------
   2. LOAN DATASET TABLES
--------------------------------------------------------- */

DROP TABLE IF EXISTS loan;
DROP TABLE IF EXISTS credit_profile;
DROP TABLE IF EXISTS customer;

/* -------------------------
   TABLE: customer
------------------------- */
CREATE TABLE customer (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    employment_type VARCHAR(255),
    marital_status VARCHAR(255),
    education VARCHAR(255),
    has_dependents VARCHAR(255),
    has_mortgage VARCHAR(255),
    has_co_signer VARCHAR(255)
);

/* -------------------------
   TABLE: credit_profile
------------------------- */
CREATE TABLE credit_profile (
    credit_profile_id INT AUTO_INCREMENT PRIMARY KEY,
    credit_score INT,
    income INT,
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
);

/* -------------------------
   TABLE: loan
------------------------- */
CREATE TABLE loan (
    loan_id VARCHAR(50) PRIMARY KEY,
    loan_amount INT,
    loan_term INT,
    interest_rate FLOAT,
    loan_purpose VARCHAR(255),
    months_employed INT,
    num_credit_lines INT,
    loan_default INT,
    loan_to_income FLOAT,
    dti_ratio FLOAT,
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customer(customer_id)
);


/* ---------------------------------------------------------
   3. UCI DATASET TABLES
--------------------------------------------------------- */

DROP TABLE IF EXISTS campaign_outcome;
DROP TABLE IF EXISTS campaign_contact;
DROP TABLE IF EXISTS customer_uci;

/* -------------------------
   TABLE: customer_uci
------------------------- */
CREATE TABLE customer_uci (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    age_group_bin VARCHAR(255),
    job_type VARCHAR(255),
    marital_status VARCHAR(255),
    education_level VARCHAR(255),
    credit_default VARCHAR(255),
    account_balance INT,
    housing_loan VARCHAR(255),
    personal_loan VARCHAR(255)
);

/* -------------------------
   TABLE: campaign_contact
------------------------- */
CREATE TABLE campaign_contact (
    contact_id INT AUTO_INCREMENT PRIMARY KEY,
    contact_type VARCHAR(255),
    contact_day INT,
    contact_month VARCHAR(255),
    call_duration_sec INT,
    num_contacts_current_campaign INT,
    days_since_last_contact INT,
    num_previous_contacts INT,
    contact_missing_flag INT,
    campaign_group VARCHAR(255),
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customer_uci(customer_id)
);

/* -------------------------
   TABLE: campaign_outcome
------------------------- */
CREATE TABLE campaign_outcome (
    outcome_id INT AUTO_INCREMENT PRIMARY KEY,
    previous_outcome VARCHAR(255),
    subscribed VARCHAR(255),
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customer_uci(customer_id)
);