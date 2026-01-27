/******************************************************************************************
    PROJECT: SEGMENTATION PIPELINE (LOAN + UCI DATASETS)
    AUTHOR: DAVY
    DESCRIPTION:
        - Creates the database segmentation_project
        - Creates all 6 tables from both ERDs
        - Includes PKs, FKs, auto-increment, and clean formatting
******************************************************************************************/

/* ---------------------------------------------------------
   1. CREATE DATABASE
--------------------------------------------------------- */

CREATE DATABASE IF NOT EXISTS segmentation_project;
USE segmentation_project;


/* ---------------------------------------------------------
   2. LOAN DATASET TABLES
--------------------------------------------------------- */


/* -------------------------
   TABLE: customer
   Description: Demographic and personal attributes of borrowers
------------------------- */

DROP TABLE IF EXISTS customer;

CREATE TABLE IF NOT EXISTS `customer` (
	`customer_id` INTEGER NOT NULL AUTO_INCREMENT UNIQUE,
	`age` INTEGER NOT NULL,
	`employment_type` VARCHAR(255) NOT NULL,
	`marital_status` VARCHAR(255) NOT NULL,
	`education` VARCHAR(255) NOT NULL,
	`has_dependents` VARCHAR(255) NOT NULL,
	` has_mortgage` VARCHAR(255) NOT NULL,
	`has_co_signer` VARCHAR(255) NOT NULL,
	PRIMARY KEY(`customer_id`)
);


/* -------------------------
   TABLE: credit_profile
   Description: Creditworthiness and financial capacity of customers
------------------------- */

DROP TABLE IF EXISTS credit_profile;

CREATE TABLE IF NOT EXISTS `credit_profile` (
	`credit_profile_id` INTEGER NOT NULL AUTO_INCREMENT UNIQUE,
	`credit_score` INTEGER NOT NULL,
	`credit_risk_group` VARCHAR(255) NOT NULL,
	`income` INTEGER NOT NULL,
	`customer_id` INTEGER NOT NULL,
	PRIMARY KEY(`credit_profile_id`)
);

/* -------------------------
   TABLE: loan
   Description: Loan-level financial attributes
------------------------- */

DROP TABLE IF EXISTS loan;

CREATE TABLE IF NOT EXISTS `loan` (
	`loan_id` VARCHAR(20) NOT NULL UNIQUE,
	`loan_amount` INTEGER NOT NULL,
	`loan_term` INTEGER NOT NULL,
	`interest_rate` FLOAT NOT NULL,
	`loan_purpose` VARCHAR(255) NOT NULL,
	`months_employed` INTEGER NOT NULL,
	`num_credit_lines` INTEGER NOT NULL,
	`loan_default` INTEGER NOT NULL,
	`loan_to_income` FLOAT NOT NULL,
	`dti_ratio` FLOAT NOT NULL,
	`customer_id` INTEGER NOT NULL,
	PRIMARY KEY(`loan_id`)
);



/* ---------------------------------------------------------
   3. UCI DATASET TABLES
--------------------------------------------------------- */


/* -------------------------
   TABLE: customer_uci
   Description: Customer demographic + financial attributes from UCI dataset
------------------------- */

DROP TABLE IF EXISTS customer_uci;

CREATE TABLE IF NOT EXISTS `customer_uci` (
	`customer_id` INTEGER NOT NULL AUTO_INCREMENT UNIQUE,
	`age` INTEGER NOT NULL,
	`age_group_bin` VARCHAR(255) NOT NULL,
	`job_type` VARCHAR(255) NOT NULL,
	`marital_status` VARCHAR(255) NOT NULL,
	`education_level` VARCHAR(255) NOT NULL,
	`credit_default` VARCHAR(255) NOT NULL,
	`account_balance` INTEGER NOT NULL,
	`housing_loan` VARCHAR(255) NOT NULL,
	`personal_loan` VARCHAR(255) NOT NULL,
	PRIMARY KEY(`customer_id`)
);


/* -------------------------
   TABLE: campaign_contact
   Description: Behavioral attributes related to campaign interactions
------------------------- */

DROP TABLE IF EXISTS campaign_contact;

CREATE TABLE IF NOT EXISTS `campaign_contact` (
	`contact_id` INTEGER NOT NULL AUTO_INCREMENT UNIQUE,
	`contact_type` VARCHAR(255) NOT NULL,
	`contact_day` INTEGER NOT NULL,
	`contact_month` VARCHAR(255) NOT NULL,
	`call_duration_sec` INTEGER NOT NULL,
	`num_contacts_current_campaign` INTEGER NOT NULL,
	`days_since_last_contact` INTEGER NOT NULL,
	`num_previous_contacts` INTEGER NOT NULL,
	`contact_missing_fla` INTEGER NOT NULL,
	`campaign_group` VARCHAR(255) NOT NULL,
	`customer_id` INTEGER NOT NULL,
	PRIMARY KEY(`contact_id`)
);


/* -------------------------
   TABLE: campaign_outcome
   Description: Final outcome of the marketing campaign
------------------------- */

DROP TABLE IF EXISTS campaign_outcome;

CREATE TABLE IF NOT EXISTS `campaign_outcome` (
	`id`INT AUTO_INCREMENT PRIMARY KEY,
	`outcome_id` INTEGER NOT NULL AUTO_INCREMENT UNIQUE,
	`previous_outcome` VARCHAR(255) NOT NULL,
	`subscribed` VARCHAR(255) NOT NULL,
	`customer_id` INTEGER,
	PRIMARY KEY(`outcome_id`)
);


ALTER TABLE `CREDIT_PROFILE`
ADD FOREIGN KEY(`customer_id`) REFERENCES `CUSTOMER`(`customer_id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;
ALTER TABLE `LOAN`
ADD FOREIGN KEY(`customer_id`) REFERENCES `CUSTOMER`(`customer_id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;
ALTER TABLE `CAMPAIGN_CONTACT`
ADD FOREIGN KEY(`customer_id`) REFERENCES `CUSTOMER`(`customer_id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;
ALTER TABLE `CAMPAIGN_OUTCOME`
ADD FOREIGN KEY(`customer_id`) REFERENCES `CUSTOMER`(`customer_id`)
ON UPDATE NO ACTION ON DELETE NO ACTION;