## U.S. Opiate Prescriptions/Overdoses  ![image title](https://img.shields.io/badge/work-in%20progress-blue.svg) ![image title](https://img.shields.io/badge/python-v3.6-green.svg) ![Image title](https://img.shields.io/badge/sklearn-0.19.1-orange.svg)
<br>
<br>
<p align="center">
  <img src="images/opioids.png", width = "180">
</p>
<br>

<p align="center">
  <a href="#p">Brief Introduction </a> •
  <a href="#d"> Dataset </a> •
  <a href="#g"> Project Goal  </a>
</p>

<a id = 'p'></a>
## Brief Introduction

Accidental death by fatal drug overdose is a rising trend in the United States. What can you do to help? (From Kaggle)

<a id = 'd'></a>
## Dataset

This dataset contains:
- Summaries of prescription records for 250 common **opioid** and ***non-opioid*** drugs written by 25,000 unique licensed medical professionals in 2014 in the United States for citizens covered under Class D Medicare
- Metadata about the doctors themselves. 
- This data here is in a format with 1 row per prescriber and 25,000 unique prescribers down to 25,000 to keep it manageable. 
- The main data is in `prescriber-info.csv`. 
- There is also `opioids.csv` that contains the names of all opioid drugs included in the data 
- There is the file `overdoses.csv` that contains information on opioid related drug overdose fatalities.


The data consists of the following characteristics for each prescriber:
- NPI – unique National Provider Identifier number
- Gender - (M/F)
- State - U.S. State by abbreviation
- Credentials - set of initials indicative of medical degree
- Specialty - description of type of medicinal practice
- A long list of drugs with numeric values indicating the total number of prescriptions written for the year by that individual
- `Opioid.Prescriber` - a boolean label indicating whether or not that individual prescribed opiate drugs more than 10 times in the yearr

<a id = 'g'></a>
## Project Goal

The increase in overdose fatalities is a well-known problem, and the search for possible solutions is an ongoing effort. This dataset is can be used to detect sources of significant quantities of opiate prescriptions. 

