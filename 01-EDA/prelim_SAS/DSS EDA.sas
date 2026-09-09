/*
# =========================================================
# Created:    2026-07-19
# Last Edit:  2026-07-20
# Author:     Aaron Niecestro
#
# Description:
# Preliminary EDA for the Data Science Salaries 2023 dataset.
# Includes data loading, structure checks, missing values,
# summary statistics, categorical exploration, and visualization.
# =========================================================
*/

/* =========================================================
   1. Load Data
   ========================================================= */

proc import datafile="ds_salaries.csv"
    out=dss23
    dbms=csv
    replace;
    guessingrows=max;
run;

/* View first rows */
proc print data=dss23(obs=5);
run;

/* Data Cleaning */
proc format;
    value $expfmt
        'EN' = 'Entry'
        'MI' = 'Mid'
        'SE' = 'Senior'
        'EX' = 'Executive';

    value $empfmt
        'FT' = 'Full-Time'
        'PT' = 'Part-Time'
        'CT' = 'Contract'
        'FL' = 'Freelance';
run;

data dss23;
    set dss23;
    format experience_level $expfmt.;
    format employment_type $empfmt.;
run;

/* View first rows */
proc print data=dss23(obs=5);
run;

/* =========================================================
   2. Basic Structure
   ========================================================= */

title "Dataset Structure";

proc contents data=dss23;
run;

/* =========================================================
   3. Missing Values
   ========================================================= */

title "Missing Values Count";

proc means data=dss23 n nmiss;
run;

/* =========================================================
   4. R-style Summary (SAS Equivalent)
   ========================================================= */

title "Summary Statistics (Numeric)";
proc means data=dss23 mean median min max p25 p75;
run;

title "Summary Statistics (Categorical)";
proc freq data=dss23;
    tables _character_ / nocum nopercent;
run;

/* =========================================================
   5. Unique Values for Categorical Columns
   ========================================================= */

title "Unique Values (Categorical)";
proc freq data=dss23 nlevels;
    tables _character_;
run;

/* =========================================================
   6. Numeric Summary (Python-style describe)
   ========================================================= */

title "Numeric Summary";
proc means data=dss23 mean median std min max p25 p75;
run;

/* =========================================================
   7. Duplicated Rows
   ========================================================= */

title "Duplicated Rows";
proc sort data=dss23 nodupkey out=unique_rows dupout=dups;
    by _all_;
run;

proc print data=dups;
run;

/* =========================================================
   8. Value Counts (Top 10)
   ========================================================= */

title "Top 10 Value Counts per Categorical Column";
proc freq data=dss23;
    tables experience_level employment_type company_size job_title / nocum;
run;

/* =========================================================
   9. Correlation Matrix
   ========================================================= */

title "Correlation Matrix";
proc corr data=dss23 pearson;
    var salary_in_usd remote_ratio work_year;
run;

/* =========================================================
   MOVING FROM SUMMARY TO VISUALIZATION
   ========================================================= */
/* Same explanation as Python/R version */

/* =========================================================
   10. Visualization (PROC SGPLOT)
   ========================================================= */

/* 0. All Continuos Variable  Histograms and Boxplots */

/* Identify numeric variables */
proc contents data=dss23 out=meta(keep=name type);
run;

data numvars;
    set meta;
    if type = 1; /* 1 = numeric */
run;

/* Macro to loop through numeric variables */
%macro plots;
    proc sql noprint;
        select name into :numlist separated by ' ' from numvars;
    quit;

    %let count = %sysfunc(countw(&numlist));

    %do i = 1 %to &count;
        %let var = %scan(&numlist, &i);

        title "Histogram of &var";
        proc sgplot data=dss23;
            histogram &var / nbins=40;
            density &var;
        run;

        title "Boxplot of &var";
        proc sgplot data=dss23;
            vbox &var;
        run;

    %end;
%mend;

%plots;

/* 1. Salary Distribution */
title "Salary Distribution";
proc sgplot data=dss23;
    histogram salary_in_usd / nbins=40;
    density salary_in_usd;
run;

/* 2. Salary by Experience Level */
title "Salary by Experience Level";
proc sgplot data=dss23;
    vbox salary_in_usd / category=experience_level;
run;

/* 3. Salary by Employment Type */
title "Salary by Employment Type";
proc sgplot data=dss23;
    vbox salary_in_usd / category=employment_type;
run;

/* 4. Salary by Remote Ratio */
title "Salary by Remote Ratio";
proc sgplot data=dss23;
    vbox salary_in_usd / category=remote_ratio;
run;

/* 5. Job Title Frequency (Top 20) */
proc sql;
    create table top_jobs as
    select job_title, count(*) as freq
    from dss23
    group by job_title
    order by freq desc;
quit;

title "Top 20 Job Titles";
proc sgplot data=top_jobs(obs=20);
    hbar job_title / response=freq;
run;

/* 6. Salary by Country */
proc sql;
    create table top_countries as
    select employee_residence, count(*) as freq
    from dss23
    group by employee_residence
    order by freq desc;
quit;

title "Salary by Employee Residence";
proc sgplot data=dss23(where=(employee_residence in (select employee_residence from top_countries(obs=20))));
    vbox salary_in_usd / category=employee_residence;
run;

/* 7. Correlation Heatmap (SAS Equivalent) */
title "Correlation Heatmap";
proc corr data=dss23 plots=matrix(histogram);
    var salary_in_usd remote_ratio work_year;
run;

/* 8. Geographic Salary Plot */
proc sql;
    create table country_salary as
    select company_location, median(salary_in_usd) as median_salary
    from dss23
    group by company_location
    order by median_salary desc;
quit;

title "Median Salary by Company Location";
proc sgplot data=country_salary;
    hbar company_location / response=median_salary;
run;

/* =========================================================
   ADDITIONAL GRAPHS (SAS Versions)
   ========================================================= */

/* Salary Over Time */
title "Salary by Work Year";
proc sgplot data=dss23;
    vbox salary_in_usd / category=work_year;
run;

/* Salary by Company Size */
title "Salary by Company Size";
proc sgplot data=dss23;
    vbox salary_in_usd / category=company_size;
run;

/* Salary vs Remote Ratio (Scatter) */
title "Salary vs Remote Ratio";
proc sgplot data=dss23;
    scatter x=remote_ratio y=salary_in_usd;
run;

/* Pairplot Equivalent */
title "Scatter Matrix";
proc sgscatter data=dss23;
    matrix salary_in_usd remote_ratio work_year;
run;

/* Countplots */
title "Experience Level Count";
proc sgplot data=dss23;
    vbar experience_level;
run;

title "Employment Type Count";
proc sgplot data=dss23;
    vbar employment_type;
run;

title "Company Size Count";
proc sgplot data=dss23;
    vbar company_size;
run;

/* Interaction Plot */
title "Salary by Experience Level and Company Size";
proc sgplot data=dss23;
    vbox salary_in_usd / category=experience_level group=company_size;
run;

/* =========================================================
   KEY INSIGHTS FROM VISUALIZATION
   ========================================================= */
/* Same insights as Python/R version */
