1. Identify and follow individuals.
    The combination of 1968 family ID number and Person Number (ER30002) provides unique identification for each individual on the data file.    
        - 1968 INTERVIEW NUMBER [(ER30001)](https://simba.isr.umich.edu/cb.aspx?vList=ER30001)   
        - Person Number [(ER30002)](https://simba.isr.umich.edu/cb.aspx?vList=ER30002)   
    Pull and merge the data by a single identifier calcualted as follow: (ER30001)*1000 + (ER30002). Then we could track individual data across years. But personal number may change through years as the relationship with family members changes.    
    The example:   
        - linking individual through time:   
            From 2011 PSID Individual File, (ER30001)*1000 + (ER30002)   
            From 2013 PSID Individual File, (ER30001)*1000 + (ER30002)   
            Which this unique identifier does not change through time in most times.   



2. Identify and follow families.   
    First we need to figure out: 1.The definition of family. 2.How to handle family composition change through time.   
    How to define the members of family is based our research interest and because the flexibility we could always change the definition.   
    Once the individuals that makes up a famliy are identified at each year, the the family can be linked over time.   
    There are some key variables:   
    - RELATIONSHIP TO THE HEAD, ex 2013 (ER34203), *As of 2017, the term "reference person" has replaced head.   
    - SAMPLE STATUS, ex 2013 (ER32006), maybe we only care about SAMPLE STATUS: oringinal sample, born-in sample, move-in sample.   
    - FAMILY COMPOSITION CHANGE, ex changes between 2011-2013 (ER53007). For simplicity, we may only care about families with no composition change.   
    The procedure of following families over time: 
    1. Define all families in a starting point. "Family Interview ID"
    2. Select the focal persons in the family that we want to follow. For simplicity here, I chose head and wife.
    3. Chose how to follow. For simplicity, I chose the families that only existed in the baseline year. 
    To start with I only followed the family without the family composition change.



3. Variables we need in individual files and variables we need in family files.   

Individual variables:   
    - 1968 INTERVIEW NUMBER (ER30001)    
    - Person Number (ER30002)   
    - RELATIONSHIP TO THE HEAD, ex 2013 (ER34203)   
    - Family Interview ID ex 1999 (ER33501), using family head as a linker to the family household.   
Pull and merge the data by a single identifier calcualted as follow: (ER30001)*1000 + (ER30002).
ER33501


Household variables:   
    - family composition change(ER53007)
	- headCount (ER13009)
	- Age (ER13010)
	- Education (ER16516)
	- Location (metropolitan and cities) (ER16431)
	- Employment status (ER10081)
	- Ethnicity, race (ER15928)
	- Marital status (ER13021)
	- Occupation and industry (ER12169) (ER23479)
	- Wealth as liquid assets (ER15020)
	- Labor income: (total income not accessible, break it down to head generated income and wife generated income)
		 - laborIncome_head (ER16463)   
		 -	laborIncome_wife (ER16465)   
	- Expenditure: (total amount not accessible, so break it down to several expenditure):
		- childCare (ER16515D1)
		- educationCost (ER16515C9)
		- foodCost (ER16515A1)
		- healthCare (ER16515D2)
		- housing (ER16515A5)
		- transportation (ER16515B6)   
	  So the total expenditure could roughly be represented by the sum of the above variables.   
    
Total wealth including the home equity and total wealth excluding the equity. 
		- wealthIncludeHomeEquity: (S416)
		This variable is constructed as sum of values of seven asset types (S403, S405, S409, S411, S413, S415, S419) net of debt value (S407).
		- wealthExcludeHomeEquity: (S417)
		This variable is constructed as sum of values of seven asset types (S403, S405, S409, S411, S413, S415, S419) net of debt value (S407) plus value of home equity.
			- farm and business(S403)
			- checking and saving account(S405)
			- debt value(S407)
			- real estate(S409)
			- invest in stock not including pension and IRA (S411)
			- market participation, whether holding stock excluding IRAs (S410)
			- vehicles value (S413)
			- other asset (S415)
			- Annuity and IRAs (S419)
				Do [you/you or anyone in your family] have any money in private annuities or Individual Retirement Accounts (IRAs)?









