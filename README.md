# Personal Financial Forecasting

## Introduction and Data Collection

How much money am I going to spend on various categories over the first three months of 2021? That is the question I would like to solve using my Data Analytics skills. This answer will come from predicting 'Amount (USD)' values from different models. The data I have available to help answer this question comes from my monthly transaction reports from my Apple Card starting from November 2019, as I use my Apple Card for most purchases.

**Quick look at the Data:**

Transaction Date	Clearing Date	Description	Merchant	Category	Type	Amount (USD)
* 04/28/2020	04/30/2020	APPLE.COM/BILL ONE APPLE PARK WAY 866-712-7753...	Apple Services	Other	Purchase	5.35
* 04/28/2020	04/28/2020	MAVERIK #517 24 S GENEVA ROAD VINEYARD 84058 U...	Maverik	Gas	Purchase	10.64
* 04/28/2020	04/28/2020	MICROSOFT*ULTIMATE 1 MONE MICROSOFT WAY MSBILL...	Microsoft	Other	Purchase	16.08

The "raw" data has 612 observations with 7 columns. The column 'Amount (USD)' is the response variable I will be trying to predict later; the other 6 are explanatory variables with varying levels of utility.

## Wrangling the Data and Feature Engineering

There is some more information we can add to the dataset, as well as some we can parse out.

**Let's take another look at the Data:**

Transaction Date	Clearing Date	Description	Merchant	Category	Type	Amount (USD)	Year	Month	Day	State	Season	InSchool	Employed
* 2020-04-28	2020-04-30	APPLE.COM/BILL ONE APPLE PARK WAY 866-712-7753...	Apple Services	Other	Purchase	5.35	2020	4	28	CA	Spring	1.0	0.0
* 2020-04-28	2020-04-28	MAVERIK #517 24 S GENEVA ROAD VINEYARD 84058 U...	Maverik	Gas	Purchase	10.64	2020	4	28	UT	Spring	1.0	0.0
* 2020-04-28	2020-04-28	MICROSOFT*ULTIMATE 1 MONE MICROSOFT WAY MSBILL...	Microsoft	Other	Purchase	16.08	2020	4	28	WA	Spring	1.0	0.0

After engineering these new features, the data now has 14 columns, with 13 of them being possible explanatory variables. Now, since the 'Description' variable has no common structure to parse, the most useful variable becomes 'Category'. This 'Category' variable decribes what kind of purchase I made. Originally, there were only 11 categories with the "Other" 'Category' being a very vague catch-all. After digging through the different 'Category' values, I created some new categories and edited some others. The end result is 16 more descriptive categories: 

'Tech Stuff', 'Gas', 'XBOX', 'Pizza', 'Fast Food or Snacks', 'Debit', 'Other', 'Grocery', 'Shopping', 'Transportation', 'Real Food', 'Entertainment', 'Insurance', 'School Stuff', 'Hotels', 'Airlines'

16 categories seems to be enough for me. While the "Other" 'Category' still exists, I believe it is more useful now that it has been somewhat spliced.

## Exploratory Data Analysis

As we should do with any data-based project, let's perform some exploratory data analytics by creating a few different types of graphs representing the data we now have. Let's start with some numeric summaries.

Based on some grpahics not shown here, we can learn the following:

- The 'Category' with the most transactions is "Fast Food or Snacks", which also was where I spent the most money (oof). "Tech Stuff" only has 22 purchases, but is the second most expensive.
- I was in school for 74% of the year, which makes sense.
- I was only employed for 63% of the year; thanks Covid!
- The most common purchase is between $5-10.
- It looks like my spending increased around the holidays in 2019. This makes perfect sense. Additionally, my purchases go down during Winter semester and plumets when I am laid off from work becasue of COVID-19. Surprisingly though, my purchases sky-rocket in June. This I can attribute to 1) receiving unemployment insurance (which was more than I was making at work), and 2) purchasing plane tickets for my travel to Virginia when my grandmother died. Post June, my expenses are still relatively high, as I am continuing to receive unemployment insurance, but the expenses steadily decrease as the year continued.
- I spent just a little more money when I was Employed than when I was not. Normally, one would probably expect a larger margin here, but as I have stated previously, I collected more money through unemployment insurance than what I was paid at work (and I'm bad at saving money). Also, I spent decently more money while I was in school than during summer vacation. This may be because I am in school for more time overall, but also because holidays like Christmas and Valentine's Day occur during the months I'm in school.

## Modeling

Now, it is time to begin modeling. We will split the data I have wrangled (also called the training data) into a train and test set to evaluate the Mean Squared Error metric from the 4 different models. I will use a 80/20 split for the data and a random state of 602. The four models I will be comparing are Linear Regression, Lasso Regression, K Nearest Neighbors, and Decision Trees. Below are the results from the train/test models:

*Pipeline Models: (Test-MSE, MSE-CV, MAE)*
- LinearRegression: 8.954670349111718e+28, 7194.375, 49063394563146.68
- Lasso: 3913.213, 7183.074, 29.514
- KNeighborsRegressor: 3146.046, 6998.254, 25.092
- DecisionTreeRegressor: 22328.48, 7757.108, 44.246

K Nearest Neighbors produces the best MSE score before and after cross validation, but all of these values are quite high. K Nearest Neighbors also produces the lowest MAE value. Regardless, becasue the first three models' scores are pretty similar, lets run all three models and compare practicality of the results.

## Model Evaluation

The Linear Regression model produced many negative values, as well as values in the billions and trillions of dollars. Of course, these values are not practical.

The Lasso Regression model produces mostly positive values, with the exception of the "Airlines" 'Category'. However, these values all are much too high. The prices do seem to vary well between most categories and months, but all values seem too high to be practical.

The K Nearest Neighbors model produces similar values as the Lasso Regression model, but are all positive. Additionally, this model seems to have used an 'over time' pattern. Across all categories, the amounts spent changes with Feb < Jan < Mar in totals. Meaning, this model predicts I will spend the most money in March and the least amount of money in February. Interestingly enough, this pattern does not follow the Mar < Feb < Jan pattern shown during the EDA segment of this analysis.

## Returning to the Original Question

So, how much money am I going to spend? Knowing myself, while I may not be the best at saving, I know I will not be spending anything close to what any of the models predict. I simply do not have the income nor the expenses. I have no intention of spending any money on "Airlines", and "XBOX" monthly payments (which I am subscribed to) are less than $20 each month, not around the hundreds. It seems that all models have erred greatly in these predictions. The only possible pattern included in these models is that within KNN, where moeny will be spent differently for each month across all categories. Perhaps I will load in new data next April and see if that pattern in evident.

## Biases, Limitations, and Future Recommendations

After reviewing the results for each model, it is clear that none of them worked perfectly. And while some have more negative values than I believe should ever have been predicted, all predict expenses astronomically higher than personally conceivable.

**Biases:**

The largest issue I see present with my models is that none of them ever predicted a value of $0.00 for any category. Every single day, I am predicted to spend money on a purchase in every single category. This, while not only not feasible nor practical, is not suggested anywhere in my original data. There were scores of days I spent less than a Hamilton bill, as well as plenty of days I did not spend a single copper piece. Somehow, the models I ran must have missed that part.

**Limitations:**

Ideally, I would have loved to incorporate some essence of a Time-Series in this project. However, seeing how I have only had the Apple Card for a year (and that the UCCU debit card I have does not produce anywear near the sort of data Apple does), trying to predict per annum values with only one year of reference would provide severely limited (if any) variance.

**Future Recommendations:**

While this project was still chalk-full of informative details about my spending habits, performing such an analysis yearly may help me save more. In addition, to someone wishing to replicate a similar project or analysis, I would recommend first having more data available, but also suggest spending the majority of time focused on the EDA segment. Producing these graphs, as well as others I cut from the notebook, was the most helpful for me to understand some of my spending habits. Of course, many large expenses are unplanned, like my trip to Virginia, but understanding smaller, more consistent spending can help one save for the bigger, one-time purchases.
