# What's for Dinner?
# Recipe Classification with NLP
 
![](images/header.PNG)

Everyone needs to eat, and while ordering delivery or dining out each day sounds like an appealing solution to this condition very few people have the means to sustain such a lifestyle.  In a post-COVID world where [92% of households plan to cook at home more often](https://www.supermarketnews.com/consumer-trends/study-most-us-consumers-stick-eating-home-post-pandemic) the market for recipe websites and cookbooks is wide open. Equally in demand are "weeknight" meals - simple easy recipes that are quick to prepare after work.  Using data scraped from Food.com my logistic regression model has 71% accuracy in predicting if a recipe is easy or not

## Business Understanding

While initially seeming the perfect tool for home chefs the practical applications of this model apply to more then one target audience:

- home cooks could use this to check if a recipe fits their energy level and schedule, possibly by creating a web interface.
websites that host recipes, or companies with collections of recipes - for example a publisher - could use this to auto-classify their recipes, which in turn could be used for SEO, for general organizational purposes, or for automating their review process.

According to the Supermarket News article the number of people eating out at least once a week has dropped by 9% since COVID with more people planning to eat out less. This audience, many of whom did not cook much before the virus struck, are ripe for guidance towards simple, week-night, dinners be it via website or cookbook.

## Data Understanding

The data used to build my model comes from [Kaggle](https://www.kaggle.com/shuyangli94/foodcom-recipes-with-search-terms-and-tags) and consists of ~500,000 user-submitted recipes scraped from Food.com. In addition to the text of the recipes' description and instructions the dataset also contains columns breaking out the ingredients, search terms, tags, and individual steps for each recipe. The tag data comes from the recipe author from a list of options provided by Food.com whereas the search terms are, from all evidence, assigned by Food.com. There is also an "id" column that can be used to search for the recipe on Food.com.

This dataset does not have a target variable included so one needs to be constructed for it by leveraging the tag and search term data to find dinner recipes that can be called easy, be it because they're quick, simple to make, or have very few steps.

![Wordcloud of tag feature frequencies](images/wordcloud.png)

## Modeling and Evaluation

The baseline for this model was created by running numeric metadata about the recipes in my dataset, such as number of ingredients or number of characters in the recipe's description, through a LogisticRegression model to make sure that a model trained on these values would not outperform a model using NLP.

Initial testing on a wide array of types resulted in the best performances from MultinomialNB and LogisticRegression models using the TfidfVectorizer, with further hyperparameter tuning resulting in a model with 71% accuracy, 16% higher then the accuracy of the baseline model. This demonstrates that NLP is the better method for recipe difficulty classification.

## Conclusion

While I would prefer higher accuracy rates on my current model I remain positive it can be improved, and more importently expanded to classify other types of recipes such as 'kid-friendly' or 'healthy'. Even with it's current target the model can be effectively used by website's and publishers to vet and classify submitted recipes in a consistent manner.

## Repository Structure

NLP_Recipe_Analysis  
|-- archive #folder containing experiment and draft notebooks  
|-|- Drafts # drafts of final model
|-+- Experiments # testing processes and ideas
|-- images  #folder containing images collected for project documents   
|-- .gitignore  
|-- README.md  
|-- To_do.ipynb #scratch notebook containing To Do list  
+-- recipe-classification-with-nlp-v4_CURRENTFINAL.ipynb  




