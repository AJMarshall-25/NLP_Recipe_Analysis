import ast # used for converting column values to lists post-import from csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from nltk import FreqDist

from sklearn.model_selection import cross_validate
from sklearn.metrics import (roc_auc_score, plot_confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

def return_to_list(df, column_names):
    ''' Takes in list of names of columns containing strings and the dataframe they sit in and returns converts each column's contents into a new
    column, called '<original column name>_list', now as lists. May only work on strings that look like lists.... 
    
    Inputs:
    df = dataframe with columns being converted to lists
    column_names = list of columns whose contents need to be transformed
    
    Returns: updated dataframe
    '''
    for col in column_names:
        col_name = col + '_list'
        df[col_name] = [ast.literal_eval(x) for x in df[col] ]
    
    return df


def lists_to_count(df, column, series = False):
    ''' takes in a column of lists and returns counts for all unique values. 
    
    Inputs:
    df - dataframe with column being converted
    column - column of lists
    series - if set to True returns pandas Series instead of a FreqDist object 
    
    Returns: 
    Series with unique value counts or FreqDist object, depending on setting of 'series' parameter
    '''

    all_col = df[column].explode()
    col_count = FreqDist(all_col)
    
    if series:
        return pd.Series(dict(col_count))
    else:
        return col_count

#function for visualizing the most common tokens within a frequency distribution

def visualize_tokens(dist, number, title):
    ''' function for visualizing the most common tokens within a frequency distribution
    From Phase 4 Project: 
    https://github.com/CGPinDC/Tweet_NLP_Project/blob/main/Tweet_Sentiment_%20Analysis_Notebook.ipynb
    
    Inputs:
    dist- pass in frequency dictionary or string of tokens. 
    number- number as integer of the top tokens to return  
    title- title of graph
    
    Returns:
    nothing
    '''
    
    if type(dist).__name__ == 'Series':
        dist.sort_values(ascending=False, inplace=True)
        keys = list(dist.keys())
        values = dist.tolist()
        top = list(zip(keys, values))
        tokens = [key[0] for key in top][:number]
        counts = [value[1] for value in top][:number]
    else:
        # get tokens and frequency counts from freq_dist
        top = list(zip(*dist.most_common(number)))
        tokens = top[0]
        counts = top[1]

    # Set up plot and plot data
    fig, ax = plt.subplots(figsize = (15, 10))
    ax.bar(tokens, counts)

    # Customize plot appearance
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=90)

# small function to relabled the 'target_tag' value for recipes with '60-minutes-or-less' 
# tags
def hour_check(x):
    '''Small, project-specific function to relabled the 'target_tag' value for recipes with '60-minutes-or-less' 
    tags.
    
    Inputs:
    x - list to be searched
    
    Returns:
    nothing '''
    if '60-minutes-or-less' in x: 
        return False
    else:
        return True

def target_check(x):
    ''' Dataframe specific function to set rows as meeting the conditions for the 
    target variable or not based on values in 'target_tag' and 'target_search_term'
    columns.
    
    Inputs:
    x - Series 
    
    Returns:
    boolean value based on series values'''

    if (x['target_tag'] == True) and (x['target_search_term'] == True):
        return 1
    else:
        return 0

def basic_cleaning(df, column):
    ''' Takes in a dataframe and the name of the column to be cleaned.  The contents of the column, 
    which need to be strings, are converted to lowercase, have their punctuation and numbers removed,
    and are finally stripped of whitespaces
    
    Input:
    df - dataframe with column to be cleaned
    column - column containing strings
    
    Returns: 
    Dataframe with new, cleaned, column added'''
    new_col = 'cleaned_' +column
    # convert to lowercase
    df[new_col] = df[column].apply(lambda x: x.lower())
    
    # remove punctuation and non-characters
    df[new_col] = df[new_col].apply(lambda x: re.sub(r'[^\w\s]','',x))
    df[new_col] = df[new_col].apply(lambda x: re.sub('[0-9\n]',' ',x))

    #strip whitespace
    df[new_col] = df[new_col].apply(lambda x: re.sub('[ ]{2,}',' ',x))
    
    return df

def remove_stop_words(count, stop_words):
    '''A small function to quickly remove stopwords from a pandas' Series

    Inputs:
        count - a Series containing stop_words in the index
        stop_words - list of words to be removed
    
    Returns:
    Series
    '''
    for x in count.index:
        if x in stop_words:
            count = count.drop(x)
    
    return count

def evaluate(estimator, X_tr, X_te, y_tr, y_te, cv=5):
    '''
    Function takes in estimator, training data, test data, 
    and the cross validation splitting strategy, and returns the accuracy, precision, recall, f1 and the ROC-AUC
    scores for the model as well as a confusion matrix visualization.  Based on Phase 3 Project 
    https://github.com/Nindorph/TanzanianWaterWells/blob/main/Modeling_Final.ipynb and Lindsey Berlin’s evaluate function
    code found at: 
    https://github.com/lindseyberlin/Cat-in-the-Dat-Project/blob/main/notebooks/Lindsey/EDA-Initial-Models.ipynb
    ------------------------------------------------------------------------------------------
    Inputs: 
    -Estimator - Estimator object  
    -X_tr – X_train dataframe
    -X_te – X_test dataframe
    -Y_tr – y_train dataframe
    -Y_te – y_test dataframe
    -Cv – If cross_val  set to true this determines the cross-validation splitting strategy.  
            Takes in all value options for sklearn.model_selection_cross_val_score “cv” parameter:
            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices


    Returns – nothing is returned '''
    
    #go through evaluation steps as normal, including cross validation
    #Cross-Validate
    output = cross_validate(estimator, X_tr, y_tr, cv=cv,
                            scoring=['accuracy', 'precision','recall', 'f1', 'roc_auc'])
    #Printing out the mean of all of our evaluating metrics across the cross validation. 
    #Accuracy, precisionc recall, f1, and roc auc
    print('Results of Cross-Validation:\n')
    print(f'Average accuracy: {output["test_accuracy"].mean()}\
    +/- {output["test_accuracy"].std()}')
    print(f'Average precision: {output["test_precision"].mean()}\
    +/- {output["test_precision"].std()}')
    print(f'Average recall: {output["test_recall"].mean()}\
    +/- {output["test_recall"].std()}')
    print(f'Average f1 score: {output["test_f1"].mean()}\
    +/- {output["test_f1"].std()}')
    print(f'Average roc_auc: {output["test_roc_auc"].mean()}\
    +/- {output["test_roc_auc"].std()}\n')
    print('+'*20)
    
        
    #Fitting the estimator to our X and y train data
    estimator.fit(X_tr, y_tr)
    #getting predictions for X train
    tr_preds = estimator.predict(X_tr)
    #getting predictions for X test
    te_preds = estimator.predict(X_te)
        
    #Creating a confusion matrix from our data with custom labels
    print('\nResults of Train-Test Split Validation:')
    plot_confusion_matrix(estimator, X_te, y_te, cmap='mako')
    plt.show()
        
    #Printing our final evaluating metrics across X train
    #Evaluating using accuracy, precision, recall, f1, roc auc
    print("\nTraining Scores:")
    print(f"Train accuracy: {accuracy_score(y_tr, tr_preds)}")
    print(f"Train precision: {precision_score(y_tr, tr_preds)}")
    print(f"Train recall: {recall_score(y_tr, tr_preds)}")
    print(f"Train f1 score: {f1_score(y_tr, tr_preds)}")
    print(f"Train roc_auc: {roc_auc_score(y_tr, tr_preds)}\n")
    print("<>"*10)
    #Printing our final evaluating metrics across X test
    #Evaluating using accuracy, precision, recall, f1, roc auc
    print("\nTesting Scores:")
    print(f"Test accuracy: {accuracy_score(y_te, te_preds)}")
    print(f"Test precision: {precision_score(y_te, te_preds)}")
    print(f"Test recall: {recall_score(y_te, te_preds)}")
    print(f"Test f1 score: {f1_score(y_te, te_preds)}")
    print(f"Test roc_auc: {roc_auc_score(y_te, te_preds)}")


def score_tracker( gscv_results, model_name, score_df=None):
    '''Takes in  GridSearchCV results  from 'cv_results_' attribute in dataframe form and cleans it up so it can be 
    appended to other results output, labeling each row with the other required attribute, 'model_name'. Has a score
    tracking dataframe as an optional argument - if passed the search results will be appended to the tracker. 
    
    Inputs: 
    gscv_results - cv_results_ output from model cross-validation
    model_name - string to use as identifier for this batch of results
    score_df - optional argument, when included the new results data will be appended to this dataframe. Note that this
    assumes the scores_df has the same structure as the dataframe generated with the gscv_results data
    
    Returns:
    Dataframe'''
    gscv_results['model'] = model_name
    results = gscv_results[['mean_fit_time','params','mean_test_score','std_test_score','model']]
    
    if score_df is not None:
        all_scores = pd.concat([score_df, results]).copy()
        return all_scores.sort_values('mean_test_score')
    else:
        return results