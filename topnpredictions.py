import pandas as pd

def return_top_n_pred_prob_df(n, model, docs, column_name):
    '''
    Function to predict the top n topics for a specific keyword with it's accuracy score
    Parameters:
      Input:
        a) n = Top n topic classes you want
        b) model = the model you have trained your dataset on
        c) docs = the keywords on which you want to predict the top n topics
        d) column_name = name of the column in the resultant df which takes in this as it's input for naming it

      Output: A dataframe with keywords and their corresponding topic names with its associated percentage accuracy.
    '''
    predictions = model.predict_proba(docs)
    preds_idx = np.argsort(-predictions, axis=1)
    top_n_preds = pd.DataFrame()

    for i in range(n):
        top_n_preds['keywords'] = docs
        top_n_preds[column_name + "_" + '{}'.format(i)] = [preds_idx[doc][i] for doc in range(len(docs))]
        top_n_preds[column_name + "_" + '{}_prob'.format(i)] = [predictions[doc][preds_idx[doc][i]] for doc in
                                                                range(len(docs))]

        top_n_preds = top_n_preds.rename(columns={'class_name': column_name + ''.format(i)})
        try:
            top_n_preds.drop(columns=['index', column_name + '_prediction_{}_num'.format(i)], inplace=True)
        except:
            pass
    return top_n_preds