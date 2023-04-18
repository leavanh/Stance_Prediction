# Visualize tables of the accuracy and f1 score results for all models

path = '/home/ubuntu/lrz/thesis/Stance_prediction/'

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the txt files containing the results

with open(path+"Models/stance_prediction/results_electra.txt") as file:
    results_electra = [line.rstrip() for line in file]

results_electra_no_semsearch = results_electra[0]
results_electra.remove(results_electra_no_semsearch)

with open(path+"Models/stance_prediction/results_bert.txt") as file:
    results_bert = [line.rstrip() for line in file]

results_bert_no_semsearch = results_bert[0]
results_bert.remove(results_bert_no_semsearch)


# function to split results into seperate parts and create df
def create_df(results):

    data = {'input': [], 'model': [], 'accuracy': [], 'f1': []}

    for item in results:
        input_model, accuracy_f1 = item.split(': ')
        input_, model = input_model.split('_', maxsplit=1)
        accuracy_f1 = accuracy_f1.split(', ')
        accuracy = float(accuracy_f1[0].split(' = ')[1])
        f1 = float(accuracy_f1[1].split(' = ')[1])
        data['input'].append(input_)
        data['model'].append(model)
        data['accuracy'].append(accuracy)
        data['f1'].append(f1)

    df = pd.DataFrame(data)
    return df

# function to create matrix from df (with correct column and row names)
def create_matrix(df, metric):

    # create matrix
    matrix = df.pivot_table(index='input', columns='model', values=str(metric), aggfunc=max)

    # create the mapping dictionary
    col_mapping = {'bertflow': 'BERT-Flow', 'isbert': 'IS-BERT', 'rankbm25': 'BM25', 'sbert': 'SBERT',
                'sbert-bertflow': 'SBERT-BERT-Flow', 'sbertwhitening': 'SBERT-Whitening', 'sbertwk': 'SBERT-WK',
                'whitening': 'Whitening', "summary_isbert": "IS-BERT Summarized"}

    row_mapping = {'input11': 'Pattern 1-1', 'input12': 'Pattern 1-2', 'input21': 'Pattern 2-1', 'input22': 'Pattern 2-2',
                'input3': 'Pattern 3', 'input4': 'Pattern 4'}

    col_order = ['BM25', 'SBERT', 'BERT-Flow', 'SBERT-BERT-Flow', 'Whitening', 'SBERT-Whitening', 'SBERT-WK', 'IS-BERT', 'IS-BERT Summarized']

    # rename the columns and change their order
    matrix = matrix.rename(columns=col_mapping).reindex(columns=col_order)

    # rename the rows
    matrix = matrix.rename(index=row_mapping)

    # add a row and a column with the means
    matrix.loc['Mean'] = matrix.mean()
    matrix['Mean'] = matrix.mean(axis=1)

    # delete columns with missing values
    matrix.dropna(axis=1,inplace=True)

    return matrix

# function to create the table from matrix
def create_plot(matrix, metric, model, results_no_semsearch):

    metric_no_semsearch = results_no_semsearch.split(': ')[1].split(', ')
    if metric == "accuracy":
        color = "Greens"
        metric_no_semsearch = float(metric_no_semsearch[0].split(' = ')[1])
    elif metric == "f1":
        color = "Blues"
        metric_no_semsearch = float(metric_no_semsearch[1].split(' = ')[1])
    else:
        raise ValueError("Metric must be accuracy or f1.")

    # set up the plot
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(matrix, annot=True, cmap=str(color), fmt='.3f', linewidths=.5, cbar=False, ax=ax, annot_kws={"size": 12})

    # Drawing the frame
    ax.axhline(y = 0, color='k',linewidth = 10, xmin=0, xmax=1-(1/8.7))
    ax.axhline(y = 6, color = 'k', linewidth = 5, xmin=0, xmax=1-(1/9))
    ax.axvline(x = 0, color = 'k', linewidth = 10, ymin=1/6.5, ymax=1)
    ax.axvline(x = 8, color = 'k', linewidth = 5, ymin=1/6.5, ymax=1)

    # set the tick labels to be horizontal and remove axis ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=12)
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='both', which='both', length=0)

    # set the axis labels and title
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Input', fontsize=14)
    ax.set_title(f'{model.upper()}: {metric.capitalize()} by input and model', fontsize=16)

    # add the no sem search values to bottom left corner
    ax.text(0, -0.05, f"{metric.capitalize()} without semantic search: {metric_no_semsearch}", fontsize=12, transform=ax.transAxes)

    plt.savefig(path+f'Plots/{model}_{metric}.png', transparent=True)


# make the plots
df_electra = create_df(results_electra)

matrix_electra_acc = create_matrix(df_electra, "accuracy")
create_plot(matrix_electra_acc, "accuracy", "electra", results_electra_no_semsearch)

matrix_electra_f1 = create_matrix(df_electra, "f1")
create_plot(matrix_electra_f1, "f1", "electra", results_electra_no_semsearch)

df_bert = create_df(results_bert)

matrix_bert_acc = create_matrix(df_bert, "accuracy")
create_plot(matrix_bert_acc, "accuracy", "bert", results_bert_no_semsearch)

matrix_bert_f1 = create_matrix(df_bert, "f1")
create_plot(matrix_bert_f1, "f1", "bert", results_bert_no_semsearch)

# visualize the results of the models with summary as context

with open(path+"Models/stance_prediction/results_electra_summary.txt") as file:
    results_electra_summary = [line.rstrip() for line in file]

df_electra_summary = create_df(results_electra_summary)
df_electra_summary = df_electra_summary.append(df_electra[df_electra.model=="isbert"]) # add isbert results without summary

# ACCURACY, do own plot
matrix_electra_summary_acc = create_matrix(df_electra_summary, "accuracy")

electra_no_semsearch = results_electra_no_semsearch.split(': ')[1].split(', ')
color = "Greens"
acc_electra_no_semsearch = float(electra_no_semsearch[0].split(' = ')[1])

# set up the plot
fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(matrix_electra_summary_acc, annot=True, cmap=str(color), fmt='.3f', linewidths=.5, cbar=False, ax=ax, annot_kws={"size": 12})

# Drawing the frame
ax.axhline(y = 0, color='k',linewidth = 10, xmin=0, xmax=1-(1/2.97))
ax.axhline(y = 6, color = 'k', linewidth = 5, xmin=0, xmax=1-(1/3))
ax.axvline(x = 0, color = 'k', linewidth = 10, ymin=1/6.5, ymax=1)
ax.axvline(x = 2, color = 'k', linewidth = 5, ymin=1/6.5, ymax=1)

# set the tick labels to be horizontal and remove axis ticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=12)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(axis='both', which='both', length=0)

# set the axis labels and title
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Input', fontsize=14)
ax.set_title('ELECTRA: Accuracy by input and model', fontsize=16)

# add the no sem search values to bottom left corner
ax.text(0, -0.05, f"Accuracy without semantic search: {acc_electra_no_semsearch}", fontsize=12, transform=ax.transAxes)

plt.savefig(path+f'Plots/electra_summary_accuracy.png', transparent=True)

# F1, do own plot
matrix_electra_summary_f1 = create_matrix(df_electra_summary, "f1")

color = "Blues"
f1_electra_no_semsearch = float(electra_no_semsearch[1].split(' = ')[1])

# set up the plot
fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(matrix_electra_summary_f1, annot=True, cmap=str(color), fmt='.3f', linewidths=.5, cbar=False, ax=ax, annot_kws={"size": 12})

# Drawing the frame
ax.axhline(y = 0, color='k',linewidth = 10, xmin=0, xmax=1-(1/2.97))
ax.axhline(y = 6, color = 'k', linewidth = 5, xmin=0, xmax=1-(1/3))
ax.axvline(x = 0, color = 'k', linewidth = 10, ymin=1/6.5, ymax=1)
ax.axvline(x = 2, color = 'k', linewidth = 5, ymin=1/6.5, ymax=1)

# set the tick labels to be horizontal and remove axis ticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=12)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.tick_params(axis='both', which='both', length=0)

# set the axis labels and title
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Input', fontsize=14)
ax.set_title('ELECTRA: F1 by input and model', fontsize=16)

# add the no sem search values to bottom left corner
ax.text(0, -0.05, f"F1 without semantic search: {f1_electra_no_semsearch}", fontsize=12, transform=ax.transAxes)

plt.savefig(path+f'Plots/electra_summary_f1.png', transparent=True)