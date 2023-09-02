# import configparser
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import json
import os


# Read configuration file

def ReadConfig(experiments_name,  config_file_name):
    # Utils absolute path
    abs_path = os.path.realpath(__file__)
    # Utils parentes path
    utils_parent_path = os.path.dirname(abs_path)
    # config file absolute path
    config_file_path = os.path.join(utils_parent_path, experiments_name, 'config',config_file_name+'.json')

    with    open(config_file_path, 'r', encoding='utf8') as fp:
        config = json.load(fp)
    return config


def GetFileList(path, filter_words=None, exclude_files=list()):
    all_files = os.listdir(path)
    rs = list()

    if len(exclude_files)>0 and filter_words:
        exclude_files = [i+filter_words for i in exclude_files]

    for each_file in all_files:
        if filter_words:
            if (filter_words in each_file) and (each_file not in exclude_files):
                rs.append(each_file)
        else:
            if len(exclude_files)>0:
                exclude = False
                for j in exclude_files:
                    if j in each_file:
                        exclude = True
                        break
                if exclude == False:
                    rs.append(each_file)
            else:
                rs.append(each_file)
    rs.sort()
    return rs

# Print score between Ytrue and Ypred
# savePath=None -> console, else to Result.txt

def PrintScore(true, pred,all_scores, savePath=None, average='macro'):
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(os.path.join(savePath,"Result.txt"), 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:")
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (metrics.accuracy_score(true, pred),
                                                              metrics.f1_score(true, pred, average=average),
                                                              metrics.cohen_kappa_score(true, pred),
                                                              F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names=['Wake', 'N1', 'N2', 'N3', 'REM']), file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    # Results of each class
    print('\nResults of each class:', file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=None), file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=None), file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=None), file=saveFile)
    print("All folds' acc:\t ", all_scores,file=saveFile)
    # score = {"Main scores": {
    #     "Acc": metrics.accuracy_score(true, pred),
    #     "Kappa": metrics.cohen_kappa_score(true, pred),
    #     "F1-score": {
    #         "value": metrics.f1_score(true, pred, average=average),
    #         "avg": average},
    #     "Precision": {
    #         "value": metrics.precision_score(true, pred, average=average),
    #         "avg": average},
    #     "Recall": {
    #         "value": metrics.recall_score(true, pred, average=average),
    #         "avg": average}
    # },
    #     "Classification report": metrics.classification_report(true, pred,
    #                                                            target_names=['Wake', 'N1', 'N2', 'N3', 'REM']),
    #     "Confusion matrix": {"Wake":metrics.confusion_matrix(true, pred)[0].tolist(),
    #                          "N1":metrics.confusion_matrix(true, pred)[1].tolist(),
    #                          "N2":metrics.confusion_matrix(true, pred)[2].tolist(),
    #                          "N3":metrics.confusion_matrix(true, pred)[3].tolist(),
    #                          "REM":metrics.confusion_matrix(true, pred)[4].tolist()},
    #     "Results of each class":{
    #         "F1-Score":metrics.f1_score(true, pred, average=None).tolist(),
    #         "Precision":metrics.precision_score(true, pred, average=None).tolist(),
    #         "Recall":metrics.recall_score(true, pred, average=None).tolist()
    #     }
    # }
    # with open(os.path.join(savePath,"result.json"), "w", encoding='utf-8') as f:
    #     # json.dump(dict_var, f)  # 写为一行
    #     json.dump(score,f,  indent=2, ensure_ascii=False)  # 写为多行

    if savePath != None:
        saveFile.close()
    return


# Print confusion matrix and save

def ConfusionMatrix(y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix")
    print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100, '.2f') + '%\n' + format(cm_n[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(savePath,title+".png"))
    plt.show()
    return ax


# Draw ACC / loss curve and save

def VariationCurve(fit, val, yLabel, savePath, figsize=(9, 6)):
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(fit) + 1), fit, label='Train')
    plt.plot(range(1, len(val) + 1), val, label='Val')
    plt.title('Model ' + yLabel)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(savePath + 'Model_' + yLabel + '.png')
    plt.show()
    return
