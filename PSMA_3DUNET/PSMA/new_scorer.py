import numpy as np
import SimpleITK as sitk 
from matplotlib import pyplot as plt
import progressbar
import glob
import os

truth_PREFIX = 'truth.nii.gz'
preds_PREFIX = 'prediction.nii.gz'
MASK_PREFIX = 'data_mask.nii.gz'
label_dict = {'1': 'boneLesion', '2':'lymphNodeLesion', '3': 'localProstateLesion'}
epslone = 1e-10

rootdir = '/home/louis/Downloads/data'
evaluation_list = glob.glob(rootdir+'/prediction_isensee2017'+'/*_case_*')
evaluation_list.sort()

def get_itk_array(filename):
    image = sitk.ReadImage(filename)
    imageArray = sitk.GetArrayFromImage(image)
    return imageArray

def load_lesion_labels():
#    data = []
    for pn in evaluation_list:
        fnl = os.path.join(pn, truth_PREFIX)
        fnm = os.path.join(pn, MASK_PREFIX)
        l = get_itk_array(fnl)
        m = get_itk_array(fnm)
        m = np.asarray((m>0),dtype='int')
        yield(np.asarray((l * m), dtype='int'))
        #data.append(np.asarray(l * m, dtype='int'))
    #return data

def load_predictions(preds_PREFIX):
#    data = []
    for pn in evaluation_list:
        fnl = os.path.join(pn, preds_PREFIX)
        fnm = os.path.join(pn, MASK_PREFIX)
        l = get_itk_array(fnl)
        m = get_itk_array(fnm)
        m = np.asarray((m>0),dtype='int')
        yield(np.asarray(l * m, dtype='int'))
        #data.append(np.asarray(l * m, dtype='int'))
    #return data

def get_regions(gtslice, labels):
    dummy = np.zeros(gtslice.shape, dtype='int')
    dslices = {}
    cnts = {}
    for label in labels:
        dslices[str(label)] = np.copy(dummy)
        cnts[str(label)] = 0

    inds = np.where(np.isin(gtslice,labels))

    for x,y in zip(inds[0], inds[1]):
        label = gtslice[x,y]
        if dslices[str(label)][x,y] == 0:
            thisRegion = np.zeros(gtslice.shape, dtype='int')
            temp = np.zeros(gtslice.shape, dtype='int')
            thisRegion[x,y] = 1
            new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)
            iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
            temp = np.copy(thisRegion)

            while iterate:
                for xi, yi in zip(new_ind[0], new_ind[1]):
                    patch = gtslice[xi-1:xi+2,yi-1:yi+2]
                    patch = np.asarray(patch == label, dtype='int')
                    thisRegion[xi-1:xi+2,yi-1:yi+2] = patch

                iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
                if iterate:
                    new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)

                temp = np.copy(thisRegion)
            cnts[str(label)] += 1
            dslices[str(label)] = dslices[str(label)] + thisRegion * cnts[str(label)]

    return dslices, cnts

def get_regions_3d(gtslice, labels):
    dummy = np.zeros(gtslice.shape, dtype='int')
    dslices = {}
    cnts = {}
    for label in labels:
        dslices[str(label)] = np.copy(dummy[16,41,42])
        cnts[str(label)] = 0

    inds = np.where(np.isin(gtslice,labels))

    for x,y,z in zip(inds[0], inds[1], inds[2]):
        label = gtslice[x,y,z]
        if dslices[str(label)][x,y,z] == 0:
            thisRegion = np.zeros(gtslice.shape, dtype='int')
            temp = np.zeros(gtslice.shape, dtype='int')
            thisRegion[x,y,z] = 1
            new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)
            iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
            temp = np.copy(thisRegion)

            while iterate:
                for xi, yi, zi in zip(new_ind[0], new_ind[1], new_ind[2]):
                    patch = gtslice[xi-1:xi+2,yi-1:yi+2,zi-1:zi+2]
                    patch = np.asarray(patch == label, dtype='int')
                    thisRegion[xi-1:xi+2,yi-1:yi+2,zi-1:zi+2] = patch

                iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
                if iterate:
                    new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)

                temp = np.copy(thisRegion)
            cnts[str(label)] += 1
            dslices[str(label)] = dslices[str(label)] + thisRegion * cnts[str(label)]

    return dslices, cnts

def get_counts(y_true, y_pred, labels, percent = 0.5, acc_area=15):
    scores = {}
    tot_slices = y_true.shape[0]
    bar = progressbar.ProgressBar(max_value=tot_slices)
    slice_cnt = 0
    for grt, pred in zip(y_true, y_pred):
        rgrts,rgcnts = get_regions(grt,labels)
        rpreds,rpcnts = get_regions(pred,labels)

        for label in labels:
            rgrt = rgrts[str(label)]
            rgcnt = rgcnts[str(label)]
            rpred = rpreds[str(label)]
            rpcnt = rpcnts[str(label)]

            TP = 0
            FP = 0
            FN = 0
            TN = 0

            for i in np.arange(1,rgcnt+1, 1):
                overlap = np.sum(np.asarray(rgrt == i, dtype='int') * np.asarray(pred == label)) * 1.0 / np.sum(np.asarray(rgrt == i, dtype='int'))
                area = np.sum(np.asarray(rgrt == i, dtype='int'))
                if overlap >= percent and area > acc_area:
                    TP += 1
                elif area > acc_area:
                    FN += 1

            for i in np.arange(1,rpcnt+1, 1):
                overlap = np.sum(np.asarray(rpred == i, dtype='int') * np.asarray(grt == label)) * 1.0 / np.sum(np.asarray(rpred == i, dtype='int'))
                area = np.sum(np.asarray(rpred == i, dtype='int'))
                if overlap < percent and area > acc_area:
                    FP += 1

            score = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

            if str(label) in scores:
                scores[str(label)]['TP'] += TP
                scores[str(label)]['FP'] += FP
                scores[str(label)]['TN'] += TN
                scores[str(label)]['FN'] += FN
            else:
                scores[str(label)] = score

        slice_cnt += 1
        bar.update(slice_cnt)
    return scores

def get_counts_3d(y_true, y_pred, labels, percent = 0.5, acc_area=15):
    scores = {}

    rgrts,rgcnts = get_regions_3d(y_true,labels)
    rpreds,rpcnts = get_regions_3d(y_pred,labels)


    for label in labels:
        rgrt = rgrts[str(label)]
        rgcnt = rgcnts[str(label)]
        rpred = rpreds[str(label)]
        rpcnt = rpcnts[str(label)]

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for i in np.arange(1,rgcnt+1, 1):
            overlap = np.sum(np.asarray(rgrt == i, dtype='int') * np.asarray(y_pred == label)) * 1.0 / np.sum(np.asarray(rgrt == i, dtype='int'))
            area = np.sum(np.asarray(rgrt == i, dtype='int'))
            if overlap >= percent and area > acc_area:
                TP += 1
            elif area > acc_area:
                FN += 1

        for i in np.arange(1,rpcnt+1, 1):
            overlap = np.sum(np.asarray(rpred == i, dtype='int') * np.asarray(y_true == label)) * 1.0 / np.sum(np.asarray(rpred == i, dtype='int'))
            area = np.sum(np.asarray(rpred == i, dtype='int'))
            if overlap < percent and area > acc_area:
                FP += 1

        score = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

        if str(label) in scores:
            scores[str(label)]['TP'] += TP
            scores[str(label)]['FP'] += FP
            scores[str(label)]['TN'] += TN
            scores[str(label)]['FN'] += FN
        else:
            scores[str(label)] = score
    return scores

def get_count_from_lists(y_true_list, y_pred_list, labels, labels_3d=None, percent=0.5):
    acc_scores = []
    cnt = 0
    print ('Calculating Scores .....')
    for y_true, y_pred in zip(y_true_list,y_pred_list):
        cnt += 1
        print ('Patient', cnt)
        scores = get_counts(y_true, y_pred, labels, percent)
        if not labels_3d is None:
            scores_3d = get_counts_3d(y_true, y_pred, labels_3d, percent)
            for lab in labels_3d:
                scores[str(lab)] = scores_3d[str(lab)]

        acc_scores.append(scores)
    return acc_scores

def calculate_scores(count_list, label_dict):
    global_scores = {}
    patient_scores = {}
    for pat, scores in enumerate(count_list):
        print ('patient :', pat + 1)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for key in scores:
            if not key in global_scores:
                global_scores[key] = {'TP': 0,'FP': 0,'FN': 0,'TN': 0}
                patient_scores[key] = {'precision': [],'recall': [],'dice': [],'accuracy': []}

            sc = scores[key]

            if sc['TP'] + sc['FN'] > 0:

                accuracy = sc['TP']*1.0 / (sc['TP'] + sc['FN'])
                precision = 0 if sc['TP'] == 0 else sc['TP']*1.0 / (sc['TP'] + sc['FP'])
                recall = sc['TP']*1.0 / (sc['TP'] + sc['FN'])
                dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

                print (label_dict[key],':')
                print ( 'precision', precision)
                print ( 'recall', recall)
                print ( 'dice', dice)
                print ( 'accuracy', accuracy)
                print ( 'total', sc['TP'] + sc['FN'])

                patient_scores[key]['precision'].append(precision)
                patient_scores[key]['recall'].append(recall)
                patient_scores[key]['dice'].append(dice)
                patient_scores[key]['accuracy'].append(accuracy)

                TP += sc['TP']
                TN += sc['TN']
                FP += sc['FP']
                FN += sc['FN']

                global_scores[key]['TP'] += sc['TP']
                global_scores[key]['TN'] += sc['TN']
                global_scores[key]['FP'] += sc['FP']
                global_scores[key]['FN'] += sc['FN']

        accuracy = 0 if TP == 0 else TP*1.0 / (TP + FN + epslone)
        precision = 0 if TP == 0 else TP*1.0 / (TP + FP + epslone)
        recall = 0 if TP == 0 else TP*1.0 / (TP + FN + epslone)
        dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall + epslone)

        print ('patient summary')
        print ( 'precision', precision)
        print ( 'recall', recall)
        print ( 'dice', dice)
        print ( 'accuracy', accuracy)

    print ('Patient Scores')
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for lab in global_scores:
        sc = global_scores[lab]
        accuracy = sc['TP']*1.0 / (sc['TP'] + sc['FN'] + epslone)
        precision = 0 if sc['TP'] == 0 else sc['TP']*1.0 / (sc['TP'] + sc['FP'] + epslone)
        recall = sc['TP']*1.0 / (sc['TP'] + sc['FN'] + epslone)
        dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall + epslone)

        print (label_dict[lab],' accuracy:',accuracy, ' precision:',precision, ' recall:',recall, ' dice:',dice, '  total:', sc['TP'] + sc['FN'])
        TP += sc['TP']
        TN += sc['TN']
        FP += sc['FP']
        FN += sc['FN']

    accuracy = 0 if TP == 0 else TP*1.0 / (TP + FN  + epslone)
    precision = 0 if TP == 0 else TP*1.0 / (TP + FP + epslone)
    recall = 0 if TP == 0 else TP*1.0 / (TP + FN + epslone)
    dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall + epslone)
    print ('Combined',' accuracy:',accuracy, ' precision:',precision, ' recall:',recall, ' dice:',dice, '  total:', TP + FN)

if __name__ == '__main__':
    labels = load_lesion_labels()
    print ('SCORES CF VNET')    
    preds = load_predictions(preds_PREFIX)
    scores = get_count_from_lists(y_true_list=labels, y_pred_list=preds, labels=[1,2,3], labels_3d=None, percent = 0.3)
    calculate_scores(scores, label_dict)
