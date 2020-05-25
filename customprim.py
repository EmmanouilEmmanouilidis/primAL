import numpy as np
from scipy.stats.mstats import mquantiles

def make_box(data):
    box = np.zeros((2,data.shape[1]))
    for i in range(data.shape[1]):
        values = data[:,i]
        box[0,i] = np.min(values)
        box[1,i] = np.amax(values)
    return box

def vol_box(box):
    return np.prod(np.absolute(np.subtract(box[1,:],box[0,:])))

def in_box(x, box, d, boolean=False):
    x_box_ind = np.ones(x.shape[0], dtype=bool)
    for i in range(d):
        if box[0][i] != box[1][i]:
            x_box_ind = x_box_ind & (np.less_equal(box[0][i], x[:,i])) & (np.less_equal(x[:,i], box[1][i]))

    x_ind = np.argwhere(x_box_ind == True)

    if boolean == True:
        return x_ind
    else:
        return x[x_ind,:]

def peel_one(x, y, box, peel_alpha, mass_min, threshold, d, n):
    x_index = np.array([])
    row_ind = 0
    box_new = box
    mass = y.size/n

    if x.size == 0:
        return None

    y_fun_val = np.mean(y)
    y_fun_peel = np.zeros((2,d))
    box_vol_peel = np.zeros((2,d))

    for j in range(d):
        box_min_new = mquantiles(x[:,j], prob=[peel_alpha], alphap=(1/3), betap=(1/3))
        box_max_new = mquantiles(x[:,j], prob=[(1-peel_alpha)], alphap=(1/3), betap=(1/3))

        y_fun_peel[0][j] = np.mean(y[np.argwhere(x[:,j] >= box_min_new)])
        y_fun_peel[1][j] = np.mean(y[np.argwhere(x[:,j] <= box_max_new)])

        box_temp1 = box
        box_temp2 = box
        box_temp1[0][j] = box_min_new
        box_temp2[1][j] = box_max_new
        box_vol_peel[0][j] = vol_box(box_temp1)
        box_vol_peel[1][j] = vol_box(box_temp2)

    y_fun_peel_max_ind = np.argwhere(y_fun_peel == np.nanmax(y_fun_peel))

    nrr = y_fun_peel_max_ind.shape[0]
    if nrr > 1:
        box_vol_peel2 = np.zeros(nrr)
        for j in range(nrr):
            box_vol_peel2[j] = box_vol_peel[y_fun_peel_max_ind[j,0], y_fun_peel_max_ind[j,1]]
        row_ind = np.argwhere(np.amax(box_vol_peel2) == box_vol_peel2).flatten()
        row_ind = row_ind[0]
    else:
        row_ind = 0

    y_fun_peel_max_ind = y_fun_peel_max_ind[row_ind,:]


    j_max = y_fun_peel_max_ind[1]
    if y_fun_peel_max_ind[0] == 0:
        box_new[0,j_max] = mquantiles(x[:,j_max], prob=[peel_alpha], alphap=(1/3), betap=(1/3))
        x_index = np.where(x[:, j_max] >= box_new[0, j_max])
    elif y_fun_peel_max_ind[0] == 1:
        box_new[1,j_max] = mquantiles(x[:,j_max], prob=[(1-peel_alpha)], alphap=(1/3), betap=(1/3))
        x_index = np.where(x[:, j_max] <= box_new[1, j_max])
    x_new = x[x_index,:]
    y_new = y[x_index]
    mass_new = y_new.size / n
    y_fun_new = np.mean(y_new)

    if((y_fun_new >= threshold) and (mass_new >= mass_min) and (mass_new < mass)):
        return [x_new,y_new,y_fun_new,box_new,mass_new]


def paste_one(x, y, box, x_init, y_init, paste_alpha, mass_min, threshold, d, n):
    x_new = x
    y_new = y
    box_new = box
    mass = y.size / n
    y_fun_val = np.mean(y)
    n_box = y.size

    box_init = make_box(x_init)

    if x.shape[1] == 0:
        x = np.transponse(x)

    y_fun_paste = np.zeros((2,d))
    mass_paste = np.zeros((2,d))
    box_paste = np.zeros((2,d))
    x_paste1_list = []
    x_paste2_list = []
    y_paste1_list = []
    y_paste2_list = []

    box_paste1 = box
    box_paste2 = box

    for j in range(d):
        box_diff = np.subtract(box_init[1,:], box_init[0,:])[j]

        box_paste[0][j] = box[0][j] - box_diff*paste_alpha
        box_paste[1][j] = box[1][j] + box_diff*paste_alpha

        x_paste1_ind = in_box(x_init, box_paste1, d, boolean=True)
        x_paste1 = x_init[x_paste1_ind,:]
        y_paste1 = y_init[x_paste1_ind]

        x_paste2_ind = in_box(x_init, box_paste2, d, boolean=True)
        x_paste2 = x_init[x_paste2_ind,:]
        y_paste2 = y_init[x_paste2_ind]

        while y_paste1.shape[0] <= y.shape[0] and box_paste1[0][j] >= box_init[0][j]:
            box_paste1[0][j] = box_paste1[0][j] - box_diff*paste_alpha
            x_paste1_ind = in_box(x_init, box_paste1, d, boolean=True)
            x_paste1 = x_init[x_paste1_ind, :]
            y_paste1 = y_init[x_paste1_ind]

        while y_paste2.shape[0] <= y.shape[0] and box_paste2[1][j] >= box_init[1][j]:
            box_paste2[1][j] = box_paste2[1][j] + box_diff*paste_alpha
            x_paste2_ind = in_box(x_init, box_paste2, d, boolean=True)
            x_paste2 = x_init[x_paste2_ind, :]
            y_paste2 = y_init[x_paste2_ind]

        y_fun_paste[0][j] = np.mean(y_paste1)
        y_fun_paste[1][j] = np.mean(y_paste2)

        mass_paste[0][j] = y_paste1.shape[0]/n
        mass_paste[1][j] = y_paste2.shape[0]/n

        x_paste1_list.append(x_paste1)
        y_paste1_list.append(y_paste1)
        x_paste2_list.append(x_paste2)
        y_paste2_list.append(y_paste2)
        box_paste[0][j] = box_paste1[0][j]
        box_paste[1][j] = box_paste2[1][j]

    y_fun_paste_max = np.argwhere(y_fun_paste == np.nanmax(y_fun_paste))
    print(y_fun_paste)
    print("y_fun_paste_max")
    print(y_fun_paste_max)
    print(mass_paste)

    #ind = np.unravel_index(np.argmax(mass_paste, axis=None), mass_paste.shape)

    if(y_fun_paste_max.shape[0] > 1):
        print("masses")
        mass_vals = []
        for i in range(y_fun_paste_max.shape[1]):
            mass_vals.append(int(mass_paste[y_fun_paste_max[i,0], y_fun_paste_max[i,1]]))
        print(mass_vals)

        #masses = np.array([int(mass_paste[y_fun_paste_max[0,0],y_fun_paste_max[0,1]]), int(mass_paste[y_fun_paste_max[1,0],y_fun_paste_max[1,1]])])

        y_fun_paste_max = np.c_[y_fun_paste_max, mass_vals]
        print("y_fun_paste_max")
        print(y_fun_paste_max)
        y_fun_paste_max_ind = y_fun_paste_max[np.argsort(-1*y_fun_paste_max[:,2])][0:1][0]
        print("y_fun_paste_max_ind")
        print(y_fun_paste_max_ind)

    else:
        y_fun_paste_max_ind = y_fun_paste_max.tolist()

    j_max = y_fun_paste_max_ind[1]
    print("jmax")
    print(j_max)

    if y_fun_paste_max_ind[0] == 0:
        x_new = x_paste1_list[j_max]
        y_new = y_paste1_list[j_max]
        box_new[0][j_max] = box_paste[0][j_max]

    elif y_fun_paste_max_ind[0] == 1:
        x_new = x_paste2_list[j_max]
        y_new = y_paste2_list[j_max]
        box_new[1][j_max] = box_paste[1][j_max]

    mass_new = y_new.shape[0]/n
    y_fun_new = np.mean(y_new)

    if y_fun_new > threshold and mass_new >= mass_min and y_fun_new >= y_fun_val and mass_new > mass:
        return [x_new, y_new, y_fun_new, box_new, mass_new]
