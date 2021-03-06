import nibabel as nib 
import numpy as np
import math
# import numba as nb

# =============================================================================
# Read nifti
# =============================================================================
mask = nib.load("C:\\Users\\linda\\Desktop\\Funtensor_code\\MNI_Mask_2mm.nii.gz")
brain_mask = mask.get_fdata() # 只取img的部分(去掉header)

vox_dimenx = np.size(brain_mask, 0)
vox_dimeny = np.size(brain_mask, 1)
vox_dimenz = np.size(brain_mask, 2)



# =============================================================================
# neib_vec2.m
# =============================================================================
def ind2sub(b_mask):
    tem_xyz = []
    for k in range(vox_dimenz):
        for j in range(vox_dimeny):
            for i in range(vox_dimenx):
                if b_mask[i, j, k] != 0:
                    tem_xyz.append([i, j, k])
    tem_xyz = np.asarray(tem_xyz)
    return tem_xyz

vox_xyz = ind2sub(brain_mask)
vox_x = vox_xyz[:, 0]
vox_y = vox_xyz[:, 1]
vox_z = vox_xyz[:, 2]
num_vox = np.size(vox_xyz, 0)
vox_neib_xyz = np.empty((num_vox, 1), dtype = object)
neib_vox_vec = np.empty((num_vox, 1), dtype = object)
    
for v in range(num_vox):
    t = 0
    tmp_neib_vox = np.zeros([27, 3])
    tmp_neib_vec = np.zeros([27, 3])
    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                if t == 13: # python不允許分母為0的除法，所以要避掉seed voxel
                    tmp_neib_vox[t, :] = [0, 0, 0]
                    tmp_neib_vec[t, :] = [0, 0, 0]
                    t += 1
                else:
                    tmp_neib_vox[t, :] = [vox_x[v] + a, vox_y[v] + b, vox_z[v] + c]
                    tmp_sqr = math.sqrt(a * a + b * b + c * c)
                    tmp_neib_vec[t, :] = [a / tmp_sqr, b / tmp_sqr, c / tmp_sqr]

                    t += 1
    tmp_neib_vox = np.delete(tmp_neib_vox, (13), axis = 0)
    tmp_neib_vec = np.delete(tmp_neib_vec, (13), axis = 0)
    
    vox_neib_xyz[v, 0] = tmp_neib_vox
    neib_vox_vec[v, 0] = tmp_neib_vec
# =============================================================================


num_vox = np.size(vox_xyz, 0) # fun_tensor2_Zac 7
sub_nii = nib.load("C:\\Users\\linda\\Desktop\\Funtensor_code\\MRN_002sregressedglobal_and_filtered_FunImg_2mmstd_func_tensor2_fisherz_fsl.nii.gz")
rest = sub_nii.get_fdata() # fun_tensor2_Zac 12


# =============================================================================
# NeibCor2.m
# =============================================================================
vox_x = vox_xyz[:, 0]
vox_y = vox_xyz[:, 1]
vox_z = vox_xyz[:, 2]
n_len = np.size(rest, 3) # t = 6
n_vox = np.size(vox_xyz, 0)
neib_cor = np.zeros([n_vox, 26]) # 236840 * 26

tc = np.empty((1, 6), dtype = float)
for i in range(n_vox): # 0 - 236839
    tmp_rest = rest
    # tc = np.empty((num_vox, 1), dtype = object) #num_vox要改～
    tc = np.squeeze(tmp_rest[vox_x[i], vox_y[i], vox_z[i], :])
    if np.sum(tc == 0) == n_len:
        continue
    for j in range(26):
        neib_xyz = vox_neib_xyz[i,0][j,:]
        neib_tc = np.squeeze(tmp_rest[int(neib_xyz[0]), int(neib_xyz[1]), int(neib_xyz[2]),:])
        if np.sum(neib_tc == 0) != n_len:
            # tmp = np.corrcoef(np.hstack((neib_tc, tc)),rowvar = False)
            A = (tc - tc.mean(axis = 0)) / tc.std(axis = 0)
            B = (neib_tc - neib_tc.mean(axis = 0))/neib_tc.std(axis = 0)
            tmp = (np.dot(B.T, A) / B.shape[0])
            neib_cor[i, j] = tmp
# =============================================================================



# =============================================================================
# fisherz.m 
# =============================================================================
def fisherz(neib_correlation):
    return ((np.log(np.divide(1 + neib_correlation, 1 - neib_correlation))) / 2)

zneib_cor = fisherz(neib_cor)
# =============================================================================


C = np.power(zneib_cor, 2) # fun_tensor2_Zac 15
T = np.zeros([num_vox, 6]) # 236840 * 6 # fun_tensor2_Zac 16


# =============================================================================
# dyadic_tensor_half
# =============================================================================
def dyadic_tensor_half(unit_vec):
    vec_x = unit_vec[0]
    vec_y = unit_vec[1]
    vec_z = unit_vec[2]
    D_tensor = np.zeros([1, 6])
    D_tensor[0, 0] = vec_x * vec_x
    D_tensor[0, 1] = vec_x * vec_y
    D_tensor[0, 2] = vec_x * vec_z
    D_tensor[0, 3] = vec_y * vec_y
    D_tensor[0, 4] = vec_y * vec_z
    D_tensor[0, 5] = vec_z * vec_z
    return D_tensor
# =============================================================================

# =============================================================================
# designM
# =============================================================================
def designM(n_vec):
    Nneib = np.size(n_vec, 0) # 26
    M = np.zeros([Nneib, 6]) # 26 * 6
    for neib in range(Nneib):
        M[neib, :] = dyadic_tensor_half(n_vec[neib, :])
    return M
# =============================================================================

for i in range(num_vox): # fun_tensor2_Zac 19
    tmp_vec = neib_vox_vec # 236840 * 1
    neib_vec = tmp_vec[i, 0] # 26 * 3
    M = designM(neib_vec) # 26 * 6
    M_transp = M.T
    neib_C = C[i, :].T
    tmp_T = (np.linalg.inv(M_transp.dot(M))).dot(M_transp).dot(neib_C)
    T[i, :] = tmp_T.T


vol1 = T[:, 0] # fun_tensor2_Zac 38
vol2 = T[:, 3]
vol3 = T[:, 5]
vol4 = T[:, 1]
vol5 = T[:, 2]
vol6 = T[:, 4] 

B = np.zeros([num_vox, 6])
B[:, 0] = vol1 # fun_tensor2_Zac 59
B[:, 1] = vol2
B[:, 2] = vol3
B[:, 3] = vol4
B[:, 4] = vol5
B[:, 5] = vol6

B2 = np.zeros([vox_dimenx, vox_dimeny, vox_dimenz, 6])
for n in range(num_vox):
    B2[vox_xyz[n, 0], vox_xyz[n, 1], vox_xyz[n, 2], :] = B[n, :]
sub_nii.img = B2 # fun_tensor2_Zac 64

sub_nii.header.structarr["dim"][4] = 6 # fun_tensor2_Zac 65
sub_nii.header.structarr["pixdim"][4] = 1 # fun_tensor2_Zac 66

new_sub_nii = nib.Nifti2Image(sub_nii.img, sub_nii.affine)
filename = "test.nii.gz"
nib.save(new_sub_nii, filename)




























