import nibabel as nib 
import numpy as np
import math

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
    tem_xyz = np.empty((0, 3), float)
    for k in range(vox_dimenz):
        for j in range(vox_dimeny):
            for i in range(vox_dimenx):
                if b_mask[i, j, k] != 0:
                    tem_xyz = np.vstack([tem_xyz, [i, j, k]])
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

num_vox = np.size(vox_xyz, 0) #fun_tensor2_Zac 7
sub_nii = nib.load("C:\\Users\\linda\\Desktop\\Funtensor_code\\MRN_002sregressedglobal_and_filtered_FunImg_2mmstd_func_tensor2_fisherz_fsl.nii.gz")
rest = sub_nii.get_fdata() #fun_tensor2_Zac 12

# 4/15 要理解NeibCor2.m和完成fisherz.m(這個比較簡單)





































