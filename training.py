from sfcn_smaller_kernel import sfcn_pyramid_small_kern
from sfcn_pyr_nopool_2 import sfcn_pyramid_strides
from sfcn_pyr_avg_plat import sfcn_pyr_avg

if __name__=='__main__':

    try:
        sfcn_pyramid_strides(1)    
    except:
        print('skipping sfcn stride')

    try:
        sfcn_pyramid_small_kern(0)
    except:
        print('skipping sfcn small kernel')

    try:
        sfcn_pyr_avg(0)
    except:
        print('skipping sfcn small kernel')