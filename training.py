#from sfcn_smaller_kernel import sfcn_pyramid_small_kern
#from sfcn_bigger_kern import sfcn_pyramid_bigger_kern
# from sfcn_pyr_strides_2 import sfcn_pyramid_strides
from sfcn_pyr_avg_plat import sfcn_pyr_avg
#from sfcn_pyr import sfcn_pyr
#from sfcn_vanilla import sfcn_vanilla
from sfcn_pyr_strides_2 import sfcn_pyramid_strides2
from sfcn_pyr_strides_3 import sfcn_pyramid_strides3

if __name__=='__main__':

    # try:
    #     sfcn_pyramid_strides(1)    
    # except:
    #     print('skipping sfcn stride')

    # try:
    #     sfcn_pyramid_small_kern(1)
    # except:
    #     print('skipping sfcn small kernel')

    # try:
    #     sfcn_pyramid_bigger_kern(0)
    # except:
    #     print('skipping sfcn big kernel')

    # try:
    #     sfcn_vanilla(14)
    # except:
    #     print('skipping sfcn vanilla')

        
    # try:
    #     sfcn_pyr(0)
    # except:
    #     print('skipping sfcn pyr')

    try:
        sfcn_pyr_avg(0)
    except:
        print('skipping sfcn pyr avg')
    #print('how you doin')
    
    try:
        sfcn_pyramid_strides2(0)
    except:
        print('skipping sfcn psfcn_pyramid_strides2')

    try:
        sfcn_pyramid_strides3(0)
    except:
        print('skipping sfcn psfcn_pyramid_strides3')