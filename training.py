# from sfcn_smaller_kernel import sfcn_pyramid_small_kern
# from sfcn_pyr_strides_2 import sfcn_pyramid_strides
# from sfcn_pyr_avg_plat import sfcn_pyr_avg
# from sfcn_pyr import sfcn_pyr
# from scripts.sfcn_vanilla import sfcn_vanilla
# from sfcn_pyr_strides_2 import sfcn_pyramid_strides2
# from sfcn_pyr_strides_3 import sfcn_pyramid_strides3

# from sfcn_large_nfil import sfcn_lfil_shallow_0, sfcn_lfil_shallow_1, sfcn_lfil_shallow_2
# from sfcn_bigger_kern import sfcn_pyramid_bigger_kern
# from sfcn_sigmoid import sfcn_sigmoid

# from sfcn_deeper import sfcn_deeper_k2_glomax
# from sfcn_pyr_avg_plat import sfcn_glomax
# from sfcn_large_nfil import sfcn_lfil_shallow_glomax

# from sfcn_res import res_sfcn_glomax

# from sfcn_shallow import sfcn_lfil_shallow_0_glomax
# from sfcn_shallow import sfcn_lfil_shallow_ds_glomax

# from sfcn_vanilla_mod import sfcn_glomax_ds

# from sfcn_deeper import sfcn_deeper_ks_glomax_us
# from sfcn_vanilla import sfcn_vanilla

# from scripts.sfcn_norm import sfcn_layer_norm
# from scripts.sfcn_norm import sfcn_group_norm16, sfcn_group_norm8
# from scripts.sfcn_ws import sfcn_ws, sfcn_ws16

from scripts.sfcn_deeper import sfcn_deeper
from scripts.sfcn_shallow import sfcn_shallow_3, sfcn_shallow_3_ds
from scripts.sfcn_vanilla_mod import sfcn_ds

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

    #print('how you doin')
    
    # try:
    #     sfcn_pyramid_strides2(0)
    # except:
    #     print('skipping sfcn psfcn_pyramid_strides2')

    # try:
    #     sfcn_pyramid_strides3(0)
    # except:
    #     print('skipping sfcn psfcn_pyramid_strides3')

    # try:
    #     sfcn_pyr_avg(2)
    # except:
    #     print('skipping sfcn pyr avg')
        
    # try:
    #     sfcn_pyramid_small_kern(1)
    # except:
    #     print('skipping smaller kernel sfcn')


    # try:
    #     sfcn_sigmoid(0)
    # except:
    #     print('skipping sigmoid sfcn')
         
    # try:
    #     sfcn_lfil_shallow(3)
    # except:
    #     print('skipping shallow sfcn')
        
    # try:
    #     sfcn_pyramid_bigger_kern(1)
    # except:
    #     print('skipping smaller kerne
 
    # try:
    #     sfcn_lfil_shallow_0(0)
    # except:
    #     print('skipping shallow sfcn 0')

    
    # try:
    #     sfcn_lfil_shallow_1(0)
    # except:
    #     print('skipping shallow sfcn 1')

        
    # try:
    #     sfcn_lfil_shallow_2(0)
    # except:
    #     print('skipping shallow sfcn 2')

    # try:
    #     sfcn_deeper_k2_nopool(1)
    # except:
    #     print('skipping deeper sfcn')

    # try:
    #     sfcn_deeper_k2_glomax(0)
    # except:
    #     print('skipping sfcn_deeper_k2_glomax')

    # try:
    #     sfcn_glomax(0)
    # except:
    #     print('skipping sfcn_glomax')
 
    # try:
    #     sfcn_lfil_shallow_glomax(0)
    # except:
    #     print('skipping sfcn_lfil_shallow_glomax')

    # try:
    #     res_sfcn_glomax(0)
    # except: 
    #     print('skipping res_sfcn')

    # try:
    #     sfcn_glomax_ds(0)
    # except:
    #     print('sfcn ds')

    # try:
    #     sfcn_lfil_shallow_ds_glomax(2)
    # except: 
    #    print('skipping ds shallow')

    # try:
    #     sfcn_lfil_shallow_0_glomax(2)
    # except: 
    #     print('shallow res_sfcn')

    # try:
    #     sfcn_deeper_ks_glomax_us(0)
    # except:
    #     print('skipping upsamples')

    # try:
    #     sfcn_vanilla(0)
    # except:
    #     print('skipping vanilla')

    # sfcn_layer_norm(0)

    #sfcn_no_norm(0)
    
    # sfcn_group_norm8(0)
    # sfcn_group_norm32(0)

    #sfcn_wnorm8(0)
    # sfcn_wnorm16(0)
    # sfcn_group_norm8(0)
    # sfcn_group_norm16(0)
    #sfcn_ws16(0)

    # MISSING 
    #sfcn_layer_norm(1)

    # sfcn_group_norm8(1)
    # sfcn_group_norm16(1)

    # sfcn_ws(1)
    # sfcn_ws16(1)

    sfcn_deeper(0)
    sfcn_shallow_3(0)
    sfcn_shallow_3_ds(0)
    sfcn_ds(0)