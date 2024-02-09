def color_group_matrix_plot(corrmat,roiidx,addrow=2,flip=0,remove_diag=1,
                            mat_colormap=None,
                            g_colormap=None,
                            to_plot=0,
                            addpatch=0,
                            roi_labels=None,
                            roi_group_label=None,
                           hypothesis_mat=None,
                            cbar=None,savefig=None,
                            grab_fig=0,
                           force_linspace=0,edge_color='w',vmin=None,vmax=None):
    # ['Pastel1', 'Pastel2', 'Paired', 'Accent', 
    #            'Dark2','Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
    #                  'tab20c'])
    # try - group_color =['#e6091c','#fcba03','#1414e3','#d0cdd1']
    # nature_palette = ['#000000', '#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7']
    # group_color_palleter =['#ff0400', '#ff7300', '#ffae00','#ffe600','#bbff00','#59ff00','#00ffbf','#00f2ff','#00b3ff',
    #                  '#006eff','#001aff','#8000ff','#d000ff','#ff00f7','#ff00a6']
    # import random > random.sample(group_color_palleter,nROIs_Groups)
	# force_linspace needed with small number of networks to ensure colors for each are visible
	# vmin and vmax are the colorbar limits   
    
    import numpy as np
    from numpy.ma import masked_array
    from matplotlib.colors import is_color_like
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib.pyplot as plt
    # add checks after
    
    if hypothesis_mat is None:
        hyp_present = 0
    else:
        hyp_present=1
        print('Hypothesis Mat is given')
        
    
    
    
    if roiidx.ndim > 1:
        roiidx = np.squeeze(roiidx)
        #print(roiidx.shape)
        assert roiidx.ndim == 1,'ROI idx should be a one dimentional array of indexes for groupings'
    
    #if roi_labels is not None:
        #if roi_labels.ndim >= 2:
         #   roi_labels = np.squeeze(roi_labels)
         #   assert roi_labels.ndim == 1, 'ROI labels should be numpy array with single dimention for ROIs'
    
    if remove_diag == 1:
        np.fill_diagonal(corrmat,np.nan);
    
    if flip == 1:
        
        corrmat = np.rot90(corrmat)
        
        if hyp_present ==1:
            hypothesis_mat = np.rot90(hypothesis_mat)
        
    
    #print(g_colormap)
    if g_colormap is None:
        group_colormap='tab20c'
    elif isinstance(g_colormap, list) and all(list(map(is_color_like,g_colormap))):
        group_colormap=ListedColormap(g_colormap[-1::-1])
    elif isinstance(g_colormap, list) and len(roiidx) < 8:
        print('You should specify a list of colors - resorting to nature palette')
        temp = ['#000000', '#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7'];
        group_colormap=ListedColormap(temp)
    elif isinstance(g_colormap, list) and len(roiidx) > 8:
        print('You should specify a list of colors - resorting to tab20c palette')
        group_colormap='tab20c'
    elif (isinstance(g_colormap,str) and g_colormap in plt.colormaps()):
        group_colormap=g_colormap
        
    if mat_colormap is None:
        matrix_colormap='viridis'
    elif (isinstance(mat_colormap,str) and mat_colormap in plt.colormaps()):
        matrix_colormap = mat_colormap
    else: matrix_colormap='viridis' 
    
    
    
    # pad a 2 rows at end and 2 columns at begining. - ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis. 
    new_mat_pad = np.pad(corrmat.astype(float),((0,addrow),(addrow,0)),mode='constant',constant_values=np.nan) # see also mode reflect
    
    if hyp_present ==1:
        new_hypothesis_mat = np.pad(hypothesis_mat.astype(float), ((0,addrow),(addrow,0)) ,mode='constant',constant_values=np.nan)
    
    
    roiidx[-1] == corrmat.shape[1] - 1
    roiidx_2 = np.insert(roiidx,0,0) + addrow;
    
    if roiidx[0] != 0:
        roiidx= np.insert(roiidx,0,0);
    
    
    # make the new mask values possibly more variable for smaller groups since they will be similar colors
    # easier to change the colormap
    new_mask_values = roiidx.copy()
    new_mask_values[0] = 1#np.max(new_mask_values) + np.std(new_mask_values)*0.3
    new_mask_values = new_mask_values.astype(float)
    if len(new_mask_values) <= 8 & force_linspace==0:
        new_mask_values = new_mask_values*10
    elif len(new_mask_values) > 8:
        new_mask_values = np.linspace(0,len(new_mask_values),len(new_mask_values))#np.arange(0,len(new_mask_values))
    elif force_linspace==1:
        new_mask_values = np.linspace(0,len(new_mask_values),len(new_mask_values))
    # new_mask_values =new_mask_values*10
   # new_mask_values = new_mask_values/np.max(new_mask_values)
    
    roiidx_cols = roiidx + addrow;
    #print(roiidx_cols)
    
    
    ## 
    #start = numpy_indx[0:-1]
    #end = numpy_indx[1:]
    #np.r_[tuple(slice(s, e) for s, e in zip(start, end))]

    # loop through indexes in new padded matrix for each group and assign them a new value
    for i in np.arange(0,roiidx.shape[0]):
        #print(i)
        if i <(roiidx.shape[0] - 1):

            
            new_mat_pad[ roiidx[i]:roiidx[i+1],0:addrow ] = new_mask_values[i]
            new_mat_pad[(-1*addrow): , roiidx_cols[i]:roiidx_cols[i+1]] = new_mask_values[i]
    
    if flip == 1:
        new_mat_pad[0:(-1*addrow),0:addrow] = new_mat_pad[(-1*addrow - 1)::-1,0:addrow]
            
            
    
    
    
    #roi_patch = masked_array(new_mat_pad,new_mat_pad<10) - this is fine for corr mats or beta matrixes that have values below 10
    #true_corr = masked_array(new_mat_pad,new_mat_pad>=10)
    
    # a bit more robust masking
    
    # get indexes but without loop so all indeces used for the grouping regardless of group
    
    #roiidx = numpy_indx[0:-1] # get first to penultimate element
    #roiidx = numpy_indx[1:]# get second to last element
    #np.r_[tuple(slice(s, e) for s, e in zip(start, end))]
    
    # Note alternatively since we now the shape of the corr matrix we are interested in we can simply do np.arange(0,corr_mat.shape[0])
    
    idx_to_be_masked_rows = np.r_[tuple(slice(s, e) for s, e in zip(roiidx[0:-1], roiidx[1:]))]
    idx_to_be_masked_cols = np.r_[tuple(slice(s, e) for s, e in zip(roiidx_cols[0:-1], roiidx_cols[1:]))]

    
    
    temp_zeros = np.zeros_like(new_mat_pad);
    temp_zeros[idx_to_be_masked_rows,0:addrow] = 1;
    temp_zeros[(-1*addrow): , idx_to_be_masked_cols] = 1;
    
    true_mat = masked_array(new_mat_pad, temp_zeros);
    group_patch = masked_array(new_mat_pad, np.logical_not(temp_zeros))
    
    #### PLOTTING
    
    if to_plot == 1:
        fig,ax = plt.subplots(figsize=(20,20))
        C=ax.imshow(true_mat,cmap=matrix_colormap,vmin=vmin,vmax=vmax)
        ax.imshow(group_patch,cmap=group_colormap)
        ax.tick_params(bottom=False,left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if cbar is not None:
            cbar = fig.colorbar(C,fraction= 0.040,pad=0.04,ax=ax,ticks=None,extend='both')# ,extend='both'
            cbar.ax.tick_params(labelsize=20)
        
        
    
        if addpatch ==1:
            import matplotlib.patches as patches
            if flip ==0:
                temp_vec = new_mat_pad[:,0];
                temp_vec = np.insert(temp_vec,0,temp_vec[0]) # here I insert a value at the begining simply because np.diff counts the i-1 to i rather than i to i+1
                bounds = np.where(abs(np.diff(temp_vec))>0)[0]
                bounds_aug = np.concatenate(([0],bounds,[new_mat_pad.shape[0]]))
            #bounds_aug[:-1] +=addrow
                bounds_aug[-1] -= addrow
                for i in range(len(bounds_aug)-1):
                    rec = patches.Rectangle((bounds_aug[i]-0.5+addrow,bounds_aug[i]-0.5), 
                                           bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],
                                           linewidth=2,facecolor='none',edgecolor='w') # (x,y) start position of square
                    # then I specify width and height of rectangle
                    ax.add_patch(rec)
            
            elif flip==1:
                temp_col=new_mat_pad[-1,:];
                temp_col=np.insert(temp_col,0,temp_col[0]) # 
                bounds_col = np.where(abs(np.diff(temp_col)) >0 )[0]
                bounds_col_aug = np.concatenate(([0],bounds_col,[new_mat_pad.shape[0]]))
                
                bounds_col_aug[0] += addrow
                
                widths = np.diff(bounds_col_aug);
                
                start_position = new_mat_pad.shape[0] - 0.5 - addrow
                list_positions = []
                list_positions.append(start_position)
                
                for iwid in widths:
                    start_position -= iwid
                    list_positions.append(start_position)
                
                for i in range(len(bounds_col_aug) -1 ):
                    rec = patches.Rectangle((bounds_col_aug[i] - 0.5, list_positions[i]),
                                           widths[i],widths[i]*-1,linewidth=2,facecolor='none',edgecolor=edge_color) # (x,y) of square note the different y position
                    # then I specify width and height. NOTE height is with minus sign to show that we are starting from bottom and going up
                    ax.add_patch(rec)
    
    
    
    #ax.axes.spines['top'].set_visible(False) # sets the top line of the figure frame as insivisuble see 
    #ax.axes.spines['right'].set_visible(False) # sets the right line of the figure frame as insivisuble see 
    #ax.axes.spines['bottom'].set_visible(False) # sets the bottom line of the figure frame as insivisuble see 
    #ax.axes.spines['left'].set_visible(False) # sets the left line of the figure frame as insivisuble see 
    

    
        if roi_labels is not None:
            
            yticks = np.arange((0),(corrmat.shape[0]))
            xticks = np.arange((0 + addrow),(corrmat.shape[0] + addrow))
    #if ticks == 1:
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(roi_labels,fontsize=12,rotation=45)
            if flip == 0:
              ax.set_yticks(yticks)
              ax.set_yticklabels(roi_labels,fontsize=20)
            elif flip == 1:
              ax.set_yticks(yticks)
              ax.set_yticklabels(roi_labels[::-1],fontsize=20)  
        
        if roi_group_label is not None:
        
            xti = [((roiidx_2[i+1] - roiidx_2[i])/2)+roiidx_2[i] for i in range(len(roiidx_2)-1)]
            ax.set_xticks(xti,minor=False)
            ax.set_xticklabels(roi_group_label,fontsize=20,rotation=65)
        
        #ax.tick_params(axis=u'both', which=u'both',length=0)
        #ax.tick_params(axis='x', which='major', pad=25)
    
        if hyp_present == 1:
            #ax.spy(new_hypothesis_mat,marker='.', markersize=20,color='white')
            ax.scatter(*np.argwhere(new_hypothesis_mat.T>=1).T,marker='.',s=80,color='white')
        if savefig == 1:
            plt.savefig('Correlation_Matrix.png')
    
    if grab_fig == 1:
      return group_patch,true_mat,new_mat_pad,fig,ax 
    else:
      return group_patch,true_mat,new_mat_pad
