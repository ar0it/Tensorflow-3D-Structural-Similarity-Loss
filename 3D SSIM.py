#######################################################################################################################

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 3D gaussian Kernel for convolution."""

    d = tfp.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size+1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j,k->ijk',
                                  vals,
                                  vals,
                                  vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)

#######################################################################################################################

def ssim_loss(y_true, y_pred, k1, k2, L, filter_size):
    
    l_weights = [0.6,0.1,0.1,0.1,0.1]
    # Make Gaussian Kernel with desired specs.
    gauss_kernel = gaussian_kernel(size = 2, mean = 0, std = 1)

    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
    gauss_kernel = gauss_kernel[ :, :,:,tf.newaxis,tf.newaxis]
    
    #varaibles
    c1 = tf.constant((k1*L)**2)
    c2 = tf.constant((k2*L)**2)
    c3 = c2/2
    
    alpha = 1
    beta = alpha
    gamma = beta
    
    if y_true.shape[2] == None:
        dim = filter_size
    else:
        dim = y_true.shape[2]
        
    v_number_of_pooling = 5
    v_number_of_reps = ((dim/filter_size)**3)
    l_number_of_pooling = range(v_number_of_pooling)
    v_limit = int(dim/filter_size)
    l_filter_number = range(0, v_limit) #falls dim < filter size -> durch aufrunden lösen
    v_ssim_endgueltig = 0
    
    for i in l_number_of_pooling:
        v_ssim = 0
        for iiii in l_filter_number:
            for iii in l_filter_number:                
                for ii in l_filter_number:

                    y_pred_cube = y_pred[0, ii*filter_size:(ii+1)*filter_size, iii*filter_size:(iii+1)*filter_size, iiii*filter_size:(iiii+1)*filter_size, 0]
                    y_true_cube = y_true[0, ii*filter_size:(ii+1)*filter_size, iii*filter_size:(iii+1)*filter_size, iiii*filter_size:(iiii+1)*filter_size, 0]

                    mean_x = tf.math.reduce_mean(tf.cast(y_true_cube, dtype = tf.float32))
                    mean_y = tf.math.reduce_mean(tf.cast(y_pred_cube, dtype = tf.float32))

                    std_x = tf.math.reduce_std(tf.cast(y_true_cube, dtype = tf.float32))
                    std_y = tf.math.reduce_std(tf.cast(y_pred_cube, dtype = tf.float32))
                    t_y_pred_flat = tf.reshape(y_pred_cube,[-1])
                    t_y_true_flat = tf.reshape(y_true_cube,[-1])
                    covar = tfp.stats.covariance(t_y_pred_flat,t_y_true_flat, sample_axis=0, event_axis=None)

                    #luminance = (2*mean_x*mean_y+c1)/(mean_x**2+mean_y**2+c1)
                    #contrast = (2*std_x*std_y+c2)/(std_x**2+std_y**2+c2)
                    #structure = (covar+c3)/(std_x*std_y+c3)
                    v_ssim = v_ssim + (((2*mean_x * mean_y+c1)*(2*covar+c2))/((mean_x**2+ mean_y**2+c1)*(std_x**2 + std_y**2+c2)))/v_number_of_reps
                    
        y_pred = tf.nn.conv3d(y_pred[0:1, :,:,:,0:1], gauss_kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        y_true = tf.nn.conv3d(y_true[0:1, :,:,:,0:1], gauss_kernel, strides=[1, 1, 1, 1, 1], padding="SAME")
        v_ssim_endgueltig = (v_ssim * l_weights[i]) + v_ssim_endgueltig
    return (1-v_ssim_endgueltig)/2

def dice_loss(k1, k2, L, filter_size):
    def dice(y_true, y_pred):
        return ssim_loss(y_true, y_pred, k1, k2, L, filter_size)
    return dice
ssim = dice_loss(0.3,0.1,1,10)

#######################################################################################################
