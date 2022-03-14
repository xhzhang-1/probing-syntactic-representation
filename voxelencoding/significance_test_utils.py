def all_parametric_test(all_sub_img, thresh):
    results_root = 'results_zh/ligang_final/results_all_factors/significance_test/parametric_test/group_level/'
    spm_t = (all_sub_img.mean(0))/(all_sub_img.std(0, ddof=1)/np.sqrt(720))
    pvals = stats.t.sf(spm_t, 719)
    mask, num, q = fdr_bh(pvals, 0.05)
    print(num)
    file_name = results_root+'all_720r_'+thresh+'_amount_'+str(amount)+'_ttest_bert7_qvals.mat'
    scio.savemat(file_name, {'qvals':q})


def story_parametric_test(all_sub_img, thresh):
    nums = []
    results_root = 'results_zh/ligang_final/results_all_factors/significance_test/parametric_test/group_level/'
    all_sub_img = all_sub_img.reshape(12, 60, 59412)
    spm_t = (all_sub_img.mean(0))/(all_sub_img.std(0, ddof=1)/np.sqrt(12))
    pvals = stats.t.sf(spm_t, 11)
    for i in range(60):
        p = pvals[i]
        mask, num, q = fdr_bh(p, 0.05)
        #print(num)
        nums.append(num)
        file_name = results_root+'story_'+str(i)+'_'+thresh+'_amount_'+str(amount)+'_ttest_bert7_qvals.mat'
        scio.savemat(file_name, {'qvals':q})
    return nums

def sub_level_60(all_sub_img, subs, permute_times, thresh):
    results_root = 'results_zh/ligang_final/results_all_factors/significance_test/permutation_test/sub_level/'
    leng, n_vox = all_sub_img.shape
    all_sub_img = all_sub_img.reshape(len(subs), int(leng/len(subs)), n_vox)
    for i in range(len(subs)):
        alarm = one_permute(all_sub_img[i].T, '', permute_times)
        p = (alarm+1)/1001
        mask, num, q = fdr_bh(p, 0.05)
        print(num)
        file_name = results_root+'sub_'+subs[i]+'_'+thresh+'_amount_'+str(amount)+'_permute_bert7_qvals.mat'
        scio.savemat(file_name, {'qvals':q})

def story_level_group_test(all_sub_img, permute_times, thresh):
    results_root = 'results_zh/ligang_final/results_all_factors/significance_test/permutation_test/group_level_new/'
    all_sub_img = all_sub_img.reshape(12, 60, 59412)
    for i in range(60):
        all_img = all_sub_img[:, i, :]
        alarm = one_permute(all_img.T, '', permute_times)
        p = (alarm+1)/1001
        mask, num, q = fdr_bh(p, 0.05)
        print(num)
        file_name = results_root+'story_'+str(i)+'_'+thresh+'_amount_'+str(amount)+'_permute_bert7_qvals.mat'
        scio.savemat(file_name, {'qvals':q})

def all_720_sign_flip(all_sub_img, permute_times, thresh):
    alarm_num_record = one_permute(all_sub_img.T, '', permute_times)
    p = (alarm_num_record+1)/permute_times
    mask, num, q = fdr_bh(p, 0.05)
    print(num)
    scio.savemat('results_zh/ligang_final/results_all_factors/significance_test/permutation_test/group_level_new/all_720_'+thresh+'_amount_'+str(amount)+'_permute_bert7_qvals.mat', {'qvals':q})

def sub_average_sign_flip(all_sub_img, permute_times, thresh):
    all_sub_img = all_sub_img.reshape(12, 60, 59412)
    all_sub = all_sub_img.mean(1)
    alarm_num_record = one_permute(all_sub.T, '', permute_times)
    p = (alarm_num_record+1)/permute_times
    mask, num, q = fdr_bh(p, 0.05)
    print(num)
    scio.savemat('results_zh/ligang_final/results_all_factors/significance_test/permutation_test/group_level_new/all_sub_average_'+thresh+'_amount_'+str(amount)+'_permute_bert7_qvals.mat', {'qvals':q})

def random_corrs(random_times, lengths):
    results = []
    for k in tqdm(range(random_times)):
        for i in lengths:
            a = np.random.rand(i, 59412)
            b = np.random.rand(i, 59412)
            results.append((zs(a)*zs(b)).mean(0))
    return np.vstack(results)

def random_response_corrs(responses, lengths):
    results = []
    temp = 0
    for i in lengths:
        for j in range(59412):
            a = np.random.rand(i)
            b = responses[temp:temp+i, j]
            results.append(np.corrcoef(a, b)[0,1])
        temp += i
    return results

def two_sample_permutation(sample1, sample2, permute_times):
    '''
    sample1: (720, 59412); sample2:(720, 59412)
    '''
    n_vox = 59412
    diff = sample1.mean(0)-sample2.mean(0)
    all_samples = np.zeros([1440, n_vox])
    all_samples[:720, :] = sample1
    all_samples[720:, :] = sample2
    all_inds = [i for i in range(1440)]
    alarm_num_record = np.zeros(n_vox)
    for i in range(permute_times):
        random.shuffle(all_inds)
        r_sample1 = all_samples[all_inds[:720], :]
        r_sample2 = all_samples[all_inds[720:], :]
        r_diff = r_sample1.mean(0) - r_sample2.mean(0)
        alarm_num_record += (r_diff > diff)
    return alarm_num_record