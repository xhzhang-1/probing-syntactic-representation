python run_valid_permutation.py \
--cuda0 1 \
--cuda1 2 \
--permute_times 10000 \
--feature_path data/encoding_data/story_bert_layer7/ \
--fmri_path data/encoding_data/fmri_encoding_dataset/ \
--res_root results/results_bert7/significant_voxels/ \
--true_corr_root results/results_bert7/significant_voxels/ \
--compute_true_corr True
